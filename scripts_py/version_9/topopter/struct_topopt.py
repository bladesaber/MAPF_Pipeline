import numpy as np
import dolfinx
import ufl
import pyvista
from dolfinx.fem.petsc import LinearProblem
from sklearn.neighbors import KDTree
from scipy import sparse
from tqdm import tqdm
import numba
from mpi4py import MPI
from line_profiler import profile
import time

from dolfinx_utils import DolfinxUtils


class StructTopOpt(object):
    @staticmethod
    def epsilon(u):
        # return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        return ufl.sym(ufl.grad(u))

    @staticmethod
    def sigma(u, lambda_par, mu_par):
        return lambda_par * ufl.tr(StructTopOpt.epsilon(u)) * ufl.Identity(len(u)) + \
            2 * mu_par * StructTopOpt.epsilon(u)

    @staticmethod
    def psi(u, lambda_par, mu_par):
        return 0.5 * lambda_par * (ufl.tr(StructTopOpt.epsilon(u)) ** 2) + \
            mu_par * ufl.tr(StructTopOpt.epsilon(u) * StructTopOpt.epsilon(u))

    @staticmethod
    def displacement_simulate(
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            bc_dict: dict, simulate_params: dict
    ):
        V = dolfinx.fem.VectorFunctionSpace(domain, element=('Lagrange', 1))

        dirichlet_bcs, neuuman_bcs = [], {}
        for name in bc_dict.keys():
            bc_info = bc_dict[name]

            if bc_info['type'] == 'dirichlet':
                if bc_info['method'] == 'marker':
                    bc = DolfinxUtils.define_dirichlet_boundary_marker(
                        V, domain, bc_info['marker'], facet_tags, bc_info['value']
                    )
                else:
                    bc = DolfinxUtils.define_dirichlet_boundary_from_fun(
                        V, domain, bc_info['value'], bc_info['function']
                    )
                dirichlet_bcs.append(bc)

            elif bc_info['type'] == 'neuuman':
                if bc_info['method'] == 'marker':
                    facet_indices = DolfinxUtils.define_neuuman_boundary_from_marker(bc_info['marker'], facet_tags)
                    marker = bc_info['marker']
                else:
                    facet_indices = DolfinxUtils.define_neuuman_boundary_from_fun(domain, bc_info['function'])
                    marker = len(neuuman_bcs) + 1
                neuuman_bcs[name] = {
                    'indices': facet_indices,
                    'marker': marker,
                    'vector': bc_info['value']
                }

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a_fun = ufl.inner(
            StructTopOpt.sigma(u, simulate_params['lambda'], simulate_params['mu']), StructTopOpt.epsilon(v)
        ) * ufl.dx

        f = dolfinx.fem.Constant(domain, simulate_params['gravity'])
        L_fun = ufl.dot(f, v) * ufl.dx

        if len(neuuman_bcs) > 0:
            facets_tag = DolfinxUtils.define_subdomain_facets_tag(domain, neuuman_bcs)
            ds = DolfinxUtils.define_boundary_area(domain, facets_tag)

            for name in neuuman_bcs.keys():
                bc_info = neuuman_bcs[name]
                force = dolfinx.fem.Constant(domain, bc_info['vector'])
                L_fun += ufl.dot(force, v) * ds(bc_info['marker'])

        uh = DolfinxUtils.solve_finite_element_problem(a_fun, L_fun, bcs=dirichlet_bcs)
        return uh

    @staticmethod
    # @profile
    def method_simp_opt(
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            bc_dict: dict, simulate_params: dict, opt_params: dict
    ):
        V = dolfinx.fem.VectorFunctionSpace(domain, element=('Lagrange', 1))
        # be careful here, DG is discontinued lagrange, function shape of different functionSpace is different
        Q = dolfinx.fem.FunctionSpace(domain, element=("DG", 0))

        dirichlet_bcs, neuuman_bcs = [], {}
        for name in bc_dict.keys():
            bc_info = bc_dict[name]

            if bc_info['type'] == 'dirichlet':
                if bc_info['method'] == 'marker':
                    bc = DolfinxUtils.define_dirichlet_boundary_marker(
                        V, domain, bc_info['marker'], facet_tags, bc_info['value']
                    )
                else:
                    bc = DolfinxUtils.define_dirichlet_boundary_from_fun(
                        V, domain, bc_info['value'], bc_info['function']
                    )
                dirichlet_bcs.append(bc)

            elif bc_info['type'] == 'neuuman':
                if bc_info['method'] == 'marker':
                    facet_indices = DolfinxUtils.define_neuuman_boundary_from_marker(bc_info['marker'], facet_tags)
                    marker = bc_info['marker']
                else:
                    facet_indices = DolfinxUtils.define_neuuman_boundary_from_fun(domain, bc_info['function'])
                    marker = len(neuuman_bcs) + 1
                neuuman_bcs[name] = {
                    'indices': facet_indices,
                    'marker': marker,
                    'vector': bc_info['value']
                }

        # ------ define function variables
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        density = dolfinx.fem.Function(Q)
        density.name = 'density'
        density_old = dolfinx.fem.Function(Q)
        density_new = dolfinx.fem.Function(Q)
        density.x.array[:] = opt_params['volfrac']
        # density.x.array[:] = 1.0

        # -------- define L function
        f = dolfinx.fem.Constant(domain, simulate_params['gravity'])
        L_fun = ufl.dot(f, v) * ufl.dx
        if len(neuuman_bcs) > 0:
            facets_tag = DolfinxUtils.define_subdomain_facets_tag(domain, neuuman_bcs)
            ds = DolfinxUtils.define_boundary_area(domain, facets_tag)

            for name in neuuman_bcs.keys():
                bc_info = neuuman_bcs[name]
                force = dolfinx.fem.Constant(domain, bc_info['vector'])
                L_fun += ufl.dot(force, v) * ds(bc_info['marker'])

        # -------- define a function
        a_fun = ufl.inner(
            density ** opt_params['penal'] *
            StructTopOpt.sigma(u, simulate_params['lambda'], simulate_params['mu']),
            StructTopOpt.epsilon(v)
        ) * ufl.dx

        # ------ define distance matrix
        domain_grid = DolfinxUtils.convert_to_grid(domain)
        cell_centers = domain_grid.cell_centers().points
        center_tree = KDTree(cell_centers, metric='minkowski')
        neighbour_idxs, dists = center_tree.query_radius(cell_centers, r=opt_params['rmin'], return_distance=True)
        row_idxs = []
        for row_id, ng_idxs in tqdm(enumerate(neighbour_idxs)):
            row_idxs.append(np.full(shape=(ng_idxs.shape[0],), fill_value=row_id))

        row_idxs = np.concatenate(row_idxs, axis=-1).reshape(-1)
        col_idxs = np.concatenate(neighbour_idxs, axis=-1).reshape(-1)
        dists = opt_params['rmin'] - np.concatenate(dists, axis=-1).reshape(-1)
        cells_num = cell_centers.shape[0]

        dist_mat = sparse.coo_matrix((dists, (row_idxs, col_idxs)), shape=(cells_num, cells_num))
        distSum_mat = dist_mat.sum(axis=1).reshape((-1, 1))

        # ------ define problem
        problem = LinearProblem(a_fun, L_fun, bcs=dirichlet_bcs, petsc_options={
            "ksp_type": "preonly", "pc_type": "lu"
        })
        volume_orig = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.TestFunction(Q) * ufl.dx))
        # volume_orig = DolfinxUtils.compute_integral_scalar(density * ufl.dx)

        # ------ begin to record and run
        xdmf_writer = dolfinx.io.XDMFFile(domain.comm, opt_params['output_file'], 'w')
        xdmf_writer.write_mesh(domain)

        for loop_step in tqdm(range(100)):
            density_old.x.array[:] = density.x.array
            uh = problem.solve()

            objective = density ** opt_params['penal'] * StructTopOpt.psi(
                uh, simulate_params['lambda'], simulate_params['mu']
            )
            ufl_expr = -ufl.diff(objective, density)
            object_grad_fun = DolfinxUtils.define_interpolate_fun(Q, ufl_expr)
            grad_np = object_grad_fun.x.array[:].reshape((-1, 1))
            density_np = density.x.array[:].reshape((-1, 1))
            sensitivity_np: np.matrix = np.divide(
                dist_mat.dot(grad_np * density_np),
                np.multiply(density_np, distSum_mat)
            )

            sensitivity_np = np.asarray(sensitivity_np).reshape(-1)
            density_np = density_np.reshape(-1)

            l1, l2, move = 0, 100000, 0.2
            current_vol = np.inf
            while l2 - l1 > 1e-4:
                l_mid = 0.5 * (l2 + l1)

                update_density = density_np * np.sqrt(-sensitivity_np / volume_orig / l_mid)
                update_density = np.minimum(density_np + move, np.maximum(density_np - move, update_density))
                update_density = np.minimum(1.0, np.maximum(1e-4, update_density))
                density_new.x.array[:] = update_density
                current_vol = DolfinxUtils.compute_integral_scalar(density_new * ufl.dx)

                if current_vol > opt_params['volfrac'] * volume_orig:
                    l1, l2 = l_mid, l2
                else:
                    l1, l2 = l1, l_mid

            density.x.array[:] = density_new.x.array[:]
            change = np.max(np.abs(density.x.array - density_old.x.array))
            print(f"[Debug] iter:{loop_step} volume:{current_vol:.2f}/{volume_orig:.2f} "
                  f"{current_vol / volume_orig:.2f} ")
            xdmf_writer.write_function(density, loop_step)

            if change < 0.01:
                break

    @staticmethod
    def show_displacement(domain: dolfinx.mesh.Mesh, uh: dolfinx.fem.function, dim, with_wrap=False):
        plotter = pyvista.Plotter()
        grid = DolfinxUtils.convert_to_grid(domain)

        if with_wrap:
            if dim == 2:
                u_value = uh.x.array.reshape((-1, dim))
                grid['u'] = np.concatenate([u_value, np.zeros(shape=(u_value.shape[0], 1))], axis=1)
            else:
                grid['u'] = uh.x.array.reshape((-1, dim))

            actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
            warped = grid.warp_by_vector("u", factor=1.0)
            actor_1 = plotter.add_mesh(warped, show_edges=True)
        else:
            grid.point_data['u'] = uh.x.array.reshape((-1, dim))
            grid.set_active_scalars('u')
            plotter.add_mesh(grid, show_edges=True)

        plotter.show_axes()
        plotter.show()


def model2D_test():
    simulate_params = {
        'lambda': 1.25,
        'mu': 1.0,
        'rho': 1,  # density of material
        'g': 0.045,  # acceleration of gravity,
        'gravity': np.array([0., 0.])  # rho * g
    }

    bc_dict = {
        # 'load0': {
        #     'method': 'marker',
        #     'type': 'neuuman',
        #     'marker': 15,
        #     'value': np.array([0., -1.0])
        # },
        'load0': {
            'method': 'function',
            'type': 'neuuman',
            'function': lambda x: np.logical_and(np.isclose(x[0], 60.0), x[1] < 2.0),
            'value': np.array([0., -1.0])
        },
        'support0': {
            'method': 'marker',
            'type': 'dirichlet',
            'marker': 5,
            'value': np.array([0., 0.])
        }
    }

    domain, cell_tags, facet_tags = DolfinxUtils.read_XDMF(
        file='/home/admin123456/Desktop/work/test/2D_top/t1.xdmf',
        cellTag_name='t1_cells',
        facetTag_name='t1_facets'
    )

    # uh = StructTopOpt.displacement_simulate(domain, cell_tags, facet_tags, bc_dict, simulate_params)
    # StructTopOpt.show_displacement(domain, uh, dim=2, with_wrap=False)

    # opt_params = {
    #     'volfrac': 0.5,
    #     'penal': 3.0,
    #     'rmin': 2.0,
    #     'output_file': '/home/admin123456/Desktop/work/test/2D_top/opt.xdmf'
    # }
    # StructTopOpt.method_simp_opt(domain, cell_tags, facet_tags, bc_dict, simulate_params, opt_params)


def model3D_test():
    simulate_params = {
        'lambda': 1.25,
        'mu': 1.0,
        'rho': 1,  # density of material
        'g': 0.045,  # acceleration of gravity,
        'gravity': np.array([0., 0., 0.])  # rho * g
    }

    bc_dict = {
        # 'load0': {
        #     'method': 'marker',
        #     'type': 'neuuman',
        #     'marker': 21,
        #     'value': np.array([0., 0., -1.0])
        # },
        'load0': {
            'method': 'function',
            'type': 'neuuman',
            'function': lambda x: np.logical_and(
                x[2] < 1.0, x[0] > 59.0,
            ),
            'value': np.array([0., 0.0, -1.0])
        },
        'support0': {
            'method': 'marker',
            'type': 'dirichlet',
            'marker': 14,
            'value': np.array([0., 0., 0.])
        }
    }

    domain, cell_tags, facet_tags = DolfinxUtils.read_XDMF(
        file='/home/admin123456/Desktop/work/test/3D_top/t1.xdmf',
        cellTag_name='t1_cells',
        facetTag_name='t1_facets'
    )

    # uh = StructTopOpt.displacement_simulate(domain, cell_tags, facet_tags, bc_dict, simulate_params)
    # StructTopOpt.show_displacement(domain, uh, dim=3, with_wrap=False)

    opt_params = {
        'volfrac': 0.5,
        'penal': 3.0,
        'rmin': 2.0,
        'output_file': '/home/admin123456/Desktop/work/test/3D_top/opt.xdmf'
    }

    StructTopOpt.method_simp_opt(domain, cell_tags, facet_tags, bc_dict, simulate_params, opt_params)


def model3D_test2():
    simulate_params = {
        'lambda': 1.25,
        'mu': 1.0,
        'rho': 1,  # density of material
        'g': 0.045,  # acceleration of gravity,
        'gravity': np.array([0., 0., 0.])  # rho * g
        # 'gravity': np.array([0., 0., -0.045])
    }

    bc_dict = {
        'load0': {
            'method': 'function',
            'type': 'neuuman',
            'function': lambda x: np.logical_or(
                np.logical_and(np.logical_and(x[0] > 59.0, x[1] < 2.0), x[2] < 2.0),
                np.logical_and(np.logical_and(x[0] > 59.0, x[1] > 18.0), x[2] < 2.0)
            ),
            'value': np.array([0., 0.0, -5.0])
        },
        'support0': {
            'method': 'marker',
            'type': 'dirichlet',
            'marker': 13,
            'value': np.array([0., 0., 0.])
        }
    }

    domain, cell_tags, facet_tags = DolfinxUtils.read_XDMF(
        file='/home/admin123456/Desktop/work/test/3D_top2/t2.xdmf',
        mesh_name='t2',
        cellTag_name='t2_cells',
        facetTag_name='t2_facets'
    )

    # uh = StructTopOpt.displacement_simulate(domain, cell_tags, facet_tags, bc_dict, simulate_params)
    # StructTopOpt.show_displacement(domain, uh, dim=3, with_wrap=True)

    opt_params = {
        'volfrac': 0.5,
        'penal': 3.0,
        'rmin': 2.0,
        'output_file': '/home/admin123456/Desktop/work/test/3D_top2/opt.xdmf'
    }
    StructTopOpt.method_simp_opt(domain, cell_tags, facet_tags, bc_dict, simulate_params, opt_params)


if __name__ == '__main__':
    # model2D_test()
    # model3D_test()
    model3D_test2()