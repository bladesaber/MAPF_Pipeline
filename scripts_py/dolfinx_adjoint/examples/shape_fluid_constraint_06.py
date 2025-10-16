import dolfinx
import ufl
from mpi4py import MPI

import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_adjoint.shape_fluid_utils.shape_fluid import (
    FluidShapeOptimizer,
    FluidPipeBoundary,
    FluidCost,
    ShapeRegularizationItem,
)
from scripts_py.dolfinx_adjoint.shape_fluid_utils.gradient_merger import ElasticShapeMerger
from scripts_py.remesh_utils import ReMesher


msh_file = "/home/admin123456/Desktop/work/gmsh_test/t6/shape_fluid_05.msh"
xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t6/shape_fluid_05.xdmf"
# MeshUtils.msh_to_XDMF(name='model', msh_file=msh_file, xdmf_file=xdmf_file, dim=2)

# --- 1. define mesh ---
mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")

# --- 2. define cost and regularization ---
cost_item = FluidCost(
    name="cost_1",
    style="MinDissipation",
    weight=1.0,
)
regularization_item = ShapeRegularizationItem(
    name="regularization_1", 
    style="Volume", 
    weight=0.5, 
    target_ratio=0.7
)

optim = FluidShapeOptimizer(
    name="shape_fluid_05",
    mesh=mesh,
    fluid_type="stoke",
    nut_init=1.0 / 400.0,
    cost_item=cost_item,
    regularization_item=regularization_item,
    with_vol_check=True,
    vol_change=0.25,
    with_angle_check=True,
    angle_tol_max=160.0,
    angle_tol_min=20.0,
    result_dir="/home/admin123456/Desktop/work/gmsh_test/t6/cache",
    scalar_product_kwargs={"method": "default"},
)

# --- 3. define boundary ---
inlet_bry = FluidPipeBoundary(
    name="inlet",
    boundary_type="velocity",
    value=[1.0, 0.0],
    marker=35,
    is_free=False,
    face_center=[0.0, 0.25],
    face_norm_vector=[1.0, 0.0],
    face_radius=0.25,
)
outlet_bry = FluidPipeBoundary(
    name="outlet", boundary_type="pressure", value=0.0, marker=36, is_free=False
)
wall_bry = FluidPipeBoundary(
    name="wall", boundary_type="wall", value=0.0, marker=37, is_free=True
)

optim.add_boundary(inlet_bry)
optim.add_boundary(outlet_bry)
optim.add_boundary(wall_bry)

# --- 4. define problem ---
optim.define_problem()

# --- 5. define obstacle ---
obstacle_mesh = pyvista.read("/home/admin123456/Desktop/work/gmsh_test/t6/shape_fluid_05_obstacle.stl")
obstacle_pcd = np.array(obstacle_mesh.points)
obstacle_radius = 0.1

# obstacle_pcd = None
# obstacle_radius = 0.0

# --- 6. define gradient merger ---
merger = ElasticShapeMerger(mesh=mesh)
merger.add_control_boundary(name=inlet_bry.name, boundary=inlet_bry)
merger.add_control_boundary(name=outlet_bry.name, boundary=outlet_bry)
merger.add_control_boundary(name=wall_bry.name, boundary=wall_bry)
merger.define_dofs()

# merger.draw_boundary()

remesher = ReMesher(mesh=mesh, surface_marker=[inlet_bry.marker, outlet_bry.marker, wall_bry.marker])
# remesher.draw_boundary()

# --- 7. run optimization ---
def run(
    line_search_steps: list[float] = [0.1, 0.05, 0.02, 0.01],
    stiff_cofs: list[float] = [400.0, 200.0, 100.0, 50.0],
    state_kwargs: dict = {"mat_assemble_method": "lift", "with_residual": False},
    adjoint_kwargs: dict = {"mat_assemble_method": "lift", "with_residual": False},
    gradient_kwargs: dict = {
        "mat_assemble_method": "identity_row",
        "with_residual": False,
    },
    scalar_product_kwargs: dict = {
        "mat_assemble_method": "lift",
        "with_residual": False,
    },
    max_iteration=100,
    with_gradient_visualization=False,
    with_remesh_visualization=False,
):
    is_solved, step = False, 0
    loss_current = 0.0
    loss_last = 0.0
    loss_dict_current = {}
    line_search_steps_np: np.ndarray = np.array(line_search_steps)
    step_length_last = line_search_steps_np[0]

    loss_current = optim.optimizer.evaluate_cost_functional(
        optim.mesh.domain.comm, update_state=True
    )
    optim.record_Upw(step)
    print(f"step/{step} loss:{loss_current}.")

    while True:
        step += 1
        if step > 1:
            loss_residual = loss_current - loss_last
            log = f"step/{step} loss:{loss_current} residual:{loss_residual:.6f} last_step_length:{step_length_last}."
            for name, loss_value in loss_dict_current.items():
                log += f" {name}:{loss_value:.3f}"
            print(log)

            if np.abs(loss_residual) < 1e-6:
                is_solved = True
                break

        report_dict = optim.compute_gradient(
            update_state=False,
            update_adjoint=True,
            state_kwargs=state_kwargs,
            adjoint_kwargs=adjoint_kwargs,
            gradient_kwargs=gradient_kwargs,
            scalar_product_kwargs=scalar_product_kwargs,
            with_norm=False,  # should be False in order to gradient merge
        )
        is_success = report_dict["is_success"]
        if not is_success:
            print(f'[Error]: {report_dict["log"]}')
            is_solved = False
            break

        shape_grad = report_dict["direction"] * -1.0  # must be multiplied by -1.0
        merger.update_grad(key='shape_grad', grad=shape_grad)

        find_best_step = False
        last_step_index = np.where(line_search_steps_np == step_length_last)[0][0]
        line_search_start_index = np.max(last_step_index - 1, 0)
        for i in range(line_search_start_index, line_search_steps_np.shape[0]):
            step_length = line_search_steps_np[i]
            
            merger_dict = merger.linear_elastic_obstacle_displacement(
                length_step=step_length + line_search_steps_np[-1],
                obs_pcd=obstacle_pcd,
                obs_radius=obstacle_radius,
                stiff_cof=stiff_cofs[i],
                lambda_v = 1.25,
                mu = 1.0,
                with_residual=False,
            )
            if not merger_dict['is_success']:
                print(f'[Error]: {merger_dict["log"]}')
                continue
            
            if merger_dict['is_obstacle_conflict'] and with_gradient_visualization:
                merger.draw_gradient(
                    obs_pcd=obstacle_pcd, 
                    obs_radius=obstacle_radius, 
                    length_step=step_length,
                    with_shape_gradient=True,
                    with_obstacle=True,
                    with_obstacle_gradient=True,
                    shape_gradient_mag=0.1 / step_length,
                    obs_gradient_mag=0.1 / step_length,
                )
            
            displacement_np = merger_dict['displacement']
            if displacement_np.shape[1] == 2:
                displacement_np = np.concatenate(
                    [displacement_np, np.zeros((displacement_np.shape[0], 1))], axis=1
                )

            deform_check = optim.mesh.deform_safe(displacement_np)
            if not deform_check["is_valid"]:
                print(deform_check["log"])
                continue

            loss_dict_current = optim.optimizer.get_cost_info(
                update_state=True,
                comm=optim.mesh.domain.comm,
                **state_kwargs,
            )
            loss_new = np.sum(list(loss_dict_current.values()))

            if loss_new < loss_current:
                find_best_step = True
                
                remesh_dict = remesher.inner_laplacian_smoothing(max_iteration=1000, tol=1e-4)
                if remesh_dict['is_success']:
                    remesher.reset_mesh(remesh_dict['coord'])
                    if with_remesh_visualization:
                        remesher.draw_boundary()
                
                loss_last = loss_current
                loss_current = loss_new
                step_length_last = step_length
                optim.record_Upw(step)
                break
            else:
                optim.mesh.deform(displacement_np * -1.0)  # recover mesh

        if not find_best_step:
            print(f"[Warning]: line search fail.")
            is_solved = False
            break

        if step > max_iteration:
            print(f"[Warning]: max iteration reached.")
            is_solved = False
            break

    return {"is_success": is_solved}

run(max_iteration=200, with_gradient_visualization=False, with_remesh_visualization=False)
