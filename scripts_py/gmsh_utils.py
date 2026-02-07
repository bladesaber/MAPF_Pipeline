import gmsh
import numpy as np


class GmshUtils(object):
    @staticmethod
    def define_gmsh_kernel():
        return '//+\nSetFactory("OpenCASCADE");'
    
    @staticmethod
    def define_gmsh_curve_physical_group(group_name: str, group_id: int, tags: list[int]) -> str:
        assert group_id > 0, "gmsh only supports positive group id"
        return f'Physical Curve(\"{group_name}\", {group_id}) = {{{", ".join([str(tag) for tag in tags])}}};'
    
    @staticmethod
    def define_gmsh_surface_physical_group(group_name: str, group_id: int, tags: list[int]) -> str:
        assert group_id > 0, "gmsh only supports positive group id"
        return f'Physical Surface(\"{group_name}\", {group_id}) = {{{", ".join([str(tag) for tag in tags])}}};'
    
    @staticmethod
    def define_gmsh_all_lines_physical_group(group_name: str, group_id: int) -> str:
        return f'Physical Curve("{group_name}", {group_id}) = Curve{{:}};'
    
    @staticmethod
    def define_gmsh_all_surfaces_physical_group(group_name: str, group_id: int) -> str:
        return f'Physical Surface("{group_name}", {group_id}) = Surface{{:}};'

    @staticmethod
    def define_gmsh_volume_physical_group(group_name: str, group_id: int, tags: list[int]) -> str:
        assert group_id > 0, "gmsh only supports positive group id"
        return f'Physical Volume(\"{group_name}\", {group_id}) = {{{", ".join([str(tag) for tag in tags])}}};'
  
    @staticmethod
    def define_gmsh_point(point_id: int, x: float, y: float, z: float, size: float = 0.1):
        assert point_id > 0, "gmsh only supports positive point id"
        return f'Point({point_id}) = {{{x}, {y}, {z}, {size}}};'
    
    @staticmethod
    def define_gmsh_line(line_id: int, point_id_1: int, point_id_2: int):
        assert line_id > 0, "gmsh only supports positive line id"
        return f'Line({line_id}) = {{{point_id_1}, {point_id_2}}};'
    
    @staticmethod
    def define_gmsh_curve_loop(curve_loop_id: int, line_ids: list[int]):
        assert curve_loop_id > 0, "gmsh only supports positive curve loop id"
        return f'Curve Loop({curve_loop_id}) = {{{", ".join([str(line_id) for line_id in line_ids])}}};'
    
    @staticmethod
    def define_gmsh_surface_from_lines(surface_id: int, line_ids: list[int]):
        assert surface_id > 0, "gmsh only supports positive surface id"
        content = f'Curve Loop({surface_id}) = {{{", ".join([str(line_id) for line_id in line_ids])}}};\n'
        content += f'Plane Surface({surface_id}) = {{{surface_id}}};'
        return content
    
    @staticmethod
    def define_gmsh_surface_from_curves(surface_id: int, curve_ids: list[int]):
        assert surface_id > 0, "gmsh only supports positive surface id"
        return f'Plane Surface({surface_id}) = {{{", ".join([str(curve_id) for curve_id in curve_ids])}}};'
    
    @staticmethod
    def define_gmsh_surface_loop(surface_loop_id: int, surface_ids: list[int]):
        assert surface_loop_id > 0, "gmsh only supports positive surface loop id"
        return f'Surface Loop({surface_loop_id}) = {{{", ".join([str(surface_id) for surface_id in surface_ids])}}};'
    
    @staticmethod
    def define_gmsh_volume_from_surfaces(volume_id: int, surface_ids: list[int]):
        assert volume_id > 0, "gmsh only supports positive volume id"
        content = f'Surface Loop({volume_id}) = {{{", ".join([str(surface_id) for surface_id in surface_ids])}}};\n'
        content += f'Volume({volume_id}) = {{{volume_id}}};'
        return content

    @staticmethod
    def define_gmsh_volume_from_surface_loops(volume_id: int, surface_loop_ids: list[int]):
        assert volume_id > 0, "gmsh only supports positive volume id"
        content = f'Volume({volume_id}) = {{{", ".join([str(surface_loop_id) for surface_loop_id in surface_loop_ids])}}};'
        return content

    @staticmethod
    def define_mesh_size(min_size: float, max_size: float):
        return f'Mesh.MeshSizeMin = {min_size};\nMesh.MeshSizeMax = {max_size};\n'

    @staticmethod
    def define_gmsh_cylinder(volume_id: int, center: np.ndarray, vector: np.ndarray, radius: float):
        return f'Cylinder({volume_id}) = {{{center[0]}, {center[1]}, {center[2]}, {vector[0]}, {vector[1]}, {vector[2]}, {radius}, 2*Pi}};'

    @staticmethod
    def union_volume(source_volume_id: int, union_volume_ids: list[int]):
        head = f'BooleanUnion{{ Volume{{{source_volume_id}}}; Delete; }}'
        union_str = ''
        for volume_id in union_volume_ids:
            union_str += f' Volume{{{volume_id}}};'
        tail = f'{{ {union_str} Delete; }}'
        return head + tail

    @staticmethod
    def define_field_mesh_size_to_curve(
        distance_field_id: int, threshold_field_id: int, curve_tags: list[int], sampling: int,
        size_min: float, size_max: float, dist_min: float, dist_max: float
    ):
        content = [
            f'Field[{distance_field_id}] = Distance;',
            f'Field[{distance_field_id}].CurvesList = {{{", ".join([str(curve_tag) for curve_tag in curve_tags])}}};',
            f'Field[{distance_field_id}].Sampling = {sampling};',
            
            f'Field[{threshold_field_id}] = Threshold;',
            f'Field[{threshold_field_id}].InField = {distance_field_id};',
            f'Field[{threshold_field_id}].SizeMin = {size_min};',
            f'Field[{threshold_field_id}].SizeMax = {size_max};',
            f'Field[{threshold_field_id}].DistMin = {dist_min};',
            f'Field[{threshold_field_id}].DistMax = {dist_max};'
        ]
        return '\n'.join(content)

    @staticmethod
    def define_field_mesh_size_to_surface(
        distance_field_id: int, threshold_field_id: int, surface_tags: list[int], sampling: int,
        size_min: float, size_max: float, dist_min: float, dist_max: float
    ):
        content = [
            f'Field[{distance_field_id}] = Distance;',
            f'Field[{distance_field_id}].SurfacesList = {{{", ".join([str(surface_tag) for surface_tag in surface_tags])}}};',
            f'Field[{distance_field_id}].Sampling = {sampling};',
            
            f'Field[{threshold_field_id}] = Threshold;',
            f'Field[{threshold_field_id}].InField = {distance_field_id};',
            f'Field[{threshold_field_id}].SizeMin = {size_min};',
            f'Field[{threshold_field_id}].SizeMax = {size_max};',
            f'Field[{threshold_field_id}].DistMin = {dist_min};',
            f'Field[{threshold_field_id}].DistMax = {dist_max};'
        ]
        return '\n'.join(content)

    @staticmethod
    def define_field_mesh_size_from_box(
        field_id: int,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
        thickness: float,
        size_min: float, size_max: float,
    ):
        content = [
            f'Field[{field_id}] = Box;',
            f'Field[{field_id}].VIn = {size_min};',
            f'Field[{field_id}].VOut = {size_max};',
            f'Field[{field_id}].XMin = {xmin};',
            f'Field[{field_id}].XMax = {xmax};',
            f'Field[{field_id}].YMin = {ymin};',
            f'Field[{field_id}].YMax = {ymax};',
            f'Field[{field_id}].ZMin = {zmin};',
            f'Field[{field_id}].ZMax = {zmax};',
            f'Field[{field_id}].Thickness = {thickness};',
        ]
        return '\n'.join(content)

    @staticmethod
    def define_minimum_field(aggregate_field_id: int, field_ids: list[int]):
        content = [
            f'Field[{aggregate_field_id}] = Min;',
            f'Field[{aggregate_field_id}].FieldsList = {{{", ".join([str(field_id) for field_id in field_ids])}}};',
            f'Background Field = {aggregate_field_id};'
        ]
        return '\n'.join(content)

    @staticmethod
    def read_geometry_from_geo_file(geo_file: str, dims: list[int] = [0, 1, 2, 3]):
        gmsh.initialize()
        gmsh.merge(geo_file)
        
        dim_cfg = {}
        for dim in dims:
            entity_dict = {}
            for entity_dim, entity_tag in gmsh.model.getEntities(dim):
                entity_item = {
                    'dim': entity_dim, 
                    'tag': entity_tag,
                    'type': gmsh.model.getType(entity_dim, entity_tag),
                }
                
                if gmsh.model.getEntityName(entity_dim, entity_tag) == '':
                    gmsh.model.setEntityName(entity_dim, entity_tag, f'dim{entity_dim}_id{entity_tag}')
                entity_item['name'] = gmsh.model.getEntityName(entity_dim, entity_tag)
                
                x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(entity_dim, entity_tag)
                if dim == 0:
                    entity_item.update({'x': x_min, 'y': y_min, 'z': z_min})
                else:
                    entity_item.update({'x_min': x_min, 'y_min': y_min, 'z_min': z_min, 'x_max': x_max, 'y_max': y_max, 'z_max': z_max})
                
                up_tags, down_tags = gmsh.model.getAdjacencies(entity_dim, entity_tag)
                entity_item.update({'up_tags': up_tags, 'down_tags': down_tags})
                
                if dim == 2:
                    (u_min, u_max), (v_min, v_max) = gmsh.model.getParametrizationBounds(entity_dim, entity_tag)
                    u_mid = (u_min + u_max) / 2
                    v_mid = (v_min + v_max) / 2
                    normal_vector = gmsh.model.getNormal(entity_tag, [u_mid, v_mid])
                    entity_item.update({'normal_vector': normal_vector})
                
                if dim > 0:
                    entity_item.update({'center': gmsh.model.occ.getCenterOfMass(entity_dim, entity_tag)})
                
                entity_dict[entity_tag] = entity_item
            dim_cfg[dim] = entity_dict
        gmsh.finalize()
        
        return dim_cfg

    @staticmethod
    def compress_extend_from_boundary():
        return f'Mesh.MeshSizeExtendFromBoundary = 0;'