import { BaseNode, GroupNode, GroupGeometryObj } from "./base_obj.ts";
import { RootObj } from "./store_obj.ts";
import { PointObj } from "./point_obj.ts";
import * as THREE from 'three'
import { BsplineObj, TorusObj } from "./line_obj.ts";
import { sample_bspline, compute_path_tangent, sample_surface_points_from_circles, interpolate_by_bspline } from "@/utils/math_utils";

export class SplineSkeletonObj extends GroupGeometryObj {
  radius: number[]
  ts: number[]
  parent_nodes: Map<string, BaseNode|GroupNode> = new Map()
  style: string = 'spline_skeleton'

  constructor(
    name: string, bspline_node: BaseNode|GroupNode, radius: number[], ts: number[], 
    node_style: BaseNode|GroupNode, rootObj: RootObj
  ) {
    super(name, node_style)
    this.parent_nodes.set('bspline_node', bspline_node)
    this.radius = radius
    this.ts = ts
    this.add_mesh(rootObj)
  }
  
  add_mesh(rootObj: RootObj) {
    let bspline_obj = rootObj.request_obj(this.parent_nodes.get('bspline_node') as BaseNode|GroupNode) as BsplineObj
    if (!bspline_obj) {
        throw new Error('BsplineObj not found')
        return
    }
    
    const control_point_objs: PointObj[] = []
    bspline_obj.control_point_nodes.forEach(node => {
        let point_obj = rootObj.request_obj(node) as PointObj
        if (!point_obj) {
            throw new Error('PointObj not found')
            return
        }
        control_point_objs.push(point_obj)
    })

    const control_points = control_point_objs.map(point => [point.x, point.y, point.z])
    const degree = bspline_obj.degree
    const knots = bspline_obj.knots

    const points = sample_bspline(control_points, degree, knots, this.ts)
    const tangents = compute_path_tangent(points)

    points.forEach((point, i) => {
      const torus_name = `${this.name}/skeleton_torus_${i}`
      const torus = new TorusObj(
        torus_name,
        point[0], point[1], point[2], this.radius[i], 0.05, 
        tangents[i][0], tangents[i][1], tangents[i][2], 
        new GroupNode(torus_name, this.name), 
        rootObj
      );
      torus.name = torus_name;
      this.mesh_objs.set(torus.name, torus)
    })

    this.mesh_objs.forEach(mesh => {
        mesh.add_mesh(rootObj)
    })
    this.update_hooks()
  }

  update_hooks() {
    const hook_names: string[] = []
    this.parent_nodes.forEach((node) => {
      if (node.style === 'BaseNode') {
        hook_names.push(node.name)
      } else if (node.style === 'GroupNode') {
        hook_names.push((node as GroupNode).group_name)
      }
    })
    super.update_hooks(hook_names, true);
  }
}

export class CircleSurfacePointsObj extends GroupGeometryObj {
  parent_nodes: (BaseNode|GroupNode)[] = []
  sample_num: number[]
  style: string = 'circle_surface_points'
  constructor(
    name: string, torus_nodes: (BaseNode|GroupNode)[], sample_num: number[], 
    node_style: BaseNode|GroupNode, rootObj: RootObj) {
    super(name, node_style)
    this.parent_nodes = torus_nodes
    this.sample_num = sample_num
    this.add_mesh(rootObj)
  }

  add_mesh(rootObj: RootObj) {
    const torus_objs: TorusObj[] = []
    this.parent_nodes.forEach(node => {
      const torus_obj = rootObj.request_obj(node) as TorusObj
      if (torus_obj) {
        torus_objs.push(torus_obj)
      }
    })

    const circle_center = torus_objs.map(obj => [obj.x, obj.y, obj.z])
    const circle_normal = torus_objs.map(obj => [obj.vec_x, obj.vec_y, obj.vec_z])
    const circle_radius = torus_objs.map(obj => obj.radius)

    const points = sample_surface_points_from_circles(circle_center, circle_normal, circle_radius, this.sample_num)
    points.forEach((point, i) => {
      const point_name = `${this.name}/surface_point_${i}`
      const point_obj = new PointObj(
        point_name, point[0], point[1], point[2],
        new GroupNode(point_name, this.name), 
        rootObj
      );
      point_obj.name = point_name;
      this.mesh_objs.set(point_obj.name, point_obj)
    })
    
    this.mesh_objs.forEach(mesh => {
      mesh.add_mesh(rootObj)
    })

    const hook_names: string[] = []
    torus_objs.forEach(obj => {
      if (obj.node_style.style === 'BaseNode') {
        hook_names.push(obj.node_style.name)
      } else if (obj.node_style.style === 'GroupNode') {
        hook_names.push((obj.node_style as GroupNode).group_name)
      }
    })
    super.update_hooks(hook_names, true)
  }
}
