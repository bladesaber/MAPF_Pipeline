import { BaseGeometryObj, BaseNode, GroupNode } from './base_obj.ts'
import { PointObj } from "./point_obj.ts";
import * as THREE from "three";
import {
  add_mesh,
  create_multi_lines,
  create_circle,
  create_arc,
  create_bspline,
  create_torus_mesh,
} from "../../utils/visulize_utils.ts";
import { RootObj } from "./store_obj.ts";

export class LineObj extends BaseGeometryObj {
  point_nodes: (BaseNode|GroupNode)[] = [];
  style: string = "line";

  constructor(name: string, points_node: (BaseNode|GroupNode)[], node_style: BaseNode|GroupNode, rootObj: RootObj) {
    super(name, node_style);
    this.point_nodes = points_node;
    this.add_mesh(rootObj);
  }

  add_var(var_obj: PointObj, index: number = -1) {
    //Todo non-finish
    // if (var_obj.node_style.style === 'BaseNode') {
    //   this.point_nodes.splice(index, 0, var_obj.node_style)
    // } else if (var_obj.node_style.style === 'GroupNode') {
    //   this.point_nodes.splice(index, 0, var_obj.node_style as GroupNode)
    // } else {
    //   throw new Error(`Unknown node style: ${var_obj.node_style.style}`)
    // }
    // this.update_hooks();
    // after adding a point, we need to update the mesh
    throw new Error("Not implemented");
  }

  remove_var(var_obj: PointObj) {
    // //Todo non-finish
    // if (var_obj.node_style.style === 'BaseNode') {
    //   this.point_nodes = this.point_nodes.filter((node) => node.name !== var_obj.node_style.name)
    // } else if (var_obj.node_style.style === 'GroupNode') {
    //   this.point_nodes = this.point_nodes.filter((node) => node.name !== (var_obj.node_style as GroupNode).group_name)
    // } else {
    //   throw new Error(`Unknown node style: ${var_obj.node_style.style}`)
    // }
    // this.update_hooks();
    // after removing a point, we need to update the mesh
    throw new Error("Not implemented");
  }

  add_mesh(rootObj: RootObj) {
    const point_objs: PointObj[] = []
    for (const node of this.point_nodes) {
      let obj = rootObj.request_obj(node)
      if (obj) {
        point_objs.push(obj as PointObj)
      }
    }
    if (point_objs.length <= 1) {
      throw new Error("LineObj must have at least 2 points")
      return;
    }

    this.mesh = create_multi_lines(
      point_objs.map((point) => [point.x, point.y, point.z]), 
      { output_mesh: true }
    ) as THREE.Mesh;
    this.mesh.name = this.name;
    add_mesh(rootObj.get_scene(), this.mesh);
    
    this.point_nodes = point_objs.map((point) => point.node_style)
    this.update_hooks();
  }

  update_hooks() {
    const hook_names: string[] = []
    this.point_nodes.forEach((node) => {
      if (node.style === 'BaseNode') {
        hook_names.push(node.name)
      } else if (node.style === 'GroupNode') {
        hook_names.push((node as GroupNode).group_name)
      }
    })
    super.update_hooks(hook_names, true);
  }

  toJSON() {
    return {
      name: this.name,
      style: this.style,
    };
  }
}

export class CircleObj extends BaseGeometryObj {
  x: number;
  y: number;
  z: number;
  radius: number;
  vec_x: number;
  vec_y: number;
  vec_z: number;
  style: string = "circle";

  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    radius: number,
    vec_x: number,
    vec_y: number,
    vec_z: number,
    node_style: BaseNode|GroupNode,
    rootObj: RootObj
  ) {
    super(name, node_style);
    this.x = x;
    this.y = y;
    this.z = z;
    this.radius = radius;
    this.vec_x = vec_x;
    this.vec_y = vec_y;
    this.vec_z = vec_z;
    this.add_mesh(rootObj);
  }

  add_mesh(rootObj: RootObj) {
    this.mesh = create_circle(
      this.x, this.y, this.z, this.radius, this.vec_x, this.vec_y, this.vec_z, 
      { output_mesh: true }
    ) as THREE.Mesh;
    this.mesh.name = this.name;
    add_mesh(rootObj.get_scene(), this.mesh);
  }

  toJSON() {
    return {
      name: this.name,
      center: [this.x, this.y, this.z],
      normal: [this.vec_x, this.vec_y, this.vec_z],
      radius: this.radius,
      style: this.style,
    };
  }
}

export class ArcObj extends BaseGeometryObj {
  x: number;
  y: number;
  z: number;
  radius: number;
  vec_x: number;
  vec_y: number;
  vec_z: number;
  start_angle: number;
  end_angle: number;
  style: string = "arc";

  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    radius: number,
    vec_x: number,
    vec_y: number,
    vec_z: number,
    start_angle: number,
    end_angle: number,
    node_style: BaseNode|GroupNode,
    rootObj: RootObj
  ) {
    super(name, node_style);
    this.x = x;
    this.y = y;
    this.z = z;
    this.radius = radius;
    this.vec_x = vec_x;
    this.vec_y = vec_y;
    this.vec_z = vec_z;
    this.start_angle = start_angle;
    this.end_angle = end_angle;
    this.add_mesh(rootObj);
  }

  add_mesh(rootObj: RootObj) {
    this.mesh = create_arc(
      this.x,
      this.y,
      this.z,
      this.radius,
      this.vec_x,
      this.vec_y,
      this.vec_z,
      this.start_angle,
      this.end_angle,
      { output_mesh: true }
    ) as THREE.Mesh;
    this.mesh.name = this.name;
    add_mesh(rootObj.get_scene(), this.mesh);
  }

  toJSON() {
    return {
      name: this.name,
      center: [this.x, this.y, this.z],
      normal: [this.vec_x, this.vec_y, this.vec_z],
      radius: this.radius,
      start_angle: this.start_angle,
      end_angle: this.end_angle,
      style: this.style,
    };
  }
}

export class BsplineObj extends BaseGeometryObj {
  control_point_nodes: (BaseNode|GroupNode)[] = [];
  control_points: number[][] = [];
  knots: number[] = [];
  degree: number = 3;
  sample_num: number = 50;
  style: string = "bspline";

  constructor(
    name: string,
    control_point_nodes: (BaseNode|GroupNode)[],
    knots: number[],
    degree: number,
    node_style: BaseNode|GroupNode,
    rootObj: RootObj,
    sample_num: number = 50
  ) {
    super(name, node_style);
    this.control_point_nodes = control_point_nodes;
    this.knots = knots;
    this.degree = degree;
    this.sample_num = sample_num;
    this.add_mesh(rootObj);
  }

  add_var(point: PointObj, index: number = -1) {
    // Todo non-finish
    throw new Error("Not implemented");
    // if (index === -1) {
    //   this.control_points.push(point);
    // } else {
    //   this.control_points.splice(index, 0, point);
    // }
  }

  remove_var(point: PointObj) {
    // Todo non-finish
    throw new Error("Not implemented");
    // this.control_points = this.control_points.filter((point) => point.name !== name);
  }

  add_mesh(rootObj: RootObj) {
    const control_point_objs: PointObj[] = []
    for (const node of this.control_point_nodes) {
      let obj = rootObj.request_obj(node)
      if (obj) {
        control_point_objs.push(obj as PointObj)
      }
    }
    if (control_point_objs.length < this.degree + 1) {
      throw new Error("BsplineObj must have at least degree + 1 control points");
    }

    this.control_points = control_point_objs.map((point) => [point.x, point.y, point.z])
    this.mesh = create_bspline(
      this.control_points,
      this.knots,
      this.degree,
      this.sample_num,
      { output_mesh: true }
    ) as THREE.Mesh;
    this.mesh.name = this.name;
    add_mesh(rootObj.get_scene(), this.mesh);

    this.control_point_nodes = control_point_objs.map((point) => point.node_style)
    this.update_hooks();
  }

  update_hooks() {
    const hook_names: string[] = []
    this.control_point_nodes.forEach((node) => {
      if (node.style === 'BaseNode') {
        hook_names.push(node.name)
      } else if (node.style === 'GroupNode') {
        hook_names.push((node as GroupNode).group_name)
      }
    })
    super.update_hooks(hook_names, true);
  }

  toJSON() {
    return {
      name: this.name,
      control_points: this.control_points,
      knots: this.knots,
      degree: this.degree,
      style: this.style,
    };
  }
}

export class TorusObj extends BaseGeometryObj {
  x: number;
  y: number;
  z: number;
  radius: number;
  tube_radius: number;
  vec_x: number;
  vec_y: number;
  vec_z: number;
  style: string = "torus";

  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    radius: number,
    tube_radius: number,
    vec_x: number,
    vec_y: number,
    vec_z: number,
    node_style: BaseNode|GroupNode,
    rootObj: RootObj
  ) {
    super(name, node_style);
    this.x = x;
    this.y = y;
    this.z = z;
    this.radius = radius;
    this.tube_radius = tube_radius;
    this.vec_x = vec_x;
    this.vec_y = vec_y;
    this.vec_z = vec_z;
    this.add_mesh(rootObj);
  }

  add_mesh(rootObj: RootObj) {
    this.mesh = create_torus_mesh(
      this.x, this.y, this.z, this.radius, this.tube_radius, this.vec_x, this.vec_y, this.vec_z,
      32, 32, 2 * Math.PI, { output_mesh: true }
    ) as THREE.Mesh;
    this.mesh.name = this.name;
    add_mesh(rootObj.get_scene(), this.mesh);
  }

  toJSON() {
    return {
      name: this.name,
      center: [this.x, this.y, this.z],
      radius: this.radius,
      tube_radius: this.tube_radius,
      normal: [this.vec_x, this.vec_y, this.vec_z],
      style: this.style,
    };
  }
}
