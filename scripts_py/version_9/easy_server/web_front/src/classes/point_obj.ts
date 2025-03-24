import { add_mesh, remove_mesh, create_sphere} from '../utils/visulize_utils'
import { BaseGeometryObj, BaseNode, GroupNode } from './base_obj.ts'
import { RootObj } from './store_obj.ts'
import * as THREE from "three";

export class PointObj extends BaseGeometryObj {
  x: number
  y: number
  z: number
  style: string = 'point'

  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    node_style: BaseNode|GroupNode,
    rootObj: RootObj
  ) {
    super(name, node_style)
    this.x = x
    this.y = y
    this.z = z
    this.add_mesh(rootObj)
  }

  add_mesh(rootObj: RootObj) {
    this.mesh = create_sphere(this.x, this.y, this.z, 0.2)
    this.mesh.name = this.name
    add_mesh(rootObj.get_scene(), this.mesh)
  }

  update_mesh(rootObj: RootObj) {
    this.mesh?.position.set(this.x, this.y, this.z)
  }

  toJSON() {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      style: this.style
    }
  }
}

export class VecPointObj extends PointObj {
  vec_x: number
  vec_y: number
  vec_z: number
  style: string = 'vec_point'
  
  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    vec_x: number,
    vec_y: number,
    vec_z: number,
    node_style: BaseNode|GroupNode,
    rootObj: RootObj
  ) {
    super(name, x, y, z, node_style, rootObj)
    this.vec_x = vec_x
    this.vec_y = vec_y
    this.vec_z = vec_z
  }

  toJSON() {
    return {
      name: this.name,
      style: this.style,
      x: this.x,
      y: this.y,
      z: this.z,
      vec_x: this.vec_x,
      vec_y: this.vec_y,
      vec_z: this.vec_z
    }
  }
}

export class VecRadiusPointObj extends VecPointObj {
  radius: number
  style: string = 'vec_radius_point'

  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    vec_x: number,
    vec_y: number,
    vec_z: number,
    radius: number,
    node_style: BaseNode|GroupNode,
    rootObj: RootObj
  ) {
    super(name, x, y, z, vec_x, vec_y, vec_z, node_style, rootObj)
    this.radius = radius
  }
  
}