import { 
  add_mesh, remove_mesh, get_mesh_from_name, 
  create_sphere, create_disk, create_torus_mesh
} from '../utils/visulize_utils'
import * as THREE from 'three'

export class BaseGeometryObj {
  name: string
  mesh: THREE.Mesh | null = null

  hooks: string[] = []
  layer: number = -1
  style: string = 'base_geometry'

  constructor(name: string) {
    this.name = name
  }

  add_mesh(scene: THREE.Scene) {
    throw new Error('Not implemented')
  }

  remove_mesh(scene: THREE.Scene) {
    if (get_mesh_from_name(scene, this.name)) {
      remove_mesh(scene, this.name)
    }
  }

  add_var(var_obj: BaseGeometryObj, index: number = -1) {
    throw new Error('Not implemented')
  }

  remove_var(name: string) {
    throw new Error('Not implemented')
  }

  destory(scene: THREE.Scene) {
    this.remove_mesh(scene)
    if (this.mesh) {
      this.mesh.geometry.dispose()
      this.mesh = null
    }
  }

  update_mesh(scene: THREE.Scene) {
    this.destory(scene)
    this.add_mesh(scene)
  }

  hide_mesh() {
    if (this.mesh) {
      this.mesh.visible = false
    }
  }

  show_mesh() {
    if (this.mesh) {
      this.mesh.visible = true
    }
  }

  update_hooks(hooks_name: string[], clear: boolean = false) {
    if (clear) {
      this.hooks = []
    }
    this.hooks.push(...hooks_name)
  }

  toJSON() {
    return {
      name: this.name,
      style: this.style
    }
  }
}

export class PointObj extends BaseGeometryObj {
  x: number
  y: number
  z: number
  radius: number
  style: string = 'point'
  layer: number = 0

  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    radius: number,
    scene: THREE.Scene
  ) {
    super(name)
    this.x = x
    this.y = y
    this.z = z
    this.radius = radius
    this.add_mesh(scene)
  }

  add_mesh(scene: THREE.Scene) {
    this.mesh = create_sphere(this.x, this.y, this.z, this.radius)
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
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
  layer: number = 0
  
  constructor(
    name: string,
    x: number,
    y: number,
    z: number,
    radius: number,
    vec_x: number,
    vec_y: number,
    vec_z: number,
    scene: THREE.Scene
  ) {
    super(name, x, y, z, radius, scene)
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