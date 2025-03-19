import { 
  add_mesh, remove_mesh, get_mesh_from_name, 
  create_multi_lines, create_sphere, create_circle, create_arc, create_bspline, create_disk, create_torus_mesh
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

export class LineObj extends BaseGeometryObj {
  points: PointObj[] = []
  style: string = 'line'
  layer: number = 1

  constructor(name: string, points: PointObj[], scene: THREE.Scene) {
    super(name)
    this.points = points
    this.add_mesh(scene)
    this.update_hooks()
  }

  add_var(var_obj: PointObj, index: number = -1) {
    if (index === -1) {
      this.points.push(var_obj)
    } else {
      this.points.splice(index, 0, var_obj)
    }
    // after adding a point, we need to update the mesh
  }

  remove_var(name: string) {
    this.points = this.points.filter(point => point.name !== name)
    // after removing a point, we need to update the mesh
  }

  add_mesh(scene: THREE.Scene) {
    if (this.points.length <= 1) {
      throw new Error('LineObj must have at least 2 points')
    }

    this.mesh = create_multi_lines(this.points.map(point => [point.x, point.y, point.z]), {output_mesh: true}) as THREE.Mesh
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
    this.update_hooks()
  }

  update_hooks() {
    let hook_name = this.points.map(point => point.name)
    super.update_hooks(hook_name, true)
  }

  toJSON() {
    return {
      name: this.name,
      points: this.points.map(point => point.toJSON()),
      style: this.style
    }
  }
}

export class CircleObj extends BaseGeometryObj {
  x: number
  y: number
  z: number
  radius: number
  vec_x: number
  vec_y: number
  vec_z: number
  style: string = 'circle'
  layer: number = 1

  constructor(
    name: string, x: number, y: number, z: number, radius: number, 
    vec_x: number, vec_y: number, vec_z: number, scene: THREE.Scene
  ) {
    super(name)
    this.x = x
    this.y = y
    this.z = z
    this.radius = radius
    this.vec_x = vec_x
    this.vec_y = vec_y
    this.vec_z = vec_z
    this.add_mesh(scene)
  }

  add_mesh(scene: THREE.Scene) {
    this.mesh = create_circle(
      this.x, this.y, this.z, this.radius, this.vec_x, this.vec_y, this.vec_z
    ) as THREE.Mesh
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
  }

  toJSON() {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      radius: this.radius,
      vec_x: this.vec_x,
      vec_y: this.vec_y,
      vec_z: this.vec_z,
      style: this.style
    }
  }
}

export class ArcObj extends BaseGeometryObj {
  x: number
  y: number
  z: number
  radius: number
  vec_x: number
  vec_y: number
  vec_z: number
  start_angle: number
  end_angle: number
  style: string = 'arc'
  layer: number = 1

  constructor(
    name: string, x: number, y: number, z: number, radius: number, 
    vec_x: number, vec_y: number, vec_z: number, start_angle: number, end_angle: number, scene: THREE.Scene
  ) {
    super(name)
    this.x = x
    this.y = y
    this.z = z
    this.radius = radius
    this.vec_x = vec_x
    this.vec_y = vec_y
    this.vec_z = vec_z
    this.start_angle = start_angle
    this.end_angle = end_angle
    this.add_mesh(scene)
  }

  add_mesh(scene: THREE.Scene) {
    this.mesh = create_arc(
      this.x, this.y, this.z, this.radius, 
      this.vec_x, this.vec_y, this.vec_z, this.start_angle, this.end_angle
    ) as THREE.Mesh
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
  }

  toJSON() {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      radius: this.radius,
      vec_x: this.vec_x,
      vec_y: this.vec_y,
      vec_z: this.vec_z,
      start_angle: this.start_angle,
      end_angle: this.end_angle,
      style: this.style
    }
  }
}

export class BsplineObj extends BaseGeometryObj {
  control_points: PointObj[] = []
  knots: number[] = []
  degree: number = 3
  sample_num: number = 50
  style: string = 'bspline'
  layer: number = 1
  
  constructor(name: string, control_points: PointObj[], knots: number[], degree: number, scene: THREE.Scene) {
    super(name)
    this.control_points = control_points
    this.knots = knots
    this.degree = degree
    this.add_mesh(scene)
    this.update_hooks()
  }

  add_var(point: PointObj, index: number = -1) {
    if (index === -1) {
      this.control_points.push(point)
    } else {
      this.control_points.splice(index, 0, point)
    }
  }

  remove_var(name: string) {
    this.control_points = this.control_points.filter(point => point.name !== name)
  }
  
  add_mesh(scene: THREE.Scene) {
    if (this.control_points.length <= this.degree + 1) {
      throw new Error('BsplineObj must have at least degree + 1 control points')
    }

    this.mesh = create_bspline(
      this.control_points.map(point => [point.x, point.y, point.z]),
      this.knots, this.degree, this.sample_num, {output_mesh: true}
    ) as THREE.Mesh
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
    this.update_hooks()
  }

  update_hooks() {
    let hook_name = this.control_points.map(point => point.name)
    super.update_hooks(hook_name, true)
  } 

  toJSON() {
    return {
      name: this.name,
      control_points: this.control_points.map(point => point.toJSON()),
      knots: this.knots,
      degree: this.degree,
      style: this.style
    }
  }
} 

export class DiskObj extends BaseGeometryObj {
  x: number
  y: number
  z: number
  radius: number
  vec_x: number
  vec_y: number
  vec_z: number
  style: string = 'disk'
  layer: number = 2
  
  constructor(
    name: string, x: number, y: number, z: number, radius: number, 
    vec_x: number, vec_y: number, vec_z: number, scene: THREE.Scene
  ) {
    super(name)
    this.x = x
    this.y = y
    this.z = z
    this.radius = radius
    this.vec_x = vec_x
    this.vec_y = vec_y
    this.vec_z = vec_z
    this.add_mesh(scene)
  }

  add_mesh(scene: THREE.Scene) {
    this.mesh = create_disk(this.x, this.y, this.z, this.radius, this.vec_x, this.vec_y, this.vec_z) as THREE.Mesh
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
  }

  toJSON() {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      radius: this.radius,
      vec_x: this.vec_x,
      vec_y: this.vec_y,
      vec_z: this.vec_z,
      style: this.style
    }
  }
}

export class TorusObj extends BaseGeometryObj {
  x: number
  y: number
  z: number
  radius: number
  tube_radius: number
  vec_x: number
  vec_y: number
  vec_z: number
  style: string = 'torus'
  layer: number = 1

  constructor(
    name: string, x: number, y: number, z: number, radius: number, tube_radius: number, 
    vec_x: number, vec_y: number, vec_z: number, scene: THREE.Scene
  ) {
    super(name)
    this.x = x
    this.y = y
    this.z = z
    this.radius = radius
    this.tube_radius = tube_radius
    this.vec_x = vec_x
    this.vec_y = vec_y
    this.vec_z = vec_z
    this.add_mesh(scene)
  }

  add_mesh(scene: THREE.Scene) {
    this.mesh = create_torus_mesh(this.x, this.y, this.z, this.radius, this.tube_radius, 
      this.vec_x, this.vec_y, this.vec_z, 32, 32, 2 * Math.PI, {output_mesh: true}
    ) as THREE.Mesh
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
  }

  toJSON() {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      radius: this.radius,
      tube_radius: this.tube_radius,
      vec_x: this.vec_x,
      vec_y: this.vec_y,
      vec_z: this.vec_z,
      style: this.style 
    }
  }
} 