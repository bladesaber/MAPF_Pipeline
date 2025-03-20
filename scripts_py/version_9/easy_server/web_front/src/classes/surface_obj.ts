import * as THREE from 'three'
import { BaseGeometryObj, VecPointObj, PointObj } from './point_obj.ts'
import { create_bspline, add_mesh, remove_mesh, read_mesh_json, create_disk } from '../utils/visulize_utils'
import { sample_bspline, compute_path_tangent } from '../utils/math_utils'
import { BsplineObj, TorusObj } from './line_obj.ts'

export class PathObj extends BaseGeometryObj {
  style: string = 'path'
  radius: number
  sample_num: number = 50
  layer: number = 1
  
  spline_mesh: BsplineObj
  skeleton_torus: TorusObj[] = []

  constructor(
    name: string,
    control_points: PointObj[],
    knots: number[] = [],
    degree: number = 3,
    radius: number,
    scene: THREE.Scene
  ) {
    super(name)
    this.radius = radius
    this.spline_mesh = new BsplineObj(`${name}_path_mesh`, control_points, knots, degree, scene)
    this.update_hooks()
  }

  add_mesh(scene: THREE.Scene) {
    this.spline_mesh.add_mesh(scene)
    this.update_skeleton(this.sample_num, scene)
    this.skeleton_torus.forEach(torus => torus.add_mesh(scene))
    this.update_hooks()
  }

  update_skeleton(sample_num: number, scene: THREE.Scene) {
    this.sample_num = sample_num
    const ts = Array.from({length: this.sample_num}, (_, i) => i / (this.sample_num - 1))
    const pcd = sample_bspline(
      this.spline_mesh.control_points.map(p => [p.x, p.y, p.z]), 
      this.spline_mesh.degree, 
      this.spline_mesh.knots, 
      ts
    )
    const tangents = compute_path_tangent(pcd)
    tangents.push([...tangents[tangents.length - 1]])

    pcd.forEach((p, i) => {
      const torus = new TorusObj(
        `${this.name}_skeleton_torus_${i}`, 
        p[0], p[1], p[2], this.radius, this.radius / 100, 
        tangents[i][0], tangents[i][1], tangents[i][2], scene
      )
      this.skeleton_torus.push(torus)
    })
  }

  remove_mesh(scene: THREE.Scene): void {
    this.spline_mesh.remove_mesh(scene)
    this.skeleton_torus.forEach(torus => torus.remove_mesh(scene))
  }

  destory(scene: THREE.Scene) {
    this.spline_mesh.destory(scene)
    this.skeleton_torus.forEach(torus => torus.destory(scene))
    this.skeleton_torus = []
  }

  hide_mesh() {
    this.spline_mesh.hide_mesh()
    this.skeleton_torus.forEach(torus => torus.hide_mesh())
  }

  show_mesh() {
    this.spline_mesh.show_mesh()
    this.skeleton_torus.forEach(torus => torus.show_mesh())
  }

  update_hooks(): void {
    super.update_hooks([this.spline_mesh.name], true)
  }

  toJSON() {
    return {
      name: this.name,
      control_pcd: this.spline_mesh.control_points.map(p => p.toJSON()),
      knots: this.spline_mesh.knots,
      degree: this.spline_mesh.degree,
      radius: this.radius,
      sample_num: this.sample_num,
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

export class HttpObj extends BaseGeometryObj {
  style: string = 'http'
  url: string
  options: any

  constructor(name: string, url: string, scene: THREE.Scene) {
    super(name)
    this.url = url
  }

  async fetchAndAddMesh(scene: THREE.Scene, options: any) {
    try {
      const response = await fetch(this.url, {
        method: options.method || 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        body: JSON.stringify({
          name: this.name,
          url: this.url,
          ...options.body
        }),
        ...options
      })
      const mesh_json = await response.json()
      return mesh_json
    } catch (error) {
      console.error('Error fetching mesh:', error)
      return null
    }
  }

  add_mesh(scene: THREE.Scene) {
    const mesh_json = this.fetchAndAddMesh(scene, this.options)
    if (!mesh_json) return
    this.mesh = read_mesh_json(mesh_json)
    this.mesh.name = this.name
    add_mesh(scene, this.mesh)
  }
}

export class PipeShellObj extends HttpObj {
  style: string = 'pipe_shell'
  torus_objs: TorusObj[]
  layer: number = 2

  constructor(name: string, url: string, scene: THREE.Scene, torus_objs: TorusObj[]) {
    super(name, url, scene)
    this.torus_objs = torus_objs
    this.add_mesh(scene)
    this.update_hooks()
  }

  add_mesh(scene: THREE.Scene) {
    // todo need more adjustment
    this.options = {
      body: {
        torus_centers: this.torus_objs.map(torus => ({
          x: torus.x,
          y: torus.y, 
          z: torus.z,
          vec_x: torus.vec_x,
          vec_y: torus.vec_y,
          vec_z: torus.vec_z,
          radius: torus.radius
        })),
        torus_radii: this.torus_objs.map(torus => torus.radius)
      }
    }
    super.add_mesh(scene)
    this.update_hooks()
  }

  update_hooks(): void {
    let hook_name = this.torus_objs.map(torus => torus.name)
    super.update_hooks(hook_name, true)
  }
}

export class BsplineSurfaceObj extends HttpObj {
  style: string = 'bspline_surface'
  boundary_objs: BaseGeometryObj[]
  guide_objs: BaseGeometryObj[]
  layer: number = 2

  constructor(
    name: string, url: string, scene: THREE.Scene, 
    boundary_objs: BaseGeometryObj[], guide_objs: BaseGeometryObj[]
  ) {
    super(name, url, scene)
    this.boundary_objs = boundary_objs
    this.guide_objs = guide_objs
    this.add_mesh(scene)
    this.update_hooks()
  }

  add_mesh(scene: THREE.Scene) {
    this.options = {
      body: {
        boundary_objs: this.boundary_objs.map(obj => obj.toJSON()),
        guide_objs: this.guide_objs.map(obj => obj.toJSON())
      }
    }
    super.add_mesh(scene) 
    this.update_hooks()
  }

  update_hooks(): void {
    let hook_name = this.boundary_objs.map(obj => obj.name)
    hook_name.push(...this.guide_objs.map(obj => obj.name))
    super.update_hooks(hook_name, true)
  }
}