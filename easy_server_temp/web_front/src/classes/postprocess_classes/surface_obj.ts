import * as THREE from 'three'
import { BaseGeometryObj, BaseNode, GroupNode } from './base_obj.ts'
import { VecPointObj, PointObj } from './point_obj.ts'
import { add_mesh, read_mesh_json, create_disk } from '../../utils/visulize_utils.ts'
import { BsplineObj, TorusObj } from './line_obj.ts'
import { RootObj } from './store_obj.ts'

export class DiskObj extends BaseGeometryObj {
  x: number
  y: number
  z: number
  radius: number
  vec_x: number
  vec_y: number
  vec_z: number
  style: string = 'disk'
  
  constructor(
    name: string, x: number, y: number, z: number, radius: number, 
    vec_x: number, vec_y: number, vec_z: number, 
    node_style: BaseNode|GroupNode, rootObj: RootObj
  ) {
    super(name, node_style)
    this.x = x
    this.y = y
    this.z = z
    this.radius = radius
    this.vec_x = vec_x
    this.vec_y = vec_y
    this.vec_z = vec_z
    this.add_mesh(rootObj)
  }

  add_mesh(rootObj: RootObj) {
    this.mesh = create_disk(this.x, this.y, this.z, this.radius, this.vec_x, this.vec_y, this.vec_z) as THREE.Mesh
    this.mesh.name = this.name
    add_mesh(rootObj.get_scene(), this.mesh)
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

  constructor(name: string, url: string, node_style: BaseNode|GroupNode, rootObj: RootObj) {
    super(name, node_style)
    this.url = url
  }

  async fetchAndAddMesh(rootObj: RootObj, options: any) {
    try {
      const response = await fetch(this.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(options)
      })
      const mesh_json = await response.json()
      return mesh_json
      
    } catch (error) {
      console.error('Error fetching mesh:', error)
      return null
    }
  }

  async add_mesh(rootObj: RootObj) {
    const mesh_json = await this.fetchAndAddMesh(rootObj, this.options)
    if (!mesh_json) return
    this.mesh = read_mesh_json(mesh_json['shapes'][0])
    this.mesh.name = this.name
    add_mesh(rootObj.get_scene(), this.mesh)
  }
}

export class PipeShellObj extends HttpObj {
  style: string = 'pipe_shell'
  skeleton_nodes: (BaseNode|GroupNode)[]

  constructor(
    name: string, url: string, skeleton_nodes: (BaseNode|GroupNode)[], 
    node_style: BaseNode|GroupNode, rootObj: RootObj
  ) {
    super(name, url, node_style, rootObj)
    this.skeleton_nodes = skeleton_nodes
    this.add_mesh(rootObj)
  }

  async add_mesh(rootObj: RootObj) {
    const torus_objs: TorusObj[] = []
    this.skeleton_nodes.forEach(obj => {
      const torus_obj = rootObj.request_obj(obj) as TorusObj
      if (torus_obj) {
        torus_objs.push(torus_obj)
      }
    })
    
    this.options = {
      name: this.name,
      skeleton_cfgs: [],
      ruled: true,
      export_edges: false
    }
    torus_objs.forEach(torus => {
      this.options.skeleton_cfgs.push(torus.toJSON())
    })
    super.add_mesh(rootObj)

    this.skeleton_nodes = torus_objs.map((torus) => torus.node_style)
    let hook_name = torus_objs.map((torus) => torus.name);
    super.update_hooks(hook_name, true)
  }
}

export class BsplineSurfaceObj extends HttpObj {
  style: string = 'bspline_surface'
  boundary_nodes: (BaseNode|GroupNode)[]
  guide_nodes: (BaseNode|GroupNode)[]

  constructor(
    name: string, url: string, 
    boundary_nodes: (BaseNode|GroupNode)[], guide_nodes: (BaseNode|GroupNode)[],
    node_style: BaseNode|GroupNode, rootObj: RootObj
  ) {
    super(name, url, node_style, rootObj)
    this.boundary_nodes = boundary_nodes
    this.guide_nodes = guide_nodes
    this.add_mesh(rootObj)
  }

  async add_mesh(rootObj: RootObj) {
    const boundary_objs: BsplineObj[] = []
    this.boundary_nodes.forEach(obj => {
      const boundary_obj = rootObj.request_obj(obj) as BsplineObj
      if (boundary_obj) {
        boundary_objs.push(boundary_obj)
      }
    })

    const guide_objs: BsplineObj[] = []
    this.guide_nodes.forEach(obj => {
      const guide_obj = rootObj.request_obj(obj) as BsplineObj
      if (guide_obj) {
        guide_objs.push(guide_obj)
      }
    })
    
    this.options = {
      name: this.name,
      body: {
        boundary_objs: boundary_objs.map(obj => ({...obj.toJSON(), type: 'edge', 'sample_num': 50})),
        guide_objs: guide_objs.map(obj => ({...obj.toJSON(), type: 'edge', 'sample_num': 50}))
      }
    }    
    super.add_mesh(rootObj) 

    this.boundary_nodes = boundary_objs.map((obj) => obj.node_style)
    this.guide_nodes = guide_objs.map((obj) => obj.node_style)
    let hook_name = this.boundary_nodes.map((obj) => obj.name)
    hook_name.push(...this.guide_nodes.map((obj) => obj.name))
    super.update_hooks(hook_name, true)
  }
}