import * as THREE from 'three'
import { Scene, Object3D, BufferGeometry, Float32BufferAttribute, CylinderGeometry,
  Points, Line, LineBasicMaterial, PointsMaterial, Mesh, MeshBasicMaterial } from 'three'
import { color, pass } from 'three/tsl'
import { sample_bspline } from './math_utils.ts'

interface VisualizationOptions {
  color?: number
  size?: number
  segments?: number
  resolution?: number
  output_mesh?: boolean
}

export function setMeshColor(mesh: THREE.Mesh, color: number) {
  if (mesh.material instanceof THREE.MeshStandardMaterial) {
    mesh.material.color.setHex(color)
  }
  else if (mesh.material instanceof THREE.MeshPhysicalMaterial) {
    mesh.material.color.setHex(color)
  }
  else if (mesh.material instanceof THREE.MeshBasicMaterial) {
    mesh.material.color.setHex(color)
  }
  else {
    throw new Error('Unsupported material type')
  }
}

export function add_mesh(scene: THREE.Scene, mesh: THREE.Mesh, force_update = false): void {
  const existingMesh = scene.getObjectByName(mesh.name)
  if (force_update) {
    if (existingMesh) {
      scene.remove(existingMesh)
    }
    scene.add(mesh)
  } else {
    if (!existingMesh) {
      scene.add(mesh)
    }
  }
}

export function get_mesh_name_list(scene: THREE.Scene): string[] {
  return scene.children.map(child => child.name)
}

export function get_mesh_from_name(scene: THREE.Scene, name: string): THREE.Mesh | undefined {
  return scene.getObjectByName(name) as THREE.Mesh | undefined
}

export function remove_mesh(scene: THREE.Scene, name: string): void {
  const mesh = scene.getObjectByName(name)
  if (mesh) {
    scene.remove(mesh)
  }
}

export function get_mesh_list(scene: THREE.Scene): THREE.Mesh[] {
  return scene.children.map(child => child as THREE.Mesh)
}

export function create_sphere(x: number, y: number, z: number, radius: number, options: VisualizationOptions = {}): Mesh {
  const geometry = new THREE.SphereGeometry(radius, 32, 32)
  const material = new MeshBasicMaterial({
    color: options.color || 0xff0000
  })
  const mesh = new Mesh(geometry, material)
  mesh.position.set(x, y, z)
  return mesh
}

export function create_torus_mesh(
  x: number, y: number, z: number, radius: number, tube_radius: number,
  vec_x: number = 0, vec_y: number = 0, vec_z: number = 1,
  radial_segments: number, tubular_segments: number, arc: number,
  options: VisualizationOptions = {}
): Mesh {
  const geometry = new THREE.TorusGeometry(radius, tube_radius, radial_segments, tubular_segments, arc)
  const material = new MeshBasicMaterial({
    color: options.color || 0xff0000
  })
  
  // Create a mesh with the geometry and material
  const mesh = new Mesh(geometry, material)
  
  // Position the mesh
  mesh.position.set(x, y, z)
  
  // Calculate rotation from direction vector
  if (vec_x !== 0 || vec_y !== 0 || vec_z !== 1) {
    const up = new THREE.Vector3(0, 0, 1)
    const direction = new THREE.Vector3(vec_x, vec_y, vec_z).normalize()
    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(up, direction)
    mesh.setRotationFromQuaternion(quaternion)
  }

  return mesh
}

export function create_tube(xyz_list: number[][], radius: number, options: VisualizationOptions = {}): Mesh {
  const curve = new THREE.CatmullRomCurve3(
    xyz_list.map(xyz => new THREE.Vector3(xyz[0], xyz[1], xyz[2]))
  )
  const geometry = new THREE.TubeGeometry(curve, options.segments, radius, options.segments, false)
  const material = new MeshBasicMaterial({color: options.color || 0xff0000})
  const mesh = new Mesh(geometry, material)
  return mesh
}

export function create_point(x: number, y: number, z: number, options: VisualizationOptions = {}): Points | Mesh {
  if (options.output_mesh) {
    const mesh = create_sphere(x, y, z, options.size || 0.1, options)
    return mesh
  } else {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute([x, y, z], 3))
    const material = new PointsMaterial({
      color: options.color || 0xff0000,
      size: options.size || 0.1,
      sizeAttenuation: true
    })
    return new Points(geometry, material)
  }
}

export function create_points(points: number[][], options: VisualizationOptions = {}): Points {
  const positions = new Float32Array(points.length * 3)
  
  points.forEach((point, i) => {
    positions[i * 3] = point[0]
    positions[i * 3 + 1] = point[1]
    positions[i * 3 + 2] = point[2]
  })
  
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  
  const material = new PointsMaterial({
    color: options.color || 0xff0000,
    size: options.size || 0.1,
    sizeAttenuation: true
  })
  return new Points(geometry, material)
}

export function create_circle(
  x: number, y: number, z: number, radius: number, vec_x: number = 0, vec_y: number = 0, vec_z: number = 1, 
  options: VisualizationOptions = {}
): Line | Mesh {
  const segments = options.segments || 32
  
  // Create normal vector and get perpendicular vectors for circle plane
  const normal = new THREE.Vector3(vec_x, vec_y, vec_z).normalize()
  const perpVector1 = new THREE.Vector3()
  const perpVector2 = new THREE.Vector3()
  
  if (Math.abs(normal.x) < Math.abs(normal.y)) {
    perpVector1.set(0, -normal.z, normal.y).normalize()
  } else {
    perpVector1.set(-normal.z, 0, normal.x).normalize()
  }
  perpVector2.crossVectors(normal, perpVector1)

  if (options.output_mesh) {
    const points: number[][] = []
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2
      const circleVector = perpVector1.clone().multiplyScalar(Math.cos(theta))
        .add(perpVector2.clone().multiplyScalar(Math.sin(theta)))
        .multiplyScalar(radius)
      const px = x + circleVector.x
      const py = y + circleVector.y
      const pz = z + circleVector.z
      points.push([px, py, pz])
    }
    // const mesh = create_tube(points, 0.2, options)
    const mesh = create_multi_lines(points, options)
    return mesh

  } else {
    const positions = new Float32Array((segments + 1) * 3)
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2
      const circleVector = perpVector1.clone().multiplyScalar(Math.cos(theta))
        .add(perpVector2.clone().multiplyScalar(Math.sin(theta)))
        .multiplyScalar(radius)
      const px = x + circleVector.x
      const py = y + circleVector.y
      const pz = z + circleVector.z
      positions[i * 3] = px
      positions[i * 3 + 1] = py
      positions[i * 3 + 2] = pz
    }
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
    const material = new LineBasicMaterial({color: options.color || 0x0000ff})
    return new Line(geometry, material)
  }
}

export function create_arc(
  x: number, y: number, z: number, radius: number, vec_x: number = 0, vec_y: number = 0, vec_z: number = 1,
  start_angle: number, end_angle: number, options: VisualizationOptions = {}
): Line | Mesh {
  const segments = options.segments || 32

  // Create normal vector and get perpendicular vectors for arc plane
  const normal = new THREE.Vector3(vec_x, vec_y, vec_z).normalize()
  const perpVector1 = new THREE.Vector3()
  const perpVector2 = new THREE.Vector3()
  
  if (Math.abs(normal.x) < Math.abs(normal.y)) {
    perpVector1.set(0, -normal.z, normal.y).normalize()
  } else {
    perpVector1.set(-normal.z, 0, normal.x).normalize()
  }
  perpVector2.crossVectors(normal, perpVector1)

  const points: number[][] = []
  for (let i = 0; i <= segments; i++) {
    const theta = ((i / segments) * (end_angle - start_angle) + start_angle) * Math.PI / 180
    const circleVector = perpVector1.clone().multiplyScalar(Math.cos(theta))
      .add(perpVector2.clone().multiplyScalar(Math.sin(theta)))
      .multiplyScalar(radius)
    const px = x + circleVector.x
    const py = y + circleVector.y
    const pz = z + circleVector.z
    points.push([px, py, pz])
  }

  if (options.output_mesh) {
    const mesh = create_multi_lines(points, options)
    return mesh
  } else {
    const positions = new Float32Array((segments + 1) * 3)
    for (let i = 0; i <= segments; i++) {      
      positions[i * 3] = points[i][0]
      positions[i * 3 + 1] = points[i][1]
      positions[i * 3 + 2] = points[i][2]
    }
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
    const material = new LineBasicMaterial({color: options.color || 0x0000ff})
    return new Line(geometry, material)
  }
}

export function create_bspline(
  controlPoints: number[][], knots: number[] = [], degree: number = 3, sample_num: number = 100, options: VisualizationOptions = {}
): Line | Mesh {
  const n = controlPoints.length
  if (knots.length === 0) {
    knots = []
    for (let i = 0; i <= n + degree; i++) {
      knots.push(i <= degree ? 0 : i >= n ? 1 : i / (n - degree))
    }
  }

  const ts = []
  for (let i = 0; i <= sample_num; i++) {
    ts.push(i / sample_num)
  }

  const curvePoints = sample_bspline(controlPoints, degree, knots, ts)
  if (options.output_mesh) {
    const mesh = create_tube(curvePoints, options.size || 0.05, options)
    return mesh
  } else {
    // Create geometry from curve points
    const positions = new Float32Array(curvePoints.length * 3)
    curvePoints.forEach((p, i) => {
      positions[i * 3] = p[0]
      positions[i * 3 + 1] = p[1]
      positions[i * 3 + 2] = p[2]
    })
    
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
    
    const material = new LineBasicMaterial({
      color: options.color || 0x00ff00
    })
    
    return new Line(geometry, material)
  }
}

export function create_multi_lines(points: number[][], options: VisualizationOptions = {}): Line | Mesh {
  if (options.output_mesh) {
    const mesh = create_tube(points, options.size || 0.05, options)
    return mesh
  } else {
    const positions = new Float32Array(points.length * 3)
    points.forEach((point, i) => {
      positions[i * 3] = point[0]
      positions[i * 3 + 1] = point[1]
      positions[i * 3 + 2] = point[2]
    })
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
    const material = new LineBasicMaterial({color: options.color || 0x000000})
    return new Line(geometry, material)
  }
}

export function create_disk(
  x: number, y: number, z: number, radius: number, vec_x: number = 0, vec_y: number = 0, vec_z: number = 1,
  options: VisualizationOptions = {}
): Mesh {
  const geometry = new THREE.CircleGeometry(radius, options.segments || 32)
  const material = new MeshBasicMaterial({color: options.color || 0x000000, side: THREE.DoubleSide})
  const mesh = new Mesh(geometry, material)
  mesh.position.set(x, y, z)

  // Calculate rotation to align with normal vector
  const normal = new THREE.Vector3(vec_x, vec_y, vec_z).normalize()
  const defaultNormal = new THREE.Vector3(0, 0, 1)
  const quaternion = new THREE.Quaternion()
  quaternion.setFromUnitVectors(defaultNormal, normal)
  mesh.setRotationFromQuaternion(quaternion)

  return mesh
}

export function read_mesh_json(mesh_json: any, options: VisualizationOptions = {}): THREE.Mesh {
  const loader = new THREE.BufferGeometryLoader()
  const geometry = loader.parse(mesh_json)
  const material = new THREE.MeshBasicMaterial({color: options.color || 0x00ff00, side: THREE.DoubleSide})
  const mesh = new THREE.Mesh(geometry, material)
  return mesh
}