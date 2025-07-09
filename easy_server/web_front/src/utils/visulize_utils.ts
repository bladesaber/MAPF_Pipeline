import * as THREE from 'three';
import {
  BufferGeometry,
  Float32BufferAttribute,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  Points,
  PointsMaterial,
} from 'three';

import { sample_bspline } from './math_utils.ts';

interface VisualizationOptions {
  color?: number
  size?: number
  segments?: number
  resolution?: number
  output_mesh?: boolean
  opacity?: number
  transparent?: boolean
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

export function add_mesh(scene: THREE.Scene, mesh: THREE.Mesh | THREE.LineSegments | THREE.Points, force_update = false): void {
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

export function create_sphere(
  x: number, y: number, z: number, radius: number, options: VisualizationOptions = { output_mesh: true }
): Mesh | THREE.LineSegments {
  if (options.output_mesh) {
    const geometry = new THREE.SphereGeometry(radius, 32, 32)
    const material = new MeshBasicMaterial({
      color: options.color || 0xff0000,
      transparent: options.transparent || false,
      opacity: options.opacity || 1.0
    })
    const mesh = new Mesh(geometry, material)
    mesh.position.set(x, y, z)
    return mesh
  } else {
    const geometry = new THREE.SphereGeometry(radius, 32, 32)
    const material = new LineBasicMaterial({ color: options.color || 0x0000ff })
    const mesh = new THREE.LineSegments(geometry, material)
    mesh.position.set(x, y, z)
    return mesh
  }
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
  const material = new THREE.MeshPhysicalMaterial({
    color: options.color || 0xff0000,
    opacity: options.opacity || 1.0,
    side: THREE.DoubleSide,
    // transparent: options.transparent || false
  })
  const mesh = new Mesh(geometry, material)
  return mesh
}

export function create_point(x: number, y: number, z: number, options: VisualizationOptions = {}): Points | Mesh | THREE.LineSegments {
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

export function create_colored_points(
  points: number[][],
  values: number[],
  options: VisualizationOptions = {}
): Points {
  const positions = new Float32Array(points.length * 3);
  const colors_array = new Float32Array(points.length * 3);

  // Calculate min/max for normalization
  const max_value = Math.max(...values);
  const min_value = Math.min(...values);

  points.forEach((point, i) => {
    // Positions
    positions[i * 3] = point[0];
    positions[i * 3 + 1] = point[1];
    positions[i * 3 + 2] = point[2];

    // Normalize value to [0, 1] range
    const normalizedValue = (values[i] - min_value) / (max_value - min_value || 1); // Avoid division by zero

    // Assign colors (gradient from blue to red)
    colors_array[i * 3] = normalizedValue;      // Red increases with value
    colors_array[i * 3 + 1] = 0;               // No green
    colors_array[i * 3 + 2] = 1 - normalizedValue; // Blue decreases with value
  });

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new Float32BufferAttribute(colors_array, 3));

  const material = new PointsMaterial({
    size: options.size || 0.1,
    sizeAttenuation: true,
    vertexColors: true, // Overrides 'color' property
  });

  return new Points(geometry, material);
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
    const material = new LineBasicMaterial({ color: options.color || 0x0000ff })
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
    const material = new LineBasicMaterial({ color: options.color || 0x0000ff })
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
    const material = new LineBasicMaterial({ color: options.color || 0x000000 })
    return new Line(geometry, material)
  }
}

export function create_disk(
  x: number, y: number, z: number, radius: number, vec_x: number = 0, vec_y: number = 0, vec_z: number = 1,
  options: VisualizationOptions = {}
): Mesh {
  const geometry = new THREE.CircleGeometry(radius, options.segments || 32)
  const material = new MeshBasicMaterial({ color: options.color || 0x000000, side: THREE.DoubleSide })
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
  // const material = new THREE.MeshBasicMaterial({color: options.color || 0x00ff00, side: THREE.DoubleSide})
  const material = new THREE.MeshPhysicalMaterial({
    color: options.color || 0x00ff00,
    side: THREE.DoubleSide
  })
  const mesh = new THREE.Mesh(geometry, material)
  return mesh
}

export function create_line_set(
  points: number[][], line_set: number[][], options: VisualizationOptions = {}
): THREE.LineSegments {
  const vertices = new Float32Array(points.flat());

  // Convert line_set (indices) to Uint16Array or Uint32Array
  const indices = new Uint16Array(line_set.flat());

  // Create a BufferGeometry
  const geometry = new THREE.BufferGeometry();

  // Set position attribute
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

  // Set index for line segments
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  // Default material (white lines)
  const material = new THREE.LineBasicMaterial({
    color: options.color !== undefined ? options.color : 0xff0000
  });

  // Return LineSegments (not Mesh, since we're rendering lines)
  return new THREE.LineSegments(geometry, material);
}

export function create_arrow_set(
  points: number[][], arrow_vectors: number[][], arrow_length: number, arrow_stress: number[], options: VisualizationOptions = {}
): THREE.LineSegments {
  // Re-range arrow_stress to [0,1]
  const min_stress = Math.min(...arrow_stress)
  const max_stress = Math.max(...arrow_stress)
  const normalized_stress = arrow_stress.map(stress =>
    (stress - min_stress) / (max_stress - min_stress)
  )

  const points_end = points.map((point, i) => {
    const vector = arrow_vectors[i]
    const point_end = [
      point[0] + vector[0] * arrow_length,
      point[1] + vector[1] * arrow_length,
      point[2] + vector[2] * arrow_length
    ]
    return point_end
  })

  // Create line set by pairing each point with its end point
  const line_vertices: number[] = []
  const line_colors: number[] = []
  const line_indices: number[] = []

  points.forEach((point, i) => {
    // Add point and end point vertices
    line_vertices.push(...point, ...points_end[i])

    // Add indices to connect them
    const baseIndex = i * 2
    line_indices.push(baseIndex, baseIndex + 1)

    // Add colors based on normalized stress value
    const stress = normalized_stress[i]
    // Use a color gradient from blue (low stress) to red (high stress) 
    const r = Math.min(1, Math.max(0, stress))
    const b = Math.min(1, Math.max(0, 1 - stress))
    line_colors.push(r, 0, b, r, 0, b) // Same color for both vertices of the line
  })

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(line_vertices, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(line_colors, 3))
  geometry.setIndex(line_indices)

  const material = new THREE.LineBasicMaterial({
    vertexColors: true
  })

  return new THREE.LineSegments(geometry, material)
}

export function color_to_hex(color: number[]) {

  // 确保颜色值在0-255范围内
  const r = Math.min(255, Math.max(0, color[0]))
  const g = Math.min(255, Math.max(0, color[1]))
  const b = Math.min(255, Math.max(0, color[2]))

  // 转换为16进制
  const r_hex = Math.floor(r).toString(16).padStart(2, '0')
  const g_hex = Math.floor(g).toString(16).padStart(2, '0')
  const b_hex = Math.floor(b).toString(16).padStart(2, '0')

  // 组合成完整的16进制颜色值
  return parseInt(`0x${r_hex}${g_hex}${b_hex}`, 16)
}

export function create_cone(
  x: number, y: number, z: number, radius: number, height: number, vec_x: number = 0, vec_y: number = 0, vec_z: number = 1,
  options: VisualizationOptions = {}
): Mesh {

  const geometry = new THREE.ConeGeometry(radius, height, options.segments || 32)
  const material = new THREE.MeshBasicMaterial({
    color: options.color !== undefined ? options.color : 0x000000,
    transparent: options.transparent || false,
    opacity: options.opacity || 1.0
  })
  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.set(x, y, z)

  // 设置颜色
  if (options.color !== undefined) {
    material.color.setHex(options.color)
  }

  // 重置旋转
  mesh.rotation.set(0, 0, 0)
  mesh.quaternion.set(0, 0, 0, 1)

  // 计算旋转
  const direction = new THREE.Vector3(vec_x, vec_y, vec_z).normalize()

  // 创建一个从Y轴到目标方向的旋转
  const up = new THREE.Vector3(0, 1, 0)
  const quaternion = new THREE.Quaternion()

  // 如果方向向量是零向量，使用默认方向
  if (direction.length() === 0) {
    direction.set(0, 1, 0)
  }

  quaternion.setFromUnitVectors(up, direction)

  // 应用方向旋转
  mesh.quaternion.multiply(quaternion)

  return mesh
}

export function create_surface_mesh_from_cells(
  vertices: number[][],
  cells: number[][],
  options: VisualizationOptions = {}
): THREE.Mesh {
  // Process vertices into Float32Array for Three.js
  const positions = new Float32Array(vertices.length * 3);
  vertices.forEach((vertex, i) => {
      positions[i * 3] = vertex[0];
      positions[i * 3 + 1] = vertex[1];
      positions[i * 3 + 2] = vertex[2];
  });

  // Process cells into indices for Three.js
  // PyVista cells are stored as [n, i1, i2, i3, ...] where n is the number of points in the cell
  const indices: number[] = [];
  for (const cell of cells) {
      const n = cell[0];
      if (n === 3) {
          // Triangle cell
          indices.push(cell[1], cell[2], cell[3]);
      } else if (n === 4) {
          // Quad cell - split into two triangles
          indices.push(cell[1], cell[2], cell[3]);
          indices.push(cell[1], cell[3], cell[4]);
      }
      // Add more cell types if needed
  }

  // Create geometry
  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(indices);
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  
  // Compute normals for proper lighting
  geometry.computeVertexNormals();

  // Create material with provided options
  const materialOptions: THREE.MeshPhongMaterialParameters = {
      side: THREE.DoubleSide,
      // wireframe: true,
      transparent: options.opacity !== undefined && options.opacity < 1.0,
  };

  if (options.opacity !== undefined) {
      materialOptions.opacity = options.opacity;
  }

  if (options.color !== undefined) {
      if (typeof options.color === 'string' || typeof options.color === 'number') {
          materialOptions.color = new THREE.Color(options.color);
      } else {
          materialOptions.color = options.color;
      }
  } else {
      materialOptions.color = new THREE.Color(0xffffff); // Default white
  }

  const material = new THREE.MeshPhongMaterial(materialOptions);

  return new THREE.Mesh(geometry, material);
}

export function create_circle_set(
  centers: number[][], radius: number[], directions: number[][], options: VisualizationOptions = {}
): Mesh {
  // centers: [x, y, z]
  // radius: number
  // directions: [x, y, z]

  // Create arrays to hold all vertices and indices
  const allVertices: number[] = []
  const allIndices: number[] = []
  let vertexOffset = 0

  // Create each circle
  centers.forEach((center, i) => {
    const x = center[0]
    const y = center[1]
    const z = center[2]
    const r = radius[i]
    const direction = directions[i]
    const segments = options.segments || 32

    // Create normal vector and get perpendicular vectors for circle plane
    const normal = new THREE.Vector3(direction[0], direction[1], direction[2]).normalize()
    const perpVector1 = new THREE.Vector3()
    const perpVector2 = new THREE.Vector3()

    if (Math.abs(normal.x) < Math.abs(normal.y)) {
      perpVector1.set(0, -normal.z, normal.y).normalize()
    } else {
      perpVector1.set(-normal.z, 0, normal.x).normalize()
    }
    perpVector2.crossVectors(normal, perpVector1)

    // Generate circle vertices
    for (let j = 0; j <= segments; j++) {
      const theta = (j / segments) * Math.PI * 2
      const circleVector = perpVector1.clone().multiplyScalar(Math.cos(theta))
        .add(perpVector2.clone().multiplyScalar(Math.sin(theta)))
        .multiplyScalar(r)
      const px = x + circleVector.x
      const py = y + circleVector.y
      const pz = z + circleVector.z

      allVertices.push(px, py, pz)
    }

    // Generate indices for this circle
    for (let j = 0; j < segments; j++) {
      allIndices.push(vertexOffset + j, vertexOffset + j + 1)
    }

    vertexOffset += segments + 1
  })

  // Create geometry
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(allVertices, 3))
  geometry.setIndex(allIndices)

  // Create material
  const material = new THREE.LineBasicMaterial({
    color: options.color || 0x0000ff
  })

  // Create and return line segments (which is a type of mesh)
  return new THREE.LineSegments(geometry, material) as any
}
