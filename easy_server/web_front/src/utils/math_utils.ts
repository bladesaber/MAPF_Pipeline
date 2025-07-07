/**
 * Math utility functions for path processing and geometric calculations
 */

export function distance3D(p1: number[], p2: number[]): number {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  const dz = p2[2] - p1[2]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

export function norm_vector(vec: number[]): number[] {
  const norm = Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
  return [vec[0] / norm, vec[1] / norm, vec[2] / norm]
}

export function simpler_path(points: number[][], sampleDensity: number): number[][] {
  if (!points || points.length === 0) return []
  if (points.length === 1) return [...points]

  const result = [points[0]] // Always include first point
  let lastPoint = points[0]
  let dist_cum = 0;
  
  // Process middle points
  for (let i = 1; i < points.length - 1; i++) {
    const currentPoint = points[i]
    dist_cum += distance3D(lastPoint, currentPoint)
    
    if (dist_cum >= sampleDensity) {
      result.push(currentPoint)
      lastPoint = currentPoint
      dist_cum = 0;
    }
  }

  // Handle last point
  const lastOriginalPoint = points[points.length - 1]
  if (distance3D(lastPoint, lastOriginalPoint) > 0) {
    result.push(lastOriginalPoint)
  }

  return result
} 

export function sample_bspline(control_pcd: number[][], degree: number, knots: number[], ts: number[]): number[][] {
  const n = control_pcd.length - 1; // Number of control points minus 1
  const m = knots.length - 1; // Number of knots minus 1
  const d = control_pcd[0].length; // Dimension of the control points

  // Ensure the number of knots is correct
  if (m !== n + degree + 1) {
    throw new Error("Invalid number of knots. Expected (n + degree + 1) knots.");
  }

  // Function to compute the basis function
  function basisFunction(i: number, p: number, t: number, knots: number[]): number {
    if (p === 0) {
      if (t === knots[m] && t === knots[i + 1]) {
        return 1;
      }
      return (knots[i] <= t && t < knots[i + 1]) ? 1 : 0;
    }
    const left = (knots[i + p] - knots[i]) === 0 ? 0 : 
      ((t - knots[i]) / (knots[i + p] - knots[i])) * basisFunction(i, p - 1, t, knots);
    const right = (knots[i + p + 1] - knots[i + 1]) === 0 ? 0 : 
      ((knots[i + p + 1] - t) / (knots[i + p + 1] - knots[i + 1])) * basisFunction(i + 1, p - 1, t, knots);
    return left + right;
  }

  // Compute the B-spline curve points
  const curvePoints: number[][] = [];
  for (const t of ts) {
    const point = new Array(d).fill(0);
    for (let i = 0; i <= n; i++) {
      const basis = basisFunction(i, degree, t, knots);
      for (let j = 0; j < d; j++) {
          point[j] += control_pcd[i][j] * basis;
      }
    }
    curvePoints.push(point);
  }

  return curvePoints;
}

export function compute_path_tangent(path_pcd: number[][]): number[][] {
  const tangents: number[][] = [];
  for (let i = 0; i < path_pcd.length - 1; i++) {
    const p1 = path_pcd[i];
    const p2 = path_pcd[i + 1];
    const tangent = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    tangents.push(norm_vector(tangent));
  }
  tangents.push(tangents[tangents.length - 1]);
  return tangents;
}

export function interpolate_by_bspline(values: number[], knots: number[], degree: number, ts: number[]): number[] {
  const n = values.length - 1; // Number of control points minus 1
  const m = knots.length - 1; // Number of knots minus 1

  // Ensure the number of knots is correct
  if (m !== n + degree + 1) {
    throw new Error("Invalid number of knots. Expected (n + degree + 1) knots.");
  }

  // Function to compute the basis function
  function basisFunction(i: number, p: number, t: number, knots: number[]): number {
    if (p === 0) {
      if (t === knots[m] && t === knots[i + 1]) {
        return 1;
      }
      return (knots[i] <= t && t < knots[i + 1]) ? 1 : 0;
    }
    const left = (knots[i + p] - knots[i]) === 0 ? 0 : 
      ((t - knots[i]) / (knots[i + p] - knots[i])) * basisFunction(i, p - 1, t, knots);
    const right = (knots[i + p + 1] - knots[i + 1]) === 0 ? 0 : 
      ((knots[i + p + 1] - t) / (knots[i + p + 1] - knots[i + 1])) * basisFunction(i + 1, p - 1, t, knots);
    return left + right;
  }

  // Compute the B-spline curve points
  const interpolatedValues: number[] = [];
  for (const t of ts) {
    let interpolated_value = 0.0
    for (let i = 0; i <= n; i++) {
      const basis = basisFunction(i, degree, t, knots);
      interpolated_value += values[i] * basis;
    }
    interpolatedValues.push(interpolated_value);
  }

  return interpolatedValues;
}

export function map_point_to_plane(point: number[], plane_point: number[], plane_normal: number[]): number[] {
  const PQ = [point[0] - plane_point[0], point[1] - plane_point[1], point[2] - plane_point[2]];
  const PQ_dot_n = PQ[0] * plane_normal[0] + PQ[1] * plane_normal[1] + PQ[2] * plane_normal[2];
  const n_dot_n = plane_normal[0] * plane_normal[0] 
                + plane_normal[1] * plane_normal[1]
                + plane_normal[2] * plane_normal[2];
  const projected_point = [
      point[0] - (PQ_dot_n / n_dot_n) * plane_normal[0],
      point[1] - (PQ_dot_n / n_dot_n) * plane_normal[1],
      point[2] - (PQ_dot_n / n_dot_n) * plane_normal[2]
  ];

  return projected_point;
}

export function find_cloest_point_to_circle(
  point: number[], circle_center: number[], circle_normal: number[], circle_radius: number
): number[] {
  const projected_point = map_point_to_plane(point, circle_center, circle_normal)
  const center_to_projected_point = [
    projected_point[0] - circle_center[0],
    projected_point[1] - circle_center[1],
    projected_point[2] - circle_center[2]
  ]
  const center_to_projected_point_length = Math.sqrt(
    center_to_projected_point[0] * center_to_projected_point[0] 
    + center_to_projected_point[1] * center_to_projected_point[1] 
    + center_to_projected_point[2] * center_to_projected_point[2]
  )
  const center_to_projected_point_unit = [
    center_to_projected_point[0] / center_to_projected_point_length,
    center_to_projected_point[1] / center_to_projected_point_length,
    center_to_projected_point[2] / center_to_projected_point_length
  ]
  const cloest_point = [
    circle_center[0] + circle_radius * center_to_projected_point_unit[0],
    circle_center[1] + circle_radius * center_to_projected_point_unit[1],
    circle_center[2] + circle_radius * center_to_projected_point_unit[2]
  ]
  return cloest_point
}

export function sample_surface_points_from_circles(
  circle_center: number[][], circle_normal: number[][], circle_radius: number[], sample_num: number[]
): number[][] {
  const points: number[][] = [];
  for (let i = 0; i < circle_center.length; i++) {
    const circle_center_i = circle_center[i];
    const circle_normal_i = circle_normal[i];
    const circle_radius_i = circle_radius[i];
    const sample_num_i = sample_num[i];

    // Get perpendicular vectors for circle plane
    const vector_len = Math.sqrt(
      circle_normal_i[0] * circle_normal_i[0] + 
      circle_normal_i[1] * circle_normal_i[1] + 
      circle_normal_i[2] * circle_normal_i[2]
    );
    const normal = [
      circle_normal_i[0] / vector_len,
      circle_normal_i[1] / vector_len, 
      circle_normal_i[2] / vector_len
    ];
    const perpVector1 = new Array(3);
    const perpVector2 = new Array(3);
    
    // Find first perpendicular vector
    if (Math.abs(normal[0]) < Math.abs(normal[1])) {
      const len = Math.sqrt(normal[1] * normal[1] + normal[2] * normal[2]);
      perpVector1[0] = 0;
      perpVector1[1] = -normal[2] / len;
      perpVector1[2] = normal[1] / len;
    } else {
      const len = Math.sqrt(normal[0] * normal[0] + normal[2] * normal[2]);
      perpVector1[0] = -normal[2] / len;
      perpVector1[1] = 0; 
      perpVector1[2] = normal[0] / len;
    }
    
    // Find second perpendicular vector using cross product
    perpVector2[0] = normal[1] * perpVector1[2] - normal[2] * perpVector1[1];
    perpVector2[1] = normal[2] * perpVector1[0] - normal[0] * perpVector1[2];
    perpVector2[2] = normal[0] * perpVector1[1] - normal[1] * perpVector1[0];

    // Sample points around circle
    for (let j = 0; j < sample_num_i; j++) {
      const theta = (j / sample_num_i) * Math.PI * 2;
      const cos_theta = Math.cos(theta);
      const sin_theta = Math.sin(theta);
      const point = [
        circle_center_i[0] + circle_radius_i * (perpVector1[0] * cos_theta + perpVector2[0] * sin_theta),
        circle_center_i[1] + circle_radius_i * (perpVector1[1] * cos_theta + perpVector2[1] * sin_theta),
        circle_center_i[2] + circle_radius_i * (perpVector1[2] * cos_theta + perpVector2[2] * sin_theta)
      ];
      points.push(point);
    }
  }

  // Filter points that are inside 0.9 * radius of any circle center
  const filtered_points = points.filter(point => {
    // Check distance to each circle center
    for (let i = 0; i < circle_center.length; i++) {
      const center = circle_center[i];
      const radius = circle_radius[i];
      const center_to_point = distance3D(point, center)
      if (center_to_point < 0.9 * radius) {
        return false;
      }
    }
    return true;
  });

  return filtered_points;
  
}
