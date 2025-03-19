/**
 * Math utility functions for path processing and geometric calculations
 */

import type { PointObj } from '../classes/geo_obj'

export function distance3D(p1: PointObj, p2: PointObj): number {
  const dx = p2.x - p1.x
  const dy = p2.y - p1.y
  const dz = p2.z - p1.z
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

export function norm_vector(vec: number[]): number[] {
  const norm = Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
  return [vec[0] / norm, vec[1] / norm, vec[2] / norm]
}

export function simpler_path(points: PointObj[], sampleDensity: number): PointObj[] {
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

// TODO: check if we need it
export function compute_path_tangent(path_pcd: number[][]): number[][] {
  const tangents: number[][] = [];
  for (let i = 0; i < path_pcd.length - 1; i++) {
    const p1 = path_pcd[i];
    const p2 = path_pcd[i + 1];
    const tangent = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    tangents.push(norm_vector(tangent));
  }
  return tangents;
}

