export interface TreeNode {
  id?: string;
  label: string;
  children?: TreeNode[];
  style?: string;
  [key: string]: any;
}

export interface PathResultData {
  [key: string]: {
    coefficients: number[][];
    segments: {
      [key: string]: {
        coefficient_idxs: number[];
        knots: number[];
        radius: number[];
        degree: number[];
      }
    }
  }
}

export const PointTypes = ['point', 'vec_point', 'vec_radius_point'] as const;
export const LineTypes = ['line', 'circle', 'arc', 'bspline', 'torus'] as const;
export const GroupTypes = ['spline_skeleton', 'circle_surface_points'] as const;
export const SurfaceTypes = ['pipe_shell', 'bspline_surface', 'disk'] as const;
