import { PointObj, VecPointObj } from '../classes/point_obj.ts'
import { LineObj, CircleObj, ArcObj, BsplineObj, TorusObj } from '../classes/line_obj.ts'
import { PathObj, PipeShellObj, BsplineSurfaceObj, DiskObj } from '../classes/surface_obj.ts'

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

export const PointTypes = ['point', 'vec_point'] as const;
export const LineTypes = ['line', 'circle', 'arc', 'bspline', 'torus', 'path'] as const;
export const SurfaceTypes = ['pipe_shell', 'bspline_surface', 'disk'] as const;
