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

export interface SimulateTreeNode {
  id?: string;
  label: string;
  type: string;
  children?: SimulateTreeNode[];
  tagDict?: Map<string, string>;
}

export interface SimulateInletCfg {
  type: 'VELOCITY_INLET';
  temperature: number;
  vel_scale: number;
  direction: number[];
}

export interface SimulateOutletCfg {
  type: 'PRESSURE_OUTLET';
  pressure: number;
}

export interface SimulateWallCfg {
  type: 'WALL';
  temperature: number;
}

export interface SimulateCondition {
  name: string;
  physical_cfg: {
    viscosity: 0.04;
  }
  material_cfg: {
    density: 887.0;
  };
  run_cfg: {
    run_iter: number;
    conv_field: 'RMS_PRESSURE';
    conv_level: -6;
    conv_filename: 'conv_history';
    restart_filename: 'restart_flow.dat';
    read_binary_restart: true;
    restart_sol: boolean;
    with_screen_output: false;
    output_wrt_freq: 100;
  }
  initialization_cfg: {
    body_force: number[];
    init_velocity: number[];
  }
  inlet_cfg: Map<string, SimulateInletCfg>;
  outlet_cfg: Map<string, SimulateOutletCfg>;
  wall_cfg: Map<string, SimulateWallCfg>;
}

export interface PathSearchTreeNode {
  id?: string;
  label: string;
  type: string;
  tagDict?: Map<string, string>;
  children?: (PathSearchTreeNode|PathSearchMainTask)[];
}

export interface PathSearchSubTask {
  label: string;
  type: string;
  start_points: number[][];
  end_points: number[][];
  run_method: string;
  run_cfg: Map<string, string|number|boolean>;
  result_points: number[][];
}

export interface PathSearchMainTask {
  label: string;
  type: string;
  priority: number;
  group_name: string;
  sub_tasks: PathSearchSubTask[];
}

