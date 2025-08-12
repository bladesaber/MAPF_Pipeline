import { SimulateNetworkMesh } from "./simulate_nodes";
import type { TreeNode } from '@/types/data_type';
import { th } from "element-plus/es/locales.mjs";
import type { Scene } from 'three';

export class SimulateTotalConditionsInletCfg {
    type: string;
    pressure: number;
    temperature: number;
    direction: number[];
    market: string;

    constructor(market: string) {
        this.type = 'TOTAL_CONDITIONS';
        this.pressure = 0;
        this.temperature = 300;
        this.direction = [1, 0, 0];
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'pressure': this.pressure,
            'temperature': this.temperature,
            'direction': this.direction,
            'market': this.market,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateInlet',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateInlet': this.market,
            },
            children: [],
        };
        return root;
    }
}

export class SimulateMassflowInletCfg {
    type: string;
    density: number;  // kg/m^3
    vel_scale: number; // m/s
    direction: number[];
    market: string;

    constructor(market: string) {
        this.type = 'MASS_FLOW';
        this.density = 1.225;
        this.vel_scale = 1;
        this.direction = [1, 0, 0];
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'density': this.density,
            'vel_scale': this.vel_scale,
            'direction': this.direction,
            'market': this.market,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateInlet',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateInlet': this.market,
            },
        };
        return root;
    }
}

export class SimulateVelocityInletCfg {
    type: string;
    temperature: number;
    vel_scale: number;
    direction: number[];
    market: string;

    constructor(market: string) {
        this.type = 'VELOCITY_INLET';
        this.temperature = 300;
        this.vel_scale = 1;
        this.direction = [1, 0, 0];
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'temperature': this.temperature,
            'vel_scale': this.vel_scale,
            'direction': this.direction,
            'market': this.market,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateInlet',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateInlet': this.market,
            },
            children: [],
        };
        return root;
    }
}

export class SimulatePressureInletCfg {
    type: string;
    temperature: number;
    pressure: number;
    direction: number[];
    market: string;

    constructor(market: string) {
        this.type = 'PRESSURE_INLET';
        this.temperature = 300;
        this.pressure = 0;
        this.direction = [1, 0, 0];
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'temperature': this.temperature,
            'pressure': this.pressure,
            'direction': this.direction,
            'market': this.market,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateInlet',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateInlet': this.market,
            },
        };
        return root;
    }
}

export class SimulatePressureOutletCfg {
    type: string;
    pressure: number;
    market: string;

    constructor(market: string) {
        this.type = 'PRESSURE_OUTLET';
        this.pressure = 0;
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'pressure': this.pressure,
            'market': this.market,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateOutlet',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateOutlet': this.market,
            },
        };
        return root;
    }
}

export class SimulateMassflowOutletCfg {
    type: string;
    mass_flow: number;  // kg/s
    market: string;

    constructor(market: string) {
        this.type = 'MASS_FLOW_OUTLET';
        this.mass_flow = 0;
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'mass_flow': this.mass_flow,
            'market': this.market,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateOutlet',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateOutlet': this.market,
            },
        };
        return root;
    }
}

export class SimulateVelocityOutletCfg {
    type: string;
    temperature: number;
    vel_scale: number;
    direction: number[];
    market: string;

    constructor(market: string) {
        this.type = 'VELOCITY_OUTLET';
        this.temperature = 300;
        this.vel_scale = 1;
        this.direction = [1, 0, 0];
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'temperature': this.temperature,
            'vel_scale': this.vel_scale,
            'direction': this.direction,
            'market': this.market,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateOutlet',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateOutlet': this.market,
            },
        };
        return root;
    }
}

export class SimulateIsothermalWallCfg {
    type: string;
    temperature: number;
    market: string;

    constructor(market: string) {
        this.type = 'MARKER_ISOTHERMAL';
        this.temperature = 300;
        this.market = market;
    }

    export_json() {
        return {
            'type': this.type,
            'temperature': this.temperature,
        };
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.market,
            style: 'SimulateWall',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateWall': this.market,
            },
        };
        return root;
    }
}

// -------------------------------------------
export class SimulateRunConfig {
    run_iter: number;
    conv_field: string;
    conv_level: number;
    conv_filename: string;
    restart_filename: string;
    read_binary_restart: boolean;
    restart_sol: boolean;
    with_screen_output: boolean;
    output_wrt_freq: number;

    constructor(
        run_iter: number=100,
        conv_field: string='RMS_PRESSURE',
        conv_level: number=-6,
        conv_filename: string='conv_history',
        restart_filename: string='restart_flow.dat',
        read_binary_restart: boolean=true,
        restart_sol: boolean=false,
        with_screen_output: boolean=false,
        output_wrt_freq: number=100,
    ) {
        this.run_iter = run_iter;
        this.conv_field = conv_field;
        this.conv_level = conv_level;
        this.conv_filename = conv_filename;
        this.restart_filename = restart_filename;
        this.read_binary_restart = read_binary_restart;
        this.restart_sol = restart_sol;
        this.with_screen_output = with_screen_output;
        this.output_wrt_freq = output_wrt_freq;
    }

    export_json() {
        return {
            'run_iter': this.run_iter,
            'conv_field': this.conv_field,
            'conv_level': this.conv_level,
            'conv_filename': this.conv_filename,
            'restart_filename': this.restart_filename,
            'read_binary_restart': this.read_binary_restart,
            'restart_sol': this.restart_sol,
            'with_screen_output': this.with_screen_output,
            'output_wrt_freq': this.output_wrt_freq,
        };
    }
}

export class TurbulenceModel {
    kind_turb_model: string;
    help_info: string = '';

    constructor(kind_turb_model: string) {
        this.kind_turb_model = kind_turb_model;
    }

    export_json() {
        return {
            'kind_turb_model': this.kind_turb_model,
        };
    }
}

export class ShearStressTransportTurbulenceModel extends TurbulenceModel {
    sst_options: string[];
    free_stream_turbulence_intensity: number;
    free_stream_turb2lam_visc_ratio: number;
    help_info = 'The following modifications are allowed:\n'
        + 'Versions:\n'
        + '    V1994m - SSTm WARNING: Our implementation has a small inconsistency with the literature, which will be resolved in the next major SU2 update (i.e. version 8).\n'
        + '    V2003m - SST-2003m (no known inconsistencies).\n'
        + 'Production modifications:\n'
        + '    VORTICITY - Uses vorticity to compute the source term instead of strain-rate magnitude.\n'
        + '    KATO_LAUNDER - Uses the Kato-Launder modification (vorticity times strain-rate).\n'
        + '    UQ - Production is computed using a modified stress tensor for uncertainty quantification. Note with this modification TKE is always included in the stress tensor.\n'
        + 'Corrections:\n'
        + '    SUSTAINING - SST with controlled decay.\n'
        + 'Curvature corrections are currently not implemented.';

    config_map: { [key: string]: string[] } = {
        'SST-v2003m-vorticity-sustaining': ['V2003m', 'VORTICITY', 'SUSTAINING'],
        'default': ['V2003m'],
    }

    constructor(
        config_name: string='default', 
        free_stream_turbulence_intensity: number=0.05, 
        free_stream_turb2lam_visc_ratio: number=10
    ) {
        super('SST');
        this.sst_options = this.get_config(config_name);
        this.free_stream_turbulence_intensity = free_stream_turbulence_intensity;
        this.free_stream_turb2lam_visc_ratio = free_stream_turb2lam_visc_ratio;
    }

    get_config(config_name: string): string[] {
        if (config_name in this.config_map) {
            return this.config_map[config_name];
        } else {
            throw new Error(`Invalid configuration name: ${config_name}`);
        }
    }

    export_json() {
        return {
            'kind_turb_model': this.kind_turb_model,
            'sst_options': this.sst_options.length > 1 ? this.sst_options.join(', ') : this.sst_options[0],
            'free_stream_turbulence_intensity': this.free_stream_turbulence_intensity,
            'free_stream_turb2lam_visc_ratio': this.free_stream_turb2lam_visc_ratio,
        };
    }
}

export class SpalartAllmarasTurbulenceModel extends TurbulenceModel {
    sa_options: string[];
    free_stream_nu_factor: number;
    help_info = 'The following modifications are allowed (refer to NASA’s TMR for further info):\n'
        + 'Versions:\n'
        + '    NEGATIVE - Negative SA model.\n'
        + '    EDWARDS - Edwards modification.\n'
        + '    BCM - BCM transitional model.\n'
        + '    WITHFT2 - SA model with ft2 term, note that by default we omit this term.\n'
        + 'Corrections:\n'
        + '    QCR2000 - Quadratic contitutive relation used in the stress tensor.\n'
        + '    COMPRESSIBILITY - Mixing layer compressibility correction.\n'
        + '    ROTATION - Dacles-Mariani et al. rotation correction.\n'
        + 'For example, to specify SA-neg-R-comp-QCR2000 use SA_OPTIONS= NEGATIVE, WITHFT2, ROTATION, COMPRESSIBILITY, QCR2000';

    config_map: { [key: string]: string[] } = {
        'SA-neg-R-comp-QCR2000': ['NEGATIVE', 'ROTATION', 'COMPRESSIBILITY', 'QCR2000'],
        'default': ['SA-noft2'],
    }

    constructor(
        config_name: string='default', 
        free_stream_nu_factor: number=3
    ) {
        super('SA');
        this.sa_options = this.get_config(config_name);
        this.free_stream_nu_factor = free_stream_nu_factor;
    }

    get_config(config_name: string): string[] {
        if (config_name in this.config_map) {
            return this.config_map[config_name];
        } else {
            throw new Error(`Invalid configuration name: ${config_name}`);
        }
    }

    export_json() {
        return {
            'kind_turb_model': this.kind_turb_model,
            'sa_options': this.sa_options.length > 1 ? this.sa_options.join(', ') : this.sa_options[0],
            'free_stream_nu_factor': this.free_stream_nu_factor,
        };
    }
}

export class FreeStreamConditions {
    help_info = 'The free-stream definition sets the reference conditions for the flow far from any boundaries (e.g., upstream velocity, turbulence levels). \n'
        + 'These values are critical for:\n'
        + '    Initializing the flow field (especially in RANS/LES simulations).\n'
        + '    Setting boundary conditions (inlet, far-field).\n'
        + '    Non-dimensionalizing results (if using Reynolds/Mach numbers).';

    free_stream_density: number;
    free_stream_pressure: number;
    free_stream_temperature: number;
    free_stream_velocity: number[];
    viscosity_model: string;
    thermodynamic_pressure: number;

    constructor(
        free_stream_density: number=1.225,
        free_stream_pressure: number=101325.0,
        free_stream_temperature: number=300.0,
        free_stream_velocity: number[]=[1.0, 0.0, 0.0],
        viscosity_model: string='CONSTANT_VISCOSITY',
        thermodynamic_pressure: number=73528282.41
    ) {
        this.free_stream_density = free_stream_density;
        this.free_stream_pressure = free_stream_pressure;
        this.free_stream_temperature = free_stream_temperature;
        this.free_stream_velocity = free_stream_velocity;
        this.viscosity_model = viscosity_model;
        this.thermodynamic_pressure = thermodynamic_pressure;
    }

    export_json() {
        return {
            'free_stream_density': this.free_stream_density,
            'free_stream_pressure': this.free_stream_pressure,
            'free_stream_temperature': this.free_stream_temperature,
            'free_stream_velocity': this.free_stream_velocity,
            'viscosity_model': this.viscosity_model,
            'thermodynamic_pressure': this.thermodynamic_pressure,
        };
    }
}

export class SimulateSolverConfig {
    solver: string;
    inc_nondim: string;

    fluid_model: string;
    inc_density_model: string;
    inc_density_init: number;

    body_force: string;
    body_force_vector: number[];
    inc_velocity_init: number[];
    inc_temperature_init: number;

    inc_energy_equation: string;
    turbulent_conductivity_model: string;
    prandtl_lam: number;
    prandtl_turb: number;

    constructor(
        solver: string='INC_RANS',
        inc_nondim: string='DIMENSIONAL',
        fluid_model: string='CONSTANT_DENSITY',
        inc_density_model: string='CONSTANT',
        inc_density_init: number=1.225,
        body_force: string='NO',
        body_force_vector: number[]=[0, 0, 0],
        inc_velocity_init: number[]=[1, 0, 0],
        inc_temperature_init: number=300,
        inc_energy_equation: string='YES',
        turbulent_conductivity_model: string='CONSTANT_PRANDTL_TURB',
        prandtl_lam: number=0.72,
        prandtl_turb: number=0.9,
    ) {
        this.solver = solver;
        this.inc_nondim = inc_nondim;
        this.fluid_model = fluid_model;
        this.inc_density_model = inc_density_model;
        this.inc_density_init = inc_density_init;
        this.body_force = body_force;
        this.body_force_vector = body_force_vector;
        this.inc_velocity_init = inc_velocity_init;
        this.inc_temperature_init = inc_temperature_init;
        this.inc_energy_equation = inc_energy_equation;
        this.turbulent_conductivity_model = turbulent_conductivity_model;
        this.prandtl_lam = prandtl_lam;
        this.prandtl_turb = prandtl_turb;
    }

    export_json() {
        return {
            'solver': this.solver,
            'inc_nondim': this.inc_nondim,
            'fluid_model': this.fluid_model,
            'inc_density_model': this.inc_density_model,
            'inc_density_init': this.inc_density_init,
            'body_force': this.body_force,
            'body_force_vector': this.body_force_vector,
            'inc_velocity_init': this.inc_velocity_init,
            'inc_temperature_init': this.inc_temperature_init,
            'inc_energy_equation': this.inc_energy_equation,
            'turbulent_conductivity_model': this.turbulent_conductivity_model,
            'prandtl_lam': this.prandtl_lam,
            'prandtl_turb': this.prandtl_turb,
        };
    }
}

export class SimulateTask {
    name: string;
    network_mesh: SimulateNetworkMesh;

    inlet_cfgs: {
        [key: string]: SimulateTotalConditionsInletCfg | SimulateMassflowInletCfg | SimulateVelocityInletCfg | SimulatePressureInletCfg;
    } = {};

    outlet_cfgs: {
        [key: string]: SimulatePressureOutletCfg | SimulateMassflowOutletCfg | SimulateVelocityOutletCfg;
    } = {};

    wall_cfgs: {
        [key: string]: SimulateIsothermalWallCfg;
    } = {};

    run_config: SimulateRunConfig = new SimulateRunConfig();
    solver_config: SimulateSolverConfig = new SimulateSolverConfig();
    free_stream_conditions: FreeStreamConditions = new FreeStreamConditions();
    turbulence_model: TurbulenceModel = new ShearStressTransportTurbulenceModel();

    constructor(
        name: string,
        points: number[][],
        cells: number[][],
        marker_dict: Map<string, Map<string, number | number[][]>>,
        scene: Scene
    ) {
        this.name = name;
        this.network_mesh = new SimulateNetworkMesh(
            name + '_network',
            points,
            cells,
            marker_dict,
            scene
        );
    }

    check_boundary_conditions(): boolean {
        const boundary_num = Object.keys(this.inlet_cfgs).length 
            + Object.keys(this.outlet_cfgs).length 
            + Object.keys(this.wall_cfgs).length;
        if (Object.keys(this.network_mesh.marker_dict).length != boundary_num) {
            return false;
        }

        const boundary_names = new Set<string>();
        for (const key in this.inlet_cfgs) {
            if (boundary_names.has(key)) {
                return false;
            }
            boundary_names.add(key);
        }
        for (const key in this.outlet_cfgs) {
            if (boundary_names.has(key)) {
                return false;
            }
            boundary_names.add(key);
        }
        for (const key in this.wall_cfgs) {
            if (boundary_names.has(key)) {
                return false;
            }
            boundary_names.add(key);
        }

        return true;
    }
    
    edit_inlet(name: string, cfg: SimulateTotalConditionsInletCfg | SimulateMassflowInletCfg | SimulateVelocityInletCfg | SimulatePressureInletCfg) {
        this.inlet_cfgs[name] = cfg;
    }

    edit_outlet(name: string, cfg: SimulatePressureOutletCfg | SimulateMassflowOutletCfg | SimulateVelocityOutletCfg) {
        this.outlet_cfgs[name] = cfg;
    }

    edit_wall(name: string, cfg: SimulateIsothermalWallCfg) {
        this.wall_cfgs[name] = cfg;
    }

    edit_run_config(cfg: SimulateRunConfig) {
        this.run_config = cfg;
    }

    edit_solver_config(cfg: SimulateSolverConfig) {
        this.solver_config = cfg;
    }

    edit_free_stream_conditions(cfg: FreeStreamConditions) {
        this.free_stream_conditions = cfg;
    }

    edit_turbulence_model(cfg: TurbulenceModel) {
        this.turbulence_model = cfg;
    }

    export_tree(project_name: string): TreeNode {
        const root: TreeNode = {
            label: this.name,
            style: 'SimulateTask',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
            children: [],
        };

        const inlet_nodes: TreeNode = {
            label: 'Inlets',
            style: 'Inlets',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
            children: Object.values(this.inlet_cfgs).map(cfg => cfg.export_tree(project_name, this.name)),
        };

        const outlet_nodes: TreeNode = {
            label: 'Outlets',
            style: 'Outlets',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
            children: Object.values(this.outlet_cfgs).map(cfg => cfg.export_tree(project_name, this.name)),
        };

        const wall_nodes: TreeNode = {
            label: 'Walls',
            style: 'Walls',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
            children: Object.values(this.wall_cfgs).map(cfg => cfg.export_tree(project_name, this.name)),
        };

        const run_config_node: TreeNode = {
            label: 'RunConfig',
            style: 'SimulateRunConfig',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
        };

        const solver_config_node: TreeNode = {
            label: 'SolverConfig',
            style: 'SimulateSolverConfig',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
        };

        const turbulence_model_node: TreeNode = {
            label: 'TurbulenceModel',
            style: 'SimulateTurbulenceModel',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
        };

        const free_stream_conditions_node: TreeNode = {
            label: 'FreeStreamConditions',
            style: 'SimulateFreeStreamConditions',
            data: {
                'Project': project_name,
                'SimulateTask': this.name,
            },
        };

        root.children = [
            turbulence_model_node, free_stream_conditions_node,
            inlet_nodes, outlet_nodes, wall_nodes, 
            run_config_node, solver_config_node,
            this.network_mesh.export_tree(project_name, this.name),
        ];

        return root;
    }

    export_simulation_json(): string {
        const simulation_json = {
            'simulation_name': this.name,
            'inlet_cfgs': Object.values(this.inlet_cfgs).map(cfg => cfg.export_json()),
            'outlet_cfgs': Object.values(this.outlet_cfgs).map(cfg => cfg.export_json()),
            'wall_cfgs': Object.values(this.wall_cfgs).map(cfg => cfg.export_json()),
            'run_config': this.run_config.export_json(),
            'solver_config': this.solver_config.export_json(),
            'free_stream_conditions': this.free_stream_conditions.export_json(),
            'turbulence_model': this.turbulence_model.export_json(),
        };
        return JSON.stringify(simulation_json);
    }

    get_boundary_list(): string[] {
        const boundary_list = [...this.network_mesh.get_boundary_list() as string[]];

        const used = new Set<string>();
        Object.values(this.inlet_cfgs).forEach(cfg => used.add(cfg.market));
        Object.values(this.outlet_cfgs).forEach(cfg => used.add(cfg.market));
        Object.values(this.wall_cfgs).forEach(cfg => used.add(cfg.market));

        const result = boundary_list.filter(market => !used.has(market));
        console.log("boundary_list:", result);
        return result;
    }
}