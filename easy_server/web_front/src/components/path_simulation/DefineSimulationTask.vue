<!-- DefineSimulationTask.vue -->
<template>
    <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Simulation Task</span>
      <el-button 
        link
        class="close-button"
        @click="handleClose"
      >
        <el-icon><Close /></el-icon>
      </el-button>
    </div>

    <div class="right-sidebar-content">
      <!-- Name Section -->
      <div class="section">
        <div class="section-header">
          <span>Name</span>
        </div>
        <div class="section-content">
          <el-input
            v-model="formData.name"
            placeholder="Enter simulation task name"
            :maxlength="50"
            show-word-limit
          />
        </div>
      </div>

      <!-- Mesh Section -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.mesh = !sectionOpen.mesh" style="cursor:pointer;">
          <span>Mesh</span>
          <el-icon :class="{'rotate': sectionOpen.mesh}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.mesh" class="section-content">
            <el-form label-width="160px">
              <el-form-item label="Min Size">
                <el-input-number v-model="formData.min_size" :min="0" />
              </el-form-item>
              <el-form-item label="Max Size">
                <el-input-number v-model="formData.max_size" :min="0" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- RunConfig Section -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.runConfig = !sectionOpen.runConfig" style="cursor:pointer;">
          <span>Run Config</span>
          <el-icon :class="{'rotate': sectionOpen.runConfig}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.runConfig" class="section-content">
            <el-form label-width="160px">
              <el-form-item label="Run Iter">
                <el-input-number v-model="runConfig.run_iter" :min="1" />
              </el-form-item>
              <el-form-item label="Conv Field">
                <el-select v-model="runConfig.conv_field" style="width: 100%">
                  <el-option label="RMS_PRESSURE" value="RMS_PRESSURE" />
                </el-select>
              </el-form-item>
              <el-form-item label="Conv Level">
                <el-input-number v-model="runConfig.conv_level" />
              </el-form-item>
              <el-form-item label="Conv Filename">
                <el-input v-model="runConfig.conv_filename" />
              </el-form-item>
              <el-form-item label="Restart Filename">
                <el-input v-model="runConfig.restart_filename" />
              </el-form-item>
              <el-form-item label="Read Binary Restart">
                <el-switch v-model="runConfig.read_binary_restart" />
              </el-form-item>
              <el-form-item label="Restart Sol">
                <el-switch v-model="runConfig.restart_sol" />
              </el-form-item>
              <el-form-item label="With Screen Output">
                <el-switch v-model="runConfig.with_screen_output" />
              </el-form-item>
              <el-form-item label="Output Wrt Freq">
                <el-input-number v-model="runConfig.output_wrt_freq" :min="1" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- SolverConfig Section -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.solverConfig = !sectionOpen.solverConfig" style="cursor:pointer;">
          <span>Solver Config</span>
          <el-icon :class="{'rotate': sectionOpen.solverConfig}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.solverConfig" class="section-content">
            <el-form label-width="160px">
              <el-form-item label="Solver">
                <el-select v-model="solverConfig.solver" style="width: 100%">
                  <el-option label="EULER" value="EULER" />
                  <el-option label="NAVIER_STOKES" value="NAVIER_STOKES" />
                  <el-option label="RANS" value="RANS" />
                  <el-option label="NEMO_EULER" value="NEMO_EULER" />
                  <el-option label="NEMO_NAVIER_STOKES" value="NEMO_NAVIER_STOKES" />
                  <el-option label="INC_EULER" value="INC_EULER" />
                  <el-option label="INC_NAVIER_STOKES" value="INC_NAVIER_STOKES" />
                  <el-option label="INC_RANS" value="INC_RANS" />
                  <el-option label="FEM_EULER" value="FEM_EULER" />
                  <el-option label="FEM_NAVIER_STOKES" value="FEM_NAVIER_STOKES" />
                </el-select>
              </el-form-item>
              <el-form-item label="Inc Nondim">
                <el-select v-model="solverConfig.inc_nondim" style="width: 100%">
                  <el-option label="DIMENSIONAL" value="DIMENSIONAL" />
                  <el-option label="REFERENCE_VALUES" value="REFERENCE_VALUES" />
                </el-select>
              </el-form-item>
              <el-form-item label="Fluid Model">
                <el-select v-model="solverConfig.fluid_model" style="width: 100%">
                  <el-option label="CONSTANT_DENSITY" value="CONSTANT_DENSITY" />
                  <el-option label="SU2_NONEQ" value="SU2_NONEQ" />
                </el-select>
              </el-form-item>
              <el-form-item label="Inc Density Model">
                <el-select v-model="solverConfig.inc_density_model" style="width: 100%">
                  <el-option label="CONSTANT" value="CONSTANT" />
                </el-select>
              </el-form-item>
              <el-form-item label="Inc Density Init">
                <el-input-number v-model="solverConfig.inc_density_init" />
              </el-form-item>
              <el-form-item label="Body Force">
                <el-select v-model="solverConfig.body_force" style="width: 100%">
                  <el-option label="YES" value="YES" />
                  <el-option label="NO" value="NO" />
                </el-select>
              </el-form-item>
              <el-form-item label="Body Force Vector">
                <div class="vector-input">
                  <el-input-number v-model="solverConfig.body_force_vector[0]" controls-position="" :controls="false" />
                  <el-input-number v-model="solverConfig.body_force_vector[1]" controls-position="" :controls="false" />
                  <el-input-number v-model="solverConfig.body_force_vector[2]" controls-position="" :controls="false" />
                </div>
              </el-form-item>
              <el-form-item label="Inc Velocity Init">
                <div class="vector-input">
                  <el-input-number v-model="solverConfig.inc_velocity_init[0]" controls-position="" :controls="false" />
                  <el-input-number v-model="solverConfig.inc_velocity_init[1]" controls-position="" :controls="false" />
                  <el-input-number v-model="solverConfig.inc_velocity_init[2]" controls-position="" :controls="false" />
                </div>
              </el-form-item>
              <el-form-item label="Inc Temperature Init">
                <el-input-number v-model="solverConfig.inc_temperature_init" />
              </el-form-item>
              <el-form-item label="Inc Energy Equation">
                <el-select v-model="solverConfig.inc_energy_equation" style="width: 100%">
                  <el-option label="YES" value="YES" />
                  <el-option label="NO" value="NO" />
                </el-select>
              </el-form-item>
              <el-form-item label="Turbulent Conductivity Model">
                <el-select v-model="solverConfig.turbulent_conductivity_model" style="width: 100%">
                  <el-option label="CONSTANT_PRANDTL_TURB" value="CONSTANT_PRANDTL_TURB" />
                </el-select>
              </el-form-item>
              <el-form-item label="Prandtl Lam">
                <el-input-number v-model="solverConfig.prandtl_lam" />
              </el-form-item>
              <el-form-item label="Prandtl Turb">
                <el-input-number v-model="solverConfig.prandtl_turb" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- TurbulenceModel Section -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.turbulenceModel = !sectionOpen.turbulenceModel" style="cursor:pointer;">
          <span>Turbulence Model</span>
          <el-icon :class="{'rotate': sectionOpen.turbulenceModel}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.turbulenceModel" class="section-content">
            <el-form label-width="160px">
              <el-form-item label="Model Type">
                <el-select v-model="turbulenceModelType" placeholder="Select Model">
                  <el-option label="SST" value="SST" />
                  <el-option label="SA" value="SA" />
                </el-select>
              </el-form-item>
              <template v-if="turbulenceModelType === 'SST'">
                <el-form-item label="Config Name">
                  <el-select v-model="turbulenceModelSST.config_name" style="width: 100%">
                    <el-option label="default" value="default" />
                    <el-option label="SST-v2003m-vorticity-sustaining" value="SST-v2003m-vorticity-sustaining" />
                  </el-select>
                </el-form-item>
                <el-form-item label="Free Stream Turbulence Intensity">
                  <el-input-number v-model="turbulenceModelSST.free_stream_turbulence_intensity" :min="0" />
                </el-form-item>
                <el-form-item label="Free Stream Turb2Lam Visc Ratio">
                  <el-input-number v-model="turbulenceModelSST.free_stream_turb2lam_visc_ratio" :min="0" />
                </el-form-item>
              </template>
              <template v-else>
                <el-form-item label="Config Name">
                  <el-select v-model="turbulenceModelSA.config_name" style="width: 100%">
                    <el-option label="default" value="default" />
                    <el-option label="SA-neg-R-comp-QCR2000" value="SA-neg-R-comp-QCR2000" />
                  </el-select>
                </el-form-item>
                <el-form-item label="Free Stream Nu Factor">
                  <el-input-number v-model="turbulenceModelSA.free_stream_nu_factor" :min="0" />
                </el-form-item>
              </template>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- OpenFOAM Parameters Section -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.openfoamParams = !sectionOpen.openfoamParams" style="cursor:pointer;">
          <span>OpenFOAM Parameters</span>
          <el-icon :class="{'rotate': sectionOpen.openfoamParams}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.openfoamParams" class="section-content">
            <el-form label-width="160px">
              <el-form-item label="Solver">
                <el-select v-model="openfoamParams.solver" style="width: 100%">
                  <el-option label="topoIncompressibleFluid" value="topoIncompressibleFluid" />
                  <el-option label="incompressibleFluid" value="incompressibleFluid" />
                </el-select>
              </el-form-item>
              <el-form-item label="Numerical Method">
                <el-select v-model="openfoamParams.numerical_method" style="width: 100%">
                  <el-option label="PIMPLE" value="PIMPLE" />
                  <el-option label="PISO" value="PISO" />
                  <el-option label="SIMPLE" value="SIMPLE" />
                </el-select>
              </el-form-item>
              <el-form-item label="Delta T">
                <el-input-number v-model="openfoamParams.deltaT" :min="0" :step="0.001" />
              </el-form-item>
              <el-form-item label="End Time">
                <el-input-number v-model="openfoamParams.endTime" :min="0" :step="0.1" />
              </el-form-item>
              <el-form-item label="Write Interval">
                <el-input-number v-model="openfoamParams.writeInterval" :min="1" />
              </el-form-item>
              <el-form-item label="Grid X">
                <el-input-number v-model="openfoamParams.grid_x" :min="1" />
              </el-form-item>
              <el-form-item label="Grid Y">
                <el-input-number v-model="openfoamParams.grid_y" :min="1" />
              </el-form-item>
              <el-form-item label="Grid Z">
                <el-input-number v-model="openfoamParams.grid_z" :min="1" />
              </el-form-item>
              <el-form-item label="Inside Point X">
                <el-input-number v-model="openfoamParams.inside_point[0]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inside Point Y">
                <el-input-number v-model="openfoamParams.inside_point[1]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inside Point Z">
                <el-input-number v-model="openfoamParams.inside_point[2]" :step="0.1" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- Boundary Conditions Section -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.boundaryConditions = !sectionOpen.boundaryConditions" style="cursor:pointer;">
          <span>Boundary Conditions</span>
          <el-icon :class="{'rotate': sectionOpen.boundaryConditions}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.boundaryConditions" class="section-content">
            <el-form label-width="160px">
              <el-form-item label="Wall X Min">
                <el-input-number v-model="boundaryConditions.wall_x_min" :step="0.1" />
              </el-form-item>
              <el-form-item label="Wall X Max">
                <el-input-number v-model="boundaryConditions.wall_x_max" :step="0.1" />
              </el-form-item>
              <el-form-item label="Wall Y Min">
                <el-input-number v-model="boundaryConditions.wall_y_min" :step="0.1" />
              </el-form-item>
              <el-form-item label="Wall Y Max">
                <el-input-number v-model="boundaryConditions.wall_y_max" :step="0.1" />
              </el-form-item>
              <el-form-item label="Wall Z Min">
                <el-input-number v-model="boundaryConditions.wall_z_min" :step="0.1" />
              </el-form-item>
              <el-form-item label="Wall Z Max">
                <el-input-number v-model="boundaryConditions.wall_z_max" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inlet Velocity X">
                <el-input-number v-model="boundaryConditions.inlet_velocity[0]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inlet Velocity Y">
                <el-input-number v-model="boundaryConditions.inlet_velocity[1]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inlet Velocity Z">
                <el-input-number v-model="boundaryConditions.inlet_velocity[2]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inlet Center X">
                <el-input-number v-model="boundaryConditions.inlet_center[0]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inlet Center Y">
                <el-input-number v-model="boundaryConditions.inlet_center[1]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inlet Center Z">
                <el-input-number v-model="boundaryConditions.inlet_center[2]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Inlet Radius">
                <el-input-number v-model="boundaryConditions.inlet_radius" :min="0" :step="0.1" />
              </el-form-item>
              <el-form-item label="Outlet Center X">
                <el-input-number v-model="boundaryConditions.outlet_center[0]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Outlet Center Y">
                <el-input-number v-model="boundaryConditions.outlet_center[1]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Outlet Center Z">
                <el-input-number v-model="boundaryConditions.outlet_center[2]" :step="0.1" />
              </el-form-item>
              <el-form-item label="Outlet Radius">
                <el-input-number v-model="boundaryConditions.outlet_radius" :min="0" :step="0.1" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- FreeStreamConditions Section -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.freeStream = !sectionOpen.freeStream" style="cursor:pointer;">
          <span>Free Stream Conditions</span>
          <el-icon :class="{'rotate': sectionOpen.freeStream}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.freeStream" class="section-content">
            <el-form label-width="160px">
              <el-form-item label="Density">
                <el-input-number v-model="freeStream.free_stream_density" :min="0" />
              </el-form-item>
              <el-form-item label="Pressure">
                <el-input-number v-model="freeStream.free_stream_pressure" :min="0" />
              </el-form-item>
              <el-form-item label="Temperature">
                <el-input-number v-model="freeStream.free_stream_temperature" :min="0" />
              </el-form-item>
              <el-form-item label="Velocity">
                <div class="vector-input">
                  <el-input-number v-model="freeStream.free_stream_velocity[0]" controls-position="" :controls="false" />
                  <el-input-number v-model="freeStream.free_stream_velocity[1]" controls-position="" :controls="false" />
                  <el-input-number v-model="freeStream.free_stream_velocity[2]" controls-position="" :controls="false" />
                </div>
              </el-form-item>
              <el-form-item label="Viscosity Model">
                <el-select v-model="freeStream.viscosity_model" style="width: 100%">
                  <el-option label="CONSTANT_VISCOSITY" value="CONSTANT_VISCOSITY" />
                </el-select>
              </el-form-item>
              <el-form-item label="Thermodynamic Pressure">
                <el-input-number v-model="freeStream.thermodynamic_pressure" :min="0" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- Generate Button -->
      <div class="button-container">
        <div class="button-group">
          <el-button type="primary" @click="handleGenerate">Generate</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  computed,
  inject,
  reactive,
  ref,
} from 'vue';

import { ElMessage } from 'element-plus';
import * as THREE from 'three';

import { CenterRoot } from '@/classes/center_root';
import { GroupCell } from '@/classes/path_smooth/smooth_nodes';
import { SmoothRunner } from '@/classes/path_smooth/smooth_runner';
import {
  ArrowDown,
  Close,
} from '@element-plus/icons-vue';

const rootObj = inject<CenterRoot>('centerRoot', new CenterRoot(new THREE.Scene()))

const props = defineProps<{
    modelValue: boolean,
    label_dict: Map<
        string,
        string | number | string[] | Map<string, string | number> | Map<
            string, Map<string, Map<string, string | number | number[] | boolean>>
        >
    >
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref({
    name: '',
    min_size: 0.1,
    max_size: 0.25,
});

// Section collapse state
const sectionOpen = reactive({
    runConfig: true,
    solverConfig: false,
    turbulenceModel: false,
    openfoamParams: false,
    boundaryConditions: false,
    freeStream: false,
    mesh: false,
});

// Form data for each section (initialize with all required fields)
const runConfig = ref({
    run_iter: 100,
    conv_field: 'RMS_PRESSURE',
    conv_level: -6,
    conv_filename: 'conv_history',
    restart_filename: 'restart_flow.dat',
    read_binary_restart: true,
    restart_sol: false,
    with_screen_output: false,
    output_wrt_freq: 100,
});
const solverConfig = ref({
    solver: 'INC_RANS',
    inc_nondim: 'DIMENSIONAL',
    fluid_model: 'CONSTANT_DENSITY',
    inc_density_model: 'CONSTANT',
    inc_density_init: 1.225,
    body_force: 'NO',
    body_force_vector: [0, 0, 0],
    inc_velocity_init: [1, 0, 0],
    inc_temperature_init: 300,
    inc_energy_equation: 'YES',
    turbulent_conductivity_model: 'CONSTANT_PRANDTL_TURB',
    prandtl_lam: 0.72,
    prandtl_turb: 0.9,
});
// Turbulence model selection
const turbulenceModelType = ref('SST');
const turbulenceModelSST = ref({
    kind_turb_model: 'SST',
    config_name: 'default',
    free_stream_turbulence_intensity: 0.05,
    free_stream_turb2lam_visc_ratio: 10,
});
const turbulenceModelSA = ref({
    kind_turb_model: 'SA',
    config_name: 'default',
    free_stream_nu_factor: 3,
});
// Use computed to get the correct turbulence model for emit
const turbulenceModel = computed(() => {
    return turbulenceModelType.value === 'SST' ? turbulenceModelSST.value : turbulenceModelSA.value;
});

// OpenFOAM Parameters
const openfoamParams = ref({
    solver: 'topoIncompressibleFluid',
    numerical_method: 'PIMPLE',
    deltaT: 0.005,
    endTime: 10.0,
    writeInterval: 100,
    grid_x: 30,
    grid_y: 30,
    grid_z: 30,
    inside_point: [2.0, 2.0, 1.0],
});

// Boundary Conditions
const boundaryConditions = ref({
    wall_x_min: 0,
    wall_x_max: 4,
    wall_y_min: 0,
    wall_y_max: 4,
    wall_z_min: 0,
    wall_z_max: 2,
    inlet_velocity: [2.0, 0.0, 0.0],
    inlet_center: [0.0, 2.0, 1.0],
    inlet_radius: 0.5,
    outlet_center: [4.0, 2.0, 1.0],
    outlet_radius: 0.5,
});

const freeStream = ref({
    free_stream_density: 1.225,
    free_stream_pressure: 101325.0,
    free_stream_temperature: 300.0,
    free_stream_velocity: [1.0, 0.0, 0.0],
    viscosity_model: 'CONSTANT_VISCOSITY',
    thermodynamic_pressure: 73528282.41,
});

const handleClose = () => {
    emit('update:modelValue', false)
}

const handleGenerate = () => {
    if (!formData.value.name.trim()) {
        ElMessage.warning('Please enter a name!')
        return
    }

    const project = props.label_dict.get('Project')!.toString()
    const group_name = props.label_dict.get('label')!.toString()
    const smooth_runner = rootObj.proj_dict
        .get(project)!
        .smooth_task_dict.get(props.label_dict.get('SmoothName') as string
        ) as SmoothRunner
    const group_cell: GroupCell = smooth_runner.group_cells.get(group_name)!

    emit('generate', {
        project_name: project,
        name: formData.value.name,
        simulation_json: group_cell.export_simulation_json(),
        run_config: runConfig.value,
        solver_config: solverConfig.value,
        turbulence_model: turbulenceModel.value,
        openfoam_params: openfoamParams.value,
        boundary_conditions: boundaryConditions.value,
        free_stream: freeStream.value,
        mesh_config: {
            min_size: formData.value.min_size,
            max_size: formData.value.max_size,
        },
    })
    emit('update:modelValue', false)
}

</script>

<style scoped>
.right-sidebar {
    position: fixed;
    right: 0;
    top: 124px;
    width: 500px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-left: 1px solid rgba(226, 232, 240, 0.6);
    height: calc(100vh - 124px);
    display: flex;
    flex-direction: column;
    z-index: 1000;
    box-shadow: -8px 0 32px rgba(0, 0, 0, 0.12);
}

.right-sidebar-header {
    padding: 16px 20px;
    border-bottom: 1px solid rgba(226, 232, 240, 0.6);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
    flex: 0 0 auto;
}

.right-sidebar-header span {
    font-weight: 600;
    color: #1e293b;
    font-size: 16px;
    letter-spacing: 0.025em;
}

.close-button {
    padding: 10px;
    border-radius: 10px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color: #64748b;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(226, 232, 240, 0.4);
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 600;
}

.close-button:hover {
    color: #ffffff;
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    border-color: #ef4444;
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
}

.right-sidebar-content {
    position: relative;
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 20px;
    padding-bottom: 100px;
    min-height: 0;
    background: rgba(248, 250, 252, 0.3);
}

/* 现代化滚动条样式 */
.right-sidebar-content::-webkit-scrollbar {
    width: 4px;
}

.right-sidebar-content::-webkit-scrollbar-track {
    background: transparent;
}

.right-sidebar-content::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #cbd5e1 0%, #94a3b8 100%);
    border-radius: 6px;
    transition: all 0.2s ease;
}

.right-sidebar-content::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #94a3b8 0%, #64748b 100%);
}

.section {
    margin-bottom: 24px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(226, 232, 240, 0.4);
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.section:hover {
    background: rgba(255, 255, 255, 0.8);
    border-color: rgba(99, 102, 241, 0.2);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.08);
    transform: translateY(-1px);
}

.section-header {
    font-weight: 600;
    margin-bottom: 16px;
    color: #1e293b;
    font-size: 15px;
    letter-spacing: 0.025em;
    border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    padding-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    transition: all 0.3s ease;
}

.section-header:hover {
    color: #6366f1;
    border-bottom-color: rgba(99, 102, 241, 0.4);
}

.section-content {
    padding: 0;
}

.vector-input,
.range-input {
    display: flex;
    align-items: center;
    gap: 8px;
}

.vector-input :deep(.el-input-number),
.range-input :deep(.el-input-number) {
    width: 60px;
}

.range-separator {
    color: #909399;
    font-size: 14px;
}

:deep(.el-form-item) {
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

:deep(.el-form-item__content) {
    display: flex;
    align-items: center;
    min-height: 32px;
    flex: 1;
    margin-left: 0;
}

:deep(.el-form-item__label) {
    padding-bottom: 6px;
    line-height: 1.4;
    font-weight: 500;
    color: #374151;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    text-align: right;
    word-break: break-word;
    hyphens: auto;
    min-height: 32px;
    padding-right: 16px;
    white-space: normal;
    overflow-wrap: break-word;
    width: 160px;
    flex-shrink: 0;
}

:deep(.el-input) {
    border-radius: 8px;
    transition: all 0.3s ease;
}

:deep(.el-input__wrapper) {
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

:deep(.el-input__wrapper:hover) {
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.15);
}

:deep(.el-input__wrapper.is-focus) {
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
}

:deep(.el-input-number) {
    width: 100%;
    display: flex;
    align-items: center;
}

:deep(.el-input-number .el-input__wrapper) {
    border-radius: 8px;
}

:deep(.el-input-number .el-input-number__decrease),
:deep(.el-input-number .el-input-number__increase) {
    border-radius: 8px;
    transition: all 0.3s ease;
}

:deep(.el-input-number .el-input-number__decrease:hover),
:deep(.el-input-number .el-input-number__increase:hover) {
    background-color: #6366f1;
    color: white;
}

:deep(.el-select) {
    width: 100%;
}

:deep(.el-select .el-input__wrapper) {
    border-radius: 8px;
}

:deep(.el-switch) {
    display: flex;
    align-items: center;
    height: 32px;
}

:deep(.el-switch__core) {
    border-radius: 16px;
    transition: all 0.3s ease;
}

:deep(.el-switch.is-checked .el-switch__core) {
    background-color: #6366f1;
    border-color: #6366f1;
}

.button-container {
    position: fixed;
    right: 0;
    bottom: 0;
    width: 500px;
    padding: 20px;
    border-top: 1px solid rgba(226, 232, 240, 0.6);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
    backdrop-filter: blur(20px);
    z-index: 1100;
}

.button-group {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
}

.button-group .el-button {
    min-width: 100px;
    height: 40px;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.button-group .el-button--primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    border: none;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.button-group .el-button--primary:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
}

.button-group .el-button--default {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(226, 232, 240, 0.6);
    color: #64748b;
}

.button-group .el-button--default:hover {
    background: rgba(255, 255, 255, 1);
    border-color: rgba(99, 102, 241, 0.3);
    color: #6366f1;
    transform: translateY(-1px);
}
</style>