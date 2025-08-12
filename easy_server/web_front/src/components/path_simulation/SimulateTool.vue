<!-- SimulateTool.vue -->
<template>
  <div class="path-search" @click="handleClickOutside">
    <el-menu mode="horizontal" class="menu-bar">
      <el-dropdown trigger="click" placement="bottom">
        <span class="el-dropdown-link">
          Help<el-icon class="el-icon--right"><arrow-down /></el-icon>
        </span>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item>About</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </el-menu>

    <DefineSimulationTask v-model="defineSimulationTaskVisible" :label_dict="label_dict"
      @generate="handleSimulationTaskGenerate" />

    <DefineBoundary v-model="defineBoundaryVisible" :mode="defineBoundaryMode" :boundary_list="boundary_list"
      :label_dict="label_dict" @generate="handleBoundaryGenerate" />

    <div v-show="groupCellContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/CreateSimulationTask')">Create Simulation
          Task</el-menu-item>
      </el-menu>
    </div>

    <div v-show="simulateTaskContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SimulateTask/RunSimulation')">Run Simulation</el-menu-item>
      </el-menu>
    </div>

    <div v-show="networkContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SimulateNetwork/hide_internal_mesh')">Hide internal
          mesh</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SimulateNetwork/show_internal_mesh')">Show internal
          mesh</el-menu-item>
      </el-menu>
    </div>

    <div v-show="boundaryContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SimulateBoundary/hide_boundary_mesh')">Hide Boundary
          Mesh</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SimulateBoundary/show_boundary_mesh')">Show Boundary
          Mesh</el-menu-item>
      </el-menu>
    </div>

    <div v-show="inletsContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Inlets/add_inlet')">Add Inlet</el-menu-item>
      </el-menu>
    </div>

    <div v-show="outletsContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Outlets/add_outlet')">Add Outlet</el-menu-item>
      </el-menu>
    </div>

    <div v-show="wallsContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Walls/add_wall')">Add Wall</el-menu-item>
      </el-menu>
    </div>

    <el-dialog v-model="progressDialogVisible" title="Progress" width="30%">
      <div>
        <p>{{ progressLog }}</p>
        <el-progress :percentage="progressPercentage" />
      </div>
    </el-dialog>

  </div>
</template>

<script setup lang="ts">
import { ArrowDown } from '@element-plus/icons-vue';

import {
  computed,
  defineEmits,
  defineExpose,
  inject,
  onMounted,
  onUnmounted,
  reactive,
  ref,
} from 'vue';
import * as THREE from 'three';

import { CenterRoot } from '@/classes/center_root.ts';
import DefineSimulationTask from './DefineSimulationTask.vue';
import { ElMessage } from 'element-plus';
import { SimulateTask } from '@/classes/path_simulate/simulate_task';
import { SimulateRunConfig } from '@/classes/path_simulate/simulate_task';
import { SimulateSolverConfig } from '@/classes/path_simulate/simulate_task';
import { ShearStressTransportTurbulenceModel, SpalartAllmarasTurbulenceModel, TurbulenceModel } from '@/classes/path_simulate/simulate_task';
import { FreeStreamConditions } from '@/classes/path_simulate/simulate_task';
import config from '@/config';

import DefineBoundary from './DefineBoundary.vue';

import {
  SimulateTotalConditionsInletCfg, SimulateMassflowInletCfg,
  SimulatePressureInletCfg, SimulateVelocityInletCfg,
  SimulateMassflowOutletCfg, SimulatePressureOutletCfg, SimulateVelocityOutletCfg,
  SimulateIsothermalWallCfg,
} from '@/classes/path_simulate/simulate_task';

const emit = defineEmits(['updateTree', 'highlightTreeNode', 'updateLog'])

const rootObj = inject<CenterRoot>(
  'centerRoot',
  new CenterRoot(new THREE.Scene())
)

const label_dict = ref(
  new Map<
    string,
    string | number | string[] | Map<string, string | number> | Map<
      string, Map<string, Map<string, string | number | number[] | boolean>>
    >
  >()
)

const groupCellContextMenuVisible = ref(false)
const networkContextMenuVisible = ref(false)
const boundaryContextMenuVisible = ref(false)
const inletsContextMenuVisible = ref(false)
const outletsContextMenuVisible = ref(false)
const wallsContextMenuVisible = ref(false)
const simulateTaskContextMenuVisible = ref(false)
const defineSimulationTaskVisible = ref(false)
const defineBoundaryVisible = ref(false)
const defineBoundaryMode = ref<'Inlet' | 'Outlet' | 'Wall'>('Inlet')
const boundary_list = ref<string[]>([])

// ------ progress dialog ------
const progressDialogVisible = ref(false)
const progressPercentage = ref(0)
const progressLog = ref('')

// ------ Left Side Bar ------
const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

const hideContextMenu = () => {
  groupCellContextMenuVisible.value = false
  networkContextMenuVisible.value = false
  boundaryContextMenuVisible.value = false
  inletsContextMenuVisible.value = false
  outletsContextMenuVisible.value = false
  wallsContextMenuVisible.value = false
  simulateTaskContextMenuVisible.value = false
}

const hideFragmentMenu = () => {
  defineSimulationTaskVisible.value = false
  defineBoundaryVisible.value = false
}

const showContextMenu = (data: any) => {
  console.log('showContextMenu', data)

  hideContextMenu()

  // Set menu position
  contextMenuPosition.x = data.x
  contextMenuPosition.y = data.y

  // Convert data.data to Map
  const newLabelDict = new Map<
    string,
    string | number | string[] | Map<string, string | number>
  >()

  // Add basic properties
  if (data.data.label) newLabelDict.set('label', data.data.label)
  if (data.data.style) newLabelDict.set('style', data.data.style)

  // Add any additional data properties
  if (data.data.data) {
    Object.entries(data.data.data).forEach(([key, value]) => {
      newLabelDict.set(
        key,
        value as string | number | string[] | Map<string, string | number>
      )
    })
  }

  label_dict.value = newLabelDict

  const style = label_dict.value.get('style')
  if (style === 'GroupCell') {
    groupCellContextMenuVisible.value = true
  }
  else if (style === 'SimulateNetwork') {
    networkContextMenuVisible.value = true
  }
  else if (style === 'SimulateBoundary') {
    boundaryContextMenuVisible.value = true
  }
  else if (style === 'SimulateTask') {
    simulateTaskContextMenuVisible.value = true
  }
  else if (style === 'Inlets') {
    inletsContextMenuVisible.value = true
  }
  else if (style === 'Outlets') {
    outletsContextMenuVisible.value = true
  }
  else if (style === 'Walls') {
    wallsContextMenuVisible.value = true
  }


  // ---- 自适应菜单位置，防止被遮挡 ----
  setTimeout(() => {
    const menus = Array.from(document.querySelectorAll('.context-menu')) as HTMLElement[];
    let menu: HTMLElement | null = null;
    menus.forEach((m: HTMLElement) => {
      if (getComputedStyle(m).display !== 'none') {
        menu = m;
      }
    });
    if (menu) {
      const menuElement = menu as HTMLElement;
      const menuHeight = menuElement.offsetHeight;
      const menuWidth = menuElement.offsetWidth;
      const windowHeight = window.innerHeight;
      const windowWidth = window.innerWidth;
      let top = data.y;
      let left = data.x;

      // 下方空间
      const spaceBelow = windowHeight - data.y;
      // 上方空间
      const spaceAbove = data.y;

      // 优先显示在鼠标下方
      if (spaceBelow >= menuHeight + 8) {
        top = data.y;
        menuElement.style.maxHeight = '';
        menuElement.style.overflowY = '';
        const elMenu = menuElement.querySelector('.el-menu') as HTMLElement | null;
        if (elMenu) {
          elMenu.style.maxHeight = '';
          elMenu.style.overflowY = '';
        }
      } else {
        // 下方空间不够，菜单底部贴近窗口底部，但顶部不会高于鼠标Y
        let maxHeight = Math.max(spaceBelow, spaceAbove) - 8;
        if (maxHeight < 80) maxHeight = 80; // 最小高度
        // 菜单顶部最多到鼠标Y
        top = Math.max(8, windowHeight - menuHeight - 8, data.y - (menuHeight - spaceBelow));
        menuElement.style.maxHeight = maxHeight + 'px';
        menuElement.style.overflowY = 'auto';
        const elMenu = menuElement.querySelector('.el-menu') as HTMLElement | null;
        if (elMenu) {
          elMenu.style.maxHeight = maxHeight + 'px';
          elMenu.style.overflowY = 'auto';
        }
      }

      // 右边界
      if (data.x + menuWidth > windowWidth) {
        left = windowWidth - menuWidth - 8;
        if (left < 0) left = 0;
        contextMenuPosition.x = left;
      }
      contextMenuPosition.y = top;
    }
  }, 0);
}

const handleTreeContextMenu = (mode: string) => {
  hideContextMenu()
  hideFragmentMenu()

  if (mode === 'GroupCell/CreateSimulationTask') {
    defineSimulationTaskVisible.value = true
  } else if (mode === 'SimulateNetwork/hide_internal_mesh') {
    const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
    simulate_task!.network_mesh!.hide_internal_mesh()
  } else if (mode === 'SimulateNetwork/show_internal_mesh') {
    const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
    simulate_task!.network_mesh!.show_internal_mesh()
  } else if (mode === 'SimulateBoundary/hide_boundary_mesh') {
    const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
    simulate_task!.network_mesh!.hide_boundary_mesh(label_dict.value.get('SimulateBoundary') as string)
  } else if (mode === 'SimulateBoundary/show_boundary_mesh') {
    const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
    simulate_task!.network_mesh!.show_boundary_mesh(label_dict.value.get('SimulateBoundary') as string)
  } else if (mode === 'SimulateTask/RunSimulation') {
    run_simulation(label_dict.value.get('SimulationName') as string)
  } else if (mode === 'Inlets/add_inlet') {
    defineBoundaryMode.value = 'Inlet'
    const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
    boundary_list.value = simulate_task!.network_mesh!.get_boundary_list()
    defineBoundaryVisible.value = true
  } else if (mode === 'Outlets/add_outlet') {
    defineBoundaryMode.value = 'Outlet'
    const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
    boundary_list.value = simulate_task!.network_mesh!.get_boundary_list()
    defineBoundaryVisible.value = true
  } else if (mode === 'Walls/add_wall') {
    defineBoundaryMode.value = 'Wall'
    const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
    boundary_list.value = simulate_task!.network_mesh!.get_boundary_list()
    defineBoundaryVisible.value = true
  }
}

const handleClickOutside = (event: MouseEvent) => {
  // Check if click is outside of any context menu
  const contextMenus = document.querySelectorAll('.context-menu')
  let clickedInsideMenu = false

  contextMenus.forEach(menu => {
    if (menu.contains(event.target as Node)) {
      clickedInsideMenu = true
    }
  })

  if (!clickedInsideMenu) {
    hideContextMenu()
  }
}

const handleSimulationTaskGenerate = async (data: any) => {
  const simulate_run_config = new SimulateRunConfig(
    data.run_config.conv_field,
    data.run_config.conv_field_tol,
    data.run_config.max_iter,
    data.run_config.min_iter,
    data.run_config.residual_tol,
    data.run_config.residual_tol_abs,
    data.run_config.residual_tol_rel,
  )

  const simulate_solver_config = new SimulateSolverConfig(
    data.solver_config.solver,
    data.solver_config.inc_nondim,
    data.solver_config.fluid_model,
    data.solver_config.inc_density_model,
    data.solver_config.inc_density_init,
  )

  let simulate_turbulence_model = null
  if (data.turbulence_model.kind_turb_model === 'SST') {
    simulate_turbulence_model = new ShearStressTransportTurbulenceModel(
      data.turbulence_model.config_name,
      data.turbulence_model.free_stream_turbulence_intensity,
      data.turbulence_model.free_stream_turb2lam_visc_ratio,
    )
  } else if (data.turbulence_model.kind_turb_model === 'SA') {
    simulate_turbulence_model = new SpalartAllmarasTurbulenceModel(
      data.turbulence_model.config_name,
      data.turbulence_model.free_stream_nu_factor,
    )
  } else {
    ElMessage.error('Invalid turbulence model')
  }

  const simulate_free_stream = new FreeStreamConditions(
    data.free_stream.free_stream_density,
    data.free_stream.free_stream_pressure,
    data.free_stream.free_stream_temperature,
    data.free_stream.free_stream_velocity,
    data.free_stream.viscosity_model,
    data.free_stream.thermodynamic_pressure,
  )

  if (!await upload_vertices(data.name, data.simulation_json.vertices)) {
    ElMessage.error('Failed to upload vertices')
    return
  }
  if (!await upload_cells(data.name, data.simulation_json.cells)) {
    ElMessage.error('Failed to upload cells')
    return
  }

  const [is_success, meshInfo, cells, points] = await upload_task(
    data.name, data.simulation_json.terminal_cfg, data.mesh_config.min_size, data.mesh_config.max_size
  )
  if (!is_success) {
    ElMessage.error('Failed to upload task')
    return
  }

  const simulateTask = new SimulateTask(
    data.name,
    points as number[][],
    cells as number[][],
    meshInfo as Map<string, Map<string, number | number[][]>>,
    rootObj.get_scene()
  )
  simulateTask.edit_free_stream_conditions(simulate_free_stream)
  simulateTask.edit_run_config(simulate_run_config)
  simulateTask.edit_solver_config(simulate_solver_config)
  simulateTask.edit_turbulence_model(simulate_turbulence_model as TurbulenceModel)

  rootObj.proj_dict.get(data.project_name)!.simulate_task_dict.set(data.name, simulateTask)

  emit('updateTree')
}

const upload_vertices = async (simulation_name: string, vertices: number[][]) => {
  const url = `${config.BASE_URL}/simulate/init_simulation_vertices`

  progressDialogVisible.value = true
  progressPercentage.value = 0
  progressLog.value = 'Uploading static obstacle pcd...'

  // Define chunk size
  const CHUNK_SIZE = 5000 // Number of points per chunk
  const total_chunks = Math.ceil(vertices.length / CHUNK_SIZE)

  for (let i = 0; i < vertices.length; i += CHUNK_SIZE) {
    const chunk_data = vertices.slice(i, i + CHUNK_SIZE)
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          simulation_name: simulation_name,
          vertices: chunk_data,
          chunk_index: i,
          total_chunks: total_chunks,
          is_last_chunk: i + CHUNK_SIZE >= vertices.length
        })
      })

      if (!response.ok) {
        const errorText = await response.text();
        ElMessage.error(errorText)
        progressDialogVisible.value = false
        return false
      }

      const data = await response.json()
      if (data.error) {
        ElMessage.error(data.error)
        progressDialogVisible.value = false
        return false
      }

      const progress = Math.round(((i + 1) / vertices.length) * 100)
      progressPercentage.value = progress
      progressLog.value = `Uploading vertices: ${progress}%`
    } catch (error) {
      ElMessage.error('Failed to upload vertices')
      progressDialogVisible.value = false
      return false
    }
    await new Promise(resolve => setTimeout(resolve, 200))
  }
  progressDialogVisible.value = false
  progressPercentage.value = 100
  progressLog.value = 'Vertices uploaded successfully'
  return true
}

const upload_cells = async (simulation_name: string, cells: number[][]) => {
  const url = `${config.BASE_URL}/simulate/init_simulation_cells`

  progressDialogVisible.value = true
  progressPercentage.value = 0
  progressLog.value = 'Uploading static obstacle pcd...'

  // Define chunk size
  const CHUNK_SIZE = 5000 // Number of points per chunk
  const total_chunks = Math.ceil(cells.length / CHUNK_SIZE)

  for (let i = 0; i < cells.length; i += CHUNK_SIZE) {
    const chunk_data = cells.slice(i, i + CHUNK_SIZE)
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          simulation_name: simulation_name,
          cells: chunk_data,
          chunk_index: i,
          total_chunks: total_chunks,
          is_last_chunk: i + CHUNK_SIZE >= cells.length
        })
      })

      if (!response.ok) {
        const errorText = await response.text();
        ElMessage.error(errorText)
        progressDialogVisible.value = false
        return false
      }

      const data = await response.json()
      if (data.error) {
        ElMessage.error(data.error)
        progressDialogVisible.value = false
        return false
      }

      const progress = Math.round(((i + 1) / cells.length) * 100)
      progressPercentage.value = progress
      progressLog.value = `Uploading cells: ${progress}%`
    } catch (error) {
      ElMessage.error('Failed to upload cells')
      progressDialogVisible.value = false
      return false
    }
    await new Promise(resolve => setTimeout(resolve, 200))
  }
  progressDialogVisible.value = false
  progressPercentage.value = 100
  progressLog.value = 'Cells uploaded successfully'
  return true
}

const upload_task = async (simulation_name: string, terminal_cfg: any, min_size: number, max_size: number) => {
  const url = `${config.BASE_URL}/simulate/init_simulation_task`

  progressDialogVisible.value = true
  progressPercentage.value = 0
  progressLog.value = 'Uploading task...'

  let meshInfo: any = null
  let cells: number[][] = []
  let points: number[][] = []
  let marker_dict: Map<string, Map<string, number | number[][]>> = new Map()

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        simulation_name: simulation_name,
        terminal_cfg: terminal_cfg,
        min_size: min_size,
        max_size: max_size
      })
    })

    if (!response.ok) {
      ElMessage.error('Network response was not ok')
      return [false, null, null, null]
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('Failed to get response reader')

    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = new TextDecoder().decode(value)
      buffer += text

      // Split buffer by newlines and process each JSON object
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep the last incomplete line in buffer

      for (const line of lines) {
        if (line.trim() === '') continue

        try {
          const data = JSON.parse(line)

          if (data.name === 'init_simulation_task') {
            emit('updateLog', data.msg)
          }
          else if (data.name === 'return_mesh_info') {
            progressLog.value = 'Processing mesh info...'
            meshInfo = data.data
            progressPercentage.value = 100
          } else if (data.name === 'return_cells') {
            progressLog.value = 'Processing cells data...'
            progressPercentage.value = data.progress
            if (data.data) {
              cells.push(...data.data)
            }
          } else if (data.name === 'return_points') {
            progressLog.value = 'Processing points data...'
            progressPercentage.value = data.progress
            if (data.data) {
              points.push(...data.data)
            }
          }

          if (data.total_finished) {
            if (meshInfo && cells.length > 0 && points.length > 0) {
              Object.entries(meshInfo.marker_dict as Record<string, Record<string, number | number[][]>>).forEach(([k, v]) => {
                marker_dict.set(k, new Map(Object.entries(v)))
              })
            } else {
              return [false, null, null, null]
            }

            break
          }

          buffer = ''
        } catch (error) {
          continue
        }

      }
    }

    progressDialogVisible.value = false
    progressPercentage.value = 100
    progressLog.value = 'Task uploaded successfully'
    return [true, marker_dict, cells, points]

  } catch (error) {
    ElMessage.error('Failed to upload task')
    progressDialogVisible.value = false
    return [false, null, null, null]
  } finally {
    progressDialogVisible.value = false
  }
}

const handleBoundaryGenerate = (data: any) => {
  const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
  if (data.mode === 'Inlet') {
    if (data.boundary.type === 'TOTAL_CONDITIONS') {
      const inlet = new SimulateTotalConditionsInletCfg(data.market)
      inlet.direction = data.boundary.direction
      inlet.pressure = data.boundary.pressure
      inlet.temperature = data.boundary.temperature
      simulate_task!.edit_inlet(inlet.market, inlet)
    }
    else if (data.boundary.type === 'MASS_FLOW_INLET') {
      const inlet = new SimulateMassflowInletCfg(data.market)
      inlet.direction = data.boundary.direction
      inlet.density = data.boundary.density
      inlet.vel_scale = data.boundary.vel_scale
      simulate_task!.edit_inlet(inlet.market, inlet)
    }
    else if (data.boundary.type === 'PRESSURE_INLET') {
      const inlet = new SimulatePressureInletCfg(data.market)
      inlet.direction = data.boundary.direction
      inlet.pressure = data.boundary.pressure
      inlet.temperature = data.boundary.temperature
      simulate_task!.edit_inlet(inlet.market, inlet)
    }
    else if (data.boundary.type === 'VELOCITY_INLET') {
      const inlet = new SimulateVelocityInletCfg(data.market)
      inlet.direction = data.boundary.direction
      inlet.vel_scale = data.boundary.vel_scale
      inlet.temperature = data.boundary.temperature
      simulate_task!.edit_inlet(inlet.market, inlet)
    }
    else {
      ElMessage.error('Invalid inlet type')
      throw new Error('Invalid inlet type')
    }

  } else if (data.mode === 'Outlet') {
    if (data.boundary.type === 'PRESSURE_OUTLET') {
      const outlet = new SimulatePressureOutletCfg(data.market)
      outlet.pressure = data.boundary.pressure
      simulate_task!.edit_outlet(outlet.market, outlet)
    }
    else if (data.boundary.type === 'MASS_FLOW_OUTLET') {
      const outlet = new SimulateMassflowOutletCfg(data.market)
      outlet.mass_flow = data.boundary.mass_flow
      simulate_task!.edit_outlet(outlet.market, outlet)
    }
    else if (data.boundary.type === 'VELOCITY_OUTLET') {
      const outlet = new SimulateVelocityOutletCfg(data.market)
      outlet.temperature = data.boundary.temperature
      outlet.vel_scale = data.boundary.vel_scale
      outlet.direction = data.boundary.direction
      simulate_task!.edit_outlet(outlet.market, outlet)
    }
    else {
      ElMessage.error('Invalid outlet type')
      throw new Error('Invalid outlet type')
    }
  } else if (data.mode === 'Wall') {
    if (data.boundary.type === 'MARKER_ISOTHERMAL') {
      const wall = new SimulateIsothermalWallCfg(data.market)
      wall.temperature = data.boundary.temperature
      simulate_task!.edit_wall(wall.market, wall)
    }
    else {
      ElMessage.error('Invalid wall type')
      throw new Error('Invalid wall type')
    }
  }

  hideFragmentMenu()
  emit('updateTree')
}

const run_simulation = async (simulation_name: string) => {
  const simulate_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.simulate_task_dict.get(label_dict.value.get('SimulateTask') as string)
  const simulation_json = simulate_task!.export_simulation_json()

  console.log("simulation_json:", simulation_json)

  const url = `${config.BASE_URL}/simulate/run_simulation`

  // const response = await fetch(url, {
  //   method: 'POST',
  // })
}

// ------ Expose ------
defineExpose({
  hideContextMenu,
  showContextMenu
})

onMounted(() => {
  document.addEventListener('mousedown', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('mousedown', handleClickOutside)
})
</script>

<style scoped>
.path-search {
  width: 100%;
  position: relative;
}

.menu-bar {
  border-bottom: 1px solid #dcdfe6;
}

.context-menu {
  position: fixed;
  background: white;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  z-index: 3000;
  max-height: 80vh;
  overflow-y: auto;
}

.context-menu :deep(.el-menu) {
  border-right: none;
}

.context-menu :deep(.el-menu-item) {
  height: 32px;
  line-height: 32px;
  padding: 0 16px;
}

.context-menu :deep(.el-menu-item:hover) {
  background-color: #f5f7fa;
}

/* Add styles for click outside handler */
.click-outside-handler {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2999;
  background: transparent;
}
</style>
