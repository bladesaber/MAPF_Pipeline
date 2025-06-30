<!-- PathSmooth.vue -->
<template>
  <div class="path-smooth" @click="handleClickOutside">
    <el-menu mode="horizontal" class="menu-bar">
      <el-sub-menu index="1">
        <template #title>Tools</template>
        <el-menu-item index="1-1">Path Analysis</el-menu-item>
      </el-sub-menu>
      <el-sub-menu index="2">
        <template #title>Help</template>
        <el-menu-item index="2-1">About</el-menu-item>
      </el-sub-menu>
    </el-menu>

    <DefineSmoothInfo v-model="defineSmoothInfoVisible" :label_dict="label_dict"
      @generate="handleDefineSmoothInfoGenerate" />

    <ReportPlotly v-model="reportPlotlyVisible" ref="reportPlotlyRef" />

    <div v-show="smoothContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/AddSmoothTask')">Add Smooth Task</el-menu-item>
      </el-menu>
    </div>

    <div v-show="runSmoothContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SmoothRunner/ManualRun')">Manual Run</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SmoothRunner/AutoRun')">Auto Run</el-menu-item>
      </el-menu>
    </div>

    <!-- progress dialog -->
    <el-dialog v-model="progressDialogVisible" title="Progress" width="30%">
      <div>
        <p>{{ progressLog }}</p>
        <el-progress :percentage="progressPercentage" />
      </div>
    </el-dialog>

    <!-- manual dialog have two parameters and two buttons: lr and max_iter, cancel and run -->
    <el-dialog v-model="manualDialogVisible" title="Manual Run" width="30%">
      <div>
        <el-form :model="{ lr: manualLr, max_iter: manualMaxIter }" label-width="120px">
          <el-form-item label="Learning Rate">
            <el-input-number v-model="manualLr" :min="0.001" :max="1.0" :step="0.001" :precision="3"
              style="width: 200px" />
          </el-form-item>
          <el-form-item label="Max Iterations">
            <el-input-number v-model="manualMaxIter" :min="100" :max="100000" :step="100" style="width: 200px" />
          </el-form-item>
          <el-form-item label="Log Iterations">
            <el-input-number v-model="manualLogIter" :min="1" :max="1000" :step="1" style="width: 200px" />
          </el-form-item>
        </el-form>
        <p>{{ manualLog }}</p>
        <el-progress :percentage="runSmoothPercentage" />
      </div>
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="manualDialogVisible = false">Cancel</el-button>
          <el-button type="primary" @click="handleManualRun">Run</el-button>
        </div>
      </template>
    </el-dialog>

  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, defineExpose, inject, defineEmits } from 'vue'
import { pass, round } from 'three/tsl'
import { setVisableDict } from '@/utils/utils'
import * as THREE from 'three'
import { ElMessage } from 'element-plus'

import { CenterRoot } from '@/classes/center_root.ts'
import DefineSmoothInfo from '@/components/path_smooth/DefineSmoothInfo.vue'
import ReportPlotly from '@/components/ReportPlotly.vue'
import type { PipeMesh } from '@/classes/path_search/pipe_mesh'
import { PipeCfg, ObstacleCell, SmoothRunner } from '@/classes/path_smooth/smooth_runner'
import { GroupCell, PathCell, ConnectorCell, SegmentCell } from '@/classes/path_smooth/smooth_nodes'

const emit = defineEmits(['updateTree', 'highlightTreeNode', 'updateLog'])

const rootObj = inject<CenterRoot>('centerRoot', new CenterRoot(new THREE.Scene()))

// ------ progress dialog ------
const progressDialogVisible = ref(false)
const progressPercentage = ref(0)
const progressLog = ref('')

// ------ manual dialog ------
const manualDialogVisible = ref(false)
const manualLog = ref('')
const runSmoothPercentage = ref(0)
const manualLr = ref(0.01)
const manualMaxIter = ref(1000)
const manualLogIter = ref(50)

// ------ Left Side Bar ------
const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

const hideContextMenu = () => {
  smoothContextMenuVisible.value = false
  runSmoothContextMenuVisible.value = false
}

const hideFragmentMenu = () => {
  defineSmoothInfoVisible.value = false
  reportPlotlyVisible.value = false
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

const smoothContextMenuVisible = ref(false)
const runSmoothContextMenuVisible = ref(false)
const contextMenuVisibleDict = {
  'smoothContextMenuVisible': smoothContextMenuVisible,
  'runSmoothContextMenuVisible': runSmoothContextMenuVisible,
}

const defineSmoothInfoVisible = ref(false)
const reportPlotlyVisible = ref(false)
const reportPlotlyRef = ref()
const fragmentVisibleDict = {
  'defineSmoothInfoVisible': defineSmoothInfoVisible,
  'reportPlotlyVisible': reportPlotlyVisible,
}

const label_dict = ref(
  new Map<
    string,
    string | number | string[] | Map<string, string | number> |
    Map<string, Map<string, Map<string, string | number | number[] | boolean>>>
  >()
)

const showContextMenu = (data: any) => {
  // Reset all menu states first
  hideContextMenu()

  // Set menu position
  contextMenuPosition.x = data.x
  contextMenuPosition.y = data.y

  // Convert data.data to Map
  const newLabelDict = new Map<string, string | number | string[] | Map<string, string | number>>()

  // Add basic properties
  if (data.data.label) newLabelDict.set('label', data.data.label)
  if (data.data.style) newLabelDict.set('style', data.data.style)

  // Add any additional data properties
  if (data.data.data) {
    Object.entries(data.data.data).forEach(([key, value]) => {
      newLabelDict.set(key, value as string | number | string[] | Map<string, string | number>)
    })
  }

  label_dict.value = newLabelDict

  // Show appropriate menu based on style
  const style = label_dict.value.get('style')

  if (style === 'SearchTask') {
    smoothContextMenuVisible.value = true
  } else if (style === 'SmoothRunner') {
    runSmoothContextMenuVisible.value = true
  }
}

const handleTreeContextMenu = (mode: string) => {
  hideContextMenu()
  hideFragmentMenu()

  if (mode === 'SearchTask/AddSmoothTask') {
    const search_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(label_dict.value.get('SearchTask') as string)
    if (!search_task!.has_result) {
      ElMessage.error('Search task has result, please clear the result first')
      return
    }

    const project_manager = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!

    const group_pipe_dict = new Map<string, Map<string, Map<string, string | number | number[] | boolean>>>()
    for (const group_name of Array.from(project_manager.pipe_dict.keys())) {
      const pipe_dict = new Map<string, Map<string, string | number | number[] | boolean>>()
      for (const pipe_name of Array.from(project_manager.pipe_dict.get(group_name)!.keys())) {
        const pipe_mesh = project_manager.pipe_dict.get(group_name)!.get(pipe_name)!
        if (pipe_mesh.pipe_mode === 'inlet' || pipe_mesh.pipe_mode === 'outlet') {
          const pipe_info = new Map<string, string | number | number[] | boolean>()
          pipe_info.set('name', pipe_mesh.name)
          pipe_info.set('radius', pipe_mesh.radius)
          pipe_info.set('position', [pipe_mesh.x, pipe_mesh.y, pipe_mesh.z])
          pipe_info.set('direction', [pipe_mesh.vector_x, pipe_mesh.vector_y, pipe_mesh.vector_z])
          pipe_info.set('is_input', pipe_mesh.pipe_mode === 'inlet')
          pipe_info.set('radius_scale', 1)
          pipe_dict.set(pipe_mesh.name, pipe_info)
        }
      }
      group_pipe_dict.set(group_name, pipe_dict)
    }
    label_dict.value.set('GroupPipeDict', group_pipe_dict)
    defineSmoothInfoVisible.value = true
  } else if (mode === 'SmoothRunner/ManualRun') {
    manualDialogVisible.value = true
  } else if (mode === 'SmoothRunner/AutoRun') {

  }
}

// -------------------------------------------------------
const handleDefineSmoothInfoGenerate = async (data: any) => {
  console.log('data:', data)

  const search_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(data.search_task)
  if (!search_task!.has_result) {
    ElMessage.error('Search task has result, please clear the result first')
    return null
  }

  // --------------------------------------------
  const search_result: Record<string, Record<string, number[][]>> = {}
  search_task!.priority_tasks.forEach((priority_task, priority_name) => {
    priority_task.group_tasks.forEach((group_task, group_name) => {
      if (!search_result[group_name]) {
        search_result[group_name] = {}
      }
      group_task.cell_tasks.forEach((cell_task) => {
        search_result[group_name][cell_task.name] = []
        for (let i = 0; i < cell_task.result!.points.length; i++) {
          search_result[group_name][cell_task.name].push([
            cell_task.result!.points[i][0], cell_task.result!.points[i][1], cell_task.result!.points[i][2],
            cell_task.result!.radius
          ])
        }
      })
    })
  })

  // --------------------------------------------
  let obstacle_xyzr_list: number[][] = []

  const grid = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid
  if (grid.discretize_source == '') {
    ElMessage.warning('Grid has not been discretized')
    return null
  }
  grid.surface_obstacle_points.forEach(point => {
    obstacle_xyzr_list.push([point[0], point[1], point[2], 0.0])
  })

  rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.forEach((obstacle, key) => {
    if (obstacle.discretize_source == '') {
      ElMessage.warning('Obstacle has not been discretized')
      return null
    }
    if (obstacle.relax_groups.length == 0) {
      obstacle.discretize_points.forEach(point => {
        obstacle_xyzr_list.push([point[0], point[1], point[2], 0.0])
      })
    }
  })

  // --------------------------------------------
  const pipes_dict = new Map<string, PipeCfg>()
  Object.entries(data.pipes_config).forEach(([pipe_name, pipe_info]: [string, any]) => {
    pipes_dict.set(
      pipe_name,
      new PipeCfg(
        pipe_name, pipe_info.radius,
        pipe_info.position, pipe_info.direction,
        pipe_info.group_name, pipe_info.is_input, pipe_info.radius_scale
      )
    )
  })
  obstacle_xyzr_list = ObstacleCell.filter_xyzr(pipes_dict, obstacle_xyzr_list)

  if (await !upload_smooth_obstacle_pcd(data.smooth_name, obstacle_xyzr_list)) {
    ElMessage.error('Failed to upload smooth obstacle pcd')
    return null
  }

  const group_setting: Record<string, any> = {}
  Object.entries(data.aid_paths).forEach(([group_name, group_aid_paths]: [string, any]) => {
    group_setting[group_name] = {
      'aid_path_info': group_aid_paths,
      'graph_cfg': data.graph_configs[group_name],
      'path_result': search_result[group_name],
    }
  })

  const pipes_cfg: Record<string, any> = {}
  Object.entries(data.pipes_config).forEach(([pipe_name, pipe_info]: [string, any]) => {
    pipes_cfg[pipe_name] = {
      group_name: pipe_info.group_name,
      is_input: pipe_info.is_input,
      radius: pipe_info.radius,
      position: pipe_info.position,
      direction: pipe_info.direction,
      radius_scale: pipe_info.radius_scale,
    }
  })

  const smooth_json = {
    pipes_cfg: pipes_cfg,
    conflict_setting: data.conflict_setting,
    setting: group_setting,
    auto_run_cfg: data.auto_run_config
  }

  if (!await init_smooth_map(
    label_dict.value.get('Project') as string,
    data.search_task, data.smooth_name,
    smooth_json, obstacle_xyzr_list
  )) {
    ElMessage.error('Failed to initialize smooth map')
    return null
  }

  return true
}

const upload_smooth_obstacle_pcd = async (smooth_name: string, obstacle_xyzr_list: number[][]) => {
  const url = 'http://192.168.110.66:9998/path_smooth/upload_smooth_obstacle_pcd'

  progressDialogVisible.value = true
  progressPercentage.value = 0
  progressLog.value = 'Uploading static obstacle pcd...'

  // Define chunk size
  const CHUNK_SIZE = 5000 // Number of points per chunk
  const total_chunks = Math.ceil(obstacle_xyzr_list.length / CHUNK_SIZE)

  for (let i = 0; i < obstacle_xyzr_list.length; i += CHUNK_SIZE) {
    const chunk_data = obstacle_xyzr_list.slice(i, i + CHUNK_SIZE)
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          smooth_name: smooth_name,
          obstacle_xyzr: chunk_data,
          chunk_index: i,
          total_chunks: total_chunks,
          is_last_chunk: i === obstacle_xyzr_list.length - CHUNK_SIZE
        })
      })

      if (!response.ok) {
        ElMessage.error('Failed to upload smooth obstacle pcd')
        return false
      }

      const data = await response.json()
      if (data.error) {
        ElMessage.error(data.error)
        return false
      }

      const progress = Math.round(((i + 1) / obstacle_xyzr_list.length) * 100)
      progressPercentage.value = progress
      progressLog.value = `Generating inner grid: ${progress}%`

    } catch (error) {
      ElMessage.error('Failed to generate inner grid')
      progressDialogVisible.value = false
      return false
    }
    await new Promise(resolve => setTimeout(resolve, 200))
  }
  progressDialogVisible.value = false
  progressPercentage.value = 100
  progressLog.value = 'Inner grid generated successfully'
  return true
}

const init_smooth_map = async (
  project_name: string, search_task_name: string, smooth_name: string,
  smooth_json: any, obstacle_xyzr_list: number[][]
) => {
  const url = 'http://192.168.110.66:9998/path_smooth/init_smooth_map'
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      smooth_name: smooth_name,
      smooth_json: smooth_json
    })
  })
  if (!response.ok) {
    ElMessage.error('Failed to initialize smooth map')
    return false
  }
  const data = await response.json()
  const reflect_smooth_json = data.smooth_json
  console.log('reflect_smooth_json:', reflect_smooth_json)

  const smooth_runner = new SmoothRunner(smooth_name, search_task_name, project_name)

  // ------ conflict setting ------
  smooth_runner.conflict_setting.obstacle_conflict_weight = reflect_smooth_json.conflict_setting.obstacle_conflict_weight
  smooth_runner.conflict_setting.path_conflict_weight = reflect_smooth_json.conflict_setting.path_conflict_weight
  smooth_runner.conflict_setting.self_conflict_weight = reflect_smooth_json.conflict_setting.self_conflict_weight

  // ------ auto run cfg ------
  smooth_runner.auto_run_cfg.cooldown = reflect_smooth_json.auto_run_cfg.cooldown
  smooth_runner.auto_run_cfg.factor = reflect_smooth_json.auto_run_cfg.factor
  smooth_runner.auto_run_cfg.patience = reflect_smooth_json.auto_run_cfg.patience
  smooth_runner.auto_run_cfg.min_lr = reflect_smooth_json.auto_run_cfg.min_lr
  smooth_runner.auto_run_cfg.outer_loop = reflect_smooth_json.auto_run_cfg.outer_loop
  smooth_runner.auto_run_cfg.max_iter = reflect_smooth_json.auto_run_cfg.max_iter
  smooth_runner.auto_run_cfg.lr = reflect_smooth_json.auto_run_cfg.lr

  // ------ pipe cfg ------
  Object.entries(reflect_smooth_json.pipes_cfg).forEach(([pipe_name, pipe_info]: [string, any]) => {
    smooth_runner.pipes_cfg.set(pipe_name, new PipeCfg(
      pipe_name, pipe_info.radius, pipe_info.position,
      pipe_info.direction, pipe_info.group_name, pipe_info.is_input, pipe_info.radius_scale
    ))
  })

  // ------ group cells ------
  Object.entries(reflect_smooth_json.setting).forEach(([group_name, group_info]: [string, any]) => {
    const group_cell = new GroupCell(
      smooth_name, search_task_name, project_name, group_name, group_info.cell_setting.group_idx
    )

    group_cell.aid_path_info = []
    for (const [aid_name, aid_info] of Object.entries(group_info.aid_path_info)) {
      group_cell.aid_path_info.push({
        name: aid_name,
        from_name: (aid_info as any).from_name,
        to_name: (aid_info as any).to_name,
      })
    }

    group_cell.graph_cfg.seg_node_max = group_info.graph_cfg.seg_node_max
    group_cell.graph_cfg.seg_node_min = group_info.graph_cfg.seg_node_min
    group_cell.c_table = group_info.cell_setting.c_table

    group_info.segments.forEach((segment_info: any) => {
      const segment_cell = new SegmentCell(
        smooth_name, search_task_name, project_name,
        `${group_name}/seg_${segment_info.seg_idx}`,
        segment_info.seg_idx,
        group_cell.group_idx!,
        segment_info.cell_setting.c_idxs,
        group_cell,
        rootObj.get_scene(),
        // just for front end visualization
        segment_info.cell_setting.left_intercept_num,
        segment_info.cell_setting.right_intercept_num
      )

      segment_cell.left_clamp = segment_info.cell_setting.left_clamp
      segment_cell.right_clamp = segment_info.cell_setting.right_clamp
      segment_cell.degree = segment_info.bspline_degree
      segment_cell.density = segment_info.bspline_density
      segment_cell.left_intercept_ratio = segment_info.left_intercept_ratio
      segment_cell.right_intercept_ratio = segment_info.right_intercept_ratio
      segment_cell.left_overlap_ratio = segment_info.left_overlap_ratio
      segment_cell.right_overlap_ratio = segment_info.right_overlap_ratio
      segment_cell.segment_scale = segment_info.segment_scale
      segment_cell.with_analytical_curvature_computed = segment_info.with_analytical_curvature_computed
      segment_cell.with_analytical_tangent_computed = segment_info.with_analytical_tangent_computed
      segment_cell.cost_setting = segment_info.cost_setting

      group_cell.seg_dict[segment_cell.name] = segment_cell
    })

    group_info.connectors.forEach((connector_info: any) => {
      const connector_cell = new ConnectorCell(
        smooth_name, search_task_name, project_name,
        `${group_name}/con_${connector_info.con_idx}`,
        connector_info.con_idx,
        group_cell.group_idx!,
        connector_info.left_seg_idx,
        connector_info.right_seg_idx,
        group_cell,
        rootObj.get_scene(),
        // just for front end visualization
        connector_info.cell_setting.c_idxs,
        connector_info.cell_setting.left_intercept_num,
        connector_info.cell_setting.right_intercept_num
      )

      connector_cell.degree = connector_info.bspline_degree
      connector_cell.density = connector_info.bspline_density
      connector_cell.with_analytical_curvature_computed = connector_info.with_analytical_curvature_computed
      connector_cell.with_analytical_tangent_computed = connector_info.with_analytical_tangent_computed
      connector_cell.cost_setting = connector_info.cost_setting

      group_cell.connector_dict[connector_cell.name] = connector_cell
    })

    group_info.paths.forEach((path_info: any) => {
      const path_cell = new PathCell(
        smooth_name, search_task_name, project_name,
        path_info.path_name,
        path_info.path_idx,
        group_cell.group_idx!,
        path_info.cell_idxs,
        path_info.cell_types,
        group_cell
      )
      path_cell.cost_cfgs = path_info.cost_setting.cost_cfgs
      path_cell.element_cost_cfgs = path_info.cost_setting.element_cost_cfgs
      group_cell.path_dict[path_cell.name] = path_cell
    })

    smooth_runner.group_cells.set(group_name, group_cell)
  })

  // ------ obstacle cfg ------
  const obstacle_cell = new ObstacleCell('smooth_obstacle', obstacle_xyzr_list, rootObj.get_scene())
  smooth_runner.obstacle_cell = obstacle_cell

  rootObj.proj_dict.get(project_name)!.smooth_task_dict.set(smooth_runner.name, smooth_runner)
  emit('updateTree')
  return true
}

const handleManualRun = async () => {
  manualDialogVisible.value = false
  manualLog.value = 'Starting manual run...'
  runSmoothPercentage.value = 0

  await manualRun(
    label_dict.value.get('Project') as string,
    label_dict.value.get('SmoothName') as string,
    manualMaxIter.value,
    manualLr.value,
    manualLogIter.value
  )
}

const manualRun = async (project_name: string, smooth_name: string, max_iter: number, lr: number, log_iter: number) => {
  const url = 'http://192.168.110.66:9998/path_smooth/manual_smooth'

  const smooth_runner = rootObj.proj_dict.get(project_name)!.smooth_task_dict.get(smooth_name) as SmoothRunner
  const smooth_json = smooth_runner.export_json()

  console.log('smooth_json:', smooth_json)

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      smooth_name: smooth_name,
      smooth_json: smooth_json,
      lr: lr,
      max_iter: max_iter,
      log_iter: log_iter,
    })
  })

  if (!response.ok) {
    ElMessage.error('Network response was not ok')
    return
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('Failed to get response reader')

  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const text = new TextDecoder().decode(value)
    buffer += text

    try {
      const data = JSON.parse(buffer)
      if (data.name === 'manual_smooth') {
        if (data.total_finished) {
          for (const [group_name, cell_setting] of Object.entries(data.result.smooth_result)) {
            smooth_runner.group_cells.get(group_name)!.update_c_table((cell_setting as any).c_table, rootObj.get_scene())
          }

          // Show report visualization
          reportPlotlyVisible.value = true
          
          // Update pie chart with loss info
          if (data.result.loss_info) {
            const lossInfo = data.result.loss_info
            const labels = Object.keys(lossInfo)
            const values = Object.values(lossInfo)
            reportPlotlyRef.value?.updatePieChart(labels, values)
          }
          
          // Update line chart with loss record
          if (data.result.loss_record) {
            const lossRecord = data.result.loss_record
            const iterations = Array.from({ length: lossRecord.length }, (_, i) => i + 1)
            reportPlotlyRef.value?.updateLineChart(iterations, lossRecord)
          }

          emit('updateLog', data.msg)
          break
        } else {
          if (data['result'] !== undefined) {
            for (const [group_name, cell_setting] of Object.entries(data.result.smooth_result)) {
              smooth_runner.group_cells.get(group_name)!.update_c_table((cell_setting as any).c_table, rootObj.get_scene())
            }
          }
          const message = `${data.name} Progress: ${data.progress.toFixed(2)}s, Message: ${data.msg}`
          emit('updateLog', message)
        }
      }
      buffer = ''

    } catch (error) {
      continue
    }
  }

  // const data = await response.json()
  // if (!data.is_success) {
  //   ElMessage.error(data.error)
  //   return
  // }
  // const result_json = data.result_json
  // const loss_info = data.loss_info

  // // update smooth runner
  // for (const [group_name, cell_setting] of Object.entries(result_json)) {
  //   smooth_runner.group_cells.get(group_name)!.update_c_table((cell_setting as any).c_table, rootObj.get_scene())
  // }
}


// ------ Expose ------
defineExpose({
  hideContextMenu,
  showContextMenu
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