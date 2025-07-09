<!-- PathSmooth.vue -->
<template>
  <div class="path-smooth" @click="handleClickOutside">
    <el-menu mode="horizontal" class="menu-bar">
      <el-sub-menu index="1">
        <template #title>
          <el-dropdown trigger="click" placement="bottom">
            <span class="el-dropdown-link">
              Tools<el-icon class="el-icon--right"><arrow-down /></el-icon>
            </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item>Path Analysis</el-dropdown-item>
                <el-dropdown-item @click="exportSmoothPathPoints">Export Path Points</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </template>
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

    <div v-show="groupCellContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/CreateSurface')">Create Surface</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/RemoveSurface')">Remove Surface</el-menu-item>
      </el-menu>
    </div>

    <div v-show="segmentCellContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/HideTubeResult')">Hide Tube Result</el-menu-item>
      </el-menu>
    </div>

    <div v-show="connectorCellContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/HideTubeResult')">Hide Tube Result</el-menu-item>
      </el-menu>
    </div>

    <div v-show="segmentCellsContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/HideTubeResult')">Hide Tube Result</el-menu-item>
      </el-menu>
    </div>

    <div v-show="connectorCellsContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/HideTubeResult')">Hide Tube Result</el-menu-item>
      </el-menu>
    </div>

    <div v-show="smoothTaskContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SmoothTask/ShowDiscretized')">Show Discretized</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SmoothTask/HideDiscretized')">Hide Discretized</el-menu-item>
      </el-menu>
    </div>

    <div v-show="smoothRootContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SmoothRoot/ShowDiscretized')">Show Discretized</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SmoothRoot/HideDiscretized')">Hide Discretized</el-menu-item>
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
      </div>
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="manualDialogVisible = false">Cancel</el-button>
          <el-button type="primary" @click="handleManualRun">Run</el-button>
        </div>
      </template>
    </el-dialog>

    <el-dialog v-model="surfaceGenerateDialogVisible" title="Surface Generate" width="30%">
      <div>
        <el-form :model="{ path_reso: surfacePathReso, point_reso: surfacePointReso }" label-width="120px">
          <el-form-item label="Path Reso">
            <el-input-number v-model="surfacePathReso" :min="0.001" :max="10.0" :step="0.01" :precision="3" style="width: 200px" />
          </el-form-item>
          <el-form-item label="Point Reso">
            <el-input-number v-model="surfacePointReso" :min="0.001" :max="10.0" :step="0.01" :precision="3" style="width: 200px" />
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="surfaceGenerateDialogVisible = false">Cancel</el-button>
          <el-button type="primary" @click="handleSurfaceGenerateRun">Run</el-button>
        </div>
      </template>
    </el-dialog>

  </div>
</template>

<script setup lang="ts">
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

import { ElMessage } from 'element-plus';
import JSZip from 'jszip';
import * as THREE from 'three';
import {
  pass,
  round,
} from 'three/tsl';

import { CenterRoot } from '@/classes/center_root.ts';
import type { PipeMesh } from '@/classes/path_search/pipe_mesh';
import {
  ConnectorCell,
  GroupCell,
  PathCell,
  SegmentCell,
} from '@/classes/path_smooth/smooth_nodes';
import {
  ObstacleCell,
  PipeCfg,
  SmoothRunner,
} from '@/classes/path_smooth/smooth_runner';
import DefineSmoothInfo from '@/components/path_smooth/DefineSmoothInfo.vue';
import ReportPlotly from '@/components/ReportPlotly.vue';
import config from '@/config';
import { setVisableDict } from '@/utils/utils';

const emit = defineEmits(['updateTree', 'highlightTreeNode', 'updateLog'])

const rootObj = inject<CenterRoot>(
  'centerRoot',
  new CenterRoot(new THREE.Scene())
)

// ------ progress dialog ------
const progressDialogVisible = ref(false)
const progressPercentage = ref(0)
const progressLog = ref('')

// ------ manual dialog ------
const manualDialogVisible = ref(false)
const manualLr = ref(0.01)
const manualMaxIter = ref(1000)
const manualLogIter = ref(50)

// ------ surface generate dialog ------
const surfaceGenerateDialogVisible = ref(false)
const surfacePathReso = ref(0.1)
const surfacePointReso = ref(0.1)
// Store context for surface generation
const surfaceGenProject = ref('')
const surfaceGenSmooth = ref('')
const surfaceGenGroup = ref('')
const surfaceGenMeshJson = ref<any>(null)

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
  segmentCellContextMenuVisible.value = false
  connectorCellContextMenuVisible.value = false
  segmentCellsContextMenuVisible.value = false
  connectorCellsContextMenuVisible.value = false
  smoothTaskContextMenuVisible.value = false
  smoothRootContextMenuVisible.value = false
  groupCellContextMenuVisible.value = false
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
const segmentCellContextMenuVisible = ref(false)
const connectorCellContextMenuVisible = ref(false)
const segmentCellsContextMenuVisible = ref(false)
const connectorCellsContextMenuVisible = ref(false)
const smoothTaskContextMenuVisible = ref(false)
const smoothRootContextMenuVisible = ref(false)
const groupCellContextMenuVisible = ref(false)
const contextMenuVisibleDict = {
  smoothContextMenuVisible: smoothContextMenuVisible,
  runSmoothContextMenuVisible: runSmoothContextMenuVisible,
  segmentCellContextMenuVisible: segmentCellContextMenuVisible,
  connectorCellContextMenuVisible: connectorCellContextMenuVisible,
  segmentCellsContextMenuVisible: segmentCellsContextMenuVisible,
  connectorCellsContextMenuVisible: connectorCellsContextMenuVisible,
  smoothTaskContextMenuVisible: smoothTaskContextMenuVisible,
  smoothRootContextMenuVisible: smoothRootContextMenuVisible,
  groupCellContextMenuVisible: groupCellContextMenuVisible
}

const defineSmoothInfoVisible = ref(false)
const reportPlotlyVisible = ref(false)
const reportPlotlyRef = ref()
const fragmentVisibleDict = {
  defineSmoothInfoVisible: defineSmoothInfoVisible,
  reportPlotlyVisible: reportPlotlyVisible
}

const label_dict = ref(
  new Map<
    string,
    | string
    | number
    | string[]
    | Map<string, string | number>
    | Map<
      string,
      Map<string, Map<string, string | number | number[] | boolean>>
    >
  >()
)

const showContextMenu = (data: any) => {
  // Reset all menu states first
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

  // Show appropriate menu based on style
  const style = label_dict.value.get('style')

  if (style === 'SearchTask') {
    smoothContextMenuVisible.value = true
  } else if (style === 'SmoothRunner') {
    runSmoothContextMenuVisible.value = true
  } else if (style === 'SegmentCell') {
    segmentCellContextMenuVisible.value = true
  } else if (style === 'ConnectorCell') {
    connectorCellContextMenuVisible.value = true
  } else if (style === 'SegmentCells') {
    segmentCellsContextMenuVisible.value = true
  } else if (style === 'ConnectorCells') {
    connectorCellsContextMenuVisible.value = true
  } else if (style === 'SmoothTask') {
    smoothTaskContextMenuVisible.value = true
  } else if (style === 'SmoothRoot') {
    smoothRootContextMenuVisible.value = true
  } else if (style === 'GroupCell') {
    groupCellContextMenuVisible.value = true
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

  if (mode === 'SearchTask/AddSmoothTask') {
    const search_task = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .search_task_dict.get(label_dict.value.get('SearchTask') as string)
    if (!search_task!.has_result) {
      ElMessage.error('Search task has result, please clear the result first')
      return
    }

    const project_manager = rootObj.proj_dict.get(
      label_dict.value.get('Project') as string
    )!

    const group_pipe_dict = new Map<
      string,
      Map<string, Map<string, string | number | number[] | boolean>>
    >()
    for (const group_name of Array.from(project_manager.pipe_dict.keys())) {
      const pipe_dict = new Map<
        string,
        Map<string, string | number | number[] | boolean>
      >()
      for (const pipe_name of Array.from(
        project_manager.pipe_dict.get(group_name)!.keys()
      )) {
        const pipe_mesh = project_manager.pipe_dict
          .get(group_name)!
          .get(pipe_name)!
        if (
          pipe_mesh.pipe_mode === 'inlet' ||
          pipe_mesh.pipe_mode === 'outlet'
        ) {
          const pipe_info = new Map<
            string,
            string | number | number[] | boolean
          >()
          pipe_info.set('name', pipe_mesh.name)
          pipe_info.set('radius', pipe_mesh.radius)
          pipe_info.set('position', [pipe_mesh.x, pipe_mesh.y, pipe_mesh.z])
          pipe_info.set('direction', [
            pipe_mesh.vector_x,
            pipe_mesh.vector_y,
            pipe_mesh.vector_z
          ])
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
  } else if (mode === 'SegmentCell/ShowTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const segment_name = label_dict.value.get('label') as string

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.seg_dict[segment_name]) {
      target_group_cell.seg_dict[segment_name].show_tube_mesh(rootObj.get_scene())
    }
  } else if (mode === 'SegmentCell/ShowLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const segment_name = label_dict.value.get('label') as string

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.seg_dict[segment_name]) {
      target_group_cell.seg_dict[segment_name].show_line_mesh()
    }
  } else if (mode === 'SegmentCell/HideLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const segment_name = label_dict.value.get('label') as string

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.seg_dict[segment_name]) {
      target_group_cell.seg_dict[segment_name].hide_line_mesh()
    }
  } else if (mode === 'SegmentCell/HideTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const segment_name = label_dict.value.get('label') as string

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.seg_dict[segment_name]) {
      target_group_cell.seg_dict[segment_name].hide_tube_mesh()
    }
  } else if (mode === 'ConnectorCell/ShowLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const connector_name = label_dict.value.get('label') as string

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.connector_dict[connector_name]) {
      target_group_cell.connector_dict[connector_name].show_line_mesh()
    }
  } else if (mode === 'ConnectorCell/ShowTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const connector_name = label_dict.value.get('label') as string

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.connector_dict[connector_name]) {
      target_group_cell.connector_dict[connector_name].show_tube_mesh(rootObj.get_scene())
    }
  } else if (mode === 'ConnectorCell/HideLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const connector_name = label_dict.value.get('label') as string

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.connector_dict[connector_name]) {
      target_group_cell.connector_dict[connector_name].hide_line_mesh()
    }
  } else if (mode === 'SegmentCells/ShowLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const segment_cell of Object.values(target_group_cell.seg_dict)) {
        segment_cell.show_line_mesh()
      }
    }
  } else if (mode === 'SegmentCells/ShowTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const segment_cell of Object.values(target_group_cell.seg_dict)) {
        segment_cell.show_tube_mesh(rootObj.get_scene())
      }
    }
  } else if (mode === 'SegmentCells/HideLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const segment_cell of Object.values(target_group_cell.seg_dict)) {
        segment_cell.hide_line_mesh()
      }
    }
  } else if (mode === 'SegmentCells/HideTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const segment_cell of Object.values(target_group_cell.seg_dict)) {
        segment_cell.hide_tube_mesh()
      }
    }
  } else if (mode === 'ConnectorCells/ShowLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const connector_cell of Object.values(target_group_cell.connector_dict)) {
        connector_cell.show_line_mesh()
      }
    }
  } else if (mode === 'ConnectorCells/ShowTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const connector_cell of Object.values(target_group_cell.connector_dict)) {
        connector_cell.show_tube_mesh(rootObj.get_scene())
      }
    }
  } else if (mode === 'ConnectorCells/HideLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const connector_cell of Object.values(target_group_cell.connector_dict)) {
        connector_cell.hide_line_mesh()
      }
    }
  } else if (mode === 'ConnectorCells/HideTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell) {
      for (const connector_cell of Object.values(target_group_cell.connector_dict)) {
        connector_cell.hide_tube_mesh()
      }
    }
  } else if (mode === 'SmoothTask/ShowDiscretized') {
    const project = label_dict.value.get('Project')?.toString()
    const smoothName = label_dict.value.get('label')?.toString()
    if (project && smoothName) {
      const proj = rootObj.proj_dict.get(project)
      const smooth_runner = proj?.smooth_task_dict.get(smoothName) as
        | SmoothRunner
        | undefined
      if (smooth_runner?.obstacle_cell) {
        smooth_runner.obstacle_cell.show_mesh()
      }
    }
  } else if (mode === 'SmoothTask/HideDiscretized') {
    const project = label_dict.value.get('Project')?.toString()
    const smoothName = label_dict.value.get('label')?.toString()
    if (project && smoothName) {
      const proj = rootObj.proj_dict.get(project)
      const smooth_runner = proj?.smooth_task_dict.get(smoothName) as
        | SmoothRunner
        | undefined
      if (smooth_runner?.obstacle_cell) {
        smooth_runner.obstacle_cell.hide_mesh()
      }
    }
  } else if (mode === 'SmoothRoot/ShowDiscretized') {
    const project = label_dict.value.get('Project')?.toString()
    if (project) {
      const proj = rootObj.proj_dict.get(project)
      if (proj) {
        for (const smooth_runner of proj.smooth_task_dict.values()) {
          if (smooth_runner.obstacle_cell) {
            smooth_runner.obstacle_cell.show_mesh()
          }
        }
      }
    }
  } else if (mode === 'SmoothRoot/HideDiscretized') {
    const project = label_dict.value.get('Project')?.toString()
    if (project) {
      const proj = rootObj.proj_dict.get(project)
      if (proj) {
        for (const smooth_runner of proj.smooth_task_dict.values()) {
          if (smooth_runner.obstacle_cell) {
            smooth_runner.obstacle_cell.hide_mesh()
          }
        }
      }
    }
  } else if (mode === 'GroupCell/CreateSurface') {
    const project = label_dict.value.get('Project')!.toString()
    const group_name = label_dict.value.get('label')!.toString()
    const smooth_runner = rootObj.proj_dict
      .get(project)!
      .smooth_task_dict.get(label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_cell: GroupCell = smooth_runner.group_cells.get(group_name)!
    surfaceGenProject.value = project
    surfaceGenSmooth.value = smooth_runner.name
    surfaceGenGroup.value = group_name
    surfaceGenMeshJson.value = group_cell.export_surface_mesh_json()
    surfaceGenerateDialogVisible.value = true
  } else if (mode === 'GroupCell/RemoveSurface') {
    const project = label_dict.value.get('Project')!.toString()
    const group_name = label_dict.value.get('label')!.toString()
    const smooth_runner = rootObj.proj_dict
      .get(project)!
      .smooth_task_dict.get(label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_cell: GroupCell = smooth_runner.group_cells.get(group_name)!
    if (group_cell.fine_surface_mesh) {
      group_cell.fine_surface_mesh.destory(rootObj.get_scene())
    }
    group_cell.fine_surface_mesh = null
  }
}

// -------------------------------------------------------
const handleDefineSmoothInfoGenerate = async (data: any) => {
  const project = rootObj.proj_dict.get(label_dict.value.get('Project') as string)
  if (project && project.smooth_task_dict.has(data.smooth_name)) {
    ElMessage.error('Smooth Name 已存在，请更换任务名！')
    return null
  }
  console.log('data:', data)

  const search_task = rootObj.proj_dict
    .get(label_dict.value.get('Project') as string)!
    .search_task_dict.get(data.search_task)
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
      group_task.cell_tasks.forEach(cell_task => {
        search_result[group_name][cell_task.name] = []
        for (let i = 0; i < cell_task.result!.points.length; i++) {
          search_result[group_name][cell_task.name].push([
            cell_task.result!.points[i][0],
            cell_task.result!.points[i][1],
            cell_task.result!.points[i][2],
            cell_task.result!.radius
          ])
        }
      })
    })
  })

  // --------------------------------------------
  let obstacle_xyzr_list: number[][] = []

  const grid = rootObj.proj_dict.get(
    label_dict.value.get('Project') as string
  )!.grid
  if (grid.discretize_source == '') {
    ElMessage.warning('Grid has not been discretized')
    return null
  }
  grid.surface_obstacle_points.forEach(point => {
    obstacle_xyzr_list.push([point[0], point[1], point[2], 0.0])
  })

  rootObj.proj_dict
    .get(label_dict.value.get('Project') as string)!
    .obstacle_dict.forEach((obstacle, key) => {
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
  Object.entries(data.pipes_config).forEach(
    ([pipe_name, pipe_info]: [string, any]) => {
      pipes_dict.set(
        pipe_name,
        new PipeCfg(
          pipe_name,
          pipe_info.radius,
          pipe_info.position,
          pipe_info.direction,
          pipe_info.group_name,
          pipe_info.is_input,
          pipe_info.radius_scale
        )
      )
    }
  )
  obstacle_xyzr_list = ObstacleCell.filter_xyzr(pipes_dict, obstacle_xyzr_list)

  if (await !upload_smooth_obstacle_pcd(data.smooth_name, obstacle_xyzr_list)) {
    ElMessage.error('Failed to upload smooth obstacle pcd')
    return null
  }

  const group_setting: Record<string, any> = {}
  Object.entries(data.aid_paths).forEach(
    ([group_name, group_aid_paths]: [string, any]) => {
      group_setting[group_name] = {
        aid_path_info: group_aid_paths,
        graph_cfg: data.graph_configs[group_name],
        path_result: search_result[group_name]
      }
    }
  )

  const pipes_cfg: Record<string, any> = {}
  Object.entries(data.pipes_config).forEach(
    ([pipe_name, pipe_info]: [string, any]) => {
      pipes_cfg[pipe_name] = {
        group_name: pipe_info.group_name,
        is_input: pipe_info.is_input,
        radius: pipe_info.radius,
        position: pipe_info.position,
        direction: pipe_info.direction,
        radius_scale: pipe_info.radius_scale
      }
    }
  )

  const smooth_json = {
    pipes_cfg: pipes_cfg,
    conflict_setting: data.conflict_setting,
    setting: group_setting,
    auto_run_cfg: data.auto_run_config
  }

  if (
    !(await init_smooth_map(
      label_dict.value.get('Project') as string,
      data.search_task,
      data.smooth_name,
      smooth_json,
      obstacle_xyzr_list
    ))
  ) {
    ElMessage.error('Failed to initialize smooth map')
    return null
  }

  return true
}

const upload_smooth_obstacle_pcd = async (smooth_name: string, obstacle_xyzr_list: number[][]) => {
  const url = `${config.BASE_URL}/path_smooth/upload_smooth_obstacle_pcd`

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
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          smooth_name: smooth_name,
          obstacle_xyzr: chunk_data,
          chunk_index: i,
          total_chunks: total_chunks,
          is_last_chunk: i + CHUNK_SIZE >= obstacle_xyzr_list.length
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
  project_name: string,
  search_task_name: string,
  smooth_name: string,
  smooth_json: any,
  obstacle_xyzr_list: number[][]
) => {
  const url = `${config.BASE_URL}/path_smooth/init_smooth_map`
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
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

  const smooth_runner = new SmoothRunner(
    smooth_name,
    search_task_name,
    project_name
  )

  // ------ conflict setting ------
  smooth_runner.conflict_setting.obstacle_conflict_weight =
    reflect_smooth_json.conflict_setting.obstacle_conflict_weight
  smooth_runner.conflict_setting.path_conflict_weight =
    reflect_smooth_json.conflict_setting.path_conflict_weight
  smooth_runner.conflict_setting.self_conflict_weight =
    reflect_smooth_json.conflict_setting.self_conflict_weight

  // ------ auto run cfg ------
  smooth_runner.auto_run_cfg.cooldown =
    reflect_smooth_json.auto_run_cfg.cooldown
  smooth_runner.auto_run_cfg.factor = reflect_smooth_json.auto_run_cfg.factor
  smooth_runner.auto_run_cfg.patience =
    reflect_smooth_json.auto_run_cfg.patience
  smooth_runner.auto_run_cfg.min_lr = reflect_smooth_json.auto_run_cfg.min_lr
  smooth_runner.auto_run_cfg.outer_loop =
    reflect_smooth_json.auto_run_cfg.outer_loop
  smooth_runner.auto_run_cfg.max_iter =
    reflect_smooth_json.auto_run_cfg.max_iter
  smooth_runner.auto_run_cfg.lr = reflect_smooth_json.auto_run_cfg.lr

  // ------ pipe cfg ------
  Object.entries(reflect_smooth_json.pipes_cfg).forEach(
    ([pipe_name, pipe_info]: [string, any]) => {
      smooth_runner.pipes_cfg.set(
        pipe_name,
        new PipeCfg(
          pipe_name,
          pipe_info.radius,
          pipe_info.position,
          pipe_info.direction,
          pipe_info.group_name,
          pipe_info.is_input,
          pipe_info.radius_scale
        )
      )
    }
  )

  // ------ group cells ------
  Object.entries(reflect_smooth_json.setting).forEach(
    ([group_name, group_info]: [string, any]) => {
      const group_cell = new GroupCell(
        smooth_name,
        search_task_name,
        project_name,
        group_name,
        group_info.cell_setting.group_idx
      )

      group_cell.aid_path_info = []
      for (const [aid_name, aid_info] of Object.entries(
        group_info.aid_path_info
      )) {
        group_cell.aid_path_info.push({
          name: aid_name,
          from_name: (aid_info as any).from_name,
          to_name: (aid_info as any).to_name
        })
      }

      group_cell.graph_cfg.seg_node_max = group_info.graph_cfg.seg_node_max
      group_cell.graph_cfg.seg_node_min = group_info.graph_cfg.seg_node_min
      group_cell.c_table = group_info.cell_setting.c_table

      group_info.segments.forEach((segment_info: any) => {
        const segment_cell = new SegmentCell(
          smooth_name,
          search_task_name,
          project_name,
          `${group_name}/seg_${segment_info.seg_idx}`,
          segment_info.seg_idx,
          group_cell.group_idx!,
          segment_info.cell_setting.c_idxs,
          group_cell,
          rootObj.get_scene(),
          // just for front end visualization
          segment_info.cell_setting.left_intercept_num,
          segment_info.cell_setting.right_intercept_num,
          segment_info.cell_setting.left_clamp,
          segment_info.cell_setting.right_clamp
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
        segment_cell.with_analytical_curvature_computed =
          segment_info.with_analytical_curvature_computed
        segment_cell.with_analytical_tangent_computed =
          segment_info.with_analytical_tangent_computed
        segment_cell.cost_setting = segment_info.cost_setting
        segment_cell.left_vector = segment_info.cell_setting.left_vector
        segment_cell.right_vector = segment_info.cell_setting.right_vector

        group_cell.seg_dict[segment_cell.name] = segment_cell
      })

      group_info.connectors.forEach((connector_info: any) => {
        const connector_cell = new ConnectorCell(
          smooth_name,
          search_task_name,
          project_name,
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
        connector_cell.with_analytical_curvature_computed =
          connector_info.with_analytical_curvature_computed
        connector_cell.with_analytical_tangent_computed =
          connector_info.with_analytical_tangent_computed
        connector_cell.cost_setting = connector_info.cost_setting

        group_cell.connector_dict[connector_cell.name] = connector_cell
      })

      group_info.paths.forEach((path_info: any) => {
        const path_cell = new PathCell(
          smooth_name,
          search_task_name,
          project_name,
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
    }
  )

  // ------ obstacle cfg ------
  const obstacle_cell = new ObstacleCell(
    'smooth_obstacle',
    obstacle_xyzr_list,
    rootObj.get_scene()
  )
  smooth_runner.obstacle_cell = obstacle_cell

  rootObj.proj_dict
    .get(project_name)!
    .smooth_task_dict.set(smooth_runner.name, smooth_runner)
  emit('updateTree')
  return true
}

const handleManualRun = async () => {
  manualDialogVisible.value = false
  await manualRun(
    label_dict.value.get('Project') as string,
    label_dict.value.get('SmoothName') as string,
    manualMaxIter.value,
    manualLr.value,
    manualLogIter.value
  )
}

const manualRun = async (
  project_name: string,
  smooth_name: string,
  max_iter: number,
  lr: number,
  log_iter: number
) => {
  const url = `${config.BASE_URL}/path_smooth/manual_smooth`

  const smooth_runner = rootObj.proj_dict
    .get(project_name)!
    .smooth_task_dict.get(smooth_name) as SmoothRunner
  const smooth_json = smooth_runner.export_json()

  console.log('smooth_json:', smooth_json)

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      smooth_name: smooth_name,
      smooth_json: smooth_json,
      lr: lr,
      max_iter: max_iter,
      log_iter: log_iter
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

    // Split buffer by newlines and process each JSON object
    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // Keep the last incomplete line in buffer

    for (const line of lines) {
      if (line.trim() === '') continue

      try {
        const data = JSON.parse(line)
        if (data.name === 'manual_smooth') {
          if (data.total_finished) {
            for (const [group_name, cell_setting] of Object.entries(
              data.result.smooth_result
            )) {
              smooth_runner.group_cells
                .get(group_name)!
                .update_c_table(
                  (cell_setting as any).c_table,
                  rootObj.get_scene()
                )
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
              const iterations = Array.from(
                { length: lossRecord.length },
                (_, i) => i + 1
              )
              reportPlotlyRef.value?.updateLineChart(iterations, lossRecord)
            }
            emit('updateLog', data.msg)
            break

          } else {
            if (data['result'] !== undefined) {
              for (const [group_name, cell_setting] of Object.entries(
                data.result.smooth_result
              )) {
                smooth_runner.group_cells
                  .get(group_name)!
                  .update_c_table(
                    (cell_setting as any).c_table,
                    rootObj.get_scene()
                  )
              }
            }
            const message = `${data.name} Progress: ${data.progress.toFixed(
              2
            )}s, Message: ${data.msg}`
            emit('updateLog', message)
          }
        }
        buffer = ''
      } catch (error) {
        continue
      }

    }
  }
}

const exportSmoothPathPoints = async () => {
  const JSZip = (await import('jszip')).default
  const segmentsZip = new JSZip()
  const connectorsZip = new JSZip()
  let hasSegment = false
  let hasConnector = false

  rootObj.proj_dict.forEach(proj => {
    proj.smooth_task_dict.forEach(smoothRunner => {
      // Segments
      smoothRunner.group_cells.forEach((groupCell, groupName) => {
        Object.values(groupCell.seg_dict).forEach((segmentCell: any) => {
          let points: number[][] = segmentCell.path_points || []
          if (points.length > 0) {
            hasSegment = true
            const txt = points.map(pt => pt.join(' ')).join('\n')
            segmentsZip.file(`${segmentCell.name}.txt`, txt)
          }
        })
        // Connectors
        Object.values(groupCell.connector_dict).forEach(
          (connectorCell: any) => {
            let points: number[][] = connectorCell.path_points || []
            if (points.length > 0) {
              hasConnector = true
              const txt = points.map(pt => pt.join(' ')).join('\n')
              connectorsZip.file(`${connectorCell.name}.txt`, txt)
            }
          }
        )
      })
    })
  })

  if (!hasSegment && !hasConnector) {
    ElMessage.warning('没有可导出的路径点！')
    return
  }
  // 下载Segments.zip
  if (hasSegment) {
    segmentsZip.generateAsync({ type: 'blob' }).then((content: Blob) => {
      const url = URL.createObjectURL(content)
      const a = document.createElement('a')
      a.href = url
      a.download = 'Segments.zip'
      a.click()
      URL.revokeObjectURL(url)
    })
  }
  // 下载Connectors.zip
  if (hasConnector) {
    connectorsZip.generateAsync({ type: 'blob' }).then((content: Blob) => {
      const url = URL.createObjectURL(content)
      const a = document.createElement('a')
      a.href = url
      a.download = 'Connectors.zip'
      a.click()
      URL.revokeObjectURL(url)
    })
  }
}

const createTubeSurface = async (
  project_name: string,
  smooth_name: string,
  group_name: string,
  surface_mesh_json: any,
  path_reso: number,
  point_reso: number
) => {
  console.log('surface_mesh_json:', surface_mesh_json)
  progressDialogVisible.value = true
  progressPercentage.value = 0

  try {
    const url = `${config.BASE_URL}/path_smooth/generate_surface_from_paths`
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        path_dict: surface_mesh_json,
        path_reso: path_reso,
        point_reso: point_reso
      })
    })

    if (!response.ok) {
      ElMessage.error('Network response was not ok')
      return
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('Failed to get response reader')

    let buffer = ''

    let cells: number[][] = []
    let points: number[][] = []

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

          if (data.total_finished) {
            break
          }

          if (data.name === 'return_cells') {
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
          } else if (data.name === 'generate_surface_from_paths') {
            progressLog.value = 'Processing mesh data...'
          }

        } catch (error) {
          continue
        }
      }
    }

    const group_cell = rootObj.proj_dict
      .get(project_name)!
      .smooth_task_dict.get(smooth_name)!
      .group_cells.get(group_name)!
    group_cell.generate_surface_mesh(points, cells, [255, 0, 0], rootObj.get_scene())


  } catch (error: any) {
    console.error('Error generating mesh:', error)
    progressDialogVisible.value = false
    progressPercentage.value = 100
    progressLog.value = 'Error generating mesh'
    ElMessage.error(error.message || 'Failed to generate mesh')
  } finally {
    // Close progress dialog after a short delay
    setTimeout(() => {
      progressDialogVisible.value = false
      progressPercentage.value = 0
      progressLog.value = 'Processing...'
    }, 2000)
  }
}

const handleSurfaceGenerateRun = async () => {
  surfaceGenerateDialogVisible.value = false
  await createTubeSurface(
    surfaceGenProject.value,
    surfaceGenSmooth.value,
    surfaceGenGroup.value,
    surfaceGenMeshJson.value,
    surfacePathReso.value,
    surfacePointReso.value
  )
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
