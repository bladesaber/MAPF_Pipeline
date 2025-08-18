<!-- PathSmooth.vue -->
<template>
  <div class="path-smooth" @click="handleClickOutside">
    <el-menu mode="horizontal" class="menu-bar">
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
        <el-menu-item @click="handleTreeContextMenu('SmoothRunner/ShowDiscretized')"
          >Show Discretized</el-menu-item
        >
        <el-menu-item @click="handleTreeContextMenu('SmoothRunner/HideDiscretized')"
          >Hide Discretized</el-menu-item
        >
        <el-menu-item @click="handleTreeContextMenu('SmoothRunner/EditSmoothName')"
          >Edit Smooth Name</el-menu-item
        >
        <el-menu-item @click="handleTreeContextMenu('SmoothRunner/DiscretizeObstacle')"
          >Discretize Obstacle</el-menu-item
        >
        <el-menu-item @click="handleTreeContextMenu('SmoothRunner/RemoveSmoothTask')"
          >Remove Smooth Task</el-menu-item
        >
      </el-menu>
    </div>

    <div v-show="segmentCellContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/Edit')">Edit Segment</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/ShowControlPoints')">Show Control Points</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCell/HideControlPoints')">Hide Control Points</el-menu-item>
      </el-menu>
    </div>

    <div v-show="connectorCellContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/Edit')">Edit Connector</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/ShowControlPoints')">Show Control Points</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCell/HideControlPoints')">Hide Control Points</el-menu-item>
      </el-menu>
    </div>

    <div v-show="segmentCellsContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/Edit')">Edit Segments</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/ShowControlPoints')">Show Control Points</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SegmentCells/HideControlPoints')">Hide Control Points</el-menu-item>
      </el-menu>
    </div>

    <div v-show="connectorCellsContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/Edit')">Edit Connectors</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/ShowControlPoints')">Show Control Points</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('ConnectorCells/HideControlPoints')">Hide Control Points</el-menu-item>
      </el-menu>
    </div>

    <div v-show="groupCellContextMenuVisible" class="context-menu" :style="contextMenuStyle" @click.stop>
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/CreateSurface')">Create Surface</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/RemoveSurface')">Remove Surface</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupCell/HideTubeResult')">Hide Tube Result</el-menu-item>
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
     <!-- 控制点右键菜单弹窗 -->
     <div
      v-show="controlPointContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleControlPointContextMenu('EditControlPoint')">Edit Control Point</el-menu-item>
      </el-menu>
    </div>

    <PipeConfigsEditor
      v-model:modelValue="pipeConfigsEditorVisible"
      :pipes_config="currentPipesConfig"
      @save="onPipeConfigsSave"
    />
    <!-- Pipe Config 节点右键菜单 -->
    <div
      v-show="pipeConfigContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('PipeConfig/Edit')"
          >Edit Pipe Configs</el-menu-item
        >
      </el-menu>
    </div>
    <AidPathsEditor
    v-model:modelValue="aidPathsEditorVisible"
    :aid_paths="currentAidPaths"
    :pipes_config="currentPipesConfig"
    @save="onAidPathsSave"
  />
  <!-- Aid Paths 节点右键菜单 -->
  <div
    v-show="aidPathsContextMenuVisible"
    class="context-menu"
    :style="contextMenuStyle"
    @click.stop
  >
    <el-menu>
      <el-menu-item @click="handleTreeContextMenu('AidPaths/Edit')"
        >Edit Aid Paths</el-menu-item
      >
    </el-menu>
  </div>
  <ConflictSettingEditor
    v-model:modelValue="conflictSettingEditorVisible"
    :conflict_setting="currentConflictSetting"
    @save="onConflictSettingSave"
  />
  <!-- 右键 Conflict Setting 菜单 -->
  <div
    v-show="conflictSettingContextMenuVisible"
    class="context-menu"
    :style="contextMenuStyle"
    @click.stop
  >
    <el-menu>
      <el-menu-item @click="handleTreeContextMenu('ConflictSetting/Edit')">
        Edit Conflict Setting
      </el-menu-item>
    </el-menu>
  </div>
  <AutoRunConfigEditor
    v-model:modelValue="autoRunConfigEditorVisible"
    :auto_run_config="currentAutoRunConfig"
    @save="onAutoRunConfigSave"
  />
  <!-- 右键 Auto Run Config 菜单 -->
  <div
    v-show="autoRunConfigContextMenuVisible"
    class="context-menu"
    :style="contextMenuStyle"
    @click.stop
  >
    <el-menu>
      <el-menu-item @click="handleTreeContextMenu('AutoRunConfig/Edit')">
        Edit Auto Run Config
      </el-menu-item>
    </el-menu>
  </div>
    <EditSmoothNameDrawer
      v-model="editSmoothNameVisible"
      :smoothName="currentEditSmoothName"
      @save="onEditSmoothNameSave"
    />
    <SegmentCellEditor
    v-model="segmentCellEditorVisible"
    :segmentData="currentSegmentData"
    @save="onSegmentCellSave"
  />
  <ConnectorCellEditor
    v-model="connectorCellEditorVisible"
    :connectorData="currentConnectorData"
    @save="onConnectorCellSave"
  />
  <ControlPointEditor
    v-model="controlPointEditorVisible"
    :controlPointInfo="currentControlPointInfo"
    @apply="handleControlPointApply"
    @update:modelValue="val => { console.log('ControlPointEditor visible:', val); controlPointEditorVisible = val; }"
  />
  
  <BatchSegmentEditor
    v-model="batchSegmentEditorVisible"
    :initialCostSettings="currentBatchSegmentCostSettings"
    @save="onBatchSegmentSave"
  />
  
  <BatchConnectorEditor
    v-model="batchConnectorEditorVisible"
    :initialCostSettings="currentBatchConnectorCostSettings"
    @save="onBatchConnectorSave"
  />
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

import AidPathsEditor from './AidPathsEditor.vue';
import AutoRunConfigEditor from './AutoRunConfigEditor.vue';
import BatchConnectorEditor from './BatchConnectorEditor.vue';
import BatchSegmentEditor from './BatchSegmentEditor.vue';
import ConflictSettingEditor from './ConflictSettingEditor.vue';
import ConnectorCellEditor from './ConnectorCellEditor.vue';
import ControlPointEditor from './ControlPointEditor.vue';
import EditSmoothNameDrawer from './EditSmoothNameDrawer.vue';
import PipeConfigsEditor from './PipeConfigsEditor.vue';
import SegmentCellEditor from './SegmentCellEditor.vue';

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
  pipeConfigContextMenuVisible.value = false
  aidPathsContextMenuVisible.value = false;
  conflictSettingContextMenuVisible.value = false;
  autoRunConfigContextMenuVisible.value = false;
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
    pipeConfigContextMenuVisible.value = false;
    aidPathsContextMenuVisible.value = false;
    conflictSettingContextMenuVisible.value = false;
    autoRunConfigContextMenuVisible.value = false;
    controlPointContextMenuVisible.value = false;
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
const conflictSettingContextMenuVisible = ref(false);
const autoRunConfigContextMenuVisible = ref(false);
const contextMenuVisibleDict = {
  smoothContextMenuVisible: smoothContextMenuVisible,
  runSmoothContextMenuVisible: runSmoothContextMenuVisible,
  segmentCellContextMenuVisible: segmentCellContextMenuVisible,
  connectorCellContextMenuVisible: connectorCellContextMenuVisible,
  segmentCellsContextMenuVisible: segmentCellsContextMenuVisible,
  connectorCellsContextMenuVisible: connectorCellsContextMenuVisible,
  smoothTaskContextMenuVisible: smoothTaskContextMenuVisible,
  smoothRootContextMenuVisible: smoothRootContextMenuVisible,
  groupCellContextMenuVisible: groupCellContextMenuVisible,
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
const projectName = computed(() => (label_dict.value.get('Project') ?? 'DefaultProject').toString())
const taskName = computed(() => (label_dict.value.get('TaskName') ?? 'DefaultTask').toString())

// 右键编辑 Pipe Configs 相关变量
const pipeConfigsEditorVisible = ref(false);
const currentPipesConfig = ref<any>({});
const pipeConfigContextMenuVisible = ref(false);

const onPipeConfigsSave = (newConfig: any) => {
  // TODO: 同步到全局 smooth_runner/tree
  ElMessage.success('Pipe Configs updated');
  pipeConfigsEditorVisible.value = false;
}

const aidPathsEditorVisible = ref(false);
const currentAidPaths = ref<any>({});
const aidPathsContextMenuVisible = ref(false);

const onAidPathsSave = (newConfig: any) => {
  // TODO: 同步到全局 smooth_runner/tree
  ElMessage.success('Aid Paths updated');
  aidPathsEditorVisible.value = false;
}

const conflictSettingEditorVisible = ref(false);
const currentConflictSetting = ref({
  obstacle_conflict_weight: 1000,
  path_conflict_weight: 100,
  self_conflict_weight: 100
});

function onConflictSettingSave(newSetting: any) {
  // 将编辑的数据保存到内存中的SmoothRunner对象
  const project = label_dict.value.get('Project') as string;
  const smoothName = label_dict.value.get('SmoothName') as string;
  
  if (project && smoothName) {
    const proj = rootObj.proj_dict.get(project);
    const smooth_runner = proj?.smooth_task_dict.get(smoothName);
    if (smooth_runner) {
      // 更新SmoothRunner的conflict_setting
      smooth_runner.conflict_setting.obstacle_conflict_weight = newSetting.obstacle_conflict_weight;
      smooth_runner.conflict_setting.path_conflict_weight = newSetting.path_conflict_weight;
      smooth_runner.conflict_setting.self_conflict_weight = newSetting.self_conflict_weight;
      
      ElMessage.success('Conflict Setting updated successfully');
      emit('updateTree'); // 更新树视图
    }
  }
  conflictSettingEditorVisible.value = false;
}

const autoRunConfigEditorVisible = ref(false);
const currentAutoRunConfig = ref({
  lr: 0.01,
  factor: 0.25,
  patience: 5000,
  cooldown: 5000,
  min_lr: 0.001,
  max_iter: 50000,
  outer_loop: 1
});

function onAutoRunConfigSave(newConfig: any) {
  // 将编辑的数据保存到内存中的SmoothRunner对象
  const project = label_dict.value.get('Project') as string;
  const smoothName = label_dict.value.get('SmoothName') as string;
  
  if (project && smoothName) {
    const proj = rootObj.proj_dict.get(project);
    const smooth_runner = proj?.smooth_task_dict.get(smoothName);
    if (smooth_runner) {
      // 更新SmoothRunner的auto_run_cfg
      smooth_runner.auto_run_cfg.lr = newConfig.lr;
      smooth_runner.auto_run_cfg.factor = newConfig.factor;
      smooth_runner.auto_run_cfg.patience = newConfig.patience;
      smooth_runner.auto_run_cfg.cooldown = newConfig.cooldown;
      smooth_runner.auto_run_cfg.min_lr = newConfig.min_lr;
      smooth_runner.auto_run_cfg.max_iter = newConfig.max_iter;
      smooth_runner.auto_run_cfg.outer_loop = newConfig.outer_loop;
      
      ElMessage.success('Auto Run Config updated successfully');
      emit('updateTree'); // 更新树视图
    }
  }
  autoRunConfigEditorVisible.value = false;
}

// ------ Smooth Name 编辑右侧栏 ------
const editSmoothNameVisible = ref(false);
const currentEditSmoothName = ref('');
function onEditSmoothNameSave(newName: string) {
  let project = label_dict.value.get('Project') as string;
  let smoothName = (label_dict.value.get('SmoothName') || '').toString();
  if (project && smoothName) {
    const proj = rootObj.proj_dict.get(project);
    if (proj && proj.smooth_task_dict.has(smoothName)) {
      const smooth_runner = proj.smooth_task_dict.get(smoothName);
      if (smooth_runner) {
        smooth_runner.name = newName;
        proj.smooth_task_dict.delete(smoothName);
        proj.smooth_task_dict.set(newName, smooth_runner);
        ElMessage.success('Smooth Name updated');
        emit('updateTree');
      }
    }
  }
  editSmoothNameVisible.value = false;
}

// ------ SegmentCell 编辑右侧栏 ------
const segmentCellEditorVisible = ref(false);
const currentSegmentData = ref<any>({});
function onSegmentCellSave(newData: any) {
  // 将编辑的数据保存到内存中的SegmentCell对象
  const project = label_dict.value.get('Project') as string;
  const smoothName = label_dict.value.get('SmoothName') as string;
  const groupName = label_dict.value.get('groupName') as string;
  const segmentName = label_dict.value.get('name') as string;
  
  if (project && smoothName && groupName && segmentName) {
    const proj = rootObj.proj_dict.get(project);
    const smooth_runner = proj?.smooth_task_dict.get(smoothName);
    if (smooth_runner) {
      const group_cell = smooth_runner.group_cells.get(groupName);
      if (group_cell && group_cell.seg_dict[segmentName]) {
        const segment_cell = group_cell.seg_dict[segmentName];
        
        // 更新SegmentCell的属性
        segment_cell.degree = newData.bspline_degree;
        segment_cell.density = newData.bspline_density;
        segment_cell.left_overlap_ratio = newData.left_overlap_ratio;
        segment_cell.right_overlap_ratio = newData.right_overlap_ratio;
        segment_cell.left_intercept_ratio = newData.left_intercept_ratio;
        segment_cell.right_intercept_ratio = newData.right_intercept_ratio;
        segment_cell.segment_scale = newData.segment_scale;
        segment_cell.with_analytical_curvature_computed = newData.with_analytical_curvature_computed;
        segment_cell.with_analytical_tangent_computed = newData.with_analytical_tangent_computed;
        segment_cell.cost_setting = newData.cost_setting;
        segment_cell.left_clamp = newData.left_clamp;
        segment_cell.right_clamp = newData.right_clamp;
        segment_cell.left_vector = newData.left_vector;
        segment_cell.right_vector = newData.right_vector;
        
        // 刷新mesh以反映更改
        segment_cell.update_mesh(rootObj.get_scene());
        
        ElMessage.success('Segment updated successfully');
        emit('updateTree'); // 更新树视图
      }
    }
  }
  segmentCellEditorVisible.value = false;
}

// ------ ConnectorCell 编辑右侧栏 ------
const connectorCellEditorVisible = ref(false);
const currentConnectorData = ref<any>({});
function onConnectorCellSave(newData: any) {
  // 将编辑的数据保存到内存中的ConnectorCell对象
  const project = label_dict.value.get('Project') as string;
  const smoothName = label_dict.value.get('SmoothName') as string;
  const groupName = label_dict.value.get('groupName') as string;
  const connectorName = label_dict.value.get('name') as string;
  
  if (project && smoothName && groupName && connectorName) {
    const proj = rootObj.proj_dict.get(project);
    const smooth_runner = proj?.smooth_task_dict.get(smoothName);
    if (smooth_runner) {
      const group_cell = smooth_runner.group_cells.get(groupName);
      if (group_cell && group_cell.connector_dict[connectorName]) {
        const connector_cell = group_cell.connector_dict[connectorName];
        
        // 更新ConnectorCell的属性
        connector_cell.degree = newData.bspline_degree;
        connector_cell.density = newData.bspline_density;
        connector_cell.with_analytical_curvature_computed = newData.with_analytical_curvature_computed;
        connector_cell.with_analytical_tangent_computed = newData.with_analytical_tangent_computed;
        connector_cell.cost_setting = newData.cost_setting;
        
        // 刷新mesh以反映更改
        connector_cell.update_mesh(rootObj.get_scene());
        
        ElMessage.success('Connector updated successfully');
        emit('updateTree'); // 更新树视图
      }
    }
  }
  connectorCellEditorVisible.value = false;
}

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
  const label = label_dict.value.get('label');
  if (style === 'PipeConfig' || label === 'Pipe Config') {
    pipeConfigContextMenuVisible.value = true;
    return;
  }
  if (style === 'AidPaths' || label === 'Aid Paths') {
    aidPathsContextMenuVisible.value = true;
    return;
  }
  if (style === 'ConflictSetting' || label === 'Conflict Setting') {
    conflictSettingContextMenuVisible.value = true;
    return;
  }
  if (style === 'AutoRunConfig' || label === 'Auto Run Config') {
    autoRunConfigContextMenuVisible.value = true;
    return;
  }
  if (style === 'SmoothRunner' || label === 'Smooth Runner') {
    runSmoothContextMenuVisible.value = true;
    return;
  }
  if (style === 'SearchTask') {
    smoothContextMenuVisible.value = true
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

const handleTreeContextMenu = async (mode: string) => {
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
  } else if (mode === 'SmoothRunner/ShowDiscretized') {
    const project = label_dict.value.get('Project')?.toString()
    const smoothName = label_dict.value.get('SmoothName')?.toString()
    if (project && smoothName) {
      const proj = rootObj.proj_dict.get(project)
      const smooth_runner = proj?.smooth_task_dict.get(smoothName) as SmoothRunner | undefined
      if (smooth_runner?.obstacle_cell) {
        smooth_runner.obstacle_cell.show_mesh()
      }
    }
  } else if (mode === 'SmoothRunner/HideDiscretized') {
    const project = label_dict.value.get('Project')?.toString()
    const smoothName = label_dict.value.get('SmoothName')?.toString()
    if (project && smoothName) {
      const proj = rootObj.proj_dict.get(project)
      const smooth_runner = proj?.smooth_task_dict.get(smoothName) as SmoothRunner | undefined
      if (smooth_runner?.obstacle_cell) {
        smooth_runner.obstacle_cell.hide_mesh()
      }
    }
  } else if (mode === 'SmoothRunner/EditSmoothName') {
    editSmoothNameVisible.value = true;
    currentEditSmoothName.value = (label_dict.value.get('SmoothName') || '').toString();
    return;
  } else if (mode === 'SmoothRunner/DiscretizeObstacle') {
    discretizeObstacle();
    return;
  } else if (mode === 'SmoothRunner/RemoveSmoothTask') {
    const project = label_dict.value.get('Project')?.toString()
    const smoothName = label_dict.value.get('SmoothName')?.toString()
    if (project && smoothName) {
      const proj = rootObj.proj_dict.get(project)
      if (proj?.smooth_task_dict.has(smoothName)) {
        // 获取平滑任务
        const smooth_runner = proj.smooth_task_dict.get(smoothName) as SmoothRunner
        
        // 清理各个组件
        if (smooth_runner.obstacle_cell) {
          smooth_runner.obstacle_cell.hide_mesh()
        }
        
        // 清理所有GroupCell
        smooth_runner.group_cells.forEach((group_cell, group_name) => {
          // 清理SegmentCell
          Object.values(group_cell.seg_dict).forEach(segment_cell => {
            segment_cell.hide_line_mesh()
            segment_cell.hide_tube_mesh()
          })
          
          // 清理ConnectorCell
          Object.values(group_cell.connector_dict).forEach(connector_cell => {
            connector_cell.hide_line_mesh()
            connector_cell.hide_tube_mesh()
          })

          // 隐藏所有控制点
          group_cell.hide_control_points()
        })
        
        // 从字典中删除
        proj.smooth_task_dict.delete(smoothName)
        emit('updateTree')
        ElMessage.success('Smooth task removed successfully')
      }
    }
  } else if (mode === 'SegmentCell/ShowTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const raw_label = label_dict.value.get('label') as string
    const segment_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

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
    const raw_label = label_dict.value.get('label') as string
    const segment_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

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
    const raw_label = label_dict.value.get('label') as string
    const segment_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

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
    const raw_label = label_dict.value.get('label') as string
    const segment_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

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
    const raw_label = label_dict.value.get('label') as string
    const connector_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

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
    const raw_label = label_dict.value.get('label') as string
    const connector_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.connector_dict[connector_name]) {
      target_group_cell.connector_dict[connector_name].show_tube_mesh()
    }
  } else if (mode === 'ConnectorCell/HideLineResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const raw_label = label_dict.value.get('label') as string
    const connector_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

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
  } else if (mode === 'ConnectorCell/HideTubeResult') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const raw_label = label_dict.value.get('label') as string
    const connector_name = raw_label.includes('/') ? raw_label.split('/').pop() as string : raw_label

    // Find the group cell by group_idx
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }

    if (target_group_cell && target_group_cell.connector_dict[connector_name]) {
      target_group_cell.connector_dict[connector_name].hide_tube_mesh()
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
        connector_cell.show_tube_mesh()
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
    const surface_mesh_json = group_cell.export_surface_mesh_json()
    surfaceGenMeshJson.value = surface_mesh_json
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
  }else if (mode === 'GroupCell/ShowLineResult') {
    const group_name = label_dict.value.get('label') as string
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(label_dict.value.get('SmoothName') as string) as SmoothRunner
    const group_cell = smooth_runner.group_cells.get(group_name)
    if (group_cell) {
      for (const segment_cell of Object.values(group_cell.seg_dict)) {
        segment_cell.show_line_mesh()
      }
      for (const connector_cell of Object.values(group_cell.connector_dict)) {
        connector_cell.show_line_mesh()
      }
    }
  } else if (mode === 'GroupCell/ShowTubeResult') {
    const group_name = label_dict.value.get('label') as string
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(label_dict.value.get('SmoothName') as string) as SmoothRunner
    const group_cell = smooth_runner.group_cells.get(group_name)
    if (group_cell) {
      for (const segment_cell of Object.values(group_cell.seg_dict)) {
        segment_cell.show_tube_mesh(rootObj.get_scene())
      }
      for (const connector_cell of Object.values(group_cell.connector_dict)) {
        connector_cell.show_tube_mesh()
      }
    }
  } else if (mode === 'GroupCell/HideLineResult') {
    const group_name = label_dict.value.get('label') as string
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(label_dict.value.get('SmoothName') as string) as SmoothRunner
    const group_cell = smooth_runner.group_cells.get(group_name)
    if (group_cell) {
      for (const segment_cell of Object.values(group_cell.seg_dict)) {
        segment_cell.hide_line_mesh()
      }
      for (const connector_cell of Object.values(group_cell.connector_dict)) {
        connector_cell.hide_line_mesh()
      }
    }
  } else if (mode === 'GroupCell/HideTubeResult') {
    const group_name = label_dict.value.get('label') as string
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(label_dict.value.get('SmoothName') as string) as SmoothRunner
    const group_cell = smooth_runner.group_cells.get(group_name)
    if (group_cell) {
      for (const segment_cell of Object.values(group_cell.seg_dict)) {
        segment_cell.hide_tube_mesh()
      }
      for (const connector_cell of Object.values(group_cell.connector_dict)) {
        connector_cell.hide_tube_mesh()
      }
    }
  } else if (mode === 'PipeConfig/Edit') {
    // 优先从 smooth_runner 获取 pipes_config
    let pipesConfig: any = label_dict.value.get('PipesConfig') as any;
    if (!pipesConfig) {
      const project = label_dict.value.get('Project') as string;
      const smoothName = label_dict.value.get('SmoothName') as string;
      if (project && smoothName) {
        const proj = rootObj.proj_dict.get(project);
        const smooth_runner = proj?.smooth_task_dict.get(smoothName);
        if (smooth_runner && smooth_runner.pipes_cfg) {
          pipesConfig = {};
          smooth_runner.pipes_cfg.forEach((cfg: any, name: string) => {
            pipesConfig[name] = {
              group_name: cfg.group_name,
              radius: cfg.radius,
              position: cfg.position,
              direction: cfg.direction,
              is_input: cfg.is_input,
              radius_scale: cfg.radius_scale
            }
          })
        }
      }
    }
    currentPipesConfig.value = pipesConfig || {};
    pipeConfigsEditorVisible.value = true;
    return;
  } else if (mode === 'AidPaths/Edit') {
    let aidPaths: any = label_dict.value.get('AidPaths') as any;
    let pipesConfig: any = label_dict.value.get('PipesConfig') as any;
    if (!aidPaths || !pipesConfig) {
      const project = label_dict.value.get('Project') as string;
      const smoothName = label_dict.value.get('SmoothName') as string;
      if (project && smoothName) {
        const proj = rootObj.proj_dict.get(project);
        const smooth_runner = proj?.smooth_task_dict.get(smoothName);
        if (smooth_runner) {
          // pipesConfig
          if (!pipesConfig && smooth_runner.pipes_cfg) {
            pipesConfig = {};
            smooth_runner.pipes_cfg.forEach((cfg: any, name: string) => {
              pipesConfig[name] = {
                group_name: cfg.group_name,
                radius: cfg.radius,
                position: cfg.position,
                direction: cfg.direction,
                is_input: cfg.is_input,
                radius_scale: cfg.radius_scale
              }
            });
          }
          // aidPaths（修正为 group_cells 结构）
          if (!aidPaths && smooth_runner.group_cells) {
            aidPaths = {};
            smooth_runner.group_cells.forEach((groupCell: any, groupName: string) => {
              aidPaths[groupName] = {};
              (groupCell.aid_path_info || []).forEach((info: any) => {
                aidPaths[groupName][info.name] = {
                  from_name: info.from_name,
                  to_name: info.to_name
                }
              });
            });
          }
        }
      }
    }
    currentAidPaths.value = aidPaths || {};
    currentPipesConfig.value = pipesConfig || {};
    aidPathsEditorVisible.value = true;
    return;
  } else if (mode === 'ConflictSetting/Edit') {
    // 取当前节点的 conflict_setting
    // 这里假设 label_dict.value.get('ConflictSetting') 存在
    const setting = label_dict.value.get('ConflictSetting') as any;
    if (setting) {
      currentConflictSetting.value = { ...setting };
    }
    conflictSettingEditorVisible.value = true;
    return;
  } else if (mode === 'AutoRunConfig/Edit') {
    // 取当前节点的 auto_run_config
    // 这里假设 label_dict.value.get('AutoRunConfig') 存在
    const config = label_dict.value.get('AutoRunConfig') as any;
    if (config) {
      currentAutoRunConfig.value = { ...config };
    }
    autoRunConfigEditorVisible.value = true;
    return;
  } else if (mode === 'SegmentCell/Edit') {
    // 获取当前 segment 的数据
    const project = label_dict.value.get('Project') as string;
    const smoothName = label_dict.value.get('SmoothName') as string;
    const groupName = label_dict.value.get('groupName') as string;
    const segmentName = label_dict.value.get('name') as string;
    
    if (project && smoothName && groupName && segmentName) {
      const proj = rootObj.proj_dict.get(project);
      const smooth_runner = proj?.smooth_task_dict.get(smoothName);
      if (smooth_runner) {
        const group_cell = smooth_runner.group_cells.get(groupName);
        if (group_cell && group_cell.seg_dict[segmentName]) {
          const segment_cell = group_cell.seg_dict[segmentName];
          currentSegmentData.value = {
            // 可编辑参数
            bspline_degree: segment_cell.degree,
            bspline_density: segment_cell.density,
            left_overlap_ratio: segment_cell.left_overlap_ratio,
            right_overlap_ratio: segment_cell.right_overlap_ratio,
            left_intercept_ratio: segment_cell.left_intercept_ratio,
            right_intercept_ratio: segment_cell.right_intercept_ratio,
            segment_scale: segment_cell.segment_scale,
            with_analytical_curvature_computed: segment_cell.with_analytical_curvature_computed,
            with_analytical_tangent_computed: segment_cell.with_analytical_tangent_computed,
            cost_setting: segment_cell.cost_setting,
            left_clamp: segment_cell.left_clamp,
            right_clamp: segment_cell.right_clamp,
            left_vector: segment_cell.left_vector,
            right_vector: segment_cell.right_vector,
            // 不可编辑但重要的信息
            seg_idx: segment_cell.idx,
            group_idx: segment_cell.group_idx,
            c_idxs: segment_cell.c_idxs,
            name: segment_cell.name
          };
          segmentCellEditorVisible.value = true;
          return;
        }
      }
    }
    ElMessage.error('Segment not found');
    return;
  } else if (mode === 'ConnectorCell/Edit') {
    // 获取当前 connector 的数据
    const project = label_dict.value.get('Project') as string;
    const smoothName = label_dict.value.get('SmoothName') as string;
    const groupName = label_dict.value.get('groupName') as string;
    const connectorName = label_dict.value.get('name') as string;
    
    if (project && smoothName && groupName && connectorName) {
      const proj = rootObj.proj_dict.get(project);
      const smooth_runner = proj?.smooth_task_dict.get(smoothName);
      if (smooth_runner) {
        const group_cell = smooth_runner.group_cells.get(groupName);
        if (group_cell && group_cell.connector_dict[connectorName]) {
          const connector_cell = group_cell.connector_dict[connectorName];
          currentConnectorData.value = {
            // 可编辑参数
            bspline_degree: connector_cell.degree,
            bspline_density: connector_cell.density,
            with_analytical_curvature_computed: connector_cell.with_analytical_curvature_computed,
            with_analytical_tangent_computed: connector_cell.with_analytical_tangent_computed,
            cost_setting: connector_cell.cost_setting,
            // 不可编辑但重要的信息
            con_idx: connector_cell.idx,
            group_idx: connector_cell.group_idx,
            left_seg_idx: connector_cell.left_seg_idx,
            right_seg_idx: connector_cell.right_seg_idx,
            name: connector_cell.name
          };
          connectorCellEditorVisible.value = true;
          return;
        }
      }
    }
    ElMessage.error('Connector not found');
    return;
  } else if (mode === 'SegmentCell/ShowControlPoints') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const segment_name = label_dict.value.get('name') as string // 使用 'name' 而不是 'label'
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell && target_group_cell.seg_dict[segment_name]) {
      target_group_cell.show_control_points_for_cell(segment_name)
    }
  } else if (mode === 'SegmentCell/HideControlPoints') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const segment_name = label_dict.value.get('name') as string // 使用 'name' 而不是 'label'
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell && target_group_cell.seg_dict[segment_name]) {
      target_group_cell.hide_control_points_for_cell(segment_name)
    }
  } else if (mode === 'ConnectorCell/ShowControlPoints') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const connector_name = label_dict.value.get('name') as string // 使用 'name' 而不是 'label'
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell && target_group_cell.connector_dict[connector_name]) {
      target_group_cell.show_control_points_for_cell(connector_name)
    }
  } else if (mode === 'ConnectorCell/HideControlPoints') {
    const smooth_runner = rootObj.proj_dict
      .get(label_dict.value.get('Project') as string)!
      .smooth_task_dict.get(
        label_dict.value.get('SmoothName') as string
      ) as SmoothRunner
    const group_idx = label_dict.value.get('GroupIdx') as number
    const connector_name = label_dict.value.get('name') as string // 使用 'name' 而不是 'label'
    let target_group_cell: GroupCell | null = null
    for (const [group_name, group_cell] of smooth_runner.group_cells) {
      if (group_cell.group_idx === group_idx) {
        target_group_cell = group_cell
        break
      }
    }
    if (target_group_cell && target_group_cell.connector_dict[connector_name]) {
      target_group_cell.hide_control_points_for_cell(connector_name)
    }
  } else if (mode === 'SegmentCells/ShowControlPoints') {
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
      target_group_cell.show_segments_control_points();
    }
  } else if (mode === 'SegmentCells/HideControlPoints') {
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
      target_group_cell.hide_segments_control_points();
    }
  } else if (mode === 'ConnectorCells/ShowControlPoints') {
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
      target_group_cell.show_connectors_control_points(rootObj.get_scene());
    }
  } else if (mode === 'ConnectorCells/HideControlPoints') {
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
      target_group_cell.hide_connectors_control_points();
    }
  } else if (mode === 'SegmentCells/Edit') {
    // 批量编辑段节点
    const project = label_dict.value.get('Project') as string;
    const smoothName = label_dict.value.get('SmoothName') as string;
    const groupIdx = label_dict.value.get('GroupIdx') as number;
    
    if (project && smoothName && groupIdx !== undefined) {
      const proj = rootObj.proj_dict.get(project);
      const smooth_runner = proj?.smooth_task_dict.get(smoothName);
      if (smooth_runner) {
        // 通过 GroupIdx 找到对应的 group_cell
        let targetGroupCell = null;
        for (const [groupName, groupCell] of smooth_runner.group_cells) {
          if (groupCell.group_idx === groupIdx) {
            targetGroupCell = groupCell;
            break;
          }
        }
        
        if (targetGroupCell && Object.keys(targetGroupCell.seg_dict).length > 0) {
          // 获取第一个段节点的Cost Settings作为初始值
          const firstSegment = Object.values(targetGroupCell.seg_dict)[0];
          currentBatchSegmentCostSettings.value = {
            obstacle_thickness: firstSegment.cost_setting.obstacle_thickness,
            path_thickness: firstSegment.cost_setting.path_thickness
          };
          batchSegmentEditorVisible.value = true;
          return;
        }
      }
    }
    ElMessage.error('未找到可编辑的段节点');
    return;
  } else if (mode === 'ConnectorCells/Edit') {
    // 批量编辑连接器节点
    const project = label_dict.value.get('Project') as string;
    const smoothName = label_dict.value.get('SmoothName') as string;
    const groupIdx = label_dict.value.get('GroupIdx') as number;
    
    if (project && smoothName && groupIdx !== undefined) {
      const proj = rootObj.proj_dict.get(project);
      const smooth_runner = proj?.smooth_task_dict.get(smoothName);
      if (smooth_runner) {
        // 通过 GroupIdx 找到对应的 group_cell
        let targetGroupCell = null;
        for (const [groupName, groupCell] of smooth_runner.group_cells) {
          if (groupCell.group_idx === groupIdx) {
            targetGroupCell = groupCell;
            break;
          }
        }
        
        if (targetGroupCell && Object.keys(targetGroupCell.connector_dict).length > 0) {
          // 获取第一个连接器节点的Cost Settings作为初始值
          const firstConnector = Object.values(targetGroupCell.connector_dict)[0];
          currentBatchConnectorCostSettings.value = {
            obstacle_thickness: firstConnector.cost_setting.obstacle_thickness,
            path_thickness: firstConnector.cost_setting.path_thickness
          };
          batchConnectorEditorVisible.value = true;
          return;
        }
      }
    }
    ElMessage.error('未找到可编辑的连接器节点');
    return;
  }
}

// -------------------------------------------------------
const handleDefineSmoothInfoGenerate = async (data: any) => {
  const allowOverwrite = data.allowOverwrite || false;
  // 深拷贝参数并去除 allowOverwrite 字段
  const dataForBackend = { ...data };
  delete dataForBackend.allowOverwrite;
  const projectName = label_dict.value.get('Project') as string;
  const searchTaskName = dataForBackend.search_task;
  const project = rootObj.proj_dict.get(projectName);
  if (!project) {
    ElMessage.error('Project not found');
    console.error('【handleDefineSmoothInfoGenerate】Project not found:', projectName);
    return null;
  }
  const search_task = project.search_task_dict.get(searchTaskName);
  if (!search_task) {
    ElMessage.error(`Search task '${searchTaskName}' not found in project '${projectName}'`);
    console.error('【handleDefineSmoothInfoGenerate】search_task not found:', searchTaskName, 'in', Array.from(project.search_task_dict.keys()));
    return null;
  }
  if (!allowOverwrite && project && project.smooth_task_dict.has(dataForBackend.smooth_name)) {
    ElMessage.error('Smooth Name already exists, please change the task name!')
    return null
  }
  if (!search_task.has_result) {
    ElMessage.error('Search task has result, please clear the result first');
    return null;
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
  Object.entries(dataForBackend.pipes_config).forEach(
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

  if (!(await upload_smooth_obstacle_pcd(dataForBackend.smooth_name, obstacle_xyzr_list))) {
    ElMessage.error('Failed to upload smooth obstacle pcd')
    return null
  }

  if (data.init_smooth_map !== undefined) {
    if (!data.init_smooth_map) {  
      const smooth_runner = rootObj.proj_dict.get(projectName)!.smooth_task_dict.get(dataForBackend.smooth_name) as SmoothRunner
      const smooth_json = smooth_runner.export_json()
      if (!await upload_setting_info(dataForBackend.smooth_name, smooth_json)) {
        ElMessage.error('Failed to upload setting info')
        return null
      }
      return true
    }
  }

  const group_setting: Record<string, any> = {}
  Object.entries(dataForBackend.aid_paths).forEach(
    ([group_name, group_aid_paths]: [string, any]) => {
      group_setting[group_name] = {
        aid_path_info: group_aid_paths,
        graph_cfg: dataForBackend.graph_configs[group_name],
        path_result: search_result[group_name]
      }
    }
  )

  const pipes_cfg: Record<string, any> = {}
  Object.entries(dataForBackend.pipes_config).forEach(
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
    conflict_setting: dataForBackend.conflict_setting,
    setting: group_setting,
    auto_run_cfg: dataForBackend.auto_run_config
  }

  if (
    !(await init_smooth_map(
      label_dict.value.get('Project') as string,
      dataForBackend.search_task,
      dataForBackend.smooth_name,
      smooth_json,
      obstacle_xyzr_list
    ))
  ) {
    ElMessage.error('Failed to initialize smooth map')
    return null
  }

  return true
}

async function upload_setting_info(smooth_name: string, smooth_json: any) {
  const url = `${config.BASE_URL}/path_smooth/reset_config`
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      smooth_name: smooth_name,
      smooth_json: smooth_json,
    })
  })
  if (!response.ok) {
    ElMessage.error('Failed to upload setting info')
    return false
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
  // if (!response.ok) {
  //   ElMessage.error('Failed to initialize smooth map')
  //   return false
  // }
  const data = await response.json()
  const reflect_smooth_json = data.smooth_json
  console.log('reflect_smooth_json:', reflect_smooth_json)
  
  // 调试：检查后端返回的连接器结构
  Object.entries(reflect_smooth_json.setting).forEach(([group_name, group_info]: [string, any]) => {
    if (group_info.connectors) {
      group_info.connectors.forEach((connector: any, index: number) => {
        console.log(`连接器 ${index}:`, {
          con_idx: connector.con_idx,
          cell_setting: connector.cell_setting,
          c_idxs: connector.cell_setting?.c_idxs,
          has_c_idxs: !!connector.cell_setting?.c_idxs,
          c_idxs_length: connector.cell_setting?.c_idxs?.length
        });
      });
    }
  });

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

      group_cell.seg_dict[segment_cell.name] = segment_cell
    })

    group_info.connectors.forEach((connector_info: any) => {
      // 调试信息：检查connector_info的结构
      console.log(`[init_smooth_map] 连接器 ${connector_info.con_idx} 的完整信息:`, connector_info);
      console.log(`[init_smooth_map] connector_info.cell_setting:`, connector_info.cell_setting);
      console.log(`[init_smooth_map] connector_info.cell_setting?.c_idxs:`, connector_info.cell_setting?.c_idxs);
      
      // 优先使用后端返回的c_idxs，如果没有才重新生成
      let c_idxs: number[] = [];
      if (connector_info.cell_setting?.c_idxs && Array.isArray(connector_info.cell_setting.c_idxs) && connector_info.cell_setting.c_idxs.length > 0) {
        // 使用后端返回的c_idxs
        c_idxs = connector_info.cell_setting.c_idxs;
        console.log(`[ConnectorCell] 使用后端返回的c_idxs:`, c_idxs);
      } else {
        console.log(`[init_smooth_map] 后端没有返回c_idxs，条件检查:`, {
          hasCellSetting: !!connector_info.cell_setting,
          hasCIdxs: !!connector_info.cell_setting?.c_idxs,
          isArray: Array.isArray(connector_info.cell_setting?.c_idxs),
          length: connector_info.cell_setting?.c_idxs?.length
        });
        
        // 从连接的 segments 的端点获取控制点
        const left_seg = group_cell.seg_dict[`seg_${connector_info.left_seg_idx}`];
        const right_seg = group_cell.seg_dict[`seg_${connector_info.right_seg_idx}`];
        
        if (left_seg && right_seg && left_seg.c_idxs && right_seg.c_idxs) {
          // 使用左段的最后一个控制点和右段的第一个控制点
          const left_last_c_idx = left_seg.c_idxs[left_seg.c_idxs.length - 1];
          const right_first_c_idx = right_seg.c_idxs[0];
          
          // 如果两个控制点不同，则使用它们
          if (left_last_c_idx !== right_first_c_idx) {
            c_idxs = [left_last_c_idx, right_first_c_idx];
          } else {
            // 如果相同，则只使用一个控制点
            c_idxs = [left_last_c_idx];
          }
        }
        console.log(`[ConnectorCell] 重新生成c_idxs:`, c_idxs);
      }
      
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
        // 使用后端返回的或生成的控制点索引
        c_idxs,
        connector_info.cell_setting?.left_intercept_num ?? 3,
        connector_info.cell_setting?.right_intercept_num ?? 3,
        connector_info.cell_setting?.left_clamp ?? false,
        connector_info.cell_setting?.right_clamp ?? false
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

  // 确保所有 ConnectorCell 都能正确渲染
  smooth_runner.group_cells.forEach((group_cell) => {
    Object.values(group_cell.connector_dict).forEach((connector_cell) => {
      // 重新调用 add_mesh 确保 ConnectorCell 能正确渲染
      connector_cell.add_mesh(rootObj.get_scene());
    });
  });

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
              const group_cell = smooth_runner.group_cells.get(group_name)!;
              group_cell.update_c_table((cell_setting as any).c_table, rootObj.get_scene());
              
              // 更新所有控制点的位置
              for (const controlPoint of Object.values(group_cell.control_point_dict)) {
                controlPoint.update_position();
              }
              
              // 1. 先刷新所有Segments
              for (const seg of Object.values(group_cell.seg_dict)) {
                seg.update_mesh(rootObj.get_scene());
              }
              // 2. 再刷新所有Connectors
              for (const con of Object.values(group_cell.connector_dict)) {
                // 清除现有的path_points，让update_mesh重新生成spline路径
                con.path_points = [];
                con.radius_list = [];
                con.update_mesh(rootObj.get_scene());
              }
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
                const group_cell = smooth_runner.group_cells.get(group_name)!;
                group_cell.update_c_table((cell_setting as any).c_table, rootObj.get_scene());
                
                // 更新所有控制点的位置
                for (const controlPoint of Object.values(group_cell.control_point_dict)) {
                  controlPoint.update_position();
                }
                
                // 1. 先刷新所有Segments
                for (const seg of Object.values(group_cell.seg_dict)) {
                  seg.update_mesh(rootObj.get_scene());
                }
                // 2. 再刷新所有Connectors
                for (const con of Object.values(group_cell.connector_dict)) {
                  // 清除现有的path_points，让update_mesh重新生成spline路径
                  con.path_points = [];
                  con.radius_list = [];
                  con.update_mesh(rootObj.get_scene());
                }
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

// 新增离散化主流程，参数组装方式完全参照 DefineSmoothInfo.vue 的 handleApply
async function discretizeObstacle() {
  try {
    let project = label_dict.value.get('Project') as string;
    let smoothName = (label_dict.value.get('SmoothName') || '').toString();
    let searchTask = label_dict.value.get('TaskName') as string; // 修正：使用 TaskName 而不是 SearchTask
    
    // 参数验证
    if (!project) {
      ElMessage.error('无法获取 Project,请检查节点数据！');
      console.error('[discretizeObstacle] Project 获取失败');
      return;
    }
    
    const proj = rootObj.proj_dict.get(project);
    if (!proj) {
      ElMessage.error(`Project '${project}' 未找到`);
      return;
    }
    
    const smooth_runner = proj.smooth_task_dict.get(smoothName);
    if (!smooth_runner) {
      ElMessage.error(`SmoothRunner${smoothName}' 未找到`);
      return;
    }

    // 修正 searchTask 获取逻辑 - 优先从 smooth_runner 获取
    if (!searchTask || searchTask === 'undefined') {
      searchTask = smooth_runner.task_name;
    }
    
    if (!searchTask) {
      ElMessage.error('无法获取 searchTask,请检查节点数据！');
      return;
    }
    
    // 验证 search_task 是否存在
    const search_task = proj.search_task_dict.get(searchTask);
    if (!search_task) {
      ElMessage.error(`Search task${searchTask}' 未找到`);
      return;
    }
    
    if (!search_task.has_result) {
      ElMessage.error('Search task 没有结果，请先运行搜索任务');
      return;
    }

    // pipes_config
    const pipes_config: Record<string, any> = {};
    if (smooth_runner.pipes_cfg) {
      smooth_runner.pipes_cfg.forEach((cfg: any, name: string) => {
        pipes_config[name] = {
          group_name: cfg.group_name,
          radius: cfg.radius,
          position: cfg.position,
          direction: cfg.direction,
          is_input: cfg.is_input,
          radius_scale: cfg.radius_scale
        };
      });
    }
    
    // aid_paths
    const aid_paths: Record<string, any> = {};
    if (smooth_runner.group_cells) {
      smooth_runner.group_cells.forEach((groupCell: any, groupName: string) => {
        aid_paths[groupName] = {};
        (groupCell.aid_path_info || []).forEach((info: any) => {
          aid_paths[groupName][info.name] = {
            from_name: info.from_name,
            to_name: info.to_name
          };
        });
      });
    }
    
    // graph_configs
    const graph_configs: Record<string, any> = {};
    if (smooth_runner.group_cells) {
      smooth_runner.group_cells.forEach((groupCell: any, groupName: string) => {
        graph_configs[groupName] = {
          seg_node_min: groupCell.graph_cfg?.seg_node_min ?? 10,
          seg_node_max: groupCell.graph_cfg?.seg_node_max ?? 50
        };
      });
    }
    
    // conflict_setting
    const conflict_setting = smooth_runner.conflict_setting || {
      obstacle_conflict_weight: 1000,
      path_conflict_weight: 100,
      self_conflict_weight: 100
    };
    
    // auto_run_config
    const auto_run_config = smooth_runner.auto_run_cfg || {
      lr: 0.01,
      factor: 0.25,
      patience: 5000,
      cooldown: 5000,
      min_lr: 0.001,
      max_iter: 50,
      outer_loop: 1
    };

    const params = {
      smooth_name: smoothName,
      aid_paths,
      pipes_config,
      graph_configs,
      project,
      search_task: searchTask,
      conflict_setting,
      auto_run_config,
      allowOverwrite: true,
      init_smooth_map : false
    };
    
    // 调用离散化函数并处理返回值
    const result = await handleDefineSmoothInfoGenerate(params);
    
    if (result) {
      // 离散化成功后，只渲染新生成的那一组line
      for (const group_cell of smooth_runner.group_cells.values()) {
        for (const seg of Object.values(group_cell.seg_dict)) {
          seg.update_mesh(rootObj.get_scene());
        }
        for (const con of Object.values(group_cell.connector_dict)) {
          // 自动推断connector的path_points
          if (!con.path_points || con.path_points.length < 2) {
            const left_seg = group_cell.seg_dict[`seg_${con.left_seg_idx}`];
            const right_seg = group_cell.seg_dict[`seg_${con.right_seg_idx}`];
            if (left_seg && right_seg && left_seg.path_points.length > 0 && right_seg.path_points.length > 0) {
              con.path_points = [
                left_seg.path_points[left_seg.path_points.length - 1],
                right_seg.path_points[0]
              ];
              con.radius_list = [
                left_seg.radius_list[left_seg.radius_list.length - 1],
                right_seg.radius_list[0]
              ];
            }
          }
          con.update_mesh(rootObj.get_scene());
        }
      }
      ElMessage.success('离散化成功完成！');
    } else {
      ElMessage.error('离散化失败，请检查参数和网络连接');
    }
  } catch (error) {
    console.error('[discretizeObstacle] 执行过程中发生错误:', error);
    ElMessage.error(`离散化过程中发生错误: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

// ------ Expose ------
defineExpose({
  hideContextMenu,
  showContextMenu,
  handleRightClickControlPoint
})

onMounted(() => {
  document.addEventListener('mousedown', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('mousedown', handleClickOutside)
})
const controlPointContextMenuVisible = ref(false)
function handleControlPointContextMenu(mode: string) {
  controlPointContextMenuVisible.value = false
  if (mode === 'EditControlPoint') {
    controlPointEditorVisible.value = true
  }
}

function handleControlPointApply(data: any) {
  // 更新c_table并刷新路径
  const info = currentControlPointInfo.value
  if (!info) return
  const { c_idx, parent } = info
  
  // 获取GroupCell
  const group_cell = parent?.parent_cell
  if (!group_cell) return
  
  // 更新c_table
  const c_table = group_cell.c_table
  if (!c_table) return
  
  c_table.x[c_idx] = data.x
  c_table.y[c_idx] = data.y
  c_table.z[c_idx] = data.z
  c_table.radius[c_idx] = data.radius
  c_table.is_fix[c_idx] = data.is_fix ? 1 : 0
  c_table.is_unique[c_idx] = data.is_unique ? 1 : 0
  
  // 更新控制点位置
  const controlPoint = group_cell.get_or_create_control_point(c_idx, parent.scene);
  controlPoint.update_position();
  
  // 更新所有使用该控制点的段
  for (const [segmentName, segment] of Object.entries(group_cell.seg_dict)) {
    const seg = segment as SegmentCell;
    if (seg.c_idxs.includes(c_idx)) {
      seg.update_mesh(parent.scene);
    }
  }
  
  // 收集需要更新的连接器，避免重复更新
  const connectorsToUpdate = new Set<string>();
  
  // 更新所有使用该控制点的连接器
  for (const [connectorName, connector] of Object.entries(group_cell.connector_dict)) {
    const con = connector as ConnectorCell;
    if (con.c_idxs.includes(c_idx)) {
      connectorsToUpdate.add(connectorName);
    }
  }
  
  // 更新所有连接到被更新段的连接器（即使它们不直接使用该控制点）
  for (const [segmentName, segment] of Object.entries(group_cell.seg_dict)) {
    const seg = segment as SegmentCell;
    if (seg.c_idxs.includes(c_idx)) {
      // 找到所有连接到这个段的连接器
      for (const [connectorName, connector] of Object.entries(group_cell.connector_dict)) {
        const con = connector as ConnectorCell;
        if (con.left_seg_idx === seg.idx || con.right_seg_idx === seg.idx) {
          connectorsToUpdate.add(connectorName);
        }
      }
    }
  }
  
  // 批量更新所有需要更新的连接器
  for (const connectorName of connectorsToUpdate) {
    const connector = group_cell.connector_dict[connectorName] as ConnectorCell;
    if (connector) {
      // 清除现有的path_points，让update_mesh重新生成spline路径
      connector.path_points = [];
      connector.radius_list = [];
      connector.update_mesh(parent.scene);
    }
  }
  
  // 确保控制点保持显示状态（如果之前是显示的）
  if (controlPoint.isVisible()) {
    controlPoint.show_control_point();
  }
}

const currentControlPointInfo = ref<any>(null)
function handleRightClickControlPoint(userData: any, x: number, y: number) {
  contextMenuPosition.x = x
  contextMenuPosition.y = y
  currentControlPointInfo.value = userData
  controlPointContextMenuVisible.value = true
}

function handleLeftClickMeshSelect() {}
function handleRightClickMeshSelect() {
  controlPointContextMenuVisible.value = false;
}

const controlPointEditorVisible = ref(false);

// ------ 批量编辑相关变量 ------
const batchSegmentEditorVisible = ref(false);
const batchConnectorEditorVisible = ref(false);
const currentBatchSegmentCostSettings = ref({
  obstacle_thickness: 0.010,
  path_thickness: 0.010
});
const currentBatchConnectorCostSettings = ref({
  obstacle_thickness: 0.010,
  path_thickness: 0.010
});

// ------ 批量编辑保存函数 ------
function onBatchSegmentSave(costSettings: any) {
  const project = label_dict.value.get('Project') as string;
  const smoothName = label_dict.value.get('SmoothName') as string;
  const groupIdx = label_dict.value.get('GroupIdx') as number;
  
  if (project && smoothName && groupIdx !== undefined) {
    const proj = rootObj.proj_dict.get(project);
    const smooth_runner = proj?.smooth_task_dict.get(smoothName);
    if (smooth_runner) {
      // 通过 GroupIdx 找到对应的 group_cell
      let targetGroupCell = null;
      for (const [groupName, groupCell] of smooth_runner.group_cells) {
        if (groupCell.group_idx === groupIdx) {
          targetGroupCell = groupCell;
          break;
        }
      }
      
      if (targetGroupCell) {
        // 批量更新所有段节点的Cost Settings
        Object.values(targetGroupCell.seg_dict).forEach(segment_cell => {
          segment_cell.cost_setting.obstacle_thickness = costSettings.obstacle_thickness;
          segment_cell.cost_setting.path_thickness = costSettings.path_thickness;
          // 刷新mesh以反映更改
          segment_cell.update_mesh(rootObj.get_scene());
        });
        
        ElMessage.success('所有段节点的Cost Settings已更新');
        emit('updateTree'); // 更新树视图
      }
    }
  }
  batchSegmentEditorVisible.value = false;
}

function onBatchConnectorSave(costSettings: any) {
  const project = label_dict.value.get('Project') as string;
  const smoothName = label_dict.value.get('SmoothName') as string;
  const groupIdx = label_dict.value.get('GroupIdx') as number;
  
  if (project && smoothName && groupIdx !== undefined) {
    const proj = rootObj.proj_dict.get(project);
    const smooth_runner = proj?.smooth_task_dict.get(smoothName);
    if (smooth_runner) {
      // 通过 GroupIdx 找到对应的 group_cell
      let targetGroupCell = null;
      for (const [groupName, groupCell] of smooth_runner.group_cells) {
        if (groupCell.group_idx === groupIdx) {
          targetGroupCell = groupCell;
          break;
        }
      }
      
      if (targetGroupCell) {
        // 批量更新所有连接器节点的Cost Settings
        Object.values(targetGroupCell.connector_dict).forEach(connector_cell => {
          connector_cell.cost_setting.obstacle_thickness = costSettings.obstacle_thickness;
          connector_cell.cost_setting.path_thickness = costSettings.path_thickness;
          // 刷新mesh以反映更改
          connector_cell.update_mesh(rootObj.get_scene());
        });
        
        ElMessage.success('所有连接器节点的Cost Settings已更新');
        emit('updateTree'); // 更新树视图
      }
    }
  }
  batchConnectorEditorVisible.value = false;
}
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
