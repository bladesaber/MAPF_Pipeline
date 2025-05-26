<!-- PathSearch.vue -->
<template>
  <div class="path-search" @click="handleClickOutside">
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

    <DefinePanel
      v-model="definePanelVisible"
      :mode="definePanelMode"
      :label_dict="label_dict"
      @generate="handleDefinePanelGenerate"
    />

    <DefineObstacle
      v-model="defineObstacleVisible"
      :label_dict="label_dict"
      @generate="handleDefineObstacleGenerate"
    />

    <div 
      v-show="gridContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Grid/EditGrid')">Edit Grid</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Grid/HideGrid')">Hide Grid</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Grid/ShowGrid')">Show Grid</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Grid/InitEnv')">Init Env</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Grid/HideDiscretized')">Hide Discretized</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Grid/ShowDiscretized')">Show Discretized</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Grid/ShowInner')">Show Inner</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Grid/HideInner')">Hide Inner</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="obstaclesContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Obstacles/AddObstacle')">Add Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacles/ShowObstacle')">Show Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacles/HideObstacle')">Hide Obstacle</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="obstacleContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/EditObstacle')">Edit Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/HideObstacle')">Hide Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/ShowObstacle')">Show Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/RemoveObstacle')">Remove Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/HideDiscretized')">Hide Discretized</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/ShowDiscretized')">Show Discretized</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="pipesContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('PipeGroups/AddPipeGroup')">Add Pipe Group</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="pipeGroupContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('PipeGroup/AddPipe')">Add Pipe</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PipeGroup/ShowPipeGroup')">Show Pipe Group</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PipeGroup/HidePipeGroup')">Hide Pipe Group</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PipeGroup/RemovePipeGroup')">Remove Pipe Group</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="pipeContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Pipe/EditPipe')">Edit Pipe</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Pipe/ShowPipe')">Show Pipe</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Pipe/HidePipe')">Hide Pipe</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Pipe/RemovePipe')">Remove Pipe</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="taskRootContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Tasks/AddSearchTask')">Add Search Task</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="searchTaskContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/AddPriorityTask')">Add Priority Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/RemoveSearchTask')">Remove Search Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/ConnectToServer')">Connect to Server</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/DiscretizeObstacle')">Discretize Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/RunSearch')">Run Search</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('SearchTask/ClearResult')">Clear Result</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="priorityTaskContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/AddGroupTask')">Add Group Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/RemovePriorityTask')">Remove Priority Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/RunSearch')">Run Search</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/RunSearch')">Run Search</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('PriorityTask/ClearResult')">Clear Result</el-menu-item>
      </el-menu>
    </div>
    
    <div 
      v-show="groupTaskContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/AddCellTask')">Add Cell Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/RemoveGroupTask')">Remove Group Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/RunSearch')">Run Search</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('GroupTask/ClearResult')">Clear Result</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="cellTaskContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
      @click.stop
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('CellTask/RemoveCellTask')">Remove Cell Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('CellTask/EditCellTask')">Edit Cell Task</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('CellTask/ShowLineResult')">Show Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('CellTask/ShowTubeResult')">Show Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('CellTask/HideLineResult')">Hide Line Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('CellTask/HideTubeResult')">Hide Tube Result</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('CellTask/RunSearch')">Run Search</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('CellTask/ClearResult')">Clear Result</el-menu-item>
      </el-menu>
    </div>

    <!-- Add a click outside handler div -->
    <div v-if="isAnyMenuVisible" class="click-outside-handler" @click="hideContextMenu"></div>

    <!-- progress dialog -->
    <el-dialog v-model="progressDialogVisible" title="Progress" width="30%">
      <div>
        <p>{{ progressLog }}</p>
        <el-progress :percentage="progressPercentage" />
      </div>
    </el-dialog>

  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, defineExpose, inject, defineEmits } from 'vue'
import { pass, round } from 'three/tsl'
import { setVisableDict } from '@/utils/utils'
import * as THREE from 'three'
import { ElMessage } from 'element-plus'

import DefinePanel from './DefinePanel.vue'
import DefineObstacle from './DefineObstacle.vue'

import { CylinderObstacle, BoxObstacle, PointsObstacle, SphereLineObstacle } from '@/classes/path_search/obstacle_mesh.ts'
import { CenterRoot } from '@/classes/center_root.ts'
import { PipeMesh } from '@/classes/path_search/pipe_mesh'
import { SearchTask, PriorityTask, GroupTask, CellTask } from '@/classes/path_search/task_runner'
import { da } from 'element-plus/es/locales.mjs'

const emit = defineEmits(['updateTree', 'highlightTreeNode', 'updateLog'])

const rootObj = inject<CenterRoot>('centerRoot', new CenterRoot(new THREE.Scene()))

// ------ progress dialog ------
const progressDialogVisible = ref(false)
const progressPercentage = ref(0)
const progressLog = ref('')

// ------ Left Side Bar ------
const searchTaskContextMenuVisible = ref(false)
const gridContextMenuVisible = ref(false)
const obstaclesContextMenuVisible = ref(false)
const obstacleContextMenuVisible = ref(false)
const pipesContextMenuVisible = ref(false)
const pipeGroupContextMenuVisible = ref(false)
const pipeContextMenuVisible = ref(false)
const taskRootContextMenuVisible = ref(false)
const priorityTaskContextMenuVisible = ref(false)
const groupTaskContextMenuVisible = ref(false)
const cellTaskContextMenuVisible = ref(false)

const contextMenuVisibleDict = {
  'searchTaskContextMenuVisible': searchTaskContextMenuVisible,
  'gridContextMenuVisible': gridContextMenuVisible,
  'obstaclesContextMenuVisible': obstaclesContextMenuVisible,
  'obstacleContextMenuVisible': obstacleContextMenuVisible,
  'pipesContextMenuVisible': pipesContextMenuVisible,
  'pipeGroupContextMenuVisible': pipeGroupContextMenuVisible,
  'pipeContextMenuVisible': pipeContextMenuVisible,
  'taskRootContextMenuVisible': taskRootContextMenuVisible,
  'priorityTaskContextMenuVisible': priorityTaskContextMenuVisible,
  'groupTaskContextMenuVisible': groupTaskContextMenuVisible,
  'cellTaskContextMenuVisible': cellTaskContextMenuVisible
}

const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

const label_dict = ref(new Map<string, string | number | string[] | Map<string, string|number>>())

// Add new ref for tracking active tree node
const activeTreeNode = ref<string | null>(null)

// Add computed property to check if any menu is visible
const isAnyMenuVisible = computed(() => {
  return Object.values(contextMenuVisibleDict).some(visible => visible.value)
})

// Add click outside handler
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

// Modify hideContextMenu to hide all menus
const hideContextMenu = () => {
  taskRootContextMenuVisible.value = false
  searchTaskContextMenuVisible.value = false
  priorityTaskContextMenuVisible.value = false
  groupTaskContextMenuVisible.value = false
  cellTaskContextMenuVisible.value = false
  gridContextMenuVisible.value = false
  obstaclesContextMenuVisible.value = false
  obstacleContextMenuVisible.value = false
  pipesContextMenuVisible.value = false
  pipeGroupContextMenuVisible.value = false
  pipeContextMenuVisible.value = false
  activeTreeNode.value = null
}

// Modify showContextMenu to handle tree node highlighting
const showContextMenu = (data: any) => {
  // Reset all menu states first
  hideContextMenu()
  
  // Set menu position
  contextMenuPosition.x = data.x
  contextMenuPosition.y = data.y

  // Set active tree node and emit highlight event
  if (data.data && data.data.id) {
    activeTreeNode.value = data.data.id
    emit('highlightTreeNode', data.data.id)
  }

  // Convert data.data to Map
  const newLabelDict = new Map<string, string | number | string[] | Map<string, string|number>>()
  
  // Add basic properties
  if (data.data.label) newLabelDict.set('label', data.data.label)
  if (data.data.style) newLabelDict.set('style', data.data.style)
  
  // Add any additional data properties
  if (data.data.data) {
    Object.entries(data.data.data).forEach(([key, value]) => {
      newLabelDict.set(key, value as string | number | string[] | Map<string, string|number>)
    })
  }

  label_dict.value = newLabelDict

  // Show appropriate menu based on style
  const style = label_dict.value.get('style')
  if (style === 'Tasks') {
    taskRootContextMenuVisible.value = true
  } else if (style === 'SearchTask') {
    searchTaskContextMenuVisible.value = true
  } else if (style === 'PriorityTask') {
    priorityTaskContextMenuVisible.value = true
  } else if (style === 'GroupTask') {
    groupTaskContextMenuVisible.value = true
  } else if (style === 'CellTask') {
    cellTaskContextMenuVisible.value = true
  } else if (style === 'Grid') {
    gridContextMenuVisible.value = true
  } else if (style === 'Obstacles') {
    obstaclesContextMenuVisible.value = true
  } else if (style === 'Obstacle') {
    obstacleContextMenuVisible.value = true
  } else if (style === 'Pipes') {
    pipesContextMenuVisible.value = true
  } else if (style === 'PipeGroup') {
    pipeGroupContextMenuVisible.value = true
  } else if (style === 'Pipe') {
    pipeContextMenuVisible.value = true
  }
}

// ------ Fragment ------
const defineObstacleVisible = ref(false)
const definePanelVisible = ref(false)

const fragmentVisibleDict = {
  'defineObstacleVisible': defineObstacleVisible,
  'definePanelVisible': definePanelVisible
}

const hideFragmentMenu = () => {
  setVisableDict('defineObstacleVisible', fragmentVisibleDict, true, false)
}

const definePanelMode = ref<'PipeGroup' | 'Pipe' | 'SearchTask' | 'PriorityTask' | 'GroupTask' | 'CellTask'>('SearchTask')

const handleTreeContextMenu = (mode: string) => {
  hideContextMenu()
  hideFragmentMenu()

  if (mode === 'Obstacles/AddObstacle') {
    label_dict.value.set('RelaxGroup', Array.from(rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.keys()))
    setVisableDict('defineObstacleVisible', fragmentVisibleDict, true, true)

  } else if (mode == 'PipeGroups/AddPipeGroup') {
    definePanelMode.value = 'PipeGroup'
    setVisableDict('definePanelVisible', fragmentVisibleDict, true, true)

  } else if (mode == 'PipeGroup/AddPipe') {
    definePanelMode.value = 'Pipe'
    setVisableDict('definePanelVisible', fragmentVisibleDict, true, true)

  } else if (mode == 'Tasks/AddSearchTask') {
    definePanelMode.value = 'SearchTask'
    setVisableDict('definePanelVisible', fragmentVisibleDict, true, true)

  } else if (mode == 'SearchTask/AddPriorityTask') {
    definePanelMode.value = 'PriorityTask'
    setVisableDict('definePanelVisible', fragmentVisibleDict, true, true)

  } else if (mode == 'SearchTask/RemoveSearchTask') {
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.delete(label_dict.value.get('SearchTask') as string)
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.reset_discretize_source()
    emit('updateTree')

  } else if (mode == 'PriorityTask/AddGroupTask') {
    definePanelMode.value = 'GroupTask'
    const project_manager = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!
    const group_names = Array.from(project_manager.pipe_dict.keys())
    label_dict.value.set('GroupNames', group_names as string[])
    setVisableDict('definePanelVisible', fragmentVisibleDict, true, true)

  } else if (mode == 'GroupTask/AddCellTask') {
    definePanelMode.value = 'CellTask'
    const project_manager = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!
    const group_pipes = project_manager.pipe_dict.get(label_dict.value.get('GroupTask') as string)!
    const pipe_names = Array.from(group_pipes.keys())
    label_dict.value.set('PipeNames', pipe_names as string[])
    setVisableDict('definePanelVisible', fragmentVisibleDict, true, true)

  } else if (mode == 'Grid/HideGrid') {
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid.hide_mesh()

  } else if (mode == 'Grid/ShowGrid') {
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid.show_mesh()

  } else if (mode == 'Grid/HideInner') {
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid.hide_inner_framework_mesh()

  } else if (mode == 'Grid/ShowInner') {
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid.show_inner_framework_mesh()

  } else if (mode == 'PipeGroup/HidePipeGroup') {
    const pipe_group = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.get(label_dict.value.get('PipeGroup') as string)!
    pipe_group.forEach(pipe => pipe.hide_mesh())

  } else if (mode == 'PipeGroup/ShowPipeGroup') {
    const pipe_group = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.get(label_dict.value.get('PipeGroup') as string)!
    pipe_group.forEach(pipe => pipe.show_mesh())

  } else if (mode == 'PipeGroup/RemovePipeGroup') {
    const project_manager = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!
    const pipe_group = project_manager.pipe_dict.get(label_dict.value.get('PipeGroup') as string)!
    pipe_group.forEach(pipe => pipe.destory(rootObj.get_scene()))
    project_manager.pipe_dict.delete(label_dict.value.get('PipeGroup') as string)
    project_manager.group_color_dict.delete(label_dict.value.get('PipeGroup') as string)
    project_manager.reset_discretize_source()

    // TODO: remove all task which is related to this pipe group
    emit('updateTree')

  } else if (mode == 'Pipe/HidePipe') {
    const pipe = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.get(
      label_dict.value.get('PipeGroup') as string)!.get(label_dict.value.get('Pipe') as string)!
    pipe.hide_mesh()

  } else if (mode == 'Pipe/ShowPipe') {
    const pipe = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.get(
      label_dict.value.get('PipeGroup') as string)!.get(label_dict.value.get('Pipe') as string)!
    pipe.show_mesh()

  } else if (mode == 'Pipe/RemovePipe') {
    const pipe = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.get(
      label_dict.value.get('PipeGroup') as string)!.get(label_dict.value.get('Pipe') as string)!
    pipe.destory(rootObj.get_scene())
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.get(label_dict.value.get('PipeGroup') as string)!.delete(label_dict.value.get('Pipe') as string)
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.reset_discretize_source()
    emit('updateTree')

  } else if (mode == 'Obstacles/RemoveObstacle') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!
    obstacle.destory(rootObj.get_scene())
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.delete(label_dict.value.get('Obstacle') as string)
    emit('updateTree')

  } else if (mode == 'Obstacles/HideObstacle') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!

  } else if (mode == 'Obstacles/ShowObstacle') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!
    obstacle.show_mesh()

  } else if (mode == 'PriorityTask/HideLineResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.hide_line_mesh()

  } else if (mode == 'PriorityTask/ShowLineResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.show_line_mesh()

  } else if (mode == 'PriorityTask/HideTubeResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.hide_tube_mesh()

  } else if (mode == 'PriorityTask/ShowTubeResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.show_tube_mesh()

  } else if (mode == 'GroupTask/ShowLineResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.show_line_mesh()
  
  } else if (mode == 'GroupTask/ShowTubeResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.show_tube_mesh()
  
  } else if (mode == 'GroupTask/HideLineResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.hide_line_mesh()
  
  } else if (mode == 'GroupTask/HideTubeResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.hide_tube_mesh()

  } else if (mode == 'CellTask/HideLineResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.cell_tasks.find(
      cell_task => cell_task.name === label_dict.value.get('CellTask') as string
    )!.hide_line_mesh()

  } else if (mode == 'CellTask/HideTubeResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.cell_tasks.find(
      cell_task => cell_task.name === label_dict.value.get('CellTask') as string
    )!.hide_tube_mesh()

  } else if (mode == 'CellTask/ShowLineResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.cell_tasks.find(
      cell_task => cell_task.name === label_dict.value.get('CellTask') as string
    )!.show_line_mesh()
  
  } else if (mode == 'CellTask/ShowTubeResult') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.cell_tasks.find(
      cell_task => cell_task.name === label_dict.value.get('CellTask') as string
    )!.show_tube_mesh()

  } else if (mode == 'CellTask/RemoveCellTask') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.cell_tasks.splice(
      priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.cell_tasks.findIndex(
        cell_task => cell_task.name === label_dict.value.get('CellTask') as string
      ), 1
    )
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.reset_discretize_source()
    emit('updateTree')

  } else if (mode == 'GroupTask/RemoveGroupTask') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.group_tasks.delete(label_dict.value.get('GroupTask') as string)
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.reset_discretize_source()
    emit('updateTree')

  } else if (mode == 'PriorityTask/RemovePriorityTask') {
    const search_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!
    search_task.priority_tasks.splice(search_task.priority_tasks.findIndex(
      task => task.name === label_dict.value.get('PriorityTask') as string
    ), 1)
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.reset_discretize_source()
    emit('updateTree')

  } else if (mode == 'Obstacle/HideObstacle') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!
    obstacle.hide_mesh()

  } else if (mode == 'Obstacle/ShowObstacle') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!
    obstacle.show_mesh()

  } else if (mode == 'Obstacle/RemoveObstacle') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!
    obstacle.destory(rootObj.get_scene())
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.delete(label_dict.value.get('Obstacle') as string)
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.reset_discretize_source()
    emit('updateTree')
  
  } else if (mode == 'SearchTask/DiscretizeObstacle') {
    const pipes_info = new Map<string, {radius: number, x: number, y: number, z: number, group_name: string}>()
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.forEach((pipes, group_name) => {
      pipes.forEach((pipe, pipe_name) => {
        pipes_info.set(pipe_name, {
          group_name: group_name,
          radius: pipe.radius,
          x: pipe.x,
          y: pipe.y,
          z: pipe.z
        })
      })
    })

    let min_task_radius = Infinity
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.forEach(ptask => {
      ptask.group_tasks.forEach(gtask => {
        gtask.cell_tasks.forEach(ctask => {
          min_task_radius = Math.min(min_task_radius, ctask.radius)
          pipes_info.get(ctask.start_name)!.radius = Math.max(
            pipes_info.get(ctask.start_name)!.radius, ctask.radius
          )
          pipes_info.get(ctask.end_name)!.radius = Math.max(
            pipes_info.get(ctask.end_name)!.radius, ctask.radius
          )
        })
      })
    })
    if (min_task_radius >= Infinity) {
      ElMessage.warning('No task found')
      return
    }

    const pipes_info_list: {radius: number, x: number, y: number, z: number}[] = []
    pipes_info.forEach((info, key) => {
      pipes_info_list.push({
        radius: info.radius,
        x: info.x,
        y: info.y,
        z: info.z
      })
    })

    // discrete pipes
    pipes_info.forEach((info, pipe_name) => {
      const pipe = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.get(
        info.group_name)!.get(pipe_name)!
      pipe.discrete(info.radius, min_task_radius * 0.5, 
      pipes_info_list, rootObj.get_scene())
    })

    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid.discretize(
      rootObj.get_scene(), min_task_radius * 0.5, 
      label_dict.value.get('SearchTask') as string,
      pipes_info_list
    )

    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.forEach((obstacle, key) => {
      obstacle.discrete(
        rootObj.get_scene(), min_task_radius * 0.5, 
        label_dict.value.get('SearchTask') as string, pipes_info_list
      )
    })

    emit('updateTree')

  } else if (mode == 'Obstacle/HideDiscretized') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!
    obstacle.hide_discretize_mesh()

  } else if (mode == 'Obstacle/ShowDiscretized') {
    const obstacle = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.get(label_dict.value.get('Obstacle') as string)!
    obstacle.show_discretize_mesh()

  } else if (mode == 'Grid/HideDiscretized') {
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid.hide_discretize_mesh()

  } else if (mode == 'Grid/ShowDiscretized') {
    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid.show_discretize_mesh()
  } else if (mode == 'Grid/InitEnv') {
    const static_obstacle_points: number[][] = []
    const dynamic_obstacle_points: { [key: string]: number[][] } = {}

    const grid = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.grid
    if (grid.discretize_source == '') {
      ElMessage.warning('Grid has not been discretized')
      return
    } else {
      grid.surface_obstacle_points.forEach(point => {
        static_obstacle_points.push([
          grid.value_to_grid_scaled(point[0]),
          grid.value_to_grid_scaled(point[1]), 
          grid.value_to_grid_scaled(point[2]),
          0.0
        ])
      })
    }

    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.obstacle_dict.forEach((obstacle, key) => {
      if (obstacle.discretize_source == '') {
        ElMessage.warning('Grid has not been discretized')
        return
      } else {
        if (obstacle.relax_groups.length == 0){
          for (let i = 0; i < obstacle.discretize_points.length; i++) {
            static_obstacle_points.push([
              grid.value_to_grid_scaled(obstacle.discretize_points[i][0]),
              grid.value_to_grid_scaled(obstacle.discretize_points[i][1]), 
              grid.value_to_grid_scaled(obstacle.discretize_points[i][2]),
              obstacle.discretize_radius[i]
            ])
          }
        } else {
          obstacle.relax_groups.forEach(group => {
            if (!dynamic_obstacle_points[group]) {
              dynamic_obstacle_points[group] = []
            }
          })
          for (let i = 0; i < obstacle.discretize_points.length; i++) {
            obstacle.relax_groups.forEach(group => {
              dynamic_obstacle_points[group].push([
                grid.value_to_grid_scaled(obstacle.discretize_points[i][0]),
                grid.value_to_grid_scaled(obstacle.discretize_points[i][1]), 
                grid.value_to_grid_scaled(obstacle.discretize_points[i][2]),
                obstacle.discretize_radius[i]
              ])
            })
          }
        }
      }
    })

    rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.pipe_dict.forEach((pipes, group_name) => {
      pipes.forEach((pipe, pipe_name) => {
        pipe.discretize_points.forEach(point => {
          static_obstacle_points.push([
            grid.value_to_grid_scaled(point[0]),
            grid.value_to_grid_scaled(point[1]), 
            grid.value_to_grid_scaled(point[2]),
            0.0
          ])
        })
      })
    })

    initEnv(
      label_dict.value.get('Project') as string, 
      static_obstacle_points, dynamic_obstacle_points,
      grid.xsize, grid.ysize, grid.zsize, grid.resolution
    )
  
  } else if (mode == 'PriorityTask/ClearResult') {
    const search_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!
    const priority_task = search_task.priority_tasks.find(
      task => task.name === label_dict.value.get('PriorityTask') as string)
    priority_task!.clear_result(rootObj.get_scene())
    search_task.has_result = false
    emit('updateTree')

  } else if (mode == 'GroupTask/ClearResult') {
    const search_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!
    const priority_task = search_task.priority_tasks.find(
      task => task.name === label_dict.value.get('PriorityTask') as string)
    const group_task = priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!
    group_task.clear_result(rootObj.get_scene())
    priority_task!.has_result = false
    search_task.has_result = false
    emit('updateTree')

  } else if (mode == 'CellTask/ClearResult') {
    const search_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!
    const priority_task = search_task.priority_tasks.find(
      task => task.name === label_dict.value.get('PriorityTask') as string)
    const group_task = priority_task!.group_tasks.get(label_dict.value.get('GroupTask') as string)!
    const cell_task = group_task.cell_tasks.find(
      cell_task => cell_task.name === label_dict.value.get('CellTask') as string
    )!
    cell_task.clear_result(rootObj.get_scene())
    group_task.has_result = false
    priority_task!.has_result = false
    search_task.has_result = false
    emit('updateTree')
  
  } else if (mode == 'SearchTask/ClearResult') {
    const search_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!
    search_task.clear_result(rootObj.get_scene())
    emit('updateTree')

  } else if (mode == 'CellTask/RunSearch') {
    const cell_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string
      )!.group_tasks.get(label_dict.value.get('GroupTask') as string)!.cell_tasks.find(
        cell_task => cell_task.name === label_dict.value.get('CellTask') as string
      )!
    if (cell_task.has_result) {
      ElMessage.warning('Cell task has already been searched')
      return
    }
    runCellTask(
      label_dict.value.get('Project') as string,
      label_dict.value.get('SearchTask') as string,
      label_dict.value.get('PriorityTask') as string,
      label_dict.value.get('GroupTask') as string,
      label_dict.value.get('CellTask') as string
    )
  } else if (mode == 'GroupTask/RunSearch') {
    const group_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string
      )!.group_tasks.get(label_dict.value.get('GroupTask') as string)!
    if (group_task.has_result) {
      ElMessage.warning('Group task has already been searched')
      return
    }
    runGroupTask(
      label_dict.value.get('Project') as string,
      label_dict.value.get('SearchTask') as string,
      label_dict.value.get('PriorityTask') as string,
      label_dict.value.get('GroupTask') as string
    )
  } else if (mode == 'PriorityTask/RunSearch') {
    const priority_task = rootObj.proj_dict.get(label_dict.value.get('Project') as string)!.search_task_dict.get(
      label_dict.value.get('SearchTask') as string)!.priority_tasks.find(
        task => task.name === label_dict.value.get('PriorityTask') as string
      )
    if (priority_task!.has_result) {
      ElMessage.warning('Priority task has already been searched')
      return
    }
    runPriorityTask(
      label_dict.value.get('Project') as string,
      label_dict.value.get('SearchTask') as string,
      label_dict.value.get('PriorityTask') as string
    )
  }
}

const handleDefineObstacleGenerate = (data: any) => {
  if (data.type == 'CylinderObstacle') {
    const obstacle_mesh = new CylinderObstacle(
      data.name, [1, 0, 0],
      data.x, data.y, data.z, 
      data.radius, data.height, 
      data.vector_x, data.vector_y, data.vector_z,
      data.relax_groups,
      rootObj.get_scene()
    );
    const current_label_dict = data.label_dict
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.obstacle_dict.set(obstacle_mesh.name, obstacle_mesh)
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()

  } else if (data.type == 'BoxObstacle') {
    const obstacle_block = new BoxObstacle(
      data.name, [1, 0, 0],
      data.xmin, data.xmax, data.ymin, data.ymax, data.zmin, data.zmax,
      data.relax_groups,
      rootObj.get_scene()
    );
    const current_label_dict = data.label_dict
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.obstacle_dict.set(obstacle_block.name, obstacle_block)
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()

  } else if (data.type == 'SphereLineObstacle') {
    const obstacle_sphere_line = new SphereLineObstacle(
      data.name, [1, 0, 1],
      data.points, data.radius,
      data.relax_groups,
      rootObj.get_scene()
    );
    const current_label_dict = data.label_dict
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.obstacle_dict.set(obstacle_sphere_line.name, obstacle_sphere_line)
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()

  }
  hideFragmentMenu()
  emit('updateTree')
}

const handleDefinePanelGenerate = (data: any) => {
  if (data.mode === 'PipeGroup') {
    const current_label_dict = data.label_dict
    if (!rootObj.proj_dict.get(current_label_dict.get('Project')!)!.pipe_dict.has(data.name)) {
      rootObj.proj_dict.get(current_label_dict.get('Project')!)!.pipe_dict.set(
        data.name, new Map<string, PipeMesh>()
      )
      rootObj.proj_dict.get(current_label_dict.get('Project')!)!.group_color_dict.set(
        data.name, data.color
      )
      rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()
    } else {
      ElMessage.warning('Pipe group already exists')
    }

  } else if (data.mode === 'Pipe') {
    const current_label_dict = data.label_dict
    if (!rootObj.proj_dict.get(current_label_dict.get('Project')!)!.pipe_dict.get(current_label_dict.get('PipeGroup')!)!.has(data.name)) {
      const group_color = rootObj.proj_dict.get(current_label_dict.get('Project')!)!.group_color_dict.get(current_label_dict.get('PipeGroup')!)!
      const pipe_mesh = new PipeMesh(
        data.name, data.x, data.y, data.z,
        data.radius, data.vec_x, data.vec_y, data.vec_z,
        group_color, data.group, data.pipe_mode,
        rootObj.get_scene()
      )
      rootObj.proj_dict.get(current_label_dict.get('Project')!)!.pipe_dict.get(current_label_dict.get('PipeGroup')!)!.set(
        data.name, pipe_mesh
      )
      rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()
    } else {
      ElMessage.warning('Pipe already exists')
    }

  } else if (data.mode === 'SearchTask') {
    const current_label_dict = data.label_dict
    const task = new SearchTask(data.name)
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.search_task_dict.set(
      task.name, task
    )
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()

  } else if (data.mode === 'PriorityTask') {
    const current_label_dict = data.label_dict

    const run_cfg = new Map<string, any>()
    for (const [key, value] of Object.entries(data.run_cfg)) {
      run_cfg.set(key, value)
    }

    const task = new PriorityTask(data.name, run_cfg)
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.search_task_dict.get(current_label_dict.get('SearchTask')!)!.priority_tasks.push(task)
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()

  } else if (data.mode === 'GroupTask') {
    const current_label_dict = data.label_dict

    const run_cfg = new Map<string, any>()
    for (const [key, value] of Object.entries(data.run_cfg)) {
      run_cfg.set(key, value)
    }

    const task = new GroupTask(data.name, run_cfg)
    let search_task = rootObj.proj_dict.get(current_label_dict.get('Project')!)!.search_task_dict.get(current_label_dict.get('SearchTask')!)!
    for (let priority_task of search_task.priority_tasks) {
      if (priority_task.name === current_label_dict.get('PriorityTask')!) {
        priority_task.group_tasks.set(task.name, task)
      }
    }
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()

  } else if (data.mode === 'CellTask') {
    const current_label_dict = data.label_dict
    const group_color = rootObj.proj_dict.get(current_label_dict.get('Project')!)!.group_color_dict.get(current_label_dict.get('GroupTask')!)!
    const task = new CellTask(
      data.name, data.start_name, data.end_name,
      data.radius, group_color
    )
    let search_task = rootObj.proj_dict.get(current_label_dict.get('Project')!)!.search_task_dict.get(current_label_dict.get('SearchTask')!)!
    for (let priority_task of search_task.priority_tasks) {
      if (priority_task.name === current_label_dict.get('PriorityTask')!) {
        priority_task.group_tasks.get(current_label_dict.get('GroupTask')!)!.cell_tasks.push(task)
      }
    }
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.reset_discretize_source()

  }
  hideFragmentMenu()
  emit('updateTree')
}

// ------ Backend ------
const initEnv = async (
  project_name: string, 
  static_obstacle_points: number[][], dynamic_obstacle_points: { [key: string]: number[][] },
  xsize: number, ysize: number, zsize: number, resolution: number
) => {
  if (static_obstacle_points.length > 0) {
    if (!await upload_static_obstacle_pcd(project_name, static_obstacle_points, xsize, ysize, zsize, resolution)) {
      return
    }
  }
  if (Object.keys(dynamic_obstacle_points).length > 0) {
    for (const [relax_group_name, points] of Object.entries(dynamic_obstacle_points)) {
      if (!await upload_relax_group_obstacle_pcd(project_name, relax_group_name, points)) {
      return
      }
    }
  }
  await requestInnerGrid(project_name)
}

const upload_static_obstacle_pcd = async (
  project_name: string, static_obstacle_points: number[][], 
  xsize: number, ysize: number, zsize: number, resolution: number
) => {
  const url = 'http://192.168.110.54:9998/path_search/upload_static_obstacle_pcd'
  
  progressDialogVisible.value = true
  progressPercentage.value = 0
  progressLog.value = 'Uploading static obstacle pcd...'
  
  // Define chunk size
  const CHUNK_SIZE = 5000 // Number of points per chunk
  const total_chunks = Math.ceil(static_obstacle_points.length / CHUNK_SIZE)
  
  for (let i = 0; i < static_obstacle_points.length; i += CHUNK_SIZE) {
    const chunk_data = static_obstacle_points.slice(i, i + CHUNK_SIZE)
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_id: project_name,
          obstacle_pcd: chunk_data,
          x_size: xsize,
          y_size: ysize,
          z_size: zsize,
          resolution: resolution,
          chunk_index: i,
          total_chunks: total_chunks,
          is_last_chunk: i === static_obstacle_points.length - CHUNK_SIZE
        })
      })
      
      if (!response.ok) {
        ElMessage.error('Failed to upload static obstacle pcd')
        return false
      }
      
      const data = await response.json()
      if (data.error) {
        ElMessage.error(data.error)
        return false
      }
      
      const progress = Math.round(((i + 1) / static_obstacle_points.length) * 100)
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

const upload_relax_group_obstacle_pcd = async (
  project_name: string, relax_group_name: string, dynamic_obstacle_points: number[][]
) => {
  const url = 'http://192.168.110.54:9998/path_search/upload_relax_group_obstacle_pcd'

  progressDialogVisible.value = true
  progressPercentage.value = 0
  progressLog.value = 'Uploading dynamic obstacle pcd...'

  // Define chunk size
  const CHUNK_SIZE = 5000 // Number of points per chunk
  const total_chunks = Math.ceil(dynamic_obstacle_points.length / CHUNK_SIZE)

  for (let i = 0; i < dynamic_obstacle_points.length; i += CHUNK_SIZE) {
    const chunk_data = dynamic_obstacle_points.slice(i, i + CHUNK_SIZE)
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_id: project_name,
          obstacle_pcd: chunk_data,
          relax_group_name: relax_group_name,
          chunk_index: i,
          total_chunks: total_chunks,
          is_last_chunk: i === dynamic_obstacle_points.length - CHUNK_SIZE
        })
      })
      
      if (!response.ok) {
        ElMessage.error('Failed to upload dynamic obstacle pcd')
        return false
      }
      
      const data = await response.json()
      if (data.error) {
        ElMessage.error(data.error)
        return false
      }
      
      const progress = Math.round(((i + 1) / dynamic_obstacle_points.length) * 100)
      progressPercentage.value = progress
      progressLog.value = `Uploading dynamic obstacle pcd: ${progress}%`

    } catch (error) {
      ElMessage.error('Failed to upload dynamic obstacle pcd')
      progressDialogVisible.value = false
      return false
    }
    await new Promise(resolve => setTimeout(resolve, 200))
  }
  progressDialogVisible.value = false
  progressPercentage.value = 100
  progressLog.value = 'Dynamic obstacle pcd uploaded successfully'
  return true
}

const requestInnerGrid = async (project_name: string) => {
  const pipe_data = rootObj.proj_dict.get(project_name)!.export_pipe_json(true)

  progressDialogVisible.value = true
  progressPercentage.value = 0
  progressLog.value = 'Requesting inner grid...'

  const url = 'http://192.168.110.54:9998/path_search/request_inner_xyzd'
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      project_id: project_name,
      pipe_data: pipe_data
      })
    })

    if (!response.ok) {
      ElMessage.error('Failed to request inner grid')
      return
    }

    /*
    const reader = response.body?.getReader()
    if (!reader) throw new Error('Failed to get response reader')

    let inner_grid: number[][] = []
    let inner_grid_distance: number[] = []
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = new TextDecoder().decode(value)      
      buffer += text

      try {
        const data = JSON.parse(buffer)
        if (data.total_finished) {
          rootObj.proj_dict.get(project_name)!.grid!.add_inner_grid(
            inner_grid, inner_grid_distance, rootObj.get_scene()
          )
          break
        }

        if (data.name === 'request_inner_xyzd') {
          progressLog.value = data.msg
          progressPercentage.value = data.progress
          data.data.forEach((item: number[]) => {
            inner_grid.push([item[0], item[1], item[2]])
            inner_grid_distance.push(item[3])
          })
        }
        buffer = ''

      } catch (error) {
        continue
      }
    }
    */

    } catch (error) {
      console.error('Error:', error)
      ElMessage.error('Failed to request inner grid')
    } finally {
      progressDialogVisible.value = false
    }
}
  
const uploadTaskCache = async (project_name: string, search_task_name: string) => {
  const search_task = rootObj.proj_dict.get(project_name)!.search_task_dict.get(search_task_name)!

  const url = 'http://192.168.110.54:9998/path_search/upload_task_cache'
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      project_id: project_name,
      cache_data: search_task.toJSON(true, rootObj.proj_dict.get(project_name)!.grid)
    })
  })

  if (!response.ok) {
    throw new Error('Network response was not ok')
  }

  const data = await response.json()
  if (data.error) {
    throw new Error(data.error)
  }

  return data
}

const runCellTask = async (
  project_name: string, search_task_name: string, priority_name: string, group_name: string, cell_name: string
) => {
  try {
    const return_data = await uploadTaskCache(project_name, search_task_name)
    if (!return_data.is_success) {
      ElMessage.error(return_data.error)
      return
    }

    const group_task = rootObj.proj_dict.get(project_name)!.search_task_dict.get(
      search_task_name)!.priority_tasks.find(priority_task => priority_task.name === priority_name)!.group_tasks.get(group_name)!
    const cell_task = group_task.cell_tasks.find(cell_task => cell_task.name === cell_name)!

    if (cell_task.has_result) {
      ElMessage.error('Cell task already has result')
      return
    }

    const response_type = 'stream' as const
    const url = 'http://192.168.110.54:9998/path_search/run_cell_task'
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        project_id: project_name,
        task_name: search_task_name,
        priority_name: priority_name,
        group_name: group_name,
        cell_name: cell_name,
        response_type: response_type,
        max_running_time: 600
      })
    })
    if (!response.ok) {
      ElMessage.error('Network response was not ok')
      return
    }

    if ((response_type as string) === 'single') {
      const data = await response.json()
      if (data.is_success) {
        cell_task.set_result(data.path, rootObj.get_scene(), true, rootObj.proj_dict.get(project_name)!.grid)
        emit('updateTree')
      }
      else {
        ElMessage.error(data.error)
      }
    } else if (response_type == 'stream') {
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
        if (data.name === 'run_cell_task') {
          if (data.total_finished) {
            const result = data.result
            if (result.is_success) {
              cell_task.set_result(result.path, rootObj.get_scene(), true, rootObj.proj_dict.get(project_name)!.grid)
              emit('updateTree')
            }
            else {
              ElMessage.error(result.error)
              emit('updateLog', result.error)
            }
            break
          } else {
            const message = `${data.name} Progress: ${data.progress.toFixed(2)}s, Message: ${data.msg}`
            emit('updateLog', message)
          }
        }
        buffer = ''

        } catch (error) {
          continue
        }
      }
      
    }

  } catch (error) {
    console.error('Error in runCellTask:', error)
    ElMessage.error(`Error in runCellTask: ${error}`)
    return
  }
}

const runGroupTask = async (project_name: string, search_task_name: string, priority_name: string, group_name: string) => {
  try {
    const return_data = await uploadTaskCache(project_name, search_task_name)
    if (!return_data.is_success) {
      ElMessage.error(return_data.error)
      return
    }

    const group_task = rootObj.proj_dict.get(project_name)!.search_task_dict.get(
      search_task_name)!.priority_tasks.find(priority_task => priority_task.name === priority_name)!.group_tasks.get(group_name)!
    
    if (group_task.has_result) {
      ElMessage.error('Group task already has result')
      return
    }

    const response_type = 'stream' as const
    const url = 'http://192.168.110.54:9998/path_search/run_group_task'
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        project_id: project_name,
        task_name: search_task_name,
        priority_name: priority_name,
        group_name: group_name,
        response_type: response_type,
        max_running_time: 600
      })
    })

    if (!response.ok) {
      ElMessage.error('Network response was not ok')
      return
    }

    if ((response_type as string) === 'single') {
      const data = await response.json()
      if (data.is_success) {
        const path_set = new Map<string, number[][]>()
        for (const [key, value] of Object.entries(data.path_set)) {
        path_set.set(key, (value as { path: number[][] }).path)
      }
      group_task.set_result(path_set, rootObj.get_scene(), true, rootObj.proj_dict.get(project_name)!.grid)
      emit('updateTree')
    }
      else {
        ElMessage.error(data.error)
      }
    } else if (response_type == 'stream') {
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
          if (data.name === 'run_group_task') {
            if (data.total_finished) {
              const result = data.result
              if (result.is_success) {
                const path_set = new Map<string, number[][]>()
                for (const [key, value] of Object.entries(result.path_set)) {
                  path_set.set(key, (value as { path: number[][] }).path)
                }
                group_task.set_result(path_set, rootObj.get_scene(), true, rootObj.proj_dict.get(project_name)!.grid)
                emit('updateTree')
              }
              else {
                ElMessage.error(result.error)
                emit('updateLog', result.error)
              }
              break
            } else {
              const message = `${data.name} Progress: ${data.progress.toFixed(2)}s, Message: ${data.msg}`
              emit('updateLog', message)
            }
          }
          buffer = ''

        } catch (error) {
          continue
        }
      }
    }
  } catch (error) {
    console.error('Error in runCellTask:', error)
    throw error
  }
}

const runPriorityTask = async (project_name: string, search_task_name: string, priority_name: string) => {
  try {
    const return_data = await uploadTaskCache(project_name, search_task_name)
    if (!return_data.is_success) {
      ElMessage.error(return_data.error)
      return
    }

    const priority_task = rootObj.proj_dict.get(project_name)!.search_task_dict.get(
      search_task_name)!.priority_tasks.find(priority_task => priority_task.name === priority_name)!
    
    if (priority_task.has_result) {
      ElMessage.error('Priority task already has result')
      return
    }

    const response_type = 'stream' as const
    const url = 'http://192.168.110.54:9998/path_search/run_priority_task'
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        project_id: project_name,
        task_name: search_task_name,
        priority_name: priority_name,
        response_type: response_type,
      })
    })

    if (!response.ok) {
      ElMessage.error('Network response was not ok')
      return
    }

    if ((response_type as string) === 'single') {
      const data = await response.json()
      if (data.is_success) {
        const path_res = new Map<string, Map<string, number[][]>>()
        for (const [group_name, group_results] of Object.entries(data.path_res)) {
          const path_set = new Map<string, number[][]>()
          for (const [cell_name, path] of Object.entries(group_results as Record<string, { path: number[][] }>)) {
            path_set.set(cell_name, path.path)
          }
          path_res.set(group_name, path_set)
        }
        priority_task.set_result(path_res, rootObj.get_scene(), true, rootObj.proj_dict.get(project_name)!.grid)
        emit('updateTree')
      }
      else {
        ElMessage.error(data.error)
      }
    } else if (response_type == 'stream') {
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
          if (data.name === 'run_priority_task') {
            if (data.total_finished) {
              priority_task.clear_result(rootObj.get_scene())
              const result = data.result
              if (result.is_success) {
                const path_res = new Map<string, Map<string, number[][]>>()
                for (const [group_name, group_results] of Object.entries(result.result)) {
                  const path_set = new Map<string, number[][]>()
                  for (const [cell_name, path] of Object.entries(group_results as Record<string, { path: number[][] }>)) {
                    path_set.set(cell_name, path.path)
                  }
                  path_res.set(group_name, path_set)
                }
                priority_task.set_result(path_res, rootObj.get_scene(), true, rootObj.proj_dict.get(project_name)!.grid)
                emit('updateTree')
              }
              else {
                ElMessage.error(result.error)
                emit('updateLog', result.error)
              }
              break
            } else {
              if (data['result'] !== undefined){
                const result = data.result
                const path_res = new Map<string, Map<string, number[][]>>()
                for (const [group_name, group_results] of Object.entries(result.result)) {
                  const path_set = new Map<string, number[][]>()
                  for (const [cell_name, path] of Object.entries(group_results as Record<string, { path: number[][] }>)) {
                    path_set.set(cell_name, path.path)
                  }
                  path_res.set(group_name, path_set)
                }
                priority_task.clear_result(rootObj.get_scene())
                priority_task.set_result(path_res, rootObj.get_scene(), true, rootObj.proj_dict.get(project_name)!.grid)
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
    }
  } catch (error) {
    console.error('Error in runCellTask:', error)
    throw error
  }
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

