<!-- PathSearch.vue -->
<template>
  <div class="path-search">
    <!-- Top Menu Bar -->
    <el-menu mode="horizontal" class="menu-bar">
      <el-sub-menu index="1">
        <template #title>File</template>
        <el-menu-item index="1-1" @click="triggerFileInput">
          <span>Open</span>
        </el-menu-item>
        <el-menu-item index="1-2">Save</el-menu-item>
        <el-menu-item index="1-3">Export</el-menu-item>
      </el-sub-menu>
      <el-sub-menu index="2">
        <template #title>View</template>
        <el-menu-item index="2-1">Show Grid</el-menu-item>
        <el-menu-item index="2-2">Show Axes</el-menu-item>
        <el-menu-item index="2-3">Reset Camera</el-menu-item>
      </el-sub-menu>
      <el-sub-menu index="3">
        <template #title>Tools</template>
        <el-menu-item index="3-1" @click="handleToolMenuClick('AddTask')">Add Task</el-menu-item>
        <el-menu-item index="3-2" @click="handleToolMenuClick('AutoDebug')">Auto Debug</el-menu-item>
      </el-sub-menu>
      <el-sub-menu index="4">
        <template #title>Setting</template>
        <el-menu-item index="4-1">Default</el-menu-item>
      </el-sub-menu>
      <el-sub-menu index="5">
        <template #title>Help</template>
        <el-menu-item index="5-1">About</el-menu-item>
        <el-menu-item index="5-2">Documentation</el-menu-item>
      </el-sub-menu>
    </el-menu>

    <!-- Main Content Area -->
    <div class="main-content">
      <!-- Resizable Left Sidebar -->
      <div 
        class="sidebar"
        :style="{ width: sidebarWidth + 'px' }"
      >
        <el-tree
          v-if="treeData.length"
          :data="treeData"
          @node-click="handleLeftTreeClick"
          @node-contextmenu="handleRightTreeClick"
          node-key="id"
        />
        <div
          class="resize-handle"
          @mousedown="handleTreeResize"
        ></div>
      </div>

      <!-- 3D Viewer -->
      <el-main class="viewer">
        <PathSearchViewer 
          @leftClick-select-object="handleLeftClickMeshSelect"
          @rightClick-select-object="handleRightClickMeshSelect"
        />
      </el-main>

      <DefineTask
        ref="defineTaskRef"
        v-model="defineTaskVisible"
        :currentTagDict="currentTagDict"
        @generate="handleGenerateTask"
      />

      <DefineTaskObstacle
        ref="defineTaskObstacleRef"
        v-model="defineTaskObstacleVisible"
        :currentTagDict="currentTagDict"
        @generate="handleGenerateObstacle"
      />

      <DefinePipeGroup
        ref="definePipeGroupRef"
        v-model="definePipeGroupVisible"
        :currentTagDict="currentTagDict"
        @generate="handleGeneratePipeGroup"
      />

      <DefineTaskPipe
        ref="defineTaskPipeRef"
        v-model="defineTaskPipeVisible"
        :currentTagDict="currentTagDict"
        @generate="handleGeneratePipe"
      />

    </div>

    <!-- Context Menu -->
    <div 
      v-show="taskManagerContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('TaskManager/RemoveTaskManager')">Remove Task</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="taskEnvironmentContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('TaskEnvironment/EditEnvironment')">Edit Environment</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('TaskEnvironment/HideEnvironment')">Hide Environment</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('TaskEnvironment/ShowEnvironment')">Show Environment</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="taskObstacleContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('TaskObstacle/AddObstacle')">Add Obstacle</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="obstacleContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/EditObstacle')">Edit Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/HideObstacle')">Hide Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/ShowObstacle')">Show Obstacle</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Obstacle/RemoveObstacle')">Remove Obstacle</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="taskPipeContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('TaskPipeGroup/AddPipeGroup')">Add Pipe Group</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="pipeGroupContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
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
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('Pipe/EditPipe')">Edit Pipe</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Pipe/ShowPipe')">Show Pipe</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Pipe/HidePipe')">Hide Pipe</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('Pipe/RemovePipe')">Remove Pipe</el-menu-item>
      </el-menu>
    </div>

    <!-- Mesh Context Menu -->
    <div 
      v-show="meshContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleMeshContextMenuClick('Hide')">Hide</el-menu-item>
        <el-menu-item @click="handleMeshContextMenuClick('Edit')">Edit</el-menu-item>
      </el-menu>
    </div>

    <!-- Hidden file input -->
    <input
      type="file"
      ref="fileInput"
      style="display: none"
      accept=".h5,.hdf5"
      @change="handleFileSelected"
    />

  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onBeforeUnmount, provide } from 'vue'
import PathSearchViewer from '@/components/pathsearch_component/PathSearchViewer.vue'
import { ElMessage } from 'element-plus'
import * as THREE from 'three'

import DefineTask from '@/components/pathsearch_component/DefineTask.vue'
import DefineTaskObstacle from '@/components/pathsearch_component/DefineObstacle.vue'
import DefineTaskPipe from '@/components/pathsearch_component/DefinePipe.vue'
import DefinePipeGroup from '@/components/pathsearch_component/DefinePipeGroup.vue'

import { BlockRoot } from '../classes/pathSearch_classes/base_root.ts'
import { TaskManager } from '../classes/pathSearch_classes/base_task.ts'
import type { PathSearchTreeNode } from '../types/data_type.ts'

// ----------------
const scene = new THREE.Scene()
const blockRoot = new BlockRoot(scene)
provide('blockRoot', blockRoot)

function setVisableDict(active_key: string, visable_dict: any, single_mode: boolean = false, set_value: boolean = true) {
  if (single_mode) {
    Object.keys(visable_dict).forEach(key => {
      if (key !== active_key) {
        visable_dict[key as keyof typeof visable_dict].value = false
      }
    })
  }
  visable_dict[active_key as keyof typeof visable_dict].value = set_value
}

const meshContextMenuVisible = ref(false)
const taskObstacleContextMenuVisible = ref(false)
const taskPipeContextMenuVisible = ref(false)
const taskManagerContextMenuVisible = ref(false)
const obstacleContextMenuVisible = ref(false)
const pipeContextMenuVisible = ref(false)
const pipeGroupContextMenuVisible = ref(false)
const taskEnvironmentContextMenuVisible = ref(false)

const contextMenuVisibleDict = {
  'taskObstacleContextMenuVisible': taskObstacleContextMenuVisible,
  'taskPipeContextMenuVisible': taskPipeContextMenuVisible,
  'meshContextMenuVisible': meshContextMenuVisible,
  'taskManagerContextMenuVisible': taskManagerContextMenuVisible,
  'obstacleContextMenuVisible': obstacleContextMenuVisible,
  'pipeContextMenuVisible': pipeContextMenuVisible,
  'pipeGroupContextMenuVisible': pipeGroupContextMenuVisible,
  'taskEnvironmentContextMenuVisible': taskEnvironmentContextMenuVisible,
}

const defineTaskObstacleVisible = ref(false)
const defineTaskPipeVisible = ref(false)
const defineTaskVisible = ref(false)
const definePipeGroupVisible = ref(false)

const rightSidebarVisibleDict = {
  'defineTaskObstacleVisible': defineTaskObstacleVisible,
  'defineTaskPipeVisible': defineTaskPipeVisible,
  'defineTaskVisible': defineTaskVisible,
  'definePipeGroupVisible': definePipeGroupVisible,
}

// ---------------- Left Sidebar Click Event Handling ----------------
const treeData = ref<PathSearchTreeNode[]>([])

const updateTreeData = () => {
  treeData.value = []
  blockRoot.get_task_manager_names().forEach(task_manager_name => {
    const task_manager = blockRoot.get_task_manager(task_manager_name)
    if (task_manager) {
      treeData.value.push(task_manager.export_tree())
    }
  })
}

const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

const handleLeftTreeClick = (data: any) => {
  console.log('Left tree click:', data)
}

const currentTagDict = ref(new Map<string, string| string[]>())
const handleRightTreeClick = (event: MouseEvent, data: any) => {
  event.preventDefault()
  contextMenuPosition.x = event.clientX
  contextMenuPosition.y = event.clientY
  setVisableDict('meshContextMenuVisible', contextMenuVisibleDict, true, false)

  currentTagDict.value = data.tagDict
  currentTagDict.value.set('label', data.label)

  if (data.type === 'TaskManager') {
    setVisableDict('taskManagerContextMenuVisible', contextMenuVisibleDict, true, true)
  }
  else if (data.type === 'TaskEnvironment') {
    setVisableDict('taskEnvironmentContextMenuVisible', contextMenuVisibleDict, true, true)
  }
  else if (data.type === 'TaskObstacle') {
    setVisableDict('taskObstacleContextMenuVisible', contextMenuVisibleDict, true, true)
  }
  else if (data.type === 'Obstacle') {
    setVisableDict('obstacleContextMenuVisible', contextMenuVisibleDict, true, true)
  }
  else if (data.type === 'TaskPipeGroup') {
    setVisableDict('taskPipeContextMenuVisible', contextMenuVisibleDict, true, true)
  }
  else if (data.type === 'PipeGroup') {
    setVisableDict('pipeGroupContextMenuVisible', contextMenuVisibleDict, true, true)
  }
  else if (data.type === 'Pipe') {
    setVisableDict('pipeContextMenuVisible', contextMenuVisibleDict, true, true)
  }
}

const sidebarWidth = ref(250)
const isResizing = ref(false)
const startX = ref(0)
const startWidth = ref(0)

const handleTreeResize = (e: MouseEvent) => {
  isResizing.value = true
  startX.value = e.clientX
  startWidth.value = sidebarWidth.value
  document.addEventListener('mousemove', startResize)
  document.addEventListener('mouseup', stopResize)
}

const startResize = (e: MouseEvent) => {
  if (!isResizing.value) return
  const diff = e.clientX - startX.value
  sidebarWidth.value = Math.max(200, Math.min(500, startWidth.value + diff))
}

const stopResize = () => {
  isResizing.value = false
  document.removeEventListener('mousemove', startResize)
  document.removeEventListener('mouseup', stopResize)
}

// ---------------- Mesh Selection Handling ----------------
const handleLeftClickMeshSelect = (objectName: string) => {
  setVisableDict('meshContextMenuVisible', contextMenuVisibleDict, true, false)
  console.log('Left click mesh select:', objectName)
}

const handleRightClickMeshSelect = (objectName: string, clickEventX: number, clickEventY: number) => {
  contextMenuPosition.x = clickEventX
  contextMenuPosition.y = clickEventY
  setVisableDict('meshContextMenuVisible', contextMenuVisibleDict, true, true)
}

const handleMeshContextMenuClick = (mode: string) => {
  if (mode === 'Hide/Vis') {
    // Handle hide/show
  } else if (mode === 'Edit') {
    // Handle edit
  }
}

// ---------------- File Open handling ----------------  
const fileInput = ref<HTMLInputElement | null>(null)

const triggerFileInput = () => {
  if (fileInput.value) {
    fileInput.value.click()
  }
}

const handleFileSelected = async (event: Event) => {
    // TODO need adjust
//   const input = event.target as HTMLInputElement
//   const file = input.files?.[0]
//   if (!file) {
//     return
//   }

//   const formData = new FormData()
//   formData.append('file', file)

//   try {
//     const response = await fetch('http://192.168.110.54:9998/cad/read_h5', {
//       method: 'POST',
//       body: formData
//     })
//     if (!response.ok) throw new Error('Failed to read file')
//     const data = await response.json()
//     // Process the data
//     ElMessage.success(`Successfully loaded file: ${file.name}`)

//   } catch (error: unknown) {
//     console.error('Error processing file:', error)
//     if (error instanceof Error) {
//       ElMessage.error('Failed to read file: ' + error.message)
//     } else {
//       ElMessage.error('Failed to read file: Unknown error')
//     }
//   } finally {
//     input.value = ''
//   }
}

// ---------------- Context Menu Handling ----------------
const handleTreeContextMenu = (mode: string) => {
  if (mode === 'TaskManager/RemoveTaskManager') {
    setVisableDict('taskManagerContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.remove_task_manager(currentTagDict.value.get('Task') as string)
  }
  else if (mode === 'TaskEnvironment/EditEnvironment') {
    setVisableDict('taskEnvironmentContextMenuVisible', contextMenuVisibleDict, true, false)
  }
  else if (mode === 'TaskObstacle/AddObstacle') {
    setVisableDict('taskObstacleContextMenuVisible', contextMenuVisibleDict, true, false)
    currentTagDict.value.set(
        'RelaxGroup', blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)!.get_group_names()
    )
    setVisableDict('defineTaskObstacleVisible', rightSidebarVisibleDict, true, true)
  }
  else if (mode === 'Obstacle/EditObstacle') {
    setVisableDict('obstacleContextMenuVisible', contextMenuVisibleDict, true, false)
  }
  else if (mode === 'Obstacle/HideObstacle') {
    setVisableDict('obstacleContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_obstacle(
        currentTagDict.value.get('Obstacle') as string
    )?.hide_mesh()
  }
  else if (mode === 'Obstacle/ShowObstacle') {
    setVisableDict('obstacleContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_obstacle(
        currentTagDict.value.get('Obstacle') as string
    )?.show_mesh()
  }
  else if (mode === 'Obstacle/RemoveObstacle') {
    setVisableDict('obstacleContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.remove_obstacle(
        currentTagDict.value.get('Obstacle') as string, blockRoot
    )
    updateTreeData()
  }
  else if (mode === 'TaskPipeGroup/AddPipeGroup') {
    setVisableDict('taskPipeContextMenuVisible', contextMenuVisibleDict, true, false)
    setVisableDict('definePipeGroupVisible', rightSidebarVisibleDict, true, true)
  }
  else if (mode === 'PipeGroup/AddPipe') {
    setVisableDict('pipeGroupContextMenuVisible', contextMenuVisibleDict, true, false)
    setVisableDict('defineTaskPipeVisible', rightSidebarVisibleDict, true, true)
  }
  else if (mode === 'PipeGroup/ShowPipeGroup') {
    setVisableDict('pipeGroupContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_pipe_group(
        currentTagDict.value.get('PipeGroup') as string
    )?.forEach(pipe => {
      blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_pipe(pipe)?.show_mesh()
    })
  }
  else if (mode === 'PipeGroup/HidePipeGroup') {
    setVisableDict('pipeGroupContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_pipe_group(
        currentTagDict.value.get('PipeGroup') as string
    )?.forEach(pipe => {
      blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_pipe(pipe)?.hide_mesh()
    })
  }
  else if (mode === 'PipeGroup/RemovePipeGroup') {
    setVisableDict('pipeGroupContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.remove_pipe_group(
        currentTagDict.value.get('PipeGroup') as string, blockRoot
    )
    updateTreeData()
  }
  else if (mode === 'Pipe/EditPipe') {
    setVisableDict('pipeContextMenuVisible', contextMenuVisibleDict, true, false)
  }
  else if (mode === 'Pipe/HidePipe') {
    setVisableDict('pipeContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_pipe(
        currentTagDict.value.get('Pipe') as string
    )?.hide_mesh()
  }
  else if (mode === 'Pipe/ShowPipe') {
    setVisableDict('pipeContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.get_pipe(
        currentTagDict.value.get('Pipe') as string
    )?.show_mesh()
  }
  else if (mode === 'Pipe/RemovePipe') {
    setVisableDict('pipeContextMenuVisible', contextMenuVisibleDict, true, false)
    blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)?.remove_pipe(
        currentTagDict.value.get('Pipe') as string, blockRoot
    )
    updateTreeData()
  }
}

const handleToolMenuClick = (mode: string) => {
  if (mode === 'AddTask') {
    setVisableDict('defineTaskVisible', rightSidebarVisibleDict, true, true)
  }
}

const handleGenerateTask = (data: any) => {
  const task_manager = new TaskManager(data.name, data.xmin, data.xmax, data.ymin, data.ymax, data.zmin, data.zmax, blockRoot)
  const [success, message] = blockRoot.add_task_manager(task_manager)
  if (!success) {
    ElMessage.error(message)
  }

  updateTreeData()
}

const handleGeneratePipeGroup = (data: any) => {
  const task_manager = blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)
  if (!task_manager) {
    ElMessage.error('Task not found')
    return
  }

  const [success, message] = task_manager.add_pipe_group(data.name)
  if (!success) {
    ElMessage.error(message)
  }

  updateTreeData()
}

const handleGenerateObstacle = (data: any) => {
  const task_manager = blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)
  if (!task_manager) {
    ElMessage.error('Task not found')
    return
  }

  const [success, message] = task_manager.add_obstacle(data.name, data, blockRoot)
  if (!success) {
    ElMessage.error(message)
  }

  updateTreeData()
}

const handleGeneratePipe = (data: any) => {
  const task_manager = blockRoot.get_task_manager(currentTagDict.value.get('Task') as string)
  if (!task_manager) {
    ElMessage.error('Task not found')
    return
  }
  
  const [success, message] = task_manager.add_pipe(
    data.name, data.x, data.y, data.z, 
    data.vec_x, data.vec_y, data.vec_z, data.radius, 
    data.group_name, data.is_inlet, 
    blockRoot
  )
  if (!success) {
    ElMessage.error(message)
  }

  updateTreeData()
}

// ---------------- Click Event handling ----------------
onMounted(() => {
  document.addEventListener('click', handleClickEvent)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickEvent)
})

const handleClickEvent = (event: MouseEvent) => {
  const target = event.target as HTMLElement
  if (!target.closest('.context-menu')) {
    setVisableDict('meshContextMenuVisible', contextMenuVisibleDict, true, false)
  }
}

</script>

<style scoped>
.path-search {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: absolute;
  left: 0;
  top: 0;
  right: 0;
  bottom: 0;
}

.menu-bar {
  flex: 0 0 auto;
  width: 100%;
}

.main-content {
  flex: 1;
  display: flex;
  min-height: 0;
  overflow: hidden;
  width: 100%;
}

.sidebar {
  position: relative;
  min-width: 200px;
  max-width: 500px;
  background: #fff;
  border-right: 1px solid #dcdfe6;
  overflow: auto;
  flex-shrink: 0;
  height: 100%;
}

.resize-handle {
  position: absolute;
  top: 0;
  right: -5px;
  width: 10px;
  height: 100%;
  cursor: col-resize;
  background: transparent;
}

.resize-handle:hover {
  background: rgba(0, 0, 0, 0.1);
}

.viewer {
  flex: 1;
  min-width: 1000px;
  padding: 0;
  overflow: hidden;
}

.context-menu {
  position: fixed;
  z-index: 3000;
  background: white;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,.1);
}

.context-menu :deep(.el-menu) {
  border: none;
}

.context-menu :deep(.el-menu-item) {
  height: 36px;
  line-height: 36px;
}

/* Tree node selection styles */
:deep(.el-tree-node.is-current > .el-tree-node__content) {
  background-color: #409eff !important;
  color: white;
}

:deep(.el-tree-node.is-current > .el-tree-node__content:hover) {
  background-color: #66b1ff !important;
}

:deep(.el-tree-node__content) {
  height: 32px;
  border-radius: 4px;
}

:deep(.el-tree-node__content:hover) {
  background-color: #f5f7fa;
}
</style> 