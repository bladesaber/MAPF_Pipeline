<!-- SimulateOptimize.vue -->
<template>
  <div class="simulate-optimize">
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
        <el-menu-item index="3-1">Simulation</el-menu-item>
        <el-menu-item index="3-2">Optimization</el-menu-item>
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
        <SimulateViewer 
          @leftClick-select-object="handleLeftClickMeshSelect"
          @rightClick-select-object="handleRightClickMeshSelect"
        />
      </el-main>

      <DefineSimulateGroup
        ref="defineSimulateGroupRef"
        v-model="defineSimulateGroupVisible"
        @generate="handleGenerateGroup"
      />

      <DefineSimulateCondition
        ref="defineSimulateConditionRef"
        v-model="defineSimulateConditionVisible"
        @generate="handleGenerateCondition"
      />

      <DefineSimulateBoundary
        ref="defineSimulateBoundaryRef"
        v-model="defineSimulateBoundaryVisible"
        @generate="handleGenerateBoundary"
      />

      <GenerateSimulateMesh
        v-model="generateSimulateMeshVisible"
        @generate="handleGenerateMesh"
      />

    </div>

    <!-- Tree Node Context Menu -->
    <div 
      v-show="nodeGroupsContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('nodeGroups/AddGroup')">Add Group</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('nodeGroups/AddMesh')">Add Mesh</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="groupContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('group/EditGroup')">Edit Group</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('group/DeleteGroup')">Delete Group</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="nodeConditionsContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('nodeConditions/AddCondition')">Add Condition</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="conditionContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('condition/EditCondition')">Edit Condition</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('condition/DeleteCondition')">Delete Condition</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="boundaryContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('boundary/EditBoundary')">Edit Boundary</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('boundary/DeleteBoundary')">Delete Boundary</el-menu-item>
      </el-menu>
    </div>

    <!-- Mesh Context Menu -->
    <div 
      v-show="meshContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item>Hide/Vis</el-menu-item>
        <el-menu-item>Edit</el-menu-item>
      </el-menu>
    </div>

    <!-- Hidden file input -->
    <input
      type="file"
      ref="fileInput"
      style="display: none"
      accept=".step,.STEP"
      @change="handleFileSelected"
    />

    <!-- Progress Bar -->
    <el-dialog
      v-model="progressVisible"
      title="Processing File"
      :close-on-click-modal="false"
      :close-on-press-escape="false"
      :show-close="false"
      width="30%"
    >
      <el-progress
        :percentage="progressPercentage"
        :status="progressStatus"
        :stroke-width="15"
        :text-format="progressFormat"
      />
    </el-dialog>

  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onBeforeUnmount, provide } from 'vue'
import SimulateViewer from '../components/simulateOpt_component/SimulateViewer.vue'
import { ElMessage } from 'element-plus'
import type { Scene, Object3D } from 'three'
import * as THREE from 'three'
import { pass } from 'three/tsl'
import { SimulateRootObj, SimulateEntity, CadEntity } from '../classes/simulateOpt_classes/simulate_entity.ts'
import type { SimulateTreeNode } from '@/types/data_type.ts'
import DefineSimulateGroup from '../components/simulateOpt_component/DefineSimulateGroup.vue'
import DefineSimulateCondition from '@/components/simulateOpt_component/DefineSimulateCondition.vue'
import DefineSimulateBoundary from '@/components/simulateOpt_component/DefineSimulateBoundary.vue'
import GenerateSimulateMesh from '@/components/simulateOpt_component/GenerateSimulateMesh.vue'
import { da } from 'element-plus/es/locales.mjs'

// ----------------
const scene = new THREE.Scene()
const simulateRootObj = new SimulateRootObj(scene)
provide('simulateRootObj', simulateRootObj)

// ---------------- Left Sidebar Click Event Handling ----------------
const treeData = ref<any[]>([])

const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

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

const updateTreeData = () => {
  const tree: SimulateTreeNode[] = []
  simulateRootObj.model_map.forEach((entity, entityName) => {
    tree.push(entity.export_tree_data())
  })
  treeData.value = tree
}

// ---------------- Tree Context Menu Visable ----------------
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

const nodeGroupsContextMenuVisible = ref(false)
const groupContextMenuVisible = ref(false)
const nodeConditionsContextMenuVisible = ref(false)
const conditionContextMenuVisible = ref(false)
const boundaryContextMenuVisible = ref(false)
const meshContextMenuVisible = ref(false)

const tree_visable_dict = {
  'nodeGroupsContextMenuVisible': nodeGroupsContextMenuVisible,
  'groupContextMenuVisible': groupContextMenuVisible,
  'nodeConditionsContextMenuVisible': nodeConditionsContextMenuVisible,
  'conditionContextMenuVisible': conditionContextMenuVisible,
  'boundaryContextMenuVisible': boundaryContextMenuVisible,
  'meshContextMenuVisible': meshContextMenuVisible
}

const handleLeftTreeClick = (data: any) => {
  setVisableDict('nodeGroupsContextMenuVisible', tree_visable_dict, true, false)
  console.log('Left tree click:', data)
}

const handleRightTreeClick = (event: MouseEvent, data: any) => {
  event.preventDefault()
  contextMenuPosition.x = event.clientX
  contextMenuPosition.y = event.clientY

  if (data.type === 'Node_Groups') {
    setVisableDict('nodeGroupsContextMenuVisible', tree_visable_dict, true, true)
  } else if (data.type === 'Group') {
    setVisableDict('groupContextMenuVisible', tree_visable_dict, true, true)
  } else if (data.type === 'Node_Conditions') {
    setVisableDict('nodeConditionsContextMenuVisible', tree_visable_dict, true, true)
  } else if (['Condition', 'PhysicalCfg', 'MaterialCfg', 'RunCfg', 'InitializationCfg'].includes(data.type)) {
    setVisableDict('conditionContextMenuVisible', tree_visable_dict, true, true)
  } else if (['InletCfg', 'OutletCfg', 'WallCfg'].includes(data.type)) {
    setVisableDict('boundaryContextMenuVisible', tree_visable_dict, true, true)
  }
  console.log('Right tree click:', data.type)
}

const handleTreeContextMenu = (mode: string) => {
  setVisableDict('nodeGroupsContextMenuVisible', tree_visable_dict, true, false)
  if (mode == 'nodeGroups/AddGroup') {
    setVisableDict('defineSimulateGroupVisible', rightSideBar_visableDict, true)
  } else if (mode == 'nodeConditions/AddCondition'){
    setVisableDict('defineSimulateConditionVisible', rightSideBar_visableDict, true)
  } else if (mode == 'boundary/AddBoundary'){
    setVisableDict('defineSimulateBoundaryVisible', rightSideBar_visableDict, true)
  } else if (mode == 'nodeGroups/AddMesh'){
    setVisableDict('generateSimulateMeshVisible', rightSideBar_visableDict, true)
  }
}

// ---------------- Mesh Click Event Handling ----------------
const defineSimulateGroupRef = ref()
const handleLeftClickMeshSelect = (name: string | undefined) => {
  if (defineSimulateGroupVisible.value) {
    defineSimulateGroupRef.value?.addItem({name: name})
  }
}

const handleRightClickMeshSelect = (name: string | undefined, x: number, y: number) => {
  if (name) {
    contextMenuPosition.x = x
    contextMenuPosition.y = y
    setVisableDict('meshContextMenuVisible', tree_visable_dict, true, true)
  }
}

// ---------------- File Open handling ----------------  
const fileInput = ref<HTMLInputElement | null>(null)

const triggerFileInput = () => {
  if (fileInput.value) {
    fileInput.value.click()
  }
}

// Progress bar state
const progressVisible = ref(false)
const progressPercentage = ref(0)
const progressStatus = ref<'success' | 'exception' | ''>('')

const progressFormat = (percentage: number) => {
  return percentage === 100 ? 'Processing Complete' : `${percentage}%`
}

const handleFileSelected = async (event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) {
    return
  }

  // Store file content for later use
  const fileContent = await file.arrayBuffer()
  
  // Reset progress state
  progressVisible.value = true
  progressPercentage.value = 0
  progressStatus.value = ''

  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await fetch('http://192.168.110.54:9998/cad/extract_step_face', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) throw new Error('Failed to read file')

    const reader = response.body?.getReader()
    if (!reader) throw new Error('Failed to get response reader')

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = new TextDecoder().decode(value)
      console.log('handleFileSelected:', text)
      const data = JSON.parse(text)
      
      if (!data.is_finished) {
        progressPercentage.value = data.progress
      } else {
        progressPercentage.value = data.progress
        progressStatus.value = 'success'
        handleSTEPData(file.name, data.shape_jsons, new Uint8Array(fileContent))
        break
      }
    }

  } catch (error: unknown) {
    console.error('Error processing file:', error)
    progressStatus.value = 'exception'
    progressPercentage.value = 0
  } finally {
    input.value = ''
    // Close progress dialog after a short delay
    setTimeout(() => {
      progressVisible.value = false
      progressPercentage.value = 0
      progressStatus.value = ''
    }, 2000)
  }
}

const handleSTEPData = (fileName: string, data: any, file_content: Uint8Array) => {
  let node_name = fileName.split('.')[0]
  const simulate_entity = new SimulateEntity(node_name)

  // Store the file content in localStorage for later use
  sessionStorage.setItem(simulate_entity.name, JSON.stringify({
    content: Array.from(file_content)
  }))

  Object.entries(data).forEach(([face_name, face_data]) => {
    const cad_entity = new CadEntity(face_name, face_data, simulateRootObj)
    simulate_entity.add_entity(cad_entity)
  })
  simulateRootObj.add_simulate_entity(simulate_entity)
  updateTreeData()
}

// ---------------- Visable Dictionary ----------------
const defineSimulateGroupVisible = ref(false)
const defineSimulateConditionVisible = ref(false)
const defineSimulateBoundaryVisible = ref(false)
const generateSimulateMeshVisible = ref(false)
const rightSideBar_visableDict = {
  'defineSimulateGroupVisible': defineSimulateGroupVisible,
  'defineSimulateConditionVisible': defineSimulateConditionVisible,
  'defineSimulateBoundaryVisible': defineSimulateBoundaryVisible,
  'generateSimulateMeshVisible': generateSimulateMeshVisible
}

// ---------------- Define Simulate Group ----------------
const handleGenerateGroup = () => {
  pass
}

const handleGenerateCondition = () => {
  pass
}

const handleGenerateBoundary = () => {
  pass
}

const handleGenerateMesh = () => {
  pass
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
    setVisableDict('meshContextMenuVisible', tree_visable_dict, true, false)
  }
}
</script>

<style scoped>
.simulate-optimize {
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
  min-width: 600px;
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