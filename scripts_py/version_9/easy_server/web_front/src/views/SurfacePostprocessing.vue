<!-- SurfacePostprocessing.vue -->
<template>
  <div class="surface-postprocessing">
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
        <el-sub-menu index="3-1">
          <template #title>PointTool</template>
          <el-menu-item index="3-1-1" @click="handlePointToolClick('3-1-1')">Generate Point</el-menu-item>
          <el-menu-item index="3-1-2" @click="handlePointToolClick('3-1-2')">Generate Vector Point</el-menu-item>
          <el-menu-item index="3-1-3" @click="handlePointToolClick('3-1-3')">Generate Point From Skeleton</el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="3-2">
          <template #title>LineTool</template>
          <el-menu-item index="3-2-1" @click="handleLineToolClick('3-2-1')">Generate Line</el-menu-item>
          <el-menu-item index="3-2-2" @click="handleLineToolClick('3-2-2')">Generate Circle</el-menu-item>
          <el-menu-item index="3-2-3" @click="handleLineToolClick('3-2-3')">Generate Arc</el-menu-item>
          <el-menu-item index="3-2-4" @click="handleLineToolClick('3-2-4')">Generate Bspline</el-menu-item>
          <el-menu-item index="3-2-5" @click="handleLineToolClick('3-2-5')">Generate Skeleton</el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="3-3">
          <template #title>SurfaceTool</template>
          <el-menu-item index="3-3-1" @click="handleSurfaceToolClick('3-3-1')">Generate PipeShell</el-menu-item>
          <el-menu-item index="3-3-2" @click="handleSurfaceToolClick('3-3-2')">Generate Bspline Surface</el-menu-item>
          <el-menu-item index="3-3-3" @click="handleSurfaceToolClick('3-3-3')">Generate Disk</el-menu-item>
        </el-sub-menu>
        <el-menu-item index="3-4">Visual Group</el-menu-item>
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
        <ThreeViewer 
          @leftClick-select-object="handleLeftClickMeshSelect"
          @rightClick-select-object="handleRightClickMeshSelect"
        />
      </el-main>

      <PointTool
        ref="pointToolRef"
        v-model="pointToolVisible"
        :point-mode="pointMode"
        @generate="handlePointTool"
      />

      <LineTool
        ref="lineToolRef"
        v-model="lineToolVisible"
        :line-mode="lineMode"
        @generate="handleLineTool"
      />

      <MeshContextTool
        v-model="meshContextToolVisible"
        :mode="meshContextToolMode"
      />

    </div>

    <!-- Context Menu -->
    <div 
      v-show="contextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item>Edit</el-menu-item>
      </el-menu>
    </div>

    <!-- Mesh Context Menu -->
    <div 
      v-show="meshContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleMeshContextMenuClick('Hide/Vis')">Hide/Vis</el-menu-item>
        <el-menu-item @click="handleMeshContextMenuClick('Move')">Move</el-menu-item>
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
import ThreeViewer from '../components/ThreeViewer.vue'
import PointTool from '../components/PointTool.vue'
import LineTool from '../components/LineTool.vue'
import MeshContextTool from '../components/MeshContextTool.vue'
import { RootObj } from '../classes/store_obj.ts'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { Scene, Object3D } from 'three'
import * as THREE from 'three'
import type { PathResultData } from '../types/data_type.ts'
import { PointTypes, LineTypes, SurfaceTypes } from '../types/data_type.ts'
import type { TreeNode } from '../types/data_type.ts'

import { PointObj } from '../classes/point_obj.ts'
import { BsplineObj } from '../classes/line_obj.ts'
import { pass } from 'three/tsl'
import { 
  generate_point, generate_vector_point, 
  generate_line_from_points, generate_circle, generate_arc, generate_bspline_from_points, 
  generate_skeleton, generate_surface_points_from_skeletons,
  generate_disk 
} from '../utils/operation_utils.ts'

// ----------------
const scene = new THREE.Scene();
const rootObj = new RootObj(scene);
provide('storeObj', rootObj);
provide('main_message', "Main Window Running...");

// ---------------- Left Sidebar Click Event Handling ----------------
const treeData = ref<TreeNode[]>([])

const contextMenuVisible = ref(false)
const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

const handleLeftTreeClick = (data: any) => {
  console.log('Left tree click:', data)
}

const handleRightTreeClick = (event: MouseEvent, data: any) => {
  event.preventDefault()
  contextMenuPosition.x = event.clientX
  contextMenuPosition.y = event.clientY
  contextMenuVisible.value = true
  meshContextMenuVisible.value = false
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

const updateTreeData = () => {
  const pointsNode: TreeNode = {
    label: 'points',
    children: []
  }

  const linesNode: TreeNode = {
    label: 'lines',
    children: []
  }

  const surfaceNode: TreeNode = {
    label: 'surface',
    children: []
  }

  rootObj.get_obj_list().forEach(obj => {
    if (PointTypes.includes(obj.style as (typeof PointTypes)[number])) {
        const node: TreeNode = {
          label: obj.name,
          ...obj
        }
        pointsNode.children?.push(node) 
      }
    else if (LineTypes.includes(obj.style as (typeof LineTypes)[number])) {
      const node: TreeNode = {
        label: obj.name,
        ...obj
      }
      linesNode.children?.push(node)
    }
    else if (SurfaceTypes.includes(obj.style as (typeof SurfaceTypes)[number])) {
      const node: TreeNode = {
        label: obj.name,
        ...obj
      }
      surfaceNode.children?.push(node)
    }
    else {
      throw new Error(`Unknown object style: ${obj.style}`)
    }
  })

  const tree: TreeNode[] = []
  tree.push(pointsNode)
  tree.push(linesNode)
  tree.push(surfaceNode)
  treeData.value = tree
}

// ---------------- Mesh Click Event Handling ----------------
function findNode(nodes: TreeNode[], objectName: string): TreeNode | null {
  for (const node of nodes) {
    if (node.label === objectName) {
      return node
    }
    if (node.children) {
      const found = findNode(node.children, objectName)
      if (found) return found
    }
  }
  return null
}

const pointToolRef = ref()
const lineToolRef = ref()
const handleLeftClickMeshSelect = (objectName: string) => {
  if (pointToolVisible.value) {
    const selectedNode = findNode(treeData.value, objectName)
    if (selectedNode) {
      pointToolRef.value?.addItem({name: selectedNode.label, style: selectedNode.style})
    }
  }
  else if (lineToolVisible.value) {
    // Only item from tree data can be selected
    const selectedNode = findNode(treeData.value, objectName)
    if (selectedNode) {
      lineToolRef.value?.addItem({name: selectedNode.label, style: selectedNode.style})
    }
  }
  // console.log('Left click detected:', objectName)
}

const meshContextMenuVisible = ref(false)
const handleRightClickMeshSelect = (objectName: string, clickEventX: number, clickEventY: number) => {
  contextMenuPosition.x = clickEventX
  contextMenuPosition.y = clickEventY
  meshContextMenuVisible.value = true
  contextMenuVisible.value = false
  console.log('Right click detected:', objectName)
}

// ---------------- File Open handling ----------------  
const fileInput = ref<HTMLInputElement | null>(null)

const triggerFileInput = () => {
  if (fileInput.value) {
    fileInput.value.click()
  }
}

const handleFileSelected = async (event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) {
    return
  }

  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await fetch('http://192.168.110.54:9998/cad/read_h5', {
      method: 'POST',
      body: formData
    })
    if (!response.ok) throw new Error('Failed to read file')
    const data = await response.json() as PathResultData
    processPathResultData(data, file.name)
    ElMessage.success(`Successfully loaded file: ${file.name}`)

  } catch (error: unknown) {
    console.error('Error processing file:', error)
    if (error instanceof Error) {
      ElMessage.error('Failed to read file: ' + error.message)
    } else {
      ElMessage.error('Failed to read file: Unknown error')
    }
  } finally {
    input.value = ''
  }
}

const processPathResultData = (data: PathResultData, fileName: string) => {
  let node_name = fileName.split('.')[0]
  Object.entries(data).forEach(([name, data]) => {
    const group_name = `${node_name}/${name}`

    const point_list: PointObj[] = []
    data.coefficients.forEach((coef: number[], i: number) => {
      const point = new PointObj(`${group_name}/${i}`, coef[0], coef[1], coef[2], 0.2, rootObj.get_scene())
      rootObj.add_obj(point)
      point_list.push(point)
    })
    
    Object.entries(data.segments).forEach(([segment_name, segment]) => {
      const control_point_list = segment.coefficient_idxs.map((idx: number) => point_list[idx])
      const path = new BsplineObj(
        `${group_name}/${segment_name}`, 
        control_point_list, 
        segment.knots, 
        segment.degree[0],
        rootObj.get_scene()
      )
      rootObj.add_obj(path)
    })

  })
  
  updateTreeData()
}

// ---------------- Visable Dictionary ----------------
const pointToolVisible = ref(false)
const lineToolVisible = ref(false)
const surfaceToolVisible = ref(false)
const meshContextToolVisible = ref(false)

const visable_dict = {
  'pointToolVisible': pointToolVisible,
  'lineToolVisible': lineToolVisible,
  'surfaceToolVisible': surfaceToolVisible,
  'meshContextToolVisible': meshContextToolVisible
}
function setVisableDict(active_key: string, single_mode: boolean = false) {
  if (single_mode) {
    Object.keys(visable_dict).forEach(key => {
      if (key !== active_key) {
        visable_dict[key as keyof typeof visable_dict].value = false
      }
    })
  }
  visable_dict[active_key as keyof typeof visable_dict].value = true
}

// ---------------- Point Tool ----------------
const handlePointTool = (data: any) => {
  if (data.name === '') {
    ElMessage.error('Name is required')
    return
  }
  let result = {success: false}
  if (data.style === 'vector_point') {
    result = generate_vector_point(
      data.name, data.x, data.y, data.z, data.vectorX, data.vectorY, data.vectorZ, rootObj
    )
  }
  else if (data.style === 'point') {
    result = generate_point(data.name, data.x, data.y, data.z, rootObj)
  }
  else if (data.style === 'PointFromSkeleton') {
    result = generate_surface_points_from_skeletons(
      data.name, 
      data.selectedItems.map((item: any) => rootObj.get_obj(item.name)), 
      data.selectedItems.map((item: any) => item.sample_num), 
      rootObj
    )
  }
  else {
    ElMessage.error('Invalid style')
    return
  }
  if (result.success) {
    updateTreeData()
  }
}

const pointMode = ref("Point")
const handlePointToolClick = (index: string) => {
  if (index === '3-1-1') {
    pointMode.value = "Point"
  }
  else if (index === '3-1-2') {
    pointMode.value = "VectorPoint"
  }
  else if (index === '3-1-3') {
    pointMode.value = "PointFromSkeleton"
  }
  setVisableDict('pointToolVisible', true)
}

// ---------------- Line Tool ----------------
const lineMode = ref<'Line' | 'Circle' | 'Arc' | 'Bspline' | 'Skeleton'>('Line')
const handleLineToolClick = (index: string) => {
  if (index === '3-2-1') {  
    lineMode.value = "Line"
  }
  else if (index === '3-2-2') {
    lineMode.value = "Circle"
  }
  else if (index === '3-2-3') {
    lineMode.value = "Arc"
  }
  else if (index === '3-2-4') {
    lineMode.value = "Bspline"
  }
  else if (index === '3-2-5') {
    lineMode.value = "Skeleton"
  }
  setVisableDict('lineToolVisible', true)
}

const handleLineTool = (data: any) => {
  if (data.name === '') {
    ElMessage.error('Name is required')
    return
  }
  let result = {success: false}
  if (data.style === 'Line') {
    result = generate_line_from_points(
      data.name, 
      data.selectedItems.map((item: any) => rootObj.get_obj(item.name)), 
      rootObj
    )
  }
  else if (data.style === 'Circle') {
    result = generate_circle(data.name, data.x, data.y, data.z, data.radius, data.vectorX, data.vectorY, data.vectorZ, rootObj)
  }
  else if (data.style === 'Arc') {
    result = generate_arc(data.name, data.x, data.y, data.z, data.radius, data.vectorX, data.vectorY, data.vectorZ, data.startAngle, data.endAngle, rootObj)
  }
  else if (data.style === 'Bspline') {
    result = generate_bspline_from_points(
      data.name, 
      data.selectedItems.map((item: any) => rootObj.get_obj(item.name)), 
      data.knots, 
      data.degree, 
      rootObj
    )
  }
  else if (data.style === 'Skeleton') {
    const radius = data.selectedItems.map((item: any) => item.radius)
    result = generate_skeleton(
      data.name, 
      rootObj.get_obj(data.spline_name) as BsplineObj, 
      data.selectedItems.map((item: any) => item.percentage), 
      radius[0], 
      rootObj
    )
  }
  else {
    ElMessage.error('Invalid style')
    return
  }
  if (result.success) {
    updateTreeData()
  }
}

// ---------------- Surface Tool ----------------
const surfaceMode = ref("PipeShell")
const handleSurfaceToolClick = (index: string) => {
  if (index === '3-3-1') {
    surfaceMode.value = "PipeShell"
  }
  else if (index === '3-3-2') {
    surfaceMode.value = "BsplineSurface"
  }
  else if (index === '3-3-3') {
    surfaceMode.value = "Disk"
  }
  setVisableDict('surfaceToolVisible', true)
}

const meshContextToolMode = ref("Move")
const handleMeshContextMenuClick = (mode: string) => {
  console.log('Mesh context menu click:', mode)
  if (mode === 'Hide/Vis') {
    pass
  }
  else if (mode === 'Move') {
    meshContextToolMode.value = "Move"
    setVisableDict('meshContextToolVisible', true)
  }
  else if (mode === 'Edit') {
    meshContextToolMode.value = "Edit"
    setVisableDict('meshContextToolVisible', true)
  }
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
    contextMenuVisible.value = false
  }
  meshContextMenuVisible.value = false
}

</script>

<style scoped>
.surface-postprocessing {
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

/* Make dialog headers draggable */
:deep(.el-dialog__header) {
  cursor: move;
  background: #f5f7fa;
  padding: 15px 20px;
  margin-right: 0;
  border-bottom: 1px solid #dcdfe6;
}

:deep(.el-dialog) {
  margin: 0 !important;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
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