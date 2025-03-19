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
          <el-menu-item index="3-1-1">Generate Point</el-menu-item>
          <el-menu-item index="3-1-2">Generate Vector Point</el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="3-2">
          <template #title>LineTool</template>
          <el-menu-item index="3-2-1">Generate Line</el-menu-item>
          <el-menu-item index="3-2-2">Generate Bspline</el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="3-3">
          <template #title>SurfaceTool</template>
          <el-menu-item index="3-3-1">Generate PipeShell</el-menu-item>
          <el-menu-item index="3-3-2">Generate Bspline Surface</el-menu-item>
        </el-sub-menu>
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
        <ThreeViewer @select-object="handleLeftClickMeshSelect" @right-click="handleRightClickMeshSelect"/>
      </el-main>
    </div>

    <!-- Context Menu -->
    <div 
      v-show="contextMenuVisible"
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
      accept=".h5,.hdf5"
      @change="handleFileSelected"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onBeforeUnmount, provide } from 'vue'
import { Document, Setting, Tools } from '@element-plus/icons-vue'
import ThreeViewer from '../components/ThreeViewer.vue'
import { RootObj } from '../classes/store_obj.ts'
import { ElMessage, ElMessageBox } from 'element-plus'
import { tr } from 'element-plus/es/locales.mjs'
import type { Scene, Object3D } from 'three'
import * as THREE from 'three'
import type { PathResultData } from '../types/data_type.ts'
import { PointTypes, LineTypes, SurfaceTypes } from '../types/data_type.ts'
import type { TreeNode } from '../types/data_type.ts'

import { PointObj } from '../classes/geo_obj.ts'
import { PathObj } from '../classes/board_obj.ts'

// ----------------
const scene = new THREE.Scene();
const rootObj = new RootObj(scene);
provide('storeObj', rootObj);
provide('main_message', "Main Window Running...");

// ---------------- Tree View ----------------
const handleLeftTreeClick = (data: any) => {
}

const handleRightTreeClick = (event: MouseEvent, data: any) => {
  event.preventDefault()
  contextMenuPosition.x = event.clientX
  contextMenuPosition.y = event.clientY
  contextMenuVisible.value = true
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

// Tree data
const treeData = ref<TreeNode[]>([])

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

// ---------------- Context menu ---------------- 
const contextMenuVisible = ref(false)
const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

const handleClickEvent = (event: MouseEvent) => {
  const target = event.target as HTMLElement
  if (!target.closest('.context-menu')) {
    // contextMenuVisible.value = false
  }
}

// ---------------- Event handling ----------------
onMounted(() => {
  document.addEventListener('click', handleClickEvent)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickEvent)
})

// ---------------- click mesh event ----------------
const handleLeftClickMeshSelect = (objectName: string) => {
  // // Find the object in the tree data
  // const findNode = (nodes: TreeNode[]): TreeNode | null => {
  //   for (const node of nodes) {
  //     if (node.label === objectName) {
  //       return node
  //     }
  //     if (node.children) {
  //       const found = findNode(node.children)
  //       if (found) return found
  //     }
  //   }
  //   return null
  // }

  // const selectedNode = findNode(treeData.value)
  // if (selectedNode) {
  //   // You can add additional handling here, like highlighting the tree node
  //   console.log('Selected node:', selectedNode)
  // }
  console.log('Left click detected:', objectName)
}

const handleRightClickMeshSelect = (objectName: string) => {
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
      const path = new PathObj(
        `${group_name}/${segment_name}`, 
        control_point_list, 
        segment.knots, 
        segment.degree[0],
        segment.radius[0],
        rootObj.get_scene()
      )
      rootObj.add_obj(path)
    })

  })
  
  updateTreeData()
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