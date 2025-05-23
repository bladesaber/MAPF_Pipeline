<!-- CenterLayout.vue -->
<template>
    <div class="center-layout">
      <div class="card-layout">
        <el-card 
          v-for="card in cards" 
          :key="card.name"
          :class="['menu-card', { active: activeCard === card.name }]"
          @click="handleCardClick(card.name)"
        >
          {{ card.name }}
        </el-card>
      </div>

      <el-menu 
        v-if="activeCard === 'Main'"
        mode="horizontal" 
        class="menu-bar"
      >
        <el-dropdown trigger="click" placement="bottom" @visible-change="(visible: boolean) => handleDropdownVisible('file', visible)">
          <span class="el-dropdown-link">
            File<el-icon class="el-icon--right" :class="{ 'is-active': dropdownStates.file }"><arrow-down /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item @click="handleExportProject">Export</el-dropdown-item>
              <el-dropdown-item @click="handleImportProject">Import</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <el-dropdown trigger="click" placement="bottom" @visible-change="(visible: boolean) => handleDropdownVisible('project', visible)">
          <span class="el-dropdown-link">
            Project<el-icon class="el-icon--right" :class="{ 'is-active': dropdownStates.project }"><arrow-down /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item @click="handleProjectMenu('New Project')">New Project</el-dropdown-item>
              <el-dropdown-item @click="handleProjectMenu('Open Project')">Open Project</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <el-dropdown trigger="click" placement="bottom" @visible-change="(visible: boolean) => handleDropdownVisible('view', visible)">
          <span class="el-dropdown-link">
            View<el-icon class="el-icon--right" :class="{ 'is-active': dropdownStates.view }"><arrow-down /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item>Show Grid</el-dropdown-item>
              <el-dropdown-item>Show Axes</el-dropdown-item>
              <el-dropdown-item>Reset Camera</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <el-dropdown trigger="click" placement="bottom" @visible-change="(visible: boolean) => handleDropdownVisible('setting', visible)">
          <span class="el-dropdown-link">
            Setting<el-icon class="el-icon--right" :class="{ 'is-active': dropdownStates.setting }"><arrow-down /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item>Default</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <el-dropdown trigger="click" placement="bottom" @visible-change="(visible: boolean) => handleDropdownVisible('help', visible)">
          <span class="el-dropdown-link">
            Help<el-icon class="el-icon--right" :class="{ 'is-active': dropdownStates.help }"><arrow-down /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item>About</el-dropdown-item>
              <el-dropdown-item>Documentation</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </el-menu>

      <PathSearch 
        v-if="activeCard === 'Design'"
        ref="pathSearchRef"
        @updateTree="update_tree"
        @updateLog="update_log"
        @highlightTreeNode="handleHighlightTreeNode"
      />

      <DebugTool
        v-if="activeCard === 'Debug'"
        ref="debugToolRef"
        @updateTree="update_tree"
        @updateLog="update_log"
        @highlightTreeNode="handleHighlightTreeNode"
      />

      <DefineProject
        v-model="defineProjectVisible"
        @generate="handleDefineProjectGenerate"
      />

      <div class="main-content">
        <div 
          class="sidebar"
          :style="{ width: sidebarWidth + 'px', height: `calc(100vh - 36px - ${bottomSidebarHeight}px)` }"
        >
          <el-tree
            v-if="treeData.length"
            :data="treeData"
            @node-click="handleLeftTreeClick"
            @node-contextmenu="handleRightTreeClick"
            node-key="id"
            ref="treeRef"
            highlight-current
            :current-node-key="currentNodeKey"
            :default-expand-all="true"
          />
          <div
            class="resize-handle"
            @mousedown="handleTreeResize"
          ></div>
        </div>
  
        <div class="viewer-container">
          <el-main class="viewer">
            <CenterViewer 
              @leftClick-select-object="handleLeftClickMeshSelect"
              @rightClick-select-object="handleRightClickMeshSelect"
            />
          </el-main>
          <div class="bottom-sidebar" :style="{ height: bottomSidebarHeight + 'px' }">
            <div class="log-header">
              <span>Log Information</span>
              <div class="resize-handle-horizontal" @mousedown="handleBottomResize"></div>
            </div>
            <div class="log-content" ref="logContent">
              <div v-for="(log, index) in logs" :key="index" class="log-item">
                {{ log }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <el-dialog
        v-model="exportJsonDialogVisible"
        title="Export JSON"
        width="30%"
      >
        <el-form>
          <el-form-item label="Filename">
            <el-input 
              v-model="exportFilename" 
              placeholder="Enter filename (e.g. export.json)"
            />
          </el-form-item>
        </el-form>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="exportJsonDialogVisible = false">Cancel</el-button>
            <el-button type="primary" @click="handleExportConfirm">Export</el-button>
          </span>
        </template>
      </el-dialog>

      <input
        type="file"
        ref="importFileInput"
        style="display: none"
        accept=".json"
        @change="handleImportConfirm"
      />

    </div>
  </template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onBeforeUnmount, provide, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { Scene, Object3D } from 'three'
import * as THREE from 'three'
import { pass } from 'three/tsl'
import { ArrowDown } from '@element-plus/icons-vue'

import { setVisableDict } from '@/utils/utils'

import CenterViewer from '../components/CenterViewer.vue'
import type { TreeNode } from '../types/data_type'
import { CenterRoot, ProjectManager } from '../classes/center_root.ts'

import PathSearch from '../components/path_search/PathSearch.vue'
import DebugTool from '../components/debug_tool/DebugTool.vue'
import DefineProject from '../components/main_fragment/DefineProject.vue'

// ---------------- Root Object ----------------
const scene = new THREE.Scene();
const rootObj = new CenterRoot(scene);

provide('centerRoot', rootObj);
provide('main_message', "Main Window Running...");

const pathSearchRef = ref()
const debugToolRef = ref()

// ---------------- Card Layout ----------------
const cards = [
  { name: 'Main' },
  { name: 'Design' },
  { name: 'Smooth' },
  { name: 'CAD' },
  { name: 'Simulate' },
  { name: 'Debug' }
]

const activeCard = ref('Main')
const handleCardClick = (cardName: string) => {
  activeCard.value = cardName
}

// ---------------- Left Sidebar ----------------
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

const handleLeftTreeClick = (event: MouseEvent, data: any) => {
  // TODO: change the color of the tree node
}

const handleRightTreeClick = (event: MouseEvent, data: any) => {
    event.preventDefault()

    if (activeCard.value === 'Design') {
      pathSearchRef.value!.showContextMenu({
        'x': event.clientX,
        'y': event.clientY,
        'data': data
      })
    } else if (activeCard.value === 'Smooth') {
      console.log('smooth')
    } else if (activeCard.value === 'CAD') {
      console.log('cad')
    } else if (activeCard.value === 'Simulate') {
      console.log('simulate')
    } else if (activeCard.value === 'Debug') {
      debugToolRef.value!.showContextMenu({
        'x': event.clientX,
        'y': event.clientY,
        'data': data
      })
    }
}

const treeData = ref<TreeNode[]>([])
const update_tree = () => {
  treeData.value = rootObj.export_tree()
}

// ---------------- Bottom Sidebar ----------------
const update_log = (message: string) => {
  const timestamp = new Date().toLocaleTimeString()
  if (logs.value.length > 2000) {
    logs.value.splice(0, 1000)
  }
  logs.value.push(`[${timestamp}] ${message}`)
  // Auto scroll to bottom
  nextTick(() => {
    if (logContent.value) {
      logContent.value.scrollTop = logContent.value.scrollHeight
    }
  })
}

// ---------------- Mesh Click Event handling ----------------
const handleLeftClickMeshSelect = () => {
    console.log('left click')
}

const handleRightClickMeshSelect = () => {
    console.log('right click')
}

// ---------------- Fragment Visible ----------------
const defineProjectVisible = ref(false)

const fragmentVisibleDict = {
  'defineProjectVisible': defineProjectVisible
}

// ---------------- Project ----------------
const handleProjectMenu = (menu: string) => {
    if (menu === 'New Project') {
        setVisableDict('defineProjectVisible', fragmentVisibleDict, true, true)
    } else if (menu === 'Open Project') {
        console.log('open project')
    }
}

const exportJsonDialogVisible = ref(false)
const exportFilename = ref('')
const handleExportProject = () => {
  exportJsonDialogVisible.value = true
}

const handleExportConfirm = () => {
  if (!exportFilename.value) {
    ElMessage.error('Please enter a filename')
    return
  }

  // Add .json extension if not present
  const filename = exportFilename.value.endsWith('.json') 
    ? exportFilename.value 
    : `${exportFilename.value}.json`

  const export_data = rootObj.export_json()
  
  // Create a Blob with the JSON data
  const blob = new Blob([JSON.stringify(export_data, null, 2)], { type: 'application/json' })
  
  // Create a URL for the Blob
  const url = URL.createObjectURL(blob)
  
  // Create a temporary link element
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  
  // Trigger the download
  document.body.appendChild(link)
  link.click()
  
  // Cleanup
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
  
  // Close dialog and reset filename
  exportJsonDialogVisible.value = false
  exportFilename.value = ''
  
  ElMessage.success(`Successfully exported to ${filename}`)
}

const importJsonDialogVisible = ref(false)
const importFileInput = ref<HTMLInputElement | null>(null)
const handleImportProject = () => {
  if (importFileInput.value) {
    importFileInput.value.click()
  }
}

const handleImportConfirm = async (event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) {
    return
  }

  const data = JSON.parse(await file.text())
  rootObj.load_json(data)
  update_tree()
}

// ---------------- Function handling ----------------
const handleDefineProjectGenerate = (data: any) => {
  if (rootObj.proj_dict.has(data.name)) {
    ElMessage.warning('Project already exists')
    return
  }
  const projectManager = new ProjectManager(
    data.name, data.xlength, data.ylength, data.zlength, data.resolution, rootObj.get_scene()
  )
  rootObj.proj_dict.set(data.name, projectManager)
  update_tree()
}

// ---------------- Click Event handling ----------------
// Add API call function
const clearCache = async () => {
  console.log('clear cache')
  try {
    const response = await fetch('http://192.168.110.54:9998/path_search/clear_cache', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        project_ids: Array.from(rootObj.proj_dict.keys())
      })
    })
    if (!response.ok) {
      console.error('Failed to clear cache')
    }
  } catch (error) {
    console.error('Error clearing cache:', error)
  }
}

onMounted(() => {
    document.addEventListener('click', handleClickEvent)
    window.addEventListener('beforeunload', clearCache)
})

onBeforeUnmount(() => {
    document.removeEventListener('click', handleClickEvent)
    window.removeEventListener('beforeunload', clearCache)
})

const handleClickEvent = (event: MouseEvent) => {
    const target = event.target as HTMLElement
    if (!target.closest('.context-menu')) {
    }
}

// Add ref for tree
const treeRef = ref()

// Add ref for current node key
const currentNodeKey = ref<string | null>(null)

// Modify handler for tree node highlighting
const handleHighlightTreeNode = (nodeId: string) => {
  currentNodeKey.value = nodeId
  if (treeRef.value) {
    treeRef.value.setCurrentKey(nodeId)
    nextTick(() => {
      if (treeRef.value) {
        const node = treeRef.value.getNode(nodeId)
        if (node) {
          node.expanded = true
        }
      }
    })
  }
}

// 修改下拉菜单状态控制
const dropdownStates = reactive({
  file: false,
  project: false,
  view: false,
  setting: false,
  help: false
})

// 监听下拉菜单的显示状态
const handleDropdownVisible = (menu: keyof typeof dropdownStates, visible: boolean) => {
  dropdownStates[menu] = visible
}

// Add new refs for bottom sidebar
const bottomSidebarHeight = ref(200)
const isBottomResizing = ref(false)
const startY = ref(0)
const startHeight = ref(0)
const logs = ref<string[]>([])
const logContent = ref<HTMLElement | null>(null)

// Add bottom resize handlers
const handleBottomResize = (e: MouseEvent) => {
  isBottomResizing.value = true
  startY.value = e.clientY
  startHeight.value = bottomSidebarHeight.value
  document.addEventListener('mousemove', startBottomResize)
  document.addEventListener('mouseup', stopBottomResize)
}

const startBottomResize = (e: MouseEvent) => {
  if (!isBottomResizing.value) return
  const diff = startY.value - e.clientY
  bottomSidebarHeight.value = Math.max(100, Math.min(400, startHeight.value + diff))
}

const stopBottomResize = () => {
  isBottomResizing.value = false
  document.removeEventListener('mousemove', startBottomResize)
  document.removeEventListener('mouseup', stopBottomResize)
}

// Add method to add logs
const addLog = (message: string) => {
  const timestamp = new Date().toLocaleTimeString()
  logs.value.push(`[${timestamp}] ${message}`)
  // Auto scroll to bottom
  nextTick(() => {
    if (logContent.value) {
      logContent.value.scrollTop = logContent.value.scrollHeight
    }
  })
}

// Expose addLog method
defineExpose({
  addLog
})

</script>

<style>
/* 全局样式，不使用 scoped */
.el-dropdown-menu {
  min-width: 120px !important;
}

.el-dropdown-menu__item {
  height: 36px !important;
  line-height: 36px !important;
  padding: 0 16px !important;
  background-color: #f0f2f5 !important;
}

.el-dropdown-menu__item:hover {
  background-color: #ecf5ff !important;
}

.el-dropdown-menu__item.is-active {
  background-color: #ecf5ff !important;
  color: #409eff !important;
}

/* 箭头动画 */
.el-icon--right {
  transition: transform 0.3s;
  font-size: 12px;
}

.el-icon--right.is-active {
  transform: rotate(180deg);
  color: #409eff;
}
</style>

<style scoped>
.center-layout {
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

.main-content {
  flex: 1;
  display: flex;
  min-height: 0;
  overflow: hidden;
  width: 100%;
  height: calc(100vh - 36px);
}

.sidebar {
  position: relative;
  min-width: 200px;
  max-width: 500px;
  background: #fff;
  border-right: 1px solid #dcdfe6;
  box-shadow: 1px 0 4px rgba(0, 0, 0, 0.1);
  overflow: auto;
  flex-shrink: 0;
  z-index: 1;
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

.viewer-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 600px;
  overflow: hidden;
}

.viewer {
  flex: 1;
  padding: 0;
  overflow: hidden;
  background: #fff;
  position: relative;
  z-index: 0;
}

.bottom-sidebar {
  position: relative;
  background: #fff;
  border-top: 1px solid #dcdfe6;
  box-shadow: 0 -1px 4px rgba(0, 0, 0, 0.1);
  min-height: 100px;
  max-height: 400px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* Update tree node selection styles */
:deep(.el-tree-node.is-current > .el-tree-node__content) {
  background-color: #ecf5ff !important;
  color: #409eff !important;
  font-weight: bold !important;
}

:deep(.el-tree-node.is-current > .el-tree-node__content:hover) {
  background-color: #ecf5ff !important;
  color: #409eff !important;
  font-weight: bold !important;
}

:deep(.el-tree-node__content:hover) {
  background-color: #f5f7fa !important;
}

.card-layout {
  display: flex;
  gap: 0;
  padding: 0;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  position: relative;
  width: 100%;
}

.card-layout::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background-color: #e4e7ed;
  z-index: 1;
}

.card-item {
  padding: 0 20px;
  height: 36px;
  line-height: 36px;
  font-size: 14px;
  color: #303133;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.3s cubic-bezier(0.645, 0.045, 0.355, 1);
  position: relative;
  z-index: 2;
  user-select: none;
}

.card-item:hover {
  color: #409eff;
  background-color: #f5f7fa;
}

.card-item.active {
  color: #409eff;
  border-bottom-color: #409eff;
  background-color: #fff;
}

.card-item .el-icon {
  margin-right: 4px;
  font-size: 14px;
  vertical-align: middle;
}

.card-item .el-icon-close {
  margin-left: 4px;
  font-size: 12px;
  border-radius: 50%;
  transition: all 0.3s cubic-bezier(0.645, 0.045, 0.355, 1);
  transform: scale(0.8);
  padding: 2px;
}

.card-item .el-icon-close:hover {
  background-color: #c0c4cc;
  color: #fff;
}

.menu-bar {
  border-bottom: 1px solid #dcdfe6;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  position: relative;
  z-index: 1;
}

/* Update menu styles */
:deep(.el-menu--horizontal) {
  border-bottom: none;
  display: flex;
  align-items: center;
  background: transparent;
}

.el-dropdown-link {
  cursor: pointer;
  display: flex;
  align-items: center;
  height: 36px;
  padding: 0 20px;
  color: #303133;
  transition: all 0.3s;
}

.el-dropdown-link:hover {
  background-color: #f5f7fa;
  color: #409eff;
}

.el-dropdown-link:hover .el-icon--right {
  color: #409eff;
}

/* Update tree styles */
:deep(.el-tree) {
  background: transparent;
  padding: 8px 0;
}

:deep(.el-tree-node__content) {
  height: 32px;
  transition: all 0.3s;
}

:deep(.el-tree-node.is-current > .el-tree-node__content) {
  background-color: #ecf5ff !important;
  color: #409eff !important;
  font-weight: bold !important;
}

:deep(.el-tree-node__content:hover) {
  background-color: #f5f7fa !important;
}

/* Add separator between sections */
.menu-bar::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background-color: #dcdfe6;
  z-index: 1;
}

.log-header {
  padding: 8px 16px;
  background: #f5f7fa;
  border-bottom: 1px solid #dcdfe6;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
  color: #303133;
  position: relative;
}

.resize-handle-horizontal {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 5px;
  cursor: row-resize;
  background: transparent;
}

.resize-handle-horizontal:hover {
  background: rgba(0, 0, 0, 0.1);
}

.log-content {
  flex: 1;
  overflow-y: auto;
  padding: 8px 16px;
  font-family: monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
}

.log-item {
  margin-bottom: 4px;
  color: #606266;
  border-bottom: 1px solid #ebeef5;
  padding-bottom: 4px;
}

.log-item:last-child {
  margin-bottom: 0;
  border-bottom: none;
}

</style>