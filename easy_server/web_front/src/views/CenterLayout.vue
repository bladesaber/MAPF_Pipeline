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

      <PathSmooth
        v-if="activeCard === 'Smooth'"
        ref="pathSmoothRef"
        @updateTree="update_tree"
        @updateLog="update_log"
        @highlightTreeNode="handleHighlightTreeNode"
      />

      <CADTool v-if="activeCard === 'CAD'" />
      <SimulateTool v-if="activeCard === 'Simulate'" />

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
          :style="{ width: sidebarWidth + 'px', height: `calc(100vh - 118px)` }"
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
            :default-expand-all="false"
            :default-expanded-keys="expandedKeys"
            @node-expand="handleNodeExpand"
            @node-collapse="handleNodeCollapse"
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
            <div class="bottom-sidebar-content">
              <div class="info-section">
                <div class="info-header">
                  <span>Mesh Information</span>
                </div>
                <div class="info-content">
                  <div v-if="selectedMeshInfo" class="info-item">
                    <div v-for="(value, key) in selectedMeshInfo" :key="key" class="info-row">
                      <span class="info-label">{{ key }}:</span>
                      <span class="info-value">{{ value }}</span>
                    </div>
                  </div>
                  <div v-else class="no-info">
                    No mesh selected
                  </div>
                </div>
              </div>
              <div class="log-section">
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
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  provide,
  reactive,
  ref,
  watch,
} from 'vue';

import {
  ElMessage,
  ElMessageBox,
} from 'element-plus';
import type {
  Object3D,
  Scene,
} from 'three';
import * as THREE from 'three';
import { pass } from 'three/tsl';

import {
  BoxObstacle,
  CylinderObstacle,
  PointsObstacle,
  SphereLineObstacle,
  SpheresObstacle,
  SquaresObstacle,
} from '@/classes/path_search/obstacle_mesh';
import config from '@/config';
import { setVisableDict } from '@/utils/utils';
import { setMeshColor } from '@/utils/visulize_utils';
import { ArrowDown } from '@element-plus/icons-vue';

import {
  CenterRoot,
  ProjectManager,
} from '../classes/center_root.ts';
import CenterViewer from '../components/CenterViewer.vue';
import DebugTool from '../components/debug_tool/DebugTool.vue';
import DefineProject from '../components/main_fragment/DefineProject.vue';
import CADTool from '../components/path_cad/CADTool.vue';
import PathSearch from '../components/path_search/PathSearch.vue';
import SimulateTool from '../components/path_simulation/SimulateTool.vue';
import PathSmooth from '../components/path_smooth/PathSmooth.vue';
import type { TreeNode } from '../types/data_type';

// ---------------- Root Object ----------------
const scene = new THREE.Scene();
const rootObj = new CenterRoot(scene);

provide('centerRoot', rootObj);
provide('main_message', "Main Window Running...");

const pathSearchRef = ref()
const pathSmoothRef = ref()
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
const sidebarWidth = ref(280)
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

const handleLeftTreeClick = (data: any, node: any) => {
  if (data.style === 'Obstacle') {
    highlightMeshAndShowInfo({
      type: 'Obstacle',
      projectName: data.data.Project,
      obstacleName: data.data.Obstacle,
      pipeGroup: undefined,
      pipeName: undefined,
      meshName: undefined
    })
  } else if (data.style === 'Pipe') {
    highlightMeshAndShowInfo({
      type: 'Pipe',
      projectName: data.data.Project,
      obstacleName: undefined,
      pipeGroup: data.data.PipeGroup,
      pipeName: data.data.Pipe,
      meshName: undefined
    })
  } else {
    updateMeshInfo(null)
  }
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
      pathSmoothRef.value!.showContextMenu({
        'x': event.clientX,
        'y': event.clientY,
        'data': data
      })
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
const expandedKeys = ref<string[]>([])

const update_tree = () => {
  treeData.value = rootObj.export_tree()
  if (treeData.value && treeData.value.length > 0 && expandedKeys.value.length === 0) {
    expandedKeys.value = [treeData.value[0].id]
  }
  // console.log('treeData:', JSON.stringify(treeData.value, null, 2))
  if (treeData.value && treeData.value.length > 0) {
    treeData.value.forEach((node, idx) => {
      console.log(`一级节点[${idx}] label:`, node.label, 'id:', node.id, 'children:', node.children)
    })
  } else {
    console.log('treeData 为空或无一级节点')
  }
}

watch(treeData, (newVal) => {
  if (newVal && newVal.length > 0 && expandedKeys.value.length === 0) {
    expandedKeys.value = [newVal[0].id]
  }
})

const handleNodeExpand = (data: any, node: any) => {
  if (!expandedKeys.value.includes(data.id)) {
    expandedKeys.value.push(data.id)
  }
}

const handleNodeCollapse = (data: any, node: any) => {
  expandedKeys.value = expandedKeys.value.filter(id => id !== data.id)
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
const handleLeftClickMeshSelect = (name: string) => {
  highlightMeshAndShowInfo({
    type: 'MeshName',
    projectName: undefined,
    obstacleName: undefined,
    pipeGroup: undefined,
    pipeName: undefined,
    meshName: name
  })
}

const handleRightClickMeshSelect = (name: string) => {
    console.log('right click', name)
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
  const smooth_ids: string[] = []
  rootObj.proj_dict.forEach((projectManager) => {
    projectManager.smooth_task_dict.forEach((smooth_task) => {
      smooth_ids.push(smooth_task.name)
    })
  })

  try {
    const response = await fetch(`${config.BASE_URL}/path_search/clear_cache`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        project_ids: Array.from(rootObj.proj_dict.keys()),
        smooth_ids: smooth_ids
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

// Add new ref for selected mesh info
const selectedMeshInfo = ref<Record<string, any> | null>(null)

// Add method to update mesh info
const updateMeshInfo = (info: Record<string, any> | null) => {
  selectedMeshInfo.value = info
}

// Expose addLog method
defineExpose({
  addLog,
  updateMeshInfo
})

// 公共高亮+信息展示方法
function highlightMeshAndShowInfo({ type = undefined, projectName = undefined, obstacleName = undefined, pipeGroup = undefined, pipeName = undefined, meshName = undefined }: { type?: any, projectName?: any, obstacleName?: any, pipeGroup?: any, pipeName?: any, meshName?: any }) {
  // 1. 先重置所有mesh颜色
  rootObj.proj_dict.forEach(project => {
    project.obstacle_dict.forEach(obstacle => {
      Object.values(obstacle.mesh_dict).forEach(mesh => {
        if (mesh instanceof THREE.Mesh) {
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach(mat => {
              if ('color' in mat) setMeshColor(mesh, 0x00ff00)
            })
          } else if (mesh.material && 'color' in mesh.material) {
            setMeshColor(mesh, 0x00ff00)
          }
        }
      })
    })
    project.pipe_dict.forEach(pipeGroup => {
      pipeGroup.forEach(pipe => {
        Object.values(pipe.mesh_dict).forEach(mesh => {
          if (mesh instanceof THREE.Mesh) {
            if (Array.isArray(mesh.material)) {
              mesh.material.forEach(mat => {
                if ('color' in mat) setMeshColor(mesh, 0x2196f3)
              })
            } else if (mesh.material && 'color' in mesh.material) {
              setMeshColor(mesh, 0x2196f3)
            }
          }
        })
      })
    })
  })

  // 2. 高亮目标mesh并展示信息
  if (type === 'Obstacle') {
    const project = rootObj.proj_dict.get(projectName)
    if (project) {
      const obstacle = project.obstacle_dict.get(obstacleName)
      if (obstacle) {
        Object.values(obstacle.mesh_dict).forEach(mesh => {
          if (mesh instanceof THREE.Mesh) {
            if (Array.isArray(mesh.material)) {
              mesh.material.forEach(mat => {
                if ('color' in mat) setMeshColor(mesh, 0xffffff)
              })
            } else if (mesh.material && 'color' in mesh.material) {
              setMeshColor(mesh, 0xffffff)
            }
          }
        })
        // 展示信息
        const info: Record<string, any> = {
          'Name': obstacle.name,
          'Type': obstacle.constructor.name
        }
        if (obstacle instanceof CylinderObstacle) {
          info['Position'] = `(${obstacle.x.toFixed(2)}, ${obstacle.y.toFixed(2)}, ${obstacle.z.toFixed(2)})`
          info['Radius'] = obstacle.radius.toFixed(2)
          info['Height'] = obstacle.height.toFixed(2)
          info['Direction'] = `(${obstacle.vector_x}, ${obstacle.vector_y}, ${obstacle.vector_z})`
          info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
        } else if (obstacle instanceof BoxObstacle) {
          info['Bounds'] = `X: [${obstacle.xmin.toFixed(2)}, ${obstacle.xmax.toFixed(2)}], Y: [${obstacle.ymin.toFixed(2)}, ${obstacle.ymax.toFixed(2)}], Z: [${obstacle.zmin.toFixed(2)}, ${obstacle.zmax.toFixed(2)}]`
          info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
        } else if (obstacle instanceof PointsObstacle) {
          info['Points Count'] = obstacle.points.length
          info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
        } else if (obstacle instanceof SpheresObstacle) {
          info['Spheres Count'] = obstacle.points.length
          info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
        } else if (obstacle instanceof SquaresObstacle) {
          info['Squares Count'] = obstacle.points.length
          info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
        } else if (obstacle instanceof SphereLineObstacle) {
          info['Points Count'] = obstacle.points.length
          info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
        }
        updateMeshInfo(info)
        return
      }
    }
  } else if (type === 'Pipe') {
    const project = rootObj.proj_dict.get(projectName)
    if (project) {
      const pipeGroupObj = project.pipe_dict.get(pipeGroup)
      if (pipeGroupObj) {
        const pipe = pipeGroupObj.get(pipeName)
        if (pipe) {
          Object.values(pipe.mesh_dict).forEach(mesh => {
            if (mesh instanceof THREE.Mesh) {
              if (Array.isArray(mesh.material)) {
                mesh.material.forEach(mat => {
                  if ('color' in mat) setMeshColor(mesh, 0xffffff)
                })
              } else if (mesh.material && 'color' in mesh.material) {
                setMeshColor(mesh, 0xffffff)
              }
            }
          })
          updateMeshInfo({
            'Name': pipe.name,
            'Type': 'Pipe',
            'Position': `(${pipe.x.toFixed(2)}, ${pipe.y.toFixed(2)}, ${pipe.z.toFixed(2)})`,
            'Radius': pipe.radius.toFixed(2),
            'Direction': `(${pipe.vector_x}, ${pipe.vector_y}, ${pipe.vector_z})`,
            'Pipe Mode': pipe.pipe_mode,
            'Group': pipe.group_name
          })
          return
        }
      }
    }
  } else if (type === 'MeshName' && meshName) {
    // 通过 meshName 反查
    rootObj.proj_dict.forEach((project_manager) => {
      // pipe
      project_manager.pipe_dict.forEach((pipe_dict) => {
        pipe_dict.forEach((pipe) => {
          if (pipe.name === meshName) {
            Object.values(pipe.mesh_dict).forEach(mesh => {
              if (mesh instanceof THREE.Mesh) {
                if (Array.isArray(mesh.material)) {
                  mesh.material.forEach(mat => {
                    if ('color' in mat) setMeshColor(mesh, 0xffffff)
                  })
                } else if (mesh.material && 'color' in mesh.material) {
                  setMeshColor(mesh, 0xffffff)
                }
              }
            })
            updateMeshInfo({
              'Name': pipe.name,
              'Type': 'Pipe',
              'Position': `(${pipe.x.toFixed(2)}, ${pipe.y.toFixed(2)}, ${pipe.z.toFixed(2)})`,
              'Radius': pipe.radius.toFixed(2),
              'Direction': `(${pipe.vector_x}, ${pipe.vector_y}, ${pipe.vector_z})`,
              'Pipe Mode': pipe.pipe_mode,
              'Group': pipe.group_name
            })
            return
          }
        })
      })
      // obstacle
      project_manager.obstacle_dict.forEach((obstacle) => {
        if (obstacle.name === meshName) {
          Object.values(obstacle.mesh_dict).forEach(mesh => {
            if (mesh instanceof THREE.Mesh) {
              if (Array.isArray(mesh.material)) {
                mesh.material.forEach(mat => {
                  if ('color' in mat) setMeshColor(mesh, 0xffffff)
                })
              } else if (mesh.material && 'color' in mesh.material) {
                setMeshColor(mesh, 0xffffff)
              }
            }
          })
          const info: Record<string, any> = {
            'Name': obstacle.name,
            'Type': obstacle.constructor.name
          }
          if (obstacle instanceof CylinderObstacle) {
            info['Position'] = `(${obstacle.x.toFixed(2)}, ${obstacle.y.toFixed(2)}, ${obstacle.z.toFixed(2)})`
            info['Radius'] = obstacle.radius.toFixed(2)
            info['Height'] = obstacle.height.toFixed(2)
            info['Direction'] = `(${obstacle.vector_x}, ${obstacle.vector_y}, ${obstacle.vector_z})`
            info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
          } else if (obstacle instanceof BoxObstacle) {
            info['Bounds'] = `X: [${obstacle.xmin.toFixed(2)}, ${obstacle.xmax.toFixed(2)}], Y: [${obstacle.ymin.toFixed(2)}, ${obstacle.ymax.toFixed(2)}], Z: [${obstacle.zmin.toFixed(2)}, ${obstacle.zmax.toFixed(2)}]`
            info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
          } else if (obstacle instanceof PointsObstacle) {
            info['Points Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
          } else if (obstacle instanceof SpheresObstacle) {
            info['Spheres Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
          } else if (obstacle instanceof SquaresObstacle) {
            info['Squares Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
          } else if (obstacle instanceof SphereLineObstacle) {
            info['Points Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(', ')})`
          }
          updateMeshInfo(info)
          return
        }
      })
      // grid
      if (project_manager.grid.name === meshName) {
        updateMeshInfo({
          'Name': project_manager.grid.name,
          'Type': 'Grid',
          'Size': `(${project_manager.grid.xlength.toFixed(2)}, ${project_manager.grid.ylength.toFixed(2)}, ${project_manager.grid.zlength.toFixed(2)})`,
          'Resolution': project_manager.grid.resolution.toFixed(2),
          'Grid Size': `(${project_manager.grid.xsize}, ${project_manager.grid.ysize}, ${project_manager.grid.zsize})`
        })
        return
      }
    })
  } else {
    updateMeshInfo(null)
  }
}

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
  height: calc(100vh - 118px);
  overflow: auto;
  display: flex;
  flex-direction: column;
  padding: 0;
  margin: 0;
  flex-shrink: 0;
  background: #fff;
  border-right: 1px solid #dcdfe6;
  box-shadow: 1px 0 4px rgba(0, 0, 0, 0.1);
  z-index: 1;
}

/* 优化横向滚动条样式 */
.sidebar::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

.sidebar::-webkit-scrollbar-thumb {
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: 3px;
}

.sidebar::-webkit-scrollbar-track {
  background-color: transparent;
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
}

.bottom-sidebar-content {
  display: flex;
  height: 100%;
}

.info-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  border-right: 1px solid #dcdfe6;
}

.log-section {
  display: flex;
  flex-direction: column;
  flex: 2;
  min-width: 0;
  padding: 8px 16px;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
}

.info-header, .log-header {
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

.info-content {
  flex: 1;
  overflow-y: auto;
  padding: 8px 16px;
  font-size: 12px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.info-row {
  display: flex;
  gap: 8px;
  padding: 4px 0;
  border-bottom: 1px solid #ebeef5;
}

.info-label {
  font-weight: bold;
  color: #606266;
  min-width: 100px;
}

.info-value {
  color: #303133;
  word-break: break-all;
}

.no-info {
  color: #909399;
  text-align: center;
  padding: 16px;
  font-style: italic;
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
  flex: 1 1 0%;
  overflow-y: auto;
  min-height: 0;
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

.card-layout {
  display: flex;
  gap: 20px;
  padding: 0 32px;
  background: #fff;
  border-bottom: 1.5px solid #e4e7ed;
  height: 56px;
  align-items: flex-end;
  position: relative;
  z-index: 10;
}

.menu-card {
  min-width: 96px;
  height: 55px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: #303133;
  background: #f8fafb;
  box-shadow: 0 1px 4px rgba(0,0,0,0.03);
  cursor: pointer;
  transition: 
    background 0.18s,
    color 0.18s,
    box-shadow 0.18s;
  user-select: none;
  border: 1px solid transparent;
  border-bottom: none;
  margin-bottom: 0;
  position: relative;
}

.menu-card:hover {
  background: #f0f6ff;
  color: #409eff;
}

.menu-card.active {
  background: #fff;
  color: #409eff;
  border: 1px solid #d0e4ff;
  border-bottom: 2.5px solid #409eff;
  font-weight: 600;
  z-index: 20;
  box-shadow: 0 4px 12px rgba(64,158,255,0.08);
}

.menu-card:not(.active) {
  border-bottom: 1.5px solid #e4e7ed;
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
  height:44px;
  padding: 0 20px;
  color: #303133;
  font-size: 15px;
  transition: all 0.3s;
  font-family: inherit;
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

</style>