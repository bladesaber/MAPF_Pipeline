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

      <SimulateTool v-if="activeCard === 'Simulate'" 
        ref="simulateToolRef"
        @updateTree="update_tree"
        @updateLog="update_log"
        @highlightTreeNode="handleHighlightTreeNode"
      />

      <CADTool v-if="activeCard === 'CAD'" />

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
            :expand-on-click-node="false"
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
              @rightClick-control-point="handleRightClickControlPoint"
              @check-highlight-status="checkHighlightedMeshStatus"
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
const simulateToolRef = ref()
const debugToolRef = ref()

// ---------------- Card Layout ----------------
const cards = [
  { name: 'Main' },
  { name: 'Design' },
  { name: 'Smooth' },
  { name: 'Simulate' },
  { name: 'CAD' },
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
  } else if (data.style === 'SegmentCell' || data.style === 'ConnectorCell') {
    // 新增：支持 Segments/Connectors 节点高亮
    const groupName = data.data.groupName;
    const name = data.data.name;
    if (groupName && name) {
      const meshName = `${groupName}/${name}`;
      highlightMeshAndShowInfo({
        type: 'MeshName',
        meshName
      });
    } else {
      console.warn('Missing groupName or name:', { groupName, name });
      updateMeshInfo(null);
    }
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
      simulateToolRef.value!.showContextMenu({
        'x': event.clientX,
        'y': event.clientY,
        'data': data
      })
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
  // 保存当前的展开状态
  const currentExpandedKeys = [...expandedKeys.value]
  
  treeData.value = rootObj.export_tree()
  
  // 恢复展开状态，如果树为空则设置默认展开
  if (treeData.value && treeData.value.length > 0) {
    if (currentExpandedKeys.length === 0) {
      if (treeData.value[0].id) {
        expandedKeys.value = [treeData.value[0].id]
      }
    } else {
      // 过滤掉在新树中不存在的节点ID
      expandedKeys.value = currentExpandedKeys.filter(key => {
        return findNodeById(treeData.value, key) !== null
      })
    }
    
    treeData.value.forEach((node, idx) => {
      console.log(`一级节点[${idx}] label:`, node.label, 'id:', node.id, 'children:', node.children)
    })
  } else {
    console.log('treeData 为空或无一级节点')
    expandedKeys.value = []
  }
}

// 递归查找节点ID的辅助函数
const findNodeById = (nodes: TreeNode[], id: string): TreeNode | null => {
  for (const node of nodes) {
    if (node.id && node.id === id) {
      return node
    }
    if (node.children) {
      const found = findNodeById(node.children, id)
      if (found) {
        return found
      }
    }
  }
  return null
}

watch(treeData, (newVal) => {
  if (newVal && newVal.length > 0 && expandedKeys.value.length === 0) {
    if (newVal[0].id) {
      expandedKeys.value = [newVal[0].id]
    }
  }
})

const handleNodeExpand = (data: any, node: any) => {
  if (!expandedKeys.value.includes(data.id)) {
    expandedKeys.value.push(data.id)
  }
}

// 递归移除所有子孙节点id
function removeDescendantIds(node: any, ids: string[]) {
  if (node.children && node.children.length) {
    for (const child of node.children) {
      ids.push(child.id);
      removeDescendantIds(child, ids);
    }
  }
}

const handleNodeCollapse = (data: any, node: any) => {
  // 递归移除所有子孙节点id
  const idsToRemove = [data.id];
  removeDescendantIds(data, idsToRemove);
  expandedKeys.value = expandedKeys.value.filter(id => !idsToRemove.includes(id));
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
// wireframe高亮管理
let lastHighlightWireframe: THREE.LineSegments | null = null
let currentHighlightedMesh: THREE.Mesh | null = null

// 清除高亮函数
function clearHighlight() {
  if (lastHighlightWireframe && lastHighlightWireframe.parent) {
    lastHighlightWireframe.parent.remove(lastHighlightWireframe)
    lastHighlightWireframe = null
  }
  currentHighlightedMesh = null
  updateMeshInfo(null)
}

// 检查当前高亮的mesh是否仍然有效
function checkHighlightedMeshStatus() {
  if (currentHighlightedMesh) {
    if (!currentHighlightedMesh.parent || !currentHighlightedMesh.visible) {
      clearHighlight()
    }
  }
}

function highlightMeshWithWireframe(mesh: THREE.Mesh) {
  if (lastHighlightWireframe && lastHighlightWireframe.parent) {
    lastHighlightWireframe.parent.remove(lastHighlightWireframe)
    lastHighlightWireframe = null
  }
  const geo = mesh.geometry.clone()
  const mat = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 })
  const wireframe = new THREE.LineSegments(new THREE.WireframeGeometry(geo), mat)
  wireframe.position.copy(mesh.position)
  wireframe.rotation.copy(mesh.rotation)
  wireframe.scale.copy(mesh.scale).multiplyScalar(1.02)
  if (mesh.parent) {
    mesh.parent.add(wireframe)
  } else {
    scene.add(wireframe)
  }
  wireframe.matrix.copy(mesh.matrix)
  wireframe.matrixAutoUpdate = false
  lastHighlightWireframe = wireframe
  currentHighlightedMesh = mesh
}

const handleLeftClickMeshSelect = (name: string) => {
  // 如果接收到空字符串，取消高亮并清除信息
  if (!name || name === '') {
    clearHighlight()
    return
  }

  // 首先检查是否是 Segments/Connectors 格式 (包含 / 分隔符)
  if (name.includes('/')) {
    // 检查是否是控制点
    const mesh = scene.getObjectByName(name) as THREE.Mesh;
    if (mesh && mesh.userData && mesh.userData.type === 'ControlPoint') {
      // 显示控制点详细信息
      const userData = mesh.userData;
      const info: Record<string, any> = {
        'Name': userData.parent?.name || name,
        'Type': 'Control Point',
        'Control Index': userData.c_idx,
        'Position': `(${userData.position[0].toFixed(3)}, ${userData.position[1].toFixed(3)}, ${userData.position[2].toFixed(3)})`,
        'Radius': userData.radius?.toFixed(3) || '1.000',
        'Is Fixed': userData.is_fix ? 'Yes' : 'No',
        'Is Unique': userData.is_unique ? 'Yes' : 'No',
        'Reference Count': userData.reference_count,
        'References': userData.references?.join(', ') || 'None',
        'Group': userData.group_name,
        'Group Index': userData.group_idx,
        'Project': userData.project_name,
        'Smooth Task': userData.smooth_name,
        'Task': userData.task_name
      };
      updateMeshInfo(info);
      highlightMeshWithWireframe(mesh);
      return;
    }
    
    // 使用 highlightMeshAndShowInfo 处理 Segments/Connectors
    highlightMeshAndShowInfo({
      type: 'MeshName',
      projectName: undefined,
      obstacleName: undefined,
      pipeGroup: undefined,
      pipeName: undefined,
      meshName: name
    })
  } else {
    // 使用原始逻辑处理障碍物、管道等
    // Reset mesh info first
    updateMeshInfo(null)
    
    // find the object in rootObj.proj_dict with the same name
    rootObj.proj_dict.forEach((project_manager) => {
      // find whether the name is in the pipe_dict
      project_manager.pipe_dict.forEach((pipe_dict) => {
        pipe_dict.forEach((pipe) => {
          if (pipe.name === name) {
            updateMeshInfo({
              Name: pipe.name,
              Type: 'Pipe',
              Position: `(${pipe.x.toFixed(2)}, ${pipe.y.toFixed(2)}, ${pipe.z.toFixed(2)})`,
              Radius: pipe.radius.toFixed(2),
              Direction: `(${pipe.vector_x}, ${pipe.vector_y}, ${pipe.vector_z})`,
              'Pipe Mode': pipe.pipe_mode,
              Group: pipe.group_name
            })
            // 添加 mesh 高亮
            const mesh = Object.values(pipe.mesh_dict)[0] as THREE.Mesh
            if (mesh) {
              highlightMeshWithWireframe(mesh)
            }
          }
        })
      })

      // find whether the name is in the obstacle_dict
      project_manager.obstacle_dict.forEach((obstacle) => {
        if (obstacle.name === name) {
          const info: Record<string, any> = {
            Name: obstacle.name,
            Type: obstacle.constructor.name
          }

          // Add properties based on obstacle type
          if (obstacle instanceof CylinderObstacle) {
            info['Position'] = `(${obstacle.x.toFixed(2)}, ${obstacle.y.toFixed(2)}, ${obstacle.z.toFixed(2)})`
            info['Radius'] = obstacle.radius.toFixed(2)
            info['Height'] = obstacle.height.toFixed(2)
            info['Direction'] = `(${obstacle.vector_x}, ${obstacle.vector_y}, ${obstacle.vector_z})`
            info['Relax Group'] = `(${obstacle.relax_groups.join(',')})`
          } else if (obstacle instanceof BoxObstacle) {
            info['Bounds'] = `X: [${obstacle.xmin.toFixed(2)}, ${obstacle.xmax.toFixed(2)}], Y: [${obstacle.ymin.toFixed(2)}, ${obstacle.ymax.toFixed(2)}], Z: [${obstacle.zmin.toFixed(2)}, ${obstacle.zmax.toFixed(2)}]`
            info['Relax Group'] = `(${obstacle.relax_groups.join(',')})`
          } else if (obstacle instanceof PointsObstacle) {
            info['Points Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(',')})`
          } else if (obstacle instanceof SpheresObstacle) {
            info['Spheres Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(',')})`
          } else if (obstacle instanceof SquaresObstacle) {
            info['Squares Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(',')})`
          } else if (obstacle instanceof SphereLineObstacle) {
            info['Points Count'] = obstacle.points.length
            info['Relax Group'] = `(${obstacle.relax_groups.join(',')})`
          }
          updateMeshInfo(info)
          // 添加 mesh 高亮
          const mesh = Object.values(obstacle.mesh_dict)[0] as THREE.Mesh
          if (mesh) {
            highlightMeshWithWireframe(mesh)
          }
        }
      })
       // find whether the name is grid
       if (project_manager.grid.name === name) {
        updateMeshInfo({
          Name: project_manager.grid.name,
          Type: 'Grid',
          Size: `(${project_manager.grid.xlength.toFixed(2)}, ${project_manager.grid.ylength.toFixed(2)}, ${project_manager.grid.zlength.toFixed(2)})`,
          Resolution: project_manager.grid.resolution.toFixed(2),
          GridSize: `(${project_manager.grid.xsize}, ${project_manager.grid.ysize}, ${project_manager.grid.zsize})`
        })
      }
    })
  }
}

const handleRightClickMeshSelect = (name: string) => {
    console.log('right click', name)
}

const handleRightClickControlPoint = (userData: any, x: number, y: number) => {
    console.log('right click control point', userData, x, y)
    // 将控制点右键事件传递给当前激活的组件
    if (activeCard.value === 'Smooth' && pathSmoothRef.value) {
        pathSmoothRef.value.handleRightClickControlPoint(userData, x, y)
    }
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
  updateMeshInfo,
  checkHighlightedMeshStatus
})

// 公共高亮+信息展示方法
function highlightMeshAndShowInfo({ type = undefined, projectName = undefined, obstacleName = undefined, pipeGroup = undefined, pipeName = undefined, meshName = undefined }: { type?: any, projectName?: any, obstacleName?: any, pipeGroup?: any, pipeName?: any, meshName?: any }) {
  // 统一 wireframe 高亮逻辑
  let meshToHighlight: THREE.Mesh | null = null
  if (type === 'Obstacle') {
    const project = rootObj.proj_dict.get(projectName)
    if (project) {
      const obstacle = project.obstacle_dict.get(obstacleName)
      if (obstacle) {
        meshToHighlight = Object.values(obstacle.mesh_dict)[0] as THREE.Mesh
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
        if (meshToHighlight) highlightMeshWithWireframe(meshToHighlight)
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
          meshToHighlight = Object.values(pipe.mesh_dict)[0] as THREE.Mesh
          updateMeshInfo({
            'Name': pipe.name,
            'Type': 'Pipe',
            'Position': `(${pipe.x.toFixed(2)}, ${pipe.y.toFixed(2)}, ${pipe.z.toFixed(2)})`,
            'Radius': pipe.radius.toFixed(2),
            'Direction': `(${pipe.vector_x}, ${pipe.vector_y}, ${pipe.vector_z})`,
            'Pipe Mode': pipe.pipe_mode,
            'Group': pipe.group_name
          })
          if (meshToHighlight) highlightMeshWithWireframe(meshToHighlight)
          return
        }
      }
    }
  } else if (type === 'MeshName' && meshName) {
    // 始终用 scene.getObjectByName 查找最新 mesh
    const mesh = scene.getObjectByName(meshName) as THREE.Mesh;
    if (mesh && mesh instanceof THREE.Mesh) {
      // 查找对应的 SegmentCell/ConnectorCell 对象
      let cellFound = false;
      let meshToHighlightLocal: THREE.Mesh | null = mesh
      rootObj.proj_dict.forEach((project_manager) => {
        project_manager.smooth_task_dict.forEach((smooth_task) => {
          smooth_task.group_cells.forEach((group) => {
            const [groupName, actualKey] = meshName.split('/');
            if (group.group_name !== groupName) return;
            const segment = group.seg_dict[actualKey];
            if (segment) {
              cellFound = true;
              updateMeshInfo({
                'Name': segment.name,
                'Type': 'Segment',
                'Project': segment.project_name,
                'Smooth Task': segment.smooth_name,
                'Group': segment.parent_cell.group_name,
                'Segment Index': segment.idx,
                'Group Index': segment.group_idx,
                'Control Points': segment.c_idxs.length,
                'B-spline Degree': segment.degree,
                'Density': segment.density,
                'Left Clamp': segment.left_clamp,
                'Right Clamp': segment.right_clamp,
                'Segment Scale': segment.segment_scale,
                'Analytical Curvature': segment.with_analytical_curvature_computed,
                'Analytical Tangent': segment.with_analytical_tangent_computed,
                'Obstacle Thickness': segment.cost_setting?.obstacle_thickness?.toFixed(3) || '0.000',
                'Path Thickness': segment.cost_setting?.path_thickness?.toFixed(3) || '0.000',
                'Cost Configs Count': segment.cost_setting?.cost_cfgs?.length || 0,
              });
            }
            const connector = group.connector_dict[actualKey];
            if (connector) {
              cellFound = true;
              updateMeshInfo({
                'Name': connector.name,
                'Type': 'Connector',
                'Project': connector.project_name,
                'Smooth Task': connector.smooth_name,
                'Group': connector.parent_cell.group_name,
                'Connector Index': connector.idx,
                'Group Index': connector.group_idx,
                'Left Segment': connector.left_seg_idx,
                'Right Segment': connector.right_seg_idx,
                'B-splineDegree': connector.degree,
                'Density': connector.density,
                'Left Clamp': connector.left_clamp,
                'Right Clamp': connector.right_clamp,
                'Analytical Curvature': connector.with_analytical_curvature_computed,
                'Analytical Tangent': connector.with_analytical_tangent_computed,
                'Obstacle Thickness': connector.cost_setting?.obstacle_thickness?.toFixed(3) || '0.000',
                'Path Thickness': connector.cost_setting?.path_thickness?.toFixed(3) || '0.000',
                'Cost Configs Count': connector.cost_setting?.cost_cfgs?.length || 0,
              });
            }
          });
        });
        // pipe
        project_manager.pipe_dict.forEach((pipe_dict) => {
          pipe_dict.forEach((pipe) => {
            if (pipe.name === meshName) {
              meshToHighlightLocal = Object.values(pipe.mesh_dict)[0] as THREE.Mesh
              updateMeshInfo({
                'Name': pipe.name,
                'Type': 'Pipe',
                'Position': `(${pipe.x.toFixed(2)}, ${pipe.y.toFixed(2)}, ${pipe.z.toFixed(2)})`,
                'Radius': pipe.radius.toFixed(2),
                'Direction': `(${pipe.vector_x}, ${pipe.vector_y}, ${pipe.vector_z})`,
                'Pipe Mode': pipe.pipe_mode,
                'Group': pipe.group_name
              })
            }
          })
        })
        // obstacle
        project_manager.obstacle_dict.forEach((obstacle) => {
          if (obstacle.name === meshName) {
            meshToHighlightLocal = Object.values(obstacle.mesh_dict)[0] as THREE.Mesh
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
          }
        })
        // grid
        if (project_manager.grid.name === meshName) {
          updateMeshInfo({
            'Name': project_manager.grid.name,
            'Type': 'Grid',
            'Size': `(${project_manager.grid.xlength.toFixed(2)}, ${project_manager.grid.ylength.toFixed(2)}, ${project_manager.grid.zlength.toFixed(2)})`,
            'Resolution': project_manager.grid.resolution.toFixed(2),
            'GridSize': `(${project_manager.grid.xsize}, ${project_manager.grid.ysize}, ${project_manager.grid.zsize})`
          })
        }
      });
      // 如果没有找到对应的 cell，但找到了 mesh，显示基本信息
      if (!cellFound && mesh) {
        updateMeshInfo({
          'Name': meshName,
          'Type': 'Mesh',
          'Found in Scene': 'Yes'
        });
      }
      if (meshToHighlightLocal) highlightMeshWithWireframe(meshToHighlightLocal)
    } else {
      updateMeshInfo(null)
    }
  } else {
    updateMeshInfo(null)
  }
}

</script>

<style>
/* 全局样式，不使用 scoped */
.el-dropdown-menu {
  min-width: 140px !important;
  background: rgba(255, 255, 255, 0.95) !important;
  backdrop-filter: blur(20px) !important;
  border: 1px solid rgba(226, 232, 240, 0.6) !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12) !important;
  padding: 8px !important;
}

.el-dropdown-menu__item {
  height: 40px !important;
  line-height: 40px !important;
  padding: 0 16px !important;
  background-color: transparent !important;
  border-radius: 8px !important;
  margin: 2px 0 !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  font-weight: 500 !important;
  color: #475569 !important;
  font-size: 14px !important;
}

.el-dropdown-menu__item:hover {
  background: rgba(99, 102, 241, 0.08) !important;
  color: #6366f1 !important;
  transform: translateX(4px) !important;
}

.el-dropdown-menu__item.is-active {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
  color: #6366f1 !important;
  border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

/* 箭头动画 */
.el-icon--right {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-size: 12px;
  color: #94a3b8;
}

.el-icon--right.is-active {
  transform: rotate(180deg);
  color: #6366f1;
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
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}

.main-content {
  flex: 1;
  display: flex;
  min-height: 0;
  overflow: hidden;
  width: 100%;
  height: calc(100vh - 36px);
  background: transparent;
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
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-right: 1px solid rgba(226, 232, 240, 0.8);
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
  z-index: 1;
}

/* 现代化滚动条样式 */
.sidebar::-webkit-scrollbar {
  width: 4px;
  height: 4px;
}

.sidebar::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #cbd5e1 0%, #94a3b8 100%);
  border-radius: 6px;
  transition: all 0.2s ease;
}

.sidebar::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, #94a3b8 0%, #64748b 100%);
}

.sidebar::-webkit-scrollbar-track {
  background: transparent;
}

.resize-handle {
  position: absolute;
  top: 0;
  right: -3px;
  width: 6px;
  height: 100%;
  cursor: col-resize;
  background: transparent;
  transition: all 0.2s ease;
}

.resize-handle:hover {
  background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.1) 50%, transparent 100%);
}

.viewer-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 600px;
  overflow: hidden;
  background: transparent;
}

.viewer {
  flex: 1;
  padding: 0;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  position: relative;
  z-index: 0;
  border-radius: 16px 0 0 0;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
}

.bottom-sidebar {
  position: relative;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-top: 1px solid rgba(226, 232, 240, 0.6);
  box-shadow: 0 -8px 32px rgba(0, 0, 0, 0.08);
  min-height: 100px;
  max-height: 400px;
  overflow: hidden;
  border-radius: 0 0 16px 0;
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
  border-right: 1px solid rgba(226, 232, 240, 0.6);
  background: rgba(248, 250, 252, 0.5);
}

.log-section {
  display: flex;
  flex-direction: column;
  flex: 2;
  min-width: 0;
  padding: 12px 20px;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-all;
  background: rgba(255, 255, 255, 0.3);
}

.info-header, .log-header {
  padding: 12px 20px;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
  border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  color: #1e293b;
  position: relative;
  font-size: 14px;
  letter-spacing: 0.025em;
}

.info-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  font-size: 13px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-row {
  display: flex;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(226, 232, 240, 0.4);
  transition: all 0.2s ease;
}

.info-row:hover {
  background: rgba(99, 102, 241, 0.02);
  border-radius: 6px;
  padding-left: 8px;
  padding-right: 8px;
}

.info-label {
  font-weight: 600;
  color: #475569;
  min-width: 120px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.info-value {
  color: #1e293b;
  word-break: break-all;
  font-weight: 500;
}

.no-info {
  color: #94a3b8;
  text-align: center;
  padding: 24px;
  font-style: italic;
  font-size: 14px;
}

.resize-handle-horizontal {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 6px;
  cursor: row-resize;
  background: transparent;
  transition: all 0.2s ease;
}

.resize-handle-horizontal:hover {
  background: linear-gradient(180deg, transparent 0%, rgba(99, 102, 241, 0.1) 50%, transparent 100%);
}

.log-content {
  flex: 1 1 0%;
  overflow-y: auto;
  min-height: 0;
}

.log-item {
  margin-bottom: 6px;
  color: #475569;
  border-bottom: 1px solid rgba(226, 232, 240, 0.3);
  padding-bottom: 6px;
  transition: all 0.2s ease;
  border-radius: 4px;
  padding: 4px 8px;
}

.log-item:hover {
  background: rgba(99, 102, 241, 0.02);
  color: #1e293b;
}

.log-item:last-child {
  margin-bottom: 0;
  border-bottom: none;
}

.card-layout {
  display: flex;
  gap: 8px;
  padding: 0 24px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  height: 64px;
  align-items: center;
  position: relative;
  z-index: 10;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
}

.menu-card {
  min-width: 100px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: #64748b;
  background: rgba(248, 250, 252, 0.8);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  user-select: none;
  border: 1px solid rgba(226, 232, 240, 0.6);
  position: relative;
  font-weight: 500;
  letter-spacing: 0.025em;
}

.menu-card:hover {
  background: rgba(99, 102, 241, 0.08);
  color: #6366f1;
  border-color: rgba(99, 102, 241, 0.2);
  transform: translateY(-1px);
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
}

.menu-card.active {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: #ffffff;
  border-color: #6366f1;
  font-weight: 600;
  z-index: 20;
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
  transform: translateY(-2px);
}

.menu-card.active::before {
  content: '';
  position: absolute;
  top: -2px;
  left: -2px;
  right: -2px;
  bottom: -2px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  border-radius: 14px;
  z-index: -1;
  opacity: 0.3;
  filter: blur(8px);
}

.menu-bar {
  border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
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
  height: 48px;
  padding: 0 24px;
  color: #475569;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-family: inherit;
  border-radius: 8px;
  margin: 0 4px;
  letter-spacing: 0.025em;
}

.el-dropdown-link:hover {
  background: rgba(99, 102, 241, 0.08);
  color: #6366f1;
  transform: translateY(-1px);
}

.el-dropdown-link:hover .el-icon--right {
  color: #6366f1;
  transform: translateY(-1px);
}

/* 现代化树形菜单样式 */
:deep(.el-tree) {
  background: transparent;
  padding: 12px 0;
}

:deep(.el-tree-node__content) {
  height: 36px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 8px;
  margin: 2px 8px;
  padding: 0 12px;
  font-size: 13px;
  font-weight: 500;
  color: #475569;
  letter-spacing: 0.025em;
}

:deep(.el-tree-node.is-current > .el-tree-node__content) {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
  color: #6366f1 !important;
  font-weight: 600 !important;
  border: 1px solid rgba(99, 102, 241, 0.2);
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
}

:deep(.el-tree-node__content:hover) {
  background: rgba(99, 102, 241, 0.05) !important;
  color: #6366f1 !important;
  transform: translateX(4px);
}

:deep(.el-tree-node__expand-icon) {
  color: #94a3b8;
  transition: all 0.3s ease;
}

:deep(.el-tree-node__expand-icon:hover) {
  color: #6366f1;
}

:deep(.el-tree-node__expand-icon.is-leaf) {
  color: transparent;
}

/* 添加分隔线 */
.menu-bar::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(226, 232, 240, 0.6) 50%, transparent 100%);
  z-index: 1;
}

/* 添加全局动画效果 */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInLeft {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* 为卡片添加进入动画 */
.menu-card {
  animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) both;
}

.menu-card:nth-child(1) { animation-delay: 0.1s; }
.menu-card:nth-child(2) { animation-delay: 0.2s; }
.menu-card:nth-child(3) { animation-delay: 0.3s; }
.menu-card:nth-child(4) { animation-delay: 0.4s; }
.menu-card:nth-child(5) { animation-delay: 0.5s; }
.menu-card:nth-child(6) { animation-delay: 0.6s; }

/* 为侧边栏添加进入动画 */
.sidebar {
  animation: slideInLeft 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

</style>