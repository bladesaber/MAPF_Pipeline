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
        <el-menu-item index="3-1">Optimization</el-menu-item>
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
        :currentTagDict="currentTagDict"
        @generate="handleGenerateGroup"
      />

      <DefineSimulateCondition
        ref="defineSimulateConditionRef"
        v-model="defineSimulateConditionVisible"
        :currentTagDict="currentTagDict"
        @generate="handleGenerateCondition"
      />

      <GenerateSimulateMesh
        ref="generateSimulateMeshRef"
        v-model="generateSimulateMeshVisible"
        :currentTagDict="currentTagDict"
        @generate="handleGenerateMesh"
      />

    </div>

    <!-- Tree Node Context Menu -->
    <div 
      v-show="nodeEntityContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('nodeEntity/HideEntity')">Hide Entity</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('nodeEntity/ShowEntity')">Show Entity</el-menu-item>
      </el-menu>
    </div>

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
        <el-menu-item @click="handleTreeContextMenu('group/HideGroup')">Hide Group</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('group/ShowGroup')">Show Group</el-menu-item>
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
        <el-menu-item @click="handleTreeContextMenu('condition/Simulate')">Simulate</el-menu-item>

      </el-menu>
    </div>

    <div 
      v-show="nodeNetworkContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('nodeNetwork/HideNetwork')">Hide</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('nodeNetwork/ShowNetwork')">Show</el-menu-item>
      </el-menu>
    </div>

    <div 
      v-show="nodeResultContextMenuVisible"
      class="context-menu"
      :style="contextMenuStyle"
    >
      <el-menu>
        <el-menu-item @click="handleTreeContextMenu('nodeResult/HideResult')">Hide</el-menu-item>
        <el-menu-item @click="handleTreeContextMenu('nodeResult/ShowResult')">Show</el-menu-item>
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
      <label>{{ progressMsg }}</label>
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
import { SimulateRootObj, SimulateEntity, CadEntity, SimulateNetwork } from '../classes/simulateOpt_classes/simulate_entity.ts'
import type { SimulateTreeNode, SimulateCondition } from '@/types/data_type.ts'
import DefineSimulateGroup from '../components/simulateOpt_component/DefineSimulateGroup.vue'
import DefineSimulateCondition from '@/components/simulateOpt_component/DefineSimulateCondition.vue'
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

const nodeEntityContextMenuVisible = ref(false)
const nodeGroupsContextMenuVisible = ref(false)
const groupContextMenuVisible = ref(false)
const nodeConditionsContextMenuVisible = ref(false)
const conditionContextMenuVisible = ref(false)
const nodeNetworkContextMenuVisible = ref(false)
const nodeResultContextMenuVisible = ref(false)
const meshContextMenuVisible = ref(false)
const tree_visable_dict = {
  'nodeEntityContextMenuVisible': nodeEntityContextMenuVisible,
  'nodeGroupsContextMenuVisible': nodeGroupsContextMenuVisible,
  'groupContextMenuVisible': groupContextMenuVisible,
  'nodeConditionsContextMenuVisible': nodeConditionsContextMenuVisible,
  'conditionContextMenuVisible': conditionContextMenuVisible,
  'nodeNetworkContextMenuVisible': nodeNetworkContextMenuVisible,
  'nodeResultContextMenuVisible': nodeResultContextMenuVisible,
  'meshContextMenuVisible': meshContextMenuVisible,
}

const handleLeftTreeClick = (data: any) => {
  setVisableDict('nodeGroupsContextMenuVisible', tree_visable_dict, true, false)
  console.log('Left tree click:', data)
}

const currentTagDict = ref(new Map<string, string| string[]>())
const handleRightTreeClick = (event: MouseEvent, data: any) => {
  event.preventDefault()
  contextMenuPosition.x = event.clientX
  contextMenuPosition.y = event.clientY

  currentTagDict.value = data.tagDict
  currentTagDict.value.set('label', data.label)

  if (data.type === 'Node_Entities') {
    setVisableDict('nodeEntityContextMenuVisible', tree_visable_dict, true, true)
  }
  else if (data.type === 'Node_Groups') {
    setVisableDict('nodeGroupsContextMenuVisible', tree_visable_dict, true, true)
  } 
  else if (data.type === 'Group') {
    setVisableDict('groupContextMenuVisible', tree_visable_dict, true, true)
  } 
  else if (data.type === 'Node_Conditions') {
    setVisableDict('nodeConditionsContextMenuVisible', tree_visable_dict, true, true)
  } 
  else if ([
    'Condition', 'PhysicalCfg', 'MaterialCfg', 'RunCfg', 'InitializationCfg',
    'InletCfg', 'OutletCfg', 'WallCfg'
  ].includes(data.type)) {
    setVisableDict('conditionContextMenuVisible', tree_visable_dict, true, true)
  }
  else if (data.type === 'Node_Network') {
    setVisableDict('nodeNetworkContextMenuVisible', tree_visable_dict, true, true)
  }
  else if (data.type === 'SimulateResult') {
    setVisableDict('nodeResultContextMenuVisible', tree_visable_dict, true, true)
  }

  console.log('Right tree click:', data)
}

const handleTreeContextMenu = (mode: string) => {
  setVisableDict('nodeGroupsContextMenuVisible', tree_visable_dict, true, false)
  
  if (mode == 'nodeGroups/AddGroup') {
    setVisableDict('defineSimulateGroupVisible', rightSideBar_visableDict, true)
  }
  else if (mode == 'nodeEntity/HideEntity') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      simulate_entity.hide()
    }
  }
  else if (mode == 'nodeEntity/ShowEntity') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      simulate_entity.show()
    }
  }

  else if (mode == 'group/HideGroup') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      simulate_entity.hide_group(currentTagDict.value.get('label') as string)
    }
  }
  else if (mode == 'group/ShowGroup') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      simulate_entity.show_group(currentTagDict.value.get('label') as string)
    }
  }
  else if (mode == 'nodeConditions/AddCondition'){
    setVisableDict('defineSimulateConditionVisible', rightSideBar_visableDict, true)
  }
  else if (mode == 'nodeGroups/AddMesh'){
    setVisableDict('generateSimulateMeshVisible', rightSideBar_visableDict, true)
  }
  else if (mode == 'condition/DeleteCondition'){
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      simulate_entity.remove_simulate_condition(currentTagDict.value.get('label') as string)
      updateTreeData()
    }
  }
  else if (mode == 'condition/Simulate') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      if (simulate_entity.has_simulate_condition(currentTagDict.value.get('label') as string)) {
        handleSimulate(simulate_entity, currentTagDict.value.get('label') as string)
      }
      else {
        ElMessage.error('No condition found')
      }
    }
  }
  else if (mode == 'condition/Simulate'){
    setVisableDict('defineSimulateBoundaryVisible', rightSideBar_visableDict, true)
  }
  else if (mode == 'nodeNetwork/HideNetwork') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      simulate_entity.hide_network()
    }
  }
  else if (mode == 'nodeNetwork/ShowNetwork') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      simulate_entity.show_network()
    }
  }
  else if (mode == 'nodeResult/HideResult') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      const condition_name = currentTagDict.value.get('SimulateCondition') as string
      if (simulate_entity.has_simulate_result(condition_name)) {
        simulate_entity.hide_result(condition_name)
      }
    }
  }
  else if (mode == 'nodeResult/ShowResult') {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      const condition_name = currentTagDict.value.get('SimulateCondition') as string
      if (simulate_entity.has_simulate_result(condition_name)) {
        simulate_entity.show_result(condition_name)
      }
    }
  }
}

// ---------------- Mesh Click Event Handling ----------------
const defineSimulateGroupRef = ref()
const handleLeftClickMeshSelect = (name: string) => {
  if (defineSimulateGroupVisible.value) {
    const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
    if (simulate_entity) {
      if (simulate_entity.has_entity(name)) {
        defineSimulateGroupRef.value?.addItem({name: name})
      }
    }
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
const progressStatus = ref<''| 'success'| 'exception'| 'warning'>('')
const progressMsg = ref('Progressing...')
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
  progressMsg.value = 'Extracting faces...'

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

    let shapeData: any = {}

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = new TextDecoder().decode(value)
      const data = JSON.parse(text)
      
      if (data.total_finished) {
        break
      }

      if (data.name === 'extract_step_face') {
        // Update progress during face extraction
        progressMsg.value = data.msg
        if (!data.is_finished) {
          progressPercentage.value = data.progress
        }
      } else if (data.name === 'return_extract_step_face') {
        // Collect shape data during return phase
        progressMsg.value = 'Processing face data...'
        progressPercentage.value = data.progress
        if (data.data) {
          const key = Object.keys(data.data)[0]
          shapeData[key] = data.data[key]
        }
      }
    }

    if (Object.keys(shapeData).length > 0) {
      progressStatus.value = 'success'
      progressMsg.value = 'Processing complete'
      handleSTEPData(file.name, shapeData, new Uint8Array(fileContent))
    } else {
      throw new Error('No shape data received')
    }

  } catch (error: unknown) {
    console.error('Error processing file:', error)
    progressStatus.value = 'exception'
    progressPercentage.value = 0
    progressMsg.value = 'Error processing file'
  } finally {
    input.value = ''
    // Close progress dialog after a short delay
    setTimeout(() => {
      progressVisible.value = false
      progressPercentage.value = 0
      progressStatus.value = ''
      progressMsg.value = 'Processing...'
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
const handleGenerateGroup = (data: any) => {
  const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
  if (simulate_entity) {
    simulate_entity.define_entity_group(data.name, data.items.map((item: any) => item.name))
    updateTreeData()
  }
}

const handleGenerateCondition = (data: any) => {
  const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
  if (simulate_entity) {
    // Convert Proxy to SimulateCondition object
    const condition: SimulateCondition = {
      name: data.name,
      physical_cfg: { ...data.physical_cfg },
      material_cfg: { ...data.material_cfg },
      run_cfg: { ...data.run_cfg },
      initialization_cfg: {
        body_force: [...data.initialization_cfg.body_force],
        init_velocity: [...data.initialization_cfg.init_velocity]
      },
      inlet_cfg: new Map(
        Array.from(data.inlet_cfg.entries() as Iterable<[string, any]>).map(([key, value]) => [
          key,
          {
            type: value.type,
            temperature: value.temperature,
            vel_scale: value.vel_scale,
            direction: [...value.direction]
          }
        ])
      ),
      outlet_cfg: new Map(
        Array.from(data.outlet_cfg.entries() as Iterable<[string, any]>).map(([key, value]) => [
          key,
          {
            type: value.type,
            pressure: value.pressure
          }
        ])
      ),
      wall_cfg: new Map(
        Array.from(data.wall_cfg.entries() as Iterable<[string, any]>).map(([key, value]) => [
          key,
          {
            type: value.type,
            temperature: value.temperature
          }
        ])
      )
    }
    simulate_entity.add_simulate_condition(condition)
    updateTreeData()
  }
}

const handleGenerateMesh = async (data: any) => {
  const simulate_entity = simulateRootObj.get_simulate_entity(currentTagDict.value.get('SimulateEntity') as string)
  if (simulate_entity) {
    const fileContent = simulate_entity.get_storage_content()
    if (!fileContent) {
      ElMessage.error('No file content found')
      return
    }

    // Reset progress state
    progressVisible.value = true
    progressPercentage.value = 0
    progressStatus.value = ''
    progressMsg.value = 'Generating mesh...'

    // Create FormData and append the file
    const formData = new FormData()
    const blob = new Blob([fileContent], { type: 'application/step' })
    formData.append('file', blob, `${simulate_entity.name}.step`)

    // Add mesh size parameters as numbers
    formData.append('max_size', data.maxMeshSize.toString())
    formData.append('min_size', data.minMeshSize.toString())

    // Convert Map to object for JSON serialization
    const groupCfgs: {[key: string]: string[]} = {}
    simulate_entity.entity_groups.forEach((value, key) => {
      groupCfgs[key] = value
    })
    formData.append('group_cfgs', JSON.stringify(groupCfgs))

    try {
      const response = await fetch('http://192.168.110.54:9998/cad/make_su2_mesh', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) throw new Error('Failed to generate mesh')

      const reader = response.body?.getReader()
      if (!reader) throw new Error('Failed to get response reader')

      let meshInfo: any = null
      let cells: number[][] = []
      let points: number[][] = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = new TextDecoder().decode(value)
        const data = JSON.parse(text)

        if (data.total_finished) {
          break
        }

        if (data.name === 'generate_mesh') {
          // Update progress during mesh generation
          progressMsg.value = data.msg
          if (!data.is_finished) {
            progressPercentage.value = data.progress
          }
          else {
            progressPercentage.value = 100
            progressStatus.value = 'success'
          }
        } else if (data.name === 'return_mesh_info') {
          progressMsg.value = 'Processing mesh info...'
          meshInfo = data.data
          progressPercentage.value = 100
        } else if (data.name === 'return_cells') {
          progressMsg.value = 'Processing cells data...'
          progressPercentage.value = data.progress
          if (data.data) {
            cells.push(...data.data)
          }
        } else if (data.name === 'return_points') {
          progressMsg.value = 'Processing points data...'
          progressPercentage.value = data.progress
          if (data.data) {
            points.push(...data.data)
          }
        }
      }

      if (meshInfo && cells.length > 0 && points.length > 0) {
        // Add new mesh
        const marker_dict = new Map<string, Map<string, number|number[][]>>()
        Object.entries(meshInfo.marker_dict as Record<string, Record<string, number|number[][]>>).forEach(([k, v]) => {
          marker_dict.set(k, new Map(Object.entries(v)))
        })

        simulate_entity.add_simulate_network(
          points, cells, marker_dict, simulateRootObj
        )

        progressStatus.value = 'success'
        progressMsg.value = 'Mesh generated successfully'
        ElMessage.success('Mesh generated successfully')
        updateTreeData()
      } else {
        throw new Error('Incomplete mesh data received')
      }

    } catch (error: any) {
      console.error('Error generating mesh:', error)
      progressStatus.value = 'exception'
      progressMsg.value = 'Error generating mesh'
      ElMessage.error(error.message || 'Failed to generate mesh')
    } finally {
      // Close progress dialog after a short delay
      setTimeout(() => {
        progressVisible.value = false
        progressPercentage.value = 0
        progressStatus.value = ''
        progressMsg.value = 'Processing...'
      }, 2000)
    }
  }
}

// ---------------- Simulate ----------------
const handleSimulate = async (simulate_entity: SimulateEntity, condition_name: string) => {
  const network = simulate_entity.simulate_network
  if (!network) {
    ElMessage.error('No mesh network found')
    return
  }

  // Convert nested Maps in marker_dict to objects
  const markerDictObj: any = {}
  network.marker_dict.forEach((value, key) => {
    markerDictObj[key] = Object.fromEntries(value)
  })

  // Get condition and convert its Maps to objects
  const condition = simulate_entity.simulate_conditions.get(condition_name)
  if (!condition) {
    ElMessage.error('Condition not found')
    return
  }

  const conditionObj = {
    ...condition,
    inlet_cfg: Object.fromEntries(condition.inlet_cfg),
    outlet_cfg: Object.fromEntries(condition.outlet_cfg),
    wall_cfg: Object.fromEntries(condition.wall_cfg)
  }

  const data = {
    name: simulate_entity.name,
    condition_name: condition_name,
    n_dim: network.n_dim,
    n_elem: network.n_elem,
    n_point: network.n_point,
    n_mark: network.n_mark,
    cells: network.cells,
    points: network.points,
    marker_dict: markerDictObj,
    condition: conditionObj
  }

  // Reset progress state
  progressVisible.value = true
  progressPercentage.value = 0
  progressStatus.value = ''
  progressMsg.value = 'Starting simulation...'

  try {
    const response = await fetch('http://192.168.110.54:9998/simulate/simulate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) throw new Error('Simulation failed')

    const reader = response.body?.getReader()
    if (!reader) throw new Error('Failed to get response reader')

    let velocity_data: number[][] = []
    let pressure_data: number[] = []

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = new TextDecoder().decode(value)
      const data = JSON.parse(text)

      if (data.total_finished) {
        break
      }

      if (data.name === 'simulate') {
        // Update progress during simulation
        progressMsg.value = data.msg
        if (!data.is_finished) {
          progressPercentage.value = data.progress
        }
        else {
          progressPercentage.value = 100
          progressStatus.value = 'success'
        }
      } else if (data.name === 'return_velocity_data') {
        // Collect velocity data during return phase
        progressMsg.value = 'Processing velocity data...'
        progressPercentage.value = data.progress
        if (data.data) {
          velocity_data.push(...data.data)
        }
      } else if (data.name === 'return_pressure_data') {
        // Collect pressure data during return phase
        progressMsg.value = 'Processing pressure data...'
        progressPercentage.value = data.progress
        if (data.data) {
          pressure_data.push(...data.data)
        }
      }
    }

    if (velocity_data.length > 0 && pressure_data.length > 0) {
      progressStatus.value = 'success'
      progressMsg.value = 'Simulation complete'
      simulate_entity.add_simulate_result(condition_name, velocity_data, pressure_data, simulateRootObj)
      updateTreeData()
    } else {
      throw new Error('Incomplete simulation data received')
    }

  } catch (error: unknown) {
    console.error('Error in simulation:', error)
    progressStatus.value = 'exception'
    progressMsg.value = 'Simulation failed'
    ElMessage.error(error instanceof Error ? error.message : 'Simulation failed')
  } finally {
    // Close progress dialog after a short delay
    setTimeout(() => {
      progressVisible.value = false
      progressPercentage.value = 0
      progressStatus.value = ''
      progressMsg.value = 'Processing...'
    }, 2000)
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