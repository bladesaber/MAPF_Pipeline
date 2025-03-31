<!-- DefineSimulateCondition.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Condition</span>
      <el-button 
        link
        class="close-button"
        @click="handleClose"
      >
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <!-- Name Section -->
      <div class="section">
        <div class="section-header">
          <span>Name</span>
        </div>
        <div class="section-content">
          <el-input
            v-model="formData.name"
            placeholder="Enter condition name"
            :maxlength="50"
            show-word-limit
          />
        </div>
      </div>

      <!-- Physical Section -->
      <div class="section">
        <div class="section-header">
          <span>Physical</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Viscosity">
              <el-input-number 
                v-model="formData.physical_cfg.viscosity"
                :precision="4"
                :step="0.0001"
              />
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Material Section -->
      <div class="section">
        <div class="section-header">
          <span>Material</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Density">
              <el-input-number 
                v-model="formData.material_cfg.density"
                :precision="1"
                :step="0.1"
              />
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Run Section -->
      <div class="section">
        <div class="section-header">
          <span>Run</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Run Iterations">
              <el-input-number 
                v-model="formData.run_cfg.run_iter"
                :min="1"
                :step="100"
              />
            </el-form-item>
            <el-form-item label="Convergence Field">
              <el-input 
                v-model="formData.run_cfg.conv_field"
                disabled
              />
            </el-form-item>
            <el-form-item label="Convergence Level">
              <el-input-number 
                v-model="formData.run_cfg.conv_level"
                :step="1"
              />
            </el-form-item>
            <el-form-item label="Output Write Frequency">
              <el-input-number 
                v-model="formData.run_cfg.output_wrt_freq"
                :min="1"
                :step="10"
              />
            </el-form-item>
            <el-form-item>
              <el-checkbox v-model="formData.run_cfg.restart_sol">Restart Solution</el-checkbox>
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Initialization Section -->
      <div class="section">
        <div class="section-header">
          <span>Initialization</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Body Force">
              <div class="vector-input">
                <el-input-number 
                  v-model="formData.initialization_cfg.body_force[0]"
                  :precision="4"
                  :step="0.0001"
                  placeholder="X"
                />
                <el-input-number 
                  v-model="formData.initialization_cfg.body_force[1]"
                  :precision="4"
                  :step="0.0001"
                  placeholder="Y"
                />
                <el-input-number 
                  v-model="formData.initialization_cfg.body_force[2]"
                  :precision="4"
                  :step="0.0001"
                  placeholder="Z"
                />
              </div>
            </el-form-item>
            <el-form-item label="Initial Velocity">
              <div class="vector-input">
                <el-input-number 
                  v-model="formData.initialization_cfg.init_velocity[0]"
                  :precision="4"
                  :step="0.0001"
                  placeholder="X"
                />
                <el-input-number 
                  v-model="formData.initialization_cfg.init_velocity[1]"
                  :precision="4"
                  :step="0.0001"
                  placeholder="Y"
                />
                <el-input-number 
                  v-model="formData.initialization_cfg.init_velocity[2]"
                  :precision="4"
                  :step="0.0001"
                  placeholder="Z"
                />
              </div>
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Inlet Section -->
      <div class="section">
        <div class="section-header">
          <div class="header-with-button">
            <span>Inlet</span>
            <el-button type="primary" size="small" @click="handleAddInlet">Add Inlet</el-button>
          </div>
        </div>
        <div class="section-content">
          <el-table :data="inletData" style="width: 100%">
            <el-table-column prop="name" label="Name" width="120">
              <template #default="scope">
                <el-select v-model="scope.row.name" placeholder="Select group" size="small">
                  <el-option
                    v-for="group in availableGroups"
                    :key="group"
                    :label="group"
                    :value="group"
                  />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="Direction" width="180">
              <template #default="scope">
                <div class="vector-input-compact">
                  <el-input-number 
                    v-model="scope.row.direction_x" 
                    :precision="2"
                    :step="0.1"
                    size="small"
                    placeholder="X"
                    controls-position="right"
                  />
                  <el-input-number 
                    v-model="scope.row.direction_y" 
                    :precision="2"
                    :step="0.1"
                    size="small"
                    placeholder="Y"
                    controls-position="right"
                  />
                  <el-input-number 
                    v-model="scope.row.direction_z" 
                    :precision="2"
                    :step="0.1"
                    size="small"
                    placeholder="Z"
                    controls-position="right"
                  />
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="scale" label="Scale" width="80">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.scale" 
                  :precision="2"
                  :step="0.1"
                  size="small"
                  controls-position="right"
                />
              </template>
            </el-table-column>
            <el-table-column prop="temperature" label="Temp" width="80">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.temperature" 
                  :precision="0"
                  :step="1"
                  size="small"
                  controls-position="right"
                />
              </template>
            </el-table-column>
            <el-table-column label="Action" width="70" fixed="right">
              <template #default="scope">
                <el-button
                  type="danger"
                  size="small"
                  @click="handleDeleteInlet(scope.$index)"
                >
                  Del
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>

      <!-- Outlet Section -->
      <div class="section">
        <div class="section-header">
          <div class="header-with-button">
            <span>Outlet</span>
            <el-button type="primary" size="small" @click="handleAddOutlet">Add Outlet</el-button>
          </div>
        </div>
        <div class="section-content">
          <el-table :data="outletData" style="width: 100%">
            <el-table-column prop="name" label="Name" width="120">
              <template #default="scope">
                <el-select v-model="scope.row.name" placeholder="Select group" size="small">
                  <el-option
                    v-for="group in availableGroups"
                    :key="group"
                    :label="group"
                    :value="group"
                  />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column prop="pressure" label="Pressure" width="120">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.pressure" 
                  :precision="2"
                  :step="1"
                  size="small"
                />
              </template>
            </el-table-column>
            <el-table-column label="Action" width="100" fixed="right">
              <template #default="scope">
                <el-button
                  type="danger"
                  size="small"
                  @click="handleDeleteOutlet(scope.$index)"
                >
                  Delete
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>

      <!-- Wall Section -->
      <div class="section">
        <div class="section-header">
          <div class="header-with-button">
            <span>Wall</span>
            <el-button type="primary" size="small" @click="handleAddWall">Add Wall</el-button>
          </div>
        </div>
        <div class="section-content">
          <el-table :data="wallData" style="width: 100%">
            <el-table-column prop="name" label="Name" width="120">
              <template #default="scope">
                <el-select v-model="scope.row.name" placeholder="Select group" size="small">
                  <el-option
                    v-for="group in availableGroups"
                    :key="group"
                    :label="group"
                    :value="group"
                  />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column prop="temperature" label="Temperature" width="120">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.temperature" 
                  :precision="2"
                  :step="1"
                  size="small"
                />
              </template>
            </el-table-column>
            <el-table-column label="Action" width="100" fixed="right">
              <template #default="scope">
                <el-button
                  type="danger"
                  size="small"
                  @click="handleDeleteWall(scope.$index)"
                >
                  Delete
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>

      <!-- Button Container -->
      <div class="button-container">
        <div class="button-group">
          <el-button type="primary" @click="handleGenerate">Generate</el-button>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, inject, computed } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import type { SimulateCondition } from '@/types/data_type'
import { SimulateRootObj } from '@/classes/simulateOpt_classes/simulate_entity'

const rootObj = inject<SimulateRootObj>('simulateRootObj')
const props = defineProps<{
  modelValue: boolean,
  currentTagDict: Map<string, string|string[]>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref<SimulateCondition>({
  name: '',
  physical_cfg: {
    viscosity: 0.04
  },
  material_cfg: {
    density: 887.0
  },
  run_cfg: {
    run_iter: 1000,
    conv_field: 'RMS_PRESSURE',
    conv_level: -6,
    conv_filename: 'conv_history',
    restart_filename: 'restart_flow.dat',
    read_binary_restart: true,
    restart_sol: false,
    with_screen_output: false,
    output_wrt_freq: 100
  },
  initialization_cfg: {
    body_force: [0, 0, 0],
    init_velocity: [1, 0, 0]
  },
  inlet_cfg: new Map(),
  outlet_cfg: new Map(),
  wall_cfg: new Map()
})

interface InletItem {
  name: string
  direction_x: number
  direction_y: number
  direction_z: number
  scale: number
  temperature: number
}

interface OutletItem {
  name: string
  pressure: number
}

interface WallItem {
  name: string
  temperature: number
}

const inletData = ref<InletItem[]>([])
const outletData = ref<OutletItem[]>([])
const wallData = ref<WallItem[]>([])

const availableGroups = computed(() => {
  const simulate_entity = rootObj?.get_simulate_entity(props.currentTagDict.get('SimulateEntity') as string)
  return simulate_entity?.get_entity_group_names() || []
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleAddInlet = () => {
  inletData.value.push({
    name: '',
    direction_x: 0,
    direction_y: 0,
    direction_z: 0,
    scale: 1,
    temperature: 300
  })
}

const handleAddOutlet = () => {
  outletData.value.push({
    name: '',
    pressure: 0
  })
}

const handleAddWall = () => {
  wallData.value.push({
    name: '',
    temperature: 300
  })
}

const handleDeleteInlet = (index: number) => {
  inletData.value.splice(index, 1)
}

const handleDeleteOutlet = (index: number) => {
  outletData.value.splice(index, 1)
}

const handleDeleteWall = (index: number) => {
  wallData.value.splice(index, 1)
}

const handleGenerate = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter a condition name')
    return
  }

  if (inletData.value.some(inlet => !inlet.name)) {
    ElMessage.warning('Please select names for all inlets')
    return
  }
  if (outletData.value.some(outlet => !outlet.name)) {
    ElMessage.warning('Please select names for all outlets')
    return
  }
  if (wallData.value.some(wall => !wall.name)) {
    ElMessage.warning('Please select names for all walls')
    return
  }

  // Convert table data to Maps with correct types
  formData.value.inlet_cfg = new Map(inletData.value.map(inlet => [
    inlet.name, 
    {
      type: "VELOCITY_INLET" as const,
      temperature: inlet.temperature,
      vel_scale: inlet.scale,
      direction: [inlet.direction_x, inlet.direction_y, inlet.direction_z]
    }
  ]))

  formData.value.outlet_cfg = new Map(outletData.value.map(outlet => [
    outlet.name,
    {
      type: "PRESSURE_OUTLET" as const,
      pressure: outlet.pressure
    }
  ]))

  formData.value.wall_cfg = new Map(wallData.value.map(wall => [
    wall.name,
    {
      type: "WALL" as const,
      temperature: wall.temperature
    }
  ]))

  emit('generate', formData.value)
  
  // Reset form and tables
  formData.value = {
    name: '',
    physical_cfg: { viscosity: 0.04 },
    material_cfg: { density: 887.0 },
    run_cfg: {
      run_iter: 1000,
      conv_field: 'RMS_PRESSURE',
      conv_level: -6,
      conv_filename: 'conv_history',
      restart_filename: 'restart_flow.dat',
      read_binary_restart: true,
      restart_sol: false,
      with_screen_output: false,
      output_wrt_freq: 100
    },
    initialization_cfg: {
      body_force: [0, 0, 0],
      init_velocity: [0, 0, 0]
    },
    inlet_cfg: new Map(),
    outlet_cfg: new Map(),
    wall_cfg: new Map()
  }
  inletData.value = []
  outletData.value = []
  wallData.value = []
}
</script>

<style scoped>
.right-sidebar {
  position: relative;
  width: 450px;
  background: #fff;
  border-left: 1px solid #dcdfe6;
  height: 100%;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.right-sidebar-header {
  padding: 12px 16px;
  border-bottom: 1px solid #dcdfe6;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #f5f7fa;
  flex-shrink: 0;
}

.right-sidebar-header span {
  font-weight: 500;
}

.close-button {
  padding: 4px;
}

.close-button:hover {
  color: #409eff;
}

.right-sidebar-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.section {
  margin-bottom: 24px;
}

.section-header {
  font-weight: 500;
  margin-bottom: 16px;
  color: #606266;
}

.section-content {
  padding: 0 8px;
}

.vector-input {
  display: flex;
  gap: 8px;
}

.vector-input :deep(.el-input-number) {
  width: 120px;
}

:deep(.el-form-item) {
  margin-bottom: 16px;
}

:deep(.el-form-item__label) {
  padding-bottom: 4px;
  line-height: 1.5;
}

.button-container {
  padding: 16px;
  border-top: 1px solid #dcdfe6;
  background-color: #f5f7fa;
}

.button-group {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.button-group .el-button {
  min-width: 90px;
}

.header-with-button {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-with-button span {
  font-weight: 500;
  font-size: 16px;
}

:deep(.el-table .cell) {
  padding: 8px;
}

:deep(.el-input-number.el-input-number--small) {
  width: 100%;
}

.vector-input-compact {
  display: flex;
  gap: 4px;
}

.vector-input-compact :deep(.el-input-number) {
  width: 56px;
}

.vector-input-compact :deep(.el-input-number .el-input__wrapper) {
  padding: 0 8px;
}

:deep(.el-input-number.el-input-number--small .el-input-number__decrease),
:deep(.el-input-number.el-input-number--small .el-input-number__increase) {
  width: 20px;
}
</style> 