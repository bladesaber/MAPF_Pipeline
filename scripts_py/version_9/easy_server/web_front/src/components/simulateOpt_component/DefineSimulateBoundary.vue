<!-- DefineSimulateBoundary.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Boundary</span>
      <el-button 
        link
        class="close-button"
        @click="handleClose"
      >
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
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
            <el-table-column prop="name" label="Name" width="120" />
            <el-table-column prop="direction_x" label="Direction X" width="100">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.direction_x" 
                  :precision="4"
                  :step="0.1"
                  size="small"
                />
              </template>
            </el-table-column>
            <el-table-column prop="direction_y" label="Direction Y" width="100">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.direction_y" 
                  :precision="4"
                  :step="0.1"
                  size="small"
                />
              </template>
            </el-table-column>
            <el-table-column prop="direction_z" label="Direction Z" width="100">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.direction_z" 
                  :precision="4"
                  :step="0.1"
                  size="small"
                />
              </template>
            </el-table-column>
            <el-table-column prop="scale" label="Scale" width="100">
              <template #default="scope">
                <el-input-number 
                  v-model="scope.row.scale" 
                  :precision="4"
                  :step="0.1"
                  size="small"
                />
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
                  @click="handleDeleteInlet(scope.$index)"
                >
                  Delete
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
            <el-table-column prop="name" label="Name" width="120" />
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
            <el-table-column prop="name" label="Name" width="120" />
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
          <el-button @click="handleClose">Cancel</el-button>
          <el-button type="primary" @click="handleGenerate">Generate</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

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

const handleAddInlet = () => {
  inletData.value.push({
    name: `Inlet_${inletData.value.length + 1}`,
    direction_x: 0,
    direction_y: 0,
    direction_z: 0,
    scale: 1,
    temperature: 300
  })
}

const handleAddOutlet = () => {
  outletData.value.push({
    name: `Outlet_${outletData.value.length + 1}`,
    pressure: 0
  })
}

const handleAddWall = () => {
  wallData.value.push({
    name: `Wall_${wallData.value.length + 1}`,
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

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  emit('generate', {
    inlet: inletData.value,
    outlet: outletData.value,
    wall: wallData.value
  })
  // Reset data
  inletData.value = []
  outletData.value = []
  wallData.value = []
  handleClose()
}
</script>

<style scoped>
.right-sidebar {
  position: relative;
  width: 800px;
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
  margin-bottom: 16px;
  color: #606266;
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

.section-content {
  padding: 0 8px;
}

:deep(.el-table .cell) {
  padding: 8px;
}

:deep(.el-input-number.el-input-number--small) {
  width: 100%;
}

.button-container {
  position: sticky;
  bottom: 0;
  margin: 0 -16px;
  padding: 16px;
  background-color: #f5f7fa;
  border-top: 1px solid #dcdfe6;
}

.button-group {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.button-group .el-button {
  min-width: 90px;
}
</style> 