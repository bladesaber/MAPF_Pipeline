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
import { ref } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import type { SimulateCondition } from '@/types/data_type'

const props = defineProps<{
  modelValue: boolean
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
    init_velocity: [0, 0, 0]
  },
  inlet_cfg: new Map(),
  outlet_cfg: new Map(),
  wall_cfg: new Map()
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter a condition name')
    return
  }
  emit('generate', formData.value)
  // Reset form
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
</style> 