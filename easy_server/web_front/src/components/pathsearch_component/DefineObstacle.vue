<!-- DefineTaskObstacle.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Obstacle</span>
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
            placeholder="Enter obstacle name"
            :maxlength="50"
            show-word-limit
          />
        </div>
      </div>

      <!-- Obstacle Type Section -->
      <div class="section">
        <div class="section-header">
          <span>Obstacle Type</span>
        </div>
        <div class="section-content">
          <el-select 
            v-model="formData.type"
            placeholder="Select obstacle type"
            style="width: 100%"
          >
            <el-option label="Cylinder" value="CylinderObstacle" />
            <el-option label="Box" value="BoxObstacle" />
          </el-select>
        </div>
      </div>

      <!-- Relax Groups Section -->
      <div class="section">
        <div class="section-header">
          <span>Relax Groups</span>
        </div>
        <div class="section-content">
          <el-select
            v-model="formData.relax_groups"
            multiple
            placeholder="Select relax groups"
            style="width: 100%"
          >
            <el-option
              v-for="group in availableRelaxGroups"
              :key="group"
              :label="group"
              :value="group"
            />
          </el-select>
        </div>
      </div>

      <!-- Resolution Parameter Section -->
      <div class="section">
        <div class="section-header">
          <span>Resolution</span>
        </div>
        <div class="section-content">
          <el-input-number 
            v-model="formData.resolution"
            :precision="2"
            :step="0.1"
            :min="0.1"
            style="width: 120px"
            placeholder="Resolution"
          />
        </div>
      </div>

      <!-- Dynamic Parameters Section -->
      <div v-if="formData.type" class="section">
        <div class="section-header">
          <span>Parameters</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <!-- Cylinder Parameters -->
            <template v-if="formData.type === 'CylinderObstacle'">
              <el-form-item label="Center Position">
                <div class="vector-input">
                  <el-input-number 
                    v-model="formData.x"
                    :precision="2"
                    :step="0.1"
                    placeholder="X"
                  />
                  <el-input-number 
                    v-model="formData.y"
                    :precision="2"
                    :step="0.1"
                    placeholder="Y"
                  />
                  <el-input-number 
                    v-model="formData.z"
                    :precision="2"
                    :step="0.1"
                    placeholder="Z"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Direction Vector">
                <div class="vector-input">
                  <el-input-number 
                    v-model="formData.vector_x"
                    :precision="2"
                    :step="0.1"
                    placeholder="X"
                  />
                  <el-input-number 
                    v-model="formData.vector_y"
                    :precision="2"
                    :step="0.1"
                    placeholder="Y"
                  />
                  <el-input-number 
                    v-model="formData.vector_z"
                    :precision="2"
                    :step="0.1"
                    placeholder="Z"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Radius">
                <el-input-number 
                  v-model="formData.radius"
                  :precision="2"
                  :step="0.1"
                  :min="0.1"
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item label="Height">
                <el-input-number 
                  v-model="formData.height"
                  :precision="2"
                  :step="0.1"
                  :min="0.1"
                  style="width: 120px"
                />
              </el-form-item>
            </template>

            <!-- Box Parameters -->
            <template v-if="formData.type === 'BoxObstacle'">
              <el-form-item label="X Range">
                <div class="range-input">
                  <el-input-number 
                    v-model="formData.xmin"
                    :precision="2"
                    :step="0.1"
                    placeholder="Min"
                  />
                  <span class="range-separator">to</span>
                  <el-input-number 
                    v-model="formData.xmax"
                    :precision="2"
                    :step="0.1"
                    placeholder="Max"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Y Range">
                <div class="range-input">
                  <el-input-number 
                    v-model="formData.ymin"
                    :precision="2"
                    :step="0.1"
                    placeholder="Min"
                  />
                  <span class="range-separator">to</span>
                  <el-input-number 
                    v-model="formData.ymax"
                    :precision="2"
                    :step="0.1"
                    placeholder="Max"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Z Range">
                <div class="range-input">
                  <el-input-number 
                    v-model="formData.zmin"
                    :precision="2"
                    :step="0.1"
                    placeholder="Min"
                  />
                  <span class="range-separator">to</span>
                  <el-input-number 
                    v-model="formData.zmax"
                    :precision="2"
                    :step="0.1"
                    placeholder="Max"
                  />
                </div>
              </el-form-item>
            </template>

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
import { ref, watch, computed } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  modelValue: boolean,
  currentTagDict: Map<string, string | string[] | Map<string, string|number> | number>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const availableRelaxGroups = computed(() => {
  const groups = props.currentTagDict.get('RelaxGroup')
  return Array.isArray(groups) ? groups : (groups ? [groups] : [])
})

const formData = ref({
  name: '',
  type: '',
  relax_groups: [] as string[],
  resolution: 0.5,
  // Cylinder parameters
  x: 0,
  y: 0,
  z: 0,
  vector_x: 0,
  vector_y: 0,
  vector_z: 1,
  radius: 1,
  height: 1,
  // Box parameters
  xmin: -1,
  xmax: 1,
  ymin: -1,
  ymax: 1,
  zmin: -1,
  zmax: 1
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const validateForm = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter an obstacle name')
    return false
  }

  if (!formData.value.type) {
    ElMessage.warning('Please select an obstacle type')
    return false
  }

  if (formData.value.type === 'CylinderObstacle') {
    if (formData.value.vector_x === 0 && formData.value.vector_y === 0 && formData.value.vector_z === 0) {
      ElMessage.warning('Direction vector cannot be zero')
      return false
    }
  } else if (formData.value.type === 'BoxObstacle') {
    if (formData.value.xmax <= formData.value.xmin) {
      ElMessage.warning('X maximum must be greater than minimum')
      return false
    }
    if (formData.value.ymax <= formData.value.ymin) {
      ElMessage.warning('Y maximum must be greater than minimum')
      return false
    }
    if (formData.value.zmax <= formData.value.zmin) {
      ElMessage.warning('Z maximum must be greater than minimum')
      return false
    }
  }

  return true
}

const handleGenerate = () => {
  if (!validateForm()) return

  const outputData = {
    name: formData.value.name,
    type: formData.value.type,
    resolution: formData.value.resolution,
    relax_groups: [...new Set(formData.value.relax_groups)],
    ...(formData.value.type === 'CylinderObstacle' ? {
      x: formData.value.x,
      y: formData.value.y,
      z: formData.value.z,
      vector_x: formData.value.vector_x,
      vector_y: formData.value.vector_y,
      vector_z: formData.value.vector_z,
      radius: formData.value.radius,
      height: formData.value.height
    } : formData.value.type === 'BoxObstacle' ? {
      xmin: formData.value.xmin,
      xmax: formData.value.xmax,
      ymin: formData.value.ymin,
      ymax: formData.value.ymax,
      zmin: formData.value.zmin,
      zmax: formData.value.zmax
    } : {})
  }

  emit('generate', outputData)
  
  // Reset form
  formData.value = {
    name: '',
    type: '',
    relax_groups: [],
    resolution: 1,
    x: 0,
    y: 0,
    z: 0,
    vector_x: 0,
    vector_y: 0,
    vector_z: 1,
    radius: 1,
    height: 1,
    xmin: -1,
    xmax: 1,
    ymin: -1,
    ymax: 1,
    zmin: -1,
    zmax: 1
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

.vector-input,
.range-input {
  display: flex;
  align-items: center;
  gap: 8px;
}

.vector-input :deep(.el-input-number),
.range-input :deep(.el-input-number) {
  width: 120px;
}

.range-separator {
  color: #909399;
  font-size: 14px;
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