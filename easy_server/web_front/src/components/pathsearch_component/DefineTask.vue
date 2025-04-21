<!-- DefineTask.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Task</span>
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
            placeholder="Enter task name"
            :maxlength="50"
            show-word-limit
          />
        </div>
      </div>

      <!-- Environment Definition Section -->
      <div class="section">
        <div class="section-header">
          <span>Environment Define:</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <!-- X Bounds -->
            <el-form-item label="X Range">
              <div class="range-input">
                <el-input-number 
                  v-model="formData.xmin"
                  :precision="2"
                  :step="0.1"
                  placeholder="Min"
                  class="range-input-field"
                />
                <span class="range-separator">to</span>
                <el-input-number 
                  v-model="formData.xmax"
                  :precision="2"
                  :step="0.1"
                  placeholder="Max"
                  class="range-input-field"
                />
              </div>
            </el-form-item>

            <!-- Y Bounds -->
            <el-form-item label="Y Range">
              <div class="range-input">
                <el-input-number 
                  v-model="formData.ymin"
                  :precision="2"
                  :step="0.1"
                  placeholder="Min"
                  class="range-input-field"
                />
                <span class="range-separator">to</span>
                <el-input-number 
                  v-model="formData.ymax"
                  :precision="2"
                  :step="0.1"
                  placeholder="Max"
                  class="range-input-field"
                />
              </div>
            </el-form-item>

            <!-- Z Bounds -->
            <el-form-item label="Z Range">
              <div class="range-input">
                <el-input-number 
                  v-model="formData.zmin"
                  :precision="2"
                  :step="0.1"
                  placeholder="Min"
                  class="range-input-field"
                />
                <span class="range-separator">to</span>
                <el-input-number 
                  v-model="formData.zmax"
                  :precision="2"
                  :step="0.1"
                  placeholder="Max"
                  class="range-input-field"
                />
              </div>
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Button Container -->
      <div class="button-container">
        <div class="button-group">
          <el-button type="primary" @click="handleApply">Apply</el-button>
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

const formData = ref({
  name: '',
  xmin: -10,
  xmax: 10,
  ymin: 0,
  ymax: 10,
  zmin: -10,
  zmax: 10
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleApply = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter a task name')
    return
  }

  // Validate ranges
  if (formData.value.xmax <= formData.value.xmin) {
    ElMessage.warning('X maximum must be greater than minimum')
    return
  }
  if (formData.value.ymax <= formData.value.ymin) {
    ElMessage.warning('Y maximum must be greater than minimum')
    return
  }
  if (formData.value.zmax <= formData.value.zmin) {
    ElMessage.warning('Z maximum must be greater than minimum')
    return
  }

  emit('generate', { ...formData.value })
  
  // Reset form
  formData.value = {
    name: '',
    xmin: -10,
    xmax: 10,
    ymin: -10,
    ymax: 10,
    zmin: 0,
    zmax: 10
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

.range-input {
  display: flex;
  align-items: center;
  gap: 8px;
}

.range-input-field {
  width: 160px;
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