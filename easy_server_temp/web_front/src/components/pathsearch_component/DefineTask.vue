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
            <!-- Grid Size -->
            <el-form-item label="X Length">
              <el-input-number 
                v-model="formData.xsize"
                :min="0.1"
                :precision="2"
                :step="0.1"
                placeholder="X Length"
                class="size-input"
              />
            </el-form-item>

            <el-form-item label="Y Length">
              <el-input-number 
                v-model="formData.ysize"
                :min="0.1"
                :precision="2"
                :step="0.1"
                placeholder="Y Length"
                class="size-input"
              />
            </el-form-item>

            <el-form-item label="Z Length">
              <el-input-number 
                v-model="formData.zsize"
                :min="0.1"
                :precision="2"
                :step="0.1"
                placeholder="Z Length"
                class="size-input"
              />
            </el-form-item>

            <el-form-item label="Resolution">
              <el-input-number 
                v-model="formData.resolution"
                :min="0.01"
                :max="1"
                :precision="2"
                :step="0.01"
                placeholder="Resolution"
                class="size-input"
              />
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

interface FormData {
  name: string
  xsize: number
  ysize: number
  zsize: number
  resolution: number
}

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref<FormData>({
  name: '',
  xsize: 20.0,
  ysize: 20.0,
  zsize: 20.0,
  resolution: 0.5
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleApply = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter a task name')
    return
  }

  // Validate sizes are positive numbers
  if (formData.value.xsize <= 0) {
    ElMessage.warning('X size must be a positive number')
    return
  }
  if (formData.value.ysize <= 0) {
    ElMessage.warning('Y size must be a positive number')
    return
  }
  if (formData.value.zsize <= 0) {
    ElMessage.warning('Z size must be a positive number')
    return
  }

  // Validate resolution
  if (formData.value.resolution <= 0 || formData.value.resolution > 1) {
    ElMessage.warning('Resolution must be between 0 and 1')
    return
  }

  emit('generate', { ...formData.value })
  
  // Reset form
  formData.value = {
    name: '',
    xsize: 20.0,
    ysize: 20.0,
    zsize: 20.0,
    resolution: 0.5
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

.size-input {
  width: 100%;
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