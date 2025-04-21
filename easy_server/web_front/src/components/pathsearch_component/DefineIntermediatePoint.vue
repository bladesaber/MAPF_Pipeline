<!-- DefineIntermediatePoint.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Intermediate Point</span>
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
            placeholder="Enter point name"
            :maxlength="50"
            show-word-limit
          />
        </div>
      </div>

      <!-- Position Section -->
      <div class="section">
        <div class="section-header">
          <span>Position</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Coordinates">
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

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref({
  name: '',
  x: 0,
  y: 0,
  z: 0,
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter a point name')
    return
  }

  emit('generate', { ...formData.value })
  handleClose()
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