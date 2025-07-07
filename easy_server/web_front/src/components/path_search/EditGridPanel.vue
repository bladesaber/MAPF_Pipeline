<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Grid</span>
      <el-button 
        link
        class="close-button"
        @click="handleClose"
      >
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <!-- Grid Parameters Section -->
      <div class="section">
        <div class="section-header">
          <span>Grid Parameters</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="X Length">
              <el-input-number 
                v-model="formData.xlength"
                :precision="2"
                :step="0.1"
                :min="0.1"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="Y Length">
              <el-input-number 
                v-model="formData.ylength"
                :precision="2"
                :step="0.1"
                :min="0.1"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="Z Length">
              <el-input-number 
                v-model="formData.zlength"
                :precision="2"
                :step="0.1"
                :min="0.1"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="Resolution">
              <el-input-number 
                v-model="formData.resolution"
                :precision="2"
                :step="0.1"
                :min="0.1"
                style="width: 100%"
              />
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Button Container -->
      <div class="button-container">
        <div class="button-group">
          <el-button type="primary" @click="handleSave">Save</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ref,
  watch,
} from 'vue';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean,
  grid: {
    xlength: number,
    ylength: number,
    zlength: number,
    resolution: number
  }
}>()

const emit = defineEmits(['update:modelValue', 'save'])

const formData = ref({
  xlength: 0,
  ylength: 0,
  zlength: 0,
  resolution: 0.1
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleSave = () => {
  emit('save', { ...formData.value })
}

watch(() => props.grid, (newGrid) => {
  if (newGrid) {
    formData.value = {
      xlength: newGrid.xlength,
      ylength: newGrid.ylength,
      zlength: newGrid.zlength,
      resolution: newGrid.resolution
    }
  }
}, { immediate: true })
</script>

<style scoped>
.right-sidebar {
  position: fixed;
  right: 0;
  top: 116px;
  width: 450px;
  background: #fff;
  border-left: 1px solid #dcdfe6;
  height: calc(100vh - 116px);
  display: flex;
  flex-direction: column;
  z-index: 1000;
  box-shadow: -2px 0 8px rgba(0, 0, 0, 0.1);
}

.right-sidebar-header {
  padding: 12px 16px;
  border-bottom: 1px solid #dcdfe6;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #f5f7fa;
  flex: 0 0 auto;
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
  position: relative;
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 16px;
  padding-bottom: 64px;
  min-height: 0;
}

.right-sidebar-content::-webkit-scrollbar {
  width: 3px;
}

.right-sidebar-content::-webkit-scrollbar-track {
  background: transparent;
}

.right-sidebar-content::-webkit-scrollbar-thumb {
  background: #dcdfe6;
  border-radius: 3px;
}

.right-sidebar-content::-webkit-scrollbar-thumb:hover {
  background: #c0c4cc;
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