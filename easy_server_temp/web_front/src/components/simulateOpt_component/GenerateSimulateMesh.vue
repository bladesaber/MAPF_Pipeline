<!-- GenerateSimulateMesh.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Generate Mesh</span>
      <el-button 
        link
        class="close-button"
        @click="handleClose"
      >
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <!-- Mesh Size Section -->
      <div class="section">
        <div class="section-header">
          <span>Mesh Size Settings</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Max Mesh Size">
              <el-input-number
                v-model="formData.maxMeshSize"
                :min="0.001"
                :precision="3"
                :step="0.1"
                placeholder="Enter max mesh size"
              />
            </el-form-item>
            <el-form-item label="Min Mesh Size">
              <el-input-number
                v-model="formData.minMeshSize"
                :min="0.001"
                :precision="3"
                :step="0.1"
                placeholder="Enter min mesh size"
              />
            </el-form-item>
          </el-form>
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
  modelValue: boolean,
  currentTagDict: Map<string, string|string[]>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref({
  maxMeshSize: 1.0,
  minMeshSize: 0.1
})

const handleClose = () => {
  formData.value.maxMeshSize = 1.0
  formData.value.minMeshSize = 0.1
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  if (formData.value.minMeshSize >= formData.value.maxMeshSize) {
    ElMessage.warning('Minimum mesh size must be smaller than maximum mesh size')
    return
  }

  emit('generate', {
    maxMeshSize: formData.value.maxMeshSize,
    minMeshSize: formData.value.minMeshSize
  })
  handleClose()
}
</script>

<style scoped>
.right-sidebar {
  position: relative;
  width: 400px;
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
  display: flex;
  flex-direction: column;
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

.button-container {
  margin-top: auto;
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

.section-content .el-form-item {
  margin-bottom: 16px;
}

.el-input-number {
  width: 100%;
}
</style> 