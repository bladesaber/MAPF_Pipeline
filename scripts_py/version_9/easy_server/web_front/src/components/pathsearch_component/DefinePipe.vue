<!-- DefineTaskPipe.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Pipe</span>
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
            placeholder="Enter pipe name"
            :maxlength="50"
            show-word-limit
          />
        </div>
      </div>

      <!-- Group Section -->
      <div class="section">
        <div class="section-header">
          <span>Group</span>
        </div>
        <div class="section-content">
          <el-input
            v-model="groupValue"
            :disabled="true"
            placeholder="Group name from current tag"
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
            <el-form-item label="Center">
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

      <!-- Direction Section -->
      <div class="section">
        <div class="section-header">
          <span>Direction</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Vector">
              <div class="vector-input">
                <el-input-number 
                  v-model="formData.vec_x"
                  :precision="2"
                  :step="0.1"
                  placeholder="X"
                />
                <el-input-number 
                  v-model="formData.vec_y"
                  :precision="2"
                  :step="0.1"
                  placeholder="Y"
                />
                <el-input-number 
                  v-model="formData.vec_z"
                  :precision="2"
                  :step="0.1"
                  placeholder="Z"
                />
              </div>
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Size Section -->
      <div class="section">
        <div class="section-header">
          <span>Size</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Radius">
              <el-input-number 
                v-model="formData.radius"
                :precision="2"
                :step="0.1"
                :min="0.1"
                style="width: 120px"
              />
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- Inlet/Outlet Section -->
      <div class="section">
        <div class="section-header">
          <span>Type</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <el-form-item label="Is Inlet">
              <el-switch
                v-model="formData.is_inlet"
                active-text="Inlet"
                inactive-text="Outlet"
              />
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
import { ref, computed } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  modelValue: boolean,
  currentTagDict: Map<string, string | string[] | Map<string, string|number>>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const groupValue = computed(() => {
  const group = props.currentTagDict.get('PipeGroup')
  return typeof group === 'string' ? group : (Array.isArray(group) ? group[0] : '')
})

const formData = ref({
  name: '',
  group: '',
  x: 0,
  y: 0,
  z: 0,
  vec_x: 1,
  vec_y: 0,
  vec_z: 0,
  radius: 0.5,
  is_inlet: true
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter a pipe name')
    return
  }

  if (!groupValue.value) {
    ElMessage.warning('No pipe group is selected')
    return
  }

  // Check if direction vector is zero
  if (formData.value.vec_x === 0 && formData.value.vec_y === 0 && formData.value.vec_z === 0) {
    ElMessage.warning('Direction vector cannot be zero')
    return
  }

  emit('generate', {
    ...formData.value,
    group: groupValue.value
  })
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