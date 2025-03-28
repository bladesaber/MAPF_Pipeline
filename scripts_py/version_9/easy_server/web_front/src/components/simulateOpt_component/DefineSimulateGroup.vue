<!-- DefineSimulateGroup.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Group</span>
      <el-button 
        link
        class="close-button"
        @click="handleClose"
      >
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="group-header">
        <el-input
          v-model="groupName"
          placeholder="Enter group name"
          :maxlength="50"
          show-word-limit
          @input="validateName"
        >
          <template #prefix>Group Name:</template>
        </el-input>
      </div>

      <div class="table-container">
        <el-table 
          :data="selectedItems" 
          style="width: 100%"
          height="calc(100% - 80px)"
        >
          <el-table-column prop="name" label="Name" />
          <el-table-column label="Action" width="120" fixed="right">
            <template #default="scope">
              <el-button
                type="danger"
                size="small"
                @click="handleDelete(scope.$index)"
              >
                Delete
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <div class="button-container">
        <div class="button-group">
          <el-button type="primary" @click="handleGenerate">Generate</el-button>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, inject } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { SimulateRootObj, SimulateEntity } from '@/classes/simulateOpt_classes/simulate_entity'

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const groupName = ref('')
const selectedItems = ref<{ name: string }[]>([])

const validateName = (value: string) => {
  if (!value.trim()) {
    groupName.value = ''
  }
}

const addItem = ({name}: {name: string}) => {
  if (!selectedItems.value.some(item => item.name === name)){
    selectedItems.value.push({ name })
  }
}

const handleDelete = (index: number) => {
  selectedItems.value.splice(index, 1)
}

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  if (!groupName.value.trim()) {
    ElMessage.warning('Please enter a group name')
    return
  }
  if (selectedItems.value.length === 0) {
    ElMessage.warning('Please add at least one item to the group')
    return
  }
  emit('generate', {
    name: groupName.value,
    items: selectedItems.value
  })
  groupName.value = ''
  selectedItems.value = []
}

</script>

<style scoped>
.right-sidebar {
  position: relative;
  width: 300px;
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
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.group-header {
  padding: 16px;
  border-bottom: 1px solid #dcdfe6;
  flex-shrink: 0;
}

.table-container {
  flex: 1;
  overflow: hidden;
  padding: 16px;
}

:deep(.el-input__wrapper) {
  box-shadow: 0 0 0 1px #dcdfe6 inset;
}

:deep(.el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px #409eff inset;
}

:deep(.el-table) {
  height: 100%;
}

:deep(.el-table__body-wrapper) {
  overflow-y: auto;
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