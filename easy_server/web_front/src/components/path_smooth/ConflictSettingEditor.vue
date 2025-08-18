<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Conflict Setting</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Conflict Setting Configuration</h3>
      </div>
      <el-form :model="conflictSetting" label-width="180px" size="small">
        <el-form-item label="Obstacle Conflict Weight">
          <el-input-number v-model="conflictSetting.obstacle_conflict_weight" :min="0" :max="10000" :step="10" style="width: 200px" />
        </el-form-item>
        <el-form-item label="Path Conflict Weight">
          <el-input-number v-model="conflictSetting.path_conflict_weight" :min="0" :max="10000" :step="10" style="width: 200px" />
        </el-form-item>
        <el-form-item label="Self Conflict Weight">
          <el-input-number v-model="conflictSetting.self_conflict_weight" :min="0" :max="10000" :step="10" style="width: 200px" />
        </el-form-item>
      </el-form>
      <div class="button-container">
        <div class="button-group">
          <el-button @click="handleClose">Cancel</el-button>
          <el-button type="primary" @click="handleSave">Save</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  defineEmits,
  defineProps,
  ref,
  watch,
} from 'vue';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean,
  conflict_setting: {
    obstacle_conflict_weight: number,
    path_conflict_weight: number,
    self_conflict_weight: number
  }
}>()
const emit = defineEmits(['update:modelValue', 'save'])

const conflictSetting = ref({
  obstacle_conflict_weight: 1000,
  path_conflict_weight: 100,
  self_conflict_weight: 100
})

watch(() => props.modelValue, (v) => {
  if (v && props.conflict_setting) {
    conflictSetting.value = { ...props.conflict_setting }
  }
})

const handleClose = () => {
  emit('update:modelValue', false)
}
const handleSave = () => {
  emit('save', { ...conflictSetting.value })
  emit('update:modelValue', false)
}
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
    padding-bottom: 80px;
    min-height: 0;
}
.section-header {
    margin: 16px 0 8px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #409eff;
}
.section-header h3 {
    margin: 0;
    color: #303133;
    font-size: 16px;
    font-weight: 600;
}
.button-container {
    position: fixed;
    right: 0;
    bottom: 0;
    width: 450px;
    padding: 16px;
    border-top: 1px solid #dcdfe6;
    background-color: #f5f7fa;
    z-index: 1100;
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