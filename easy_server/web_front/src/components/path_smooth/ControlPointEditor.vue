<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Control Point</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <el-form :model="form" label-width="120px" size="small">
        <el-form-item label="Index">
          <el-input-number v-model="form.c_idx" disabled style="width: 150px" />
        </el-form-item>
        <el-form-item label="X">
          <el-input-number v-model="form.x" :step="0.01" :precision="3" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Y">
          <el-input-number v-model="form.y" :step="0.01" :precision="3" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Z">
          <el-input-number v-model="form.z" :step="0.01" :precision="3" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Radius">
          <el-input-number v-model="form.radius" :min="0.01" :step="0.01" :precision="3" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Is Fix">
          <el-switch v-model="form.is_fix" />
        </el-form-item>
      </el-form>
      <div class="button-container">
        <div class="button-group">
          <el-button @click="handleClose">Cancel</el-button>
          <el-button type="primary" @click="handleApply">Apply</el-button>
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
  controlPointInfo: any,
}>();
const emit = defineEmits(['update:modelValue', 'apply'])

const form = ref({
  c_idx: 0,
  x: 0,
  y: 0,
  z: 0,
  radius: 1,
  is_fix: false,
});

watch(() => props.modelValue, (v) => {
  if (v && props.controlPointInfo) {
    // 需要从controlPointInfo和parent_cell.c_table中获取所有属性
    const { c_idx, parent } = props.controlPointInfo;
    const c_table = parent?.parent_cell?.c_table || parent?.c_table;
    form.value.c_idx = c_idx;
    form.value.x = c_table?.x?.[c_idx] ?? 0;
    form.value.y = c_table?.y?.[c_idx] ?? 0;
    form.value.z = c_table?.z?.[c_idx] ?? 0;
    form.value.radius = c_table?.radius?.[c_idx] ?? 1;
    form.value.is_fix = !!(c_table?.is_fix?.[c_idx]);
  }
}, { immediate: true });

function handleClose() {
  emit('update:modelValue', false)
}
function handleApply() {
  emit('apply', { ...form.value })
  emit('update:modelValue', false)
}
</script>

<style scoped>
.right-sidebar {
    position: fixed;
    right: 0;
    top: 124px;
    width: 450px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-left: 1px solid rgba(226, 232, 240, 0.6);
    height: calc(100vh - 124px);
    display: flex;
    flex-direction: column;
    z-index: 1000;
    box-shadow: -8px 0 32px rgba(0, 0, 0, 0.12);
}
.right-sidebar-header {
    padding: 16px 20px;
    border-bottom: 1px solid rgba(226, 232, 240, 0.6);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
    flex: 0 0 auto;
}
.right-sidebar-header span {
    font-weight: 600;
    color: #1e293b;
    font-size: 16px;
    letter-spacing: 0.025em;
}
.close-button {
    padding: 10px;
    border-radius: 10px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color: #64748b;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(226, 232, 240, 0.4);
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 600;
}
.close-button:hover {
    color: #ffffff;
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    border-color: #ef4444;
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
}
.right-sidebar-content {
    position: relative;
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 16px;
    padding-bottom: 80px;
    min-height: 0;
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