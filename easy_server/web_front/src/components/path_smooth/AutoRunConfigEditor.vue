<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Auto Run Config</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Auto Run Configuration</h3>
      </div>
      <el-form :model="autoRunConfig" label-width="120px" size="small">
        <el-form-item label="Learning Rate">
          <el-input-number v-model="autoRunConfig.lr" :min="0.001" :max="1.0" :step="0.001" :precision="3" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Factor">
          <el-input-number v-model="autoRunConfig.factor" :min="0.1" :max="1.0" :step="0.01" :precision="2" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Patience">
          <el-input-number v-model="autoRunConfig.patience" :min="100" :max="100000" :step="100" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Cooldown">
          <el-input-number v-model="autoRunConfig.cooldown" :min="100" :max="100000" :step="100" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Min Learning Rate">
          <el-input-number v-model="autoRunConfig.min_lr" :min="0.0001" :max="0.1" :step="0.0001" :precision="4" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Max Iterations">
          <el-input-number v-model="autoRunConfig.max_iter" :min="1000" :max="1000000" :step="1000" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Outer Loop">
          <el-input-number v-model="autoRunConfig.outer_loop" :min="1" :max="100" :step="1" style="width: 150px" />
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
  auto_run_config: {
    lr: number,
    factor: number,
    patience: number,
    cooldown: number,
    min_lr: number,
    max_iter: number,
    outer_loop: number
  }
}>()
const emit = defineEmits(['update:modelValue', 'save'])

const autoRunConfig = ref({
  lr: 0.01,
  factor: 0.25,
  patience: 5000,
  cooldown: 5000,
  min_lr: 0.001,
  max_iter: 50000,
  outer_loop: 1
})

watch(() => props.modelValue, (v) => {
  if (v && props.auto_run_config) {
    autoRunConfig.value = { ...props.auto_run_config }
  }
})

const handleClose = () => {
  emit('update:modelValue', false)
}
const handleSave = () => {
  emit('save', { ...autoRunConfig.value })
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
.section-header {
    margin: 20px 0 12px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    position: relative;
}

.section-header::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 1px;
}

.section-header h3 {
    margin: 0;
    color: #1e293b;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.025em;
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