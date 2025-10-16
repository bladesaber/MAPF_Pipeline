<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Smooth Name</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <el-form :model="form" label-width="120px" size="small">
        <el-form-item label="Smooth Name" required>
          <el-input v-model="form.name" placeholder="Enter smooth name" />
        </el-form-item>
      </el-form>
      <div class="button-container">
        <el-button @click="handleClose">Cancel</el-button>
        <el-button type="primary" @click="handleApply">Apply</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ref,
  watch,
} from 'vue';

import { ElMessage } from 'element-plus';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean,
  smoothName: string
}>();
const emit = defineEmits(['update:modelValue', 'save']);

const form = ref({ name: props.smoothName });

watch(() => props.smoothName, (val) => {
  form.value.name = val;
});

const handleClose = () => {
  emit('update:modelValue', false);
};

const handleApply = () => {
  if (!form.value.name.trim()) {
    ElMessage.error('Smooth name is required');
    return;
  }
  emit('save', form.value.name.trim());
  handleClose();
};
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
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}
</style> 