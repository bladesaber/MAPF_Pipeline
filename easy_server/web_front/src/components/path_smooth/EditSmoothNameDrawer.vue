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