<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Connectors</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Batch Edit Connectors</h3>
        <p class="section-description">修改所有连接器节点的Cost Settings属性</p>
      </div>
      
      <!-- Cost Settings -->
      <div class="section-header">
        <h3>Cost Settings</h3>
      </div>
      <el-form :model="costSettings" label-width="120px" size="small">
        <el-form-item label="Obstacle Thickness">
          <el-input-number 
            v-model="costSettings.obstacle_thickness" 
            :min="0.0" 
            :max="10.0" 
            :step="0.01" 
            :precision="3" 
            style="width: 150px" 
          />
        </el-form-item>
        <el-form-item label="Path Thickness">
          <el-input-number 
            v-model="costSettings.path_thickness" 
            :min="0.0" 
            :max="10.0" 
            :step="0.01" 
            :precision="3" 
            style="width: 150px" 
          />
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
  ref,
  watch,
} from 'vue';

import { Close } from '@element-plus/icons-vue';

interface CostSettings {
  obstacle_thickness: number
  path_thickness: number
}

interface Props {
  modelValue: boolean
  initialCostSettings?: CostSettings
}

interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'save', costSettings: CostSettings): void
}

const props = withDefaults(defineProps<Props>(), {
  initialCostSettings: () => ({
    obstacle_thickness: 0.010,
    path_thickness: 0.010
  })
})

const emit = defineEmits<Emits>()

const costSettings = ref<CostSettings>({
  obstacle_thickness: 0.010,
  path_thickness: 0.010
})

// 监听初始值变化
watch(() => props.initialCostSettings, (newSettings) => {
  if (newSettings) {
    costSettings.value = { ...newSettings }
  }
}, { immediate: true })

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleSave = () => {
  emit('save', { ...costSettings.value })
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
.section-description {
    margin: 8px 0 0 0;
    font-size: 12px;
    color: #909399;
    line-height: 1.4;
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
