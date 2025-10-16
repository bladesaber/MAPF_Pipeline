<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Segments</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Batch Edit Segments</h3>
        <p class="section-description">修改所有段节点的Cost Settings属性</p>
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
    padding: 20px;
    padding-bottom: 80px;
    min-height: 0;
    background: rgba(248, 250, 252, 0.3);
}

.section-header {
    margin: 20px 0 12px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.2);
}

.section-header h3 {
    margin: 0;
    color: #1e293b;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.025em;
}

.section-description {
    margin: 8px 0 0 0;
    font-size: 13px;
    color: #64748b;
    line-height: 1.5;
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
