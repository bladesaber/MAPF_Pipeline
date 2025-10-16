<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Control Points</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="segment-info">
        <el-tag type="info" size="small">Segment: {{ segmentInfo?.name || 'Unknown' }}</el-tag>
        <el-tag type="success" size="small">Control Points: {{ controlPoints.length }}</el-tag>
      </div>
      
      <el-divider />
      
      <el-form :model="form" label-width="120px" size="small">
        <el-form-item label="Apply to All">
          <el-switch v-model="applyToAll" @change="handleApplyToAllChange" />
        </el-form-item>
        
        <el-form-item label="Radius">
          <el-input-number 
            v-model="form.radius" 
            :min="0.01" 
            :step="0.01" 
            :precision="3" 
            style="width: 150px"
            :disabled="!applyToAll"
          />
        </el-form-item>
      </el-form>

      <el-divider />
      
      <div class="control-points-list">
        <div class="list-header">
          <span>Individual Control Points</span>
          <el-button size="small" @click="expandAll = !expandAll">
            {{ expandAll ? 'Collapse All' : 'Expand All' }}
          </el-button>
        </div>
        
        <el-collapse v-model="activeNames" accordion>
          <el-collapse-item 
            v-for="(point, index) in controlPoints" 
            :key="point.c_idx"
            :name="point.c_idx"
            :title="`Control Point ${point.c_idx}`"
          >
            <template #title>
              <div class="collapse-title">
                <span>Control Point {{ point.c_idx }}</span>
                <el-tag 
                  :type="point.is_fix ? 'warning' : 'success'" 
                  size="small"
                >
                  {{ point.is_fix ? 'Fixed' : 'Free' }}
                </el-tag>
              </div>
            </template>
            
            <el-form :model="point" label-width="80px" size="small">
              <el-form-item label="X">
                <el-input-number 
                  v-model="point.x" 
                  :step="0.01" 
                  :precision="3" 
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item label="Y">
                <el-input-number 
                  v-model="point.y" 
                  :step="0.01" 
                  :precision="3" 
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item label="Z">
                <el-input-number 
                  v-model="point.z" 
                  :step="0.01" 
                  :precision="3" 
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item label="Radius">
                <el-input-number 
                  v-model="point.radius" 
                  :min="0.01" 
                  :step="0.01" 
                  :precision="3" 
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item label="Is Fix">
                <el-switch v-model="point.is_fix" />
              </el-form-item>
            </el-form>
          </el-collapse-item>
        </el-collapse>
      </div>
      
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
  computed,
  defineEmits,
  defineProps,
  ref,
  watch,
} from 'vue';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean,
  segmentInfo: any,
  controlPointInfo: any,
}>();

const emit = defineEmits(['update:modelValue', 'apply'])

const applyToAll = ref(false);
const expandAll = ref(false);
const activeNames = ref<string[]>([]);

const form = ref({
  radius: 1,
});

// 计算当前段的所有控制点
const controlPoints = computed(() => {
  if (!props.controlPointInfo) return [];
  
  const { parent } = props.controlPointInfo;
  const c_table = parent?.parent_cell?.c_table || parent?.c_table;
  
  if (!c_table) return [];
  
  // 获取当前段的控制点索引列表
  let c_idxs = [];
  
  if (parent && parent.c_idxs) {
    // 如果parent是段信息，直接使用c_idxs
    c_idxs = parent.c_idxs;
  } else if (props.segmentInfo && props.segmentInfo.c_idxs) {
    // 使用传入的段信息中的c_idxs
    c_idxs = props.segmentInfo.c_idxs;
  } else {
    // 如果没有c_idxs信息，返回空数组
    return [];
  }
  
  // 根据c_idxs获取对应的控制点
  const points = [];
  for (const c_idx of c_idxs) {
    if (c_table.x && c_table.x[c_idx] !== undefined) {
      points.push({
        c_idx: c_idx,
        x: c_table.x[c_idx] ?? 0,
        y: c_table.y?.[c_idx] ?? 0,
        z: c_table.z?.[c_idx] ?? 0,
        radius: c_table.radius?.[c_idx] ?? 1,
        is_fix: !!(c_table.is_fix?.[c_idx]),
      });
    }
  }
  
  return points;
});

watch(() => props.modelValue, (v) => {
  if (v && props.controlPointInfo) {
    // 初始化表单数据
    const { c_idx, parent } = props.controlPointInfo;
    const c_table = parent?.parent_cell?.c_table || parent?.c_table;
    
    if (c_table) {
      form.value.radius = c_table.radius?.[c_idx] ?? 1;
    }
    
    // 重置状态
    applyToAll.value = false;
    expandAll.value = false;
    activeNames.value = [];
  }
}, { immediate: true });

// 监听展开/折叠状态
watch(expandAll, (newVal) => {
  if (newVal) {
    activeNames.value = controlPoints.value.map(p => p.c_idx.toString());
  } else {
    activeNames.value = [];
  }
});

// 监听半径值变化，当Apply to All开启时自动应用到所有控制点
watch(() => form.value.radius, (newRadius) => {
  if (applyToAll.value) {
    controlPoints.value.forEach(point => {
      point.radius = newRadius;
    });
  }
});

function handleApplyToAllChange() {
  if (applyToAll.value) {
    // 当启用"应用到全部"时，将当前半径值应用到所有控制点
    controlPoints.value.forEach(point => {
      point.radius = form.value.radius;
    });
  }
}

function handleClose() {
  emit('update:modelValue', false)
}

function handleApply() {
  // 准备应用数据
  const updates = controlPoints.value.map(point => ({
    c_idx: point.c_idx,
    x: point.x,
    y: point.y,
    z: point.z,
    radius: point.radius,
    is_fix: point.is_fix,
  }));
  
  emit('apply', updates)
  emit('update:modelValue', false)
}
</script>

<style scoped>
.right-sidebar {
    position: fixed;
    right: 0;
    top: 124px;
    width: 500px;
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

.segment-info {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}

.control-points-list {
    margin-top: 16px;
}

.list-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-weight: 500;
}

.collapse-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding-right: 20px;
}

.button-container {
    position: fixed;
    right: 0;
    bottom: 0;
    width: 500px;
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

:deep(.el-collapse-item__header) {
    padding-left: 12px;
}

:deep(.el-collapse-item__content) {
    padding: 12px;
    background-color: #fafafa;
}
</style>
