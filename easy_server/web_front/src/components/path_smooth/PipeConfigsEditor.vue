<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Pipe Configs</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Pipe Configs</h3>
      </div>
      <div v-for="(pipes, groupName) in groupPipeDict" :key="`pipe-${groupName}`" class="group-section">
        <el-collapse v-model="activeGroups" accordion>
          <el-collapse-item :name="`pipe-${groupName}`">
            <template #title>
              <span class="group-title">{{ groupName }} - Pipes</span>
            </template>
            <div class="group-content">
              <el-table :data="pipes" style="width: 100%" size="small">
                <el-table-column label="Name" width="100">
                  <template #default="scope">
                    <span>{{ scope.row.name }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="Radius" width="80">
                  <template #default="scope">
                    <el-input-number v-model="scope.row.radius" :min="0.1" :max="100" :step="0.1" size="small" />
                  </template>
                </el-table-column>
                <el-table-column label="Position" width="120">
                  <template #default="scope">
                    <span>{{ formatPosition(scope.row.position) }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="Direction" width="120">
                  <template #default="scope">
                    <el-input v-model="directionValues[`${groupName}_${scope.row.name}`]" placeholder="x,y,z" size="small" @input="(value: string) => updateDirection(`${groupName}_${scope.row.name}`, value)" />
                  </template>
                </el-table-column>
                <el-table-column label="Is Input" width="80">
                  <template #default="scope">
                    <span>{{ scope.row.is_input ? 'Yes' : 'No' }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="Radius Scale" width="100">
                  <template #default="scope">
                    <el-input-number v-model="radiusScaleValues[`${groupName}_${scope.row.name}`]" :min="0.1" :max="10" :step="0.1" size="small" @change="(value: number) => updateRadiusScale(`${groupName}_${scope.row.name}`, value)" />
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </el-collapse-item>
        </el-collapse>
      </div>
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

import { ElMessage } from 'element-plus';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{ modelValue: boolean, pipes_config: any }>()
const emit = defineEmits(['update:modelValue', 'save'])
const activeGroups = ref<string[]>([])
const groupPipeDict = ref<Record<string, Array<any>>>({})
const directionValues = ref<Record<string, string>>({})
const radiusScaleValues = ref<Record<string, number>>({})

watch(() => props.modelValue, v => {
  if (v) {
    // 初始化分组和表单数据
    const groupMap: Record<string, Array<any>> = {}
    const dirVals: Record<string, string> = {}
    const scaleVals: Record<string, number> = {}
    Object.entries(props.pipes_config || {}).forEach(([pipeName, info]: [string, any]) => {
      if (!groupMap[info.group_name]) groupMap[info.group_name] = []
      const pipe = { ...info, name: pipeName }
      groupMap[info.group_name].push(pipe)
      dirVals[`${info.group_name}_${pipeName}`] = (info.direction || []).join(',')
      scaleVals[`${info.group_name}_${pipeName}`] = info.radius_scale || 1.0
    })
    groupPipeDict.value = groupMap
    directionValues.value = dirVals
    radiusScaleValues.value = scaleVals
    if (Object.keys(groupMap).length > 0) {
      activeGroups.value = [`pipe-${Object.keys(groupMap)[0]}`]
    }
  }
})

const formatPosition = (position: number[] | undefined): string => {
  if (!position || !Array.isArray(position)) return 'N/A'
  return `(${position[0]?.toFixed(2)}, ${position[1]?.toFixed(2)}, ${position[2]?.toFixed(2)})`
}
const updateDirection = (key: string, value: string) => {
  directionValues.value[key] = value
}
const updateRadiusScale = (key: string, value: number) => {
  radiusScaleValues.value[key] = value
}
const handleClose = () => {
  emit('update:modelValue', false)
}
const handleSave = () => {
  // 校验并组装 pipes_config
  const newConfig: Record<string, any> = {}
  for (const [groupName, pipes] of Object.entries(groupPipeDict.value)) {
    for (const pipe of pipes) {
      const key = `${groupName}_${pipe.name}`
      const direction = (directionValues.value[key] || '').split(',').map(Number)
      if (direction.length !== 3 || direction.some(isNaN)) {
        ElMessage.error(`管道 ${pipe.name} 的 direction 格式错误`)
        return
      }
      newConfig[pipe.name] = {
        ...pipe,
        direction,
        radius_scale: radiusScaleValues.value[key] || 1.0
      }
    }
  }
  emit('save', newConfig)
  emit('update:modelValue', false)
}
</script>

<style scoped>
.right-sidebar {
    position: fixed;
    right: 0;
    top: 124px;
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
    color: #6366f1;
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
.group-section {
    margin-bottom: 16px;
}
.group-title {
    font-weight: 500;
    color: #303133;
}
.group-content {
    padding: 16px 0;
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
:deep(.el-collapse-item__header) {
    background-color: #f5f7fa;
    border-bottom: 1px solid #dcdfe6;
}
:deep(.el-collapse-item__content) {
    padding: 0;
}
:deep(.el-table) {
    border: 1px solid #dcdfe6;
}
:deep(.el-table th) {
    background-color: #f5f7fa;
}
:deep(.el-input-number) {
    width: 100%;
}
</style> 