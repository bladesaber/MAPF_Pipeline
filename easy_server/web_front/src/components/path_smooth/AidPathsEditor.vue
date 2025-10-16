<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Aid Paths</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Aid Paths</h3>
      </div>
      <div v-for="(pipes, groupName) in groupPipes" :key="groupName" class="group-section">
        <el-collapse v-model="activeGroups" accordion>
          <el-collapse-item :name="`aid-${groupName}`">
            <template #title>
              <span class="group-title">{{ groupName }} - Aid Paths</span>
            </template>
            <div class="group-content">
              <!-- Graph Configuration -->
              <div class="graph-config-section">
                <h4>Graph Configuration</h4>
                <el-form :model="graphConfigs[groupName]" label-width="120px" size="small" inline>
                  <el-form-item label="Seg Node Min">
                    <el-input-number v-model="graphConfigs[groupName].seg_node_min" :min="5" :max="100" :step="5" style="width: 120px" />
                  </el-form-item>
                  <el-form-item label="Seg Node Max">
                    <el-input-number v-model="graphConfigs[groupName].seg_node_max" :min="10" :max="200" :step="5" style="width: 120px" />
                  </el-form-item>
                </el-form>
              </div>
              <div class="add-row-container">
                <el-button type="primary" size="small" @click="addAidPath(groupName)">
                  Add Aid Path
                </el-button>
              </div>
              <el-table :data="aidPaths[groupName] || []" style="width: 100%" size="small">
                <el-table-column label="Name" width="120">
                  <template #default="scope">
                    <el-input v-model="scope.row.name" placeholder="Enter name" size="small" @blur="validateName(scope.row)" />
                  </template>
                </el-table-column>
                <el-table-column label="From" width="120">
                  <template #default="scope">
                    <el-select v-model="scope.row.from_name" placeholder="Select from" size="small" style="width: 100%">
                      <el-option v-for="pipe in pipes" :key="pipe.name" :label="pipe.name" :value="pipe.name" :disabled="pipe.name === scope.row.to_name" />
                    </el-select>
                  </template>
                </el-table-column>
                <el-table-column label="To" width="120">
                  <template #default="scope">
                    <el-select v-model="scope.row.to_name" placeholder="Select to" size="small" style="width: 100%">
                      <el-option v-for="pipe in pipes" :key="pipe.name" :label="pipe.name" :value="pipe.name" :disabled="pipe.name === scope.row.from_name" />
                    </el-select>
                  </template>
                </el-table-column>
                <el-table-column label="Action" width="60">
                  <template #default="scope">
                    <el-button type="danger" size="small" @click="removeAidPath(groupName, scope.$index)" :icon="Delete" circle />
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

import { Close, Delete } from '@element-plus/icons-vue';

const props = defineProps<{ modelValue: boolean, aid_paths: any, pipes_config: any }>()
const emit = defineEmits(['update:modelValue', 'save'])
const activeGroups = ref<string[]>([])
const groupPipes = ref<Record<string, Array<{name: string}>>>({})
const aidPaths = ref<Record<string, Array<{name: string, from_name: string, to_name: string}>>>({})
const graphConfigs = ref<Record<string, {seg_node_min: number, seg_node_max: number}>>({})

watch(() => props.modelValue, v => {
  if (v) {
    // pipes_config: { pipeName: { group_name, ... } }
    const groupPipeMap: Record<string, Array<{name: string}>> = {}
    Object.entries(props.pipes_config || {}).forEach(([pipeName, info]: [string, any]) => {
      if (!groupPipeMap[info.group_name]) groupPipeMap[info.group_name] = []
      groupPipeMap[info.group_name].push({ name: pipeName })
    })
    groupPipes.value = groupPipeMap
    // aid_paths: { groupName: { aidName: {from_name, to_name} } }
    const apObj: Record<string, Array<{name: string, from_name: string, to_name: string}>> = {}
    Object.entries(props.aid_paths || {}).forEach(([groupName, aids]: [string, any]) => {
      apObj[groupName] = []
      Object.entries(aids).forEach(([aidName, info]: [string, any]) => {
        apObj[groupName].push({ name: aidName, from_name: info.from_name, to_name: info.to_name })
      })
    })
    aidPaths.value = apObj
    // graphConfigs
    const graphCfg: Record<string, {seg_node_min: number, seg_node_max: number}> = {}
    Object.keys(groupPipeMap).forEach(groupName => {
      graphCfg[groupName] = { seg_node_min: 10, seg_node_max: 50 }
    })
    graphConfigs.value = graphCfg
    if (Object.keys(groupPipeMap).length > 0) {
      activeGroups.value = [`aid-${Object.keys(groupPipeMap)[0]}`]
    }
  }
})

const addAidPath = (groupName: string) => {
  if (!aidPaths.value[groupName]) aidPaths.value[groupName] = []
  aidPaths.value[groupName].push({ name: '', from_name: '', to_name: '' })
}
const removeAidPath = (groupName: string, idx: number) => {
  aidPaths.value[groupName].splice(idx, 1)
}
const validateName = (row: {name: string, from_name: string, to_name: string}) => {
  if (!row.name.trim()) {
    ElMessage.warning('Name cannot be empty')
  }
}
const handleClose = () => {
  emit('update:modelValue', false)
}
const handleSave = () => {
  // 校验所有 group 的 pipe 都被覆盖
  for (const [groupName, pipes] of Object.entries(groupPipes.value)) {
    const allPipeNames: string[] = pipes.map(p => p.name)
    const usedPipeNames = new Set<string>()
    const paths = aidPaths.value[groupName]
    if (Array.isArray(paths)) {
      paths.forEach((row: any) => {
        if (row.name && row.from_name && row.to_name) {
          usedPipeNames.add(row.from_name.trim())
          usedPipeNames.add(row.to_name.trim())
        }
      })
    }
    const missing = allPipeNames.filter((name: string) => !usedPipeNames.has(name))
    if (missing.length > 0) {
      ElMessage.error(`Group ${groupName} pipes not covered by any path: ${missing.join(', ')}`)
      return
    }
  }
  // 校验所有 aid path
  let isValid = true
  const newAidPaths: Record<string, Record<string, {from_name: string, to_name: string}>> = {}
  for (const [groupName, paths] of Object.entries(aidPaths.value)) {
    newAidPaths[groupName] = {}
    for (const path of paths) {
      if (!path.name.trim()) {
        ElMessage.error(`Name is required for aid path in group ${groupName}`)
        isValid = false
        break
      }
      if (!path.from_name) {
        ElMessage.error(`From pipe is required for aid path ${path.name} in group ${groupName}`)
        isValid = false
        break
      }
      if (!path.to_name) {
        ElMessage.error(`To pipe is required for aid path ${path.name} in group ${groupName}`)
        isValid = false
        break
      }
      newAidPaths[groupName][path.name] = {
        from_name: path.from_name,
        to_name: path.to_name
      }
    }
    if (!isValid) break
  }
  if (isValid) {
    emit('save', newAidPaths)
    emit('update:modelValue', false)
  }
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
.add-row-container {
    margin-bottom: 16px;
    display: flex;
    justify-content: flex-end;
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