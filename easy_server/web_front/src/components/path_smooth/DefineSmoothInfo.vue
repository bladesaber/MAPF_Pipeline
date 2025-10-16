<!-- DefinePanel.vue -->
<template>
    <div v-if="modelValue" class="right-sidebar">
        <div class="right-sidebar-header">
            <span>{{ label_dict!.get('Project') + ' - ' + label_dict!.get('SearchTask') }}</span>
            <el-button link class="close-button" @click="handleClose">
                <el-icon>
                    <Close />
                </el-icon>
            </el-button>
        </div>
        <div class="right-sidebar-content">
            <!-- Smooth Name Section -->
            <div class="smooth-name-section">
                <el-form :model="smoothName" label-width="120px" size="small">
                    <el-form-item label="Smooth Name" required>
                        <el-input 
                            v-model="smoothName.name" 
                            placeholder="Enter smooth name"
                            style="width: 100%"
                            @blur="validateSmoothName"
                        />
                    </el-form-item>
                </el-form>
            </div>

            <!-- Pipe Configs Section -->
            <div class="section-header">
                <h3>Pipe Configs</h3>
            </div>
            <div v-for="(pipes, groupName) in GroupPipeDict" :key="`pipe-${groupName}`" class="group-section">
                <el-collapse v-model="activeGroups" accordion>
                    <el-collapse-item :name="`pipe-${groupName}`">
                        <template #title>
                            <span class="group-title">{{ groupName }} - Pipes</span>
                        </template>
                        <div class="group-content">
                            <!-- Table for All Pipes in Group -->
                            <el-table :data="pipes" style="width: 100%" size="small">
                                <el-table-column label="Name" width="100">
                                    <template #default="scope">
                                        <span>{{ scope.row.name }}</span>
                                    </template>
                                </el-table-column>
                                <el-table-column label="Radius" width="80">
                                    <template #default="scope">
                                        <span>{{ scope.row.radius }}</span>
                                    </template>
                                </el-table-column>
                                <el-table-column label="Position" width="120">
                                    <template #default="scope">
                                        <span>{{ formatPosition(scope.row.position) }}</span>
                                    </template>
                                </el-table-column>
                                <el-table-column label="Direction" width="120">
                                    <template #default="scope">
                                        <el-input 
                                            v-model="directionValues[`${groupName}_${scope.row.name}`]" 
                                            placeholder="x,y,z"
                                            size="small"
                                            @input="(value: string) => updateDirection(`${groupName}_${scope.row.name}`, value)"
                                        />
                                    </template>
                                </el-table-column>
                                <el-table-column label="Is Input" width="80">
                                    <template #default="scope">
                                        <span>{{ scope.row.is_input ? 'Yes' : 'No' }}</span>
                                    </template>
                                </el-table-column>
                                <el-table-column label="Radius Scale" width="100">
                                    <template #default="scope">
                                        <el-input-number 
                                            v-model="radiusScaleValues[`${groupName}_${scope.row.name}`]" 
                                            :min="0.1"
                                            :max="10"
                                            :step="0.1"
                                            size="small"
                                            @change="(value: number) => updateRadiusScale(`${groupName}_${scope.row.name}`, value)"
                                        />
                                    </template>
                                </el-table-column>
                            </el-table>
                        </div>
                    </el-collapse-item>
                </el-collapse>
            </div>

            <!-- Aid Paths Section -->
            <div class="section-header">
                <h3>Aid Paths</h3>
            </div>
            <div v-for="(pipes, groupName) in GroupPipeDict" :key="`aid-${groupName}`" class="group-section">
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
                                        <el-input-number 
                                            v-model="graphConfigs[groupName].seg_node_min"
                                            :min="5"
                                            :max="100"
                                            :step="5"
                                            style="width: 120px"
                                        />
                                    </el-form-item>
                                    <el-form-item label="Seg Node Max">
                                        <el-input-number 
                                            v-model="graphConfigs[groupName].seg_node_max"
                                            :min="10"
                                            :max="200"
                                            :step="5"
                                            style="width: 120px"
                                        />
                                    </el-form-item>
                                </el-form>
                            </div>

                            <!-- Add New Row Button -->
                            <div class="add-row-container">
                                <el-button type="primary" size="small" @click="addAidPath(groupName)">
                                    <el-icon><Plus /></el-icon>
                                    Add Aid Path
                                </el-button>
                            </div>
                            
                            <!-- Table for Aid Paths -->
                            <el-table :data="aidPaths[groupName] || []" style="width: 100%" size="small">
                                <el-table-column label="Name" width="120">
                                    <template #default="scope">
                                        <el-input 
                                            v-model="scope.row.name" 
                                            placeholder="Enter name"
                                            size="small"
                                            @blur="validateName(scope.row)"
                                        />
                                    </template>
                                </el-table-column>
                                
                                <el-table-column label="From" width="120">
                                    <template #default="scope">
                                        <el-select 
                                            v-model="scope.row.from_name" 
                                            placeholder="Select from"
                                            size="small"
                                            style="width: 100%"
                                        >
                                            <el-option
                                                v-for="pipe in pipes"
                                                :key="pipe.name"
                                                :label="pipe.name"
                                                :value="pipe.name"
                                                :disabled="pipe.name === scope.row.to_name"
                                            />
                                        </el-select>
                                    </template>
                                </el-table-column>
                                
                                <el-table-column label="To" width="120">
                                    <template #default="scope">
                                        <el-select 
                                            v-model="scope.row.to_name" 
                                            placeholder="Select to"
                                            size="small"
                                            style="width: 100%"
                                        >
                                            <el-option
                                                v-for="pipe in pipes"
                                                :key="pipe.name"
                                                :label="pipe.name"
                                                :value="pipe.name"
                                                :disabled="pipe.name === scope.row.from_name"
                                            />
                                        </el-select>
                                    </template>
                                </el-table-column>
                                
                                <el-table-column label="Action" width="80">
                                    <template #default="scope">
                                        <el-button 
                                            type="danger" 
                                            size="small" 
                                            @click="removeAidPath(groupName, scope.$index)"
                                        >
                                            <el-icon><Delete /></el-icon>
                                        </el-button>
                                    </template>
                                </el-table-column>
                            </el-table>
                        </div>
                    </el-collapse-item>
                </el-collapse>
            </div>

            <!-- Conflict Setting Section -->
            <div class="section-header">
                <h3>Conflict Setting</h3>
            </div>
            <el-collapse v-model="activeCollapseSections" accordion>
                <el-collapse-item name="conflict-setting">
                    <template #title>
                        <span class="collapse-title">Conflict Setting Configuration</span>
                    </template>
                    <div class="conflict-setting-section">
                        <el-form :model="conflictSettings" label-width="180px" size="small">
                            <el-form-item label="Obstacle Conflict Weight">
                                <el-input-number 
                                    v-model="conflictSettings.obstacle_conflict_weight"
                                    :min="0"
                                    :max="10000"
                                    :step="10"
                                    style="width: 200px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Path Conflict Weight">
                                <el-input-number 
                                    v-model="conflictSettings.path_conflict_weight"
                                    :min="0"
                                    :max="10000"
                                    :step="10"
                                    style="width: 200px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Self Conflict Weight">
                                <el-input-number 
                                    v-model="conflictSettings.self_conflict_weight"
                                    :min="0"
                                    :max="10000"
                                    :step="10"
                                    style="width: 200px"
                                />
                            </el-form-item>
                        </el-form>
                    </div>
                </el-collapse-item>
            </el-collapse>

            <!-- Auto Run Configuration Section -->
            <div class="section-header">
                <h3>Auto Run Configuration</h3>
            </div>
            <el-collapse v-model="activeCollapseSections" accordion>
                <el-collapse-item name="auto-run-config">
                    <template #title>
                        <span class="collapse-title">Auto Run Configuration</span>
                    </template>
                    <div class="auto-run-config-section">
                        <el-form :model="autoRunConfig" label-width="120px" size="small">
                            <el-form-item label="Learning Rate">
                                <el-input-number 
                                    v-model="autoRunConfig.lr"
                                    :min="0.001"
                                    :max="1.0"
                                    :step="0.001"
                                    :precision="3"
                                    style="width: 150px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Factor">
                                <el-input-number 
                                    v-model="autoRunConfig.factor"
                                    :min="0.1"
                                    :max="1.0"
                                    :step="0.01"
                                    :precision="2"
                                    style="width: 150px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Patience">
                                <el-input-number 
                                    v-model="autoRunConfig.patience"
                                    :min="100"
                                    :max="100000"
                                    :step="100"
                                    style="width: 150px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Cooldown">
                                <el-input-number 
                                    v-model="autoRunConfig.cooldown"
                                    :min="100"
                                    :max="100000"
                                    :step="100"
                                    style="width: 150px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Min Learning Rate">
                                <el-input-number 
                                    v-model="autoRunConfig.min_lr"
                                    :min="0.0001"
                                    :max="0.1"
                                    :step="0.0001"
                                    :precision="4"
                                    style="width: 150px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Max Iterations">
                                <el-input-number 
                                    v-model="autoRunConfig.max_iter"
                                    :min="1000"
                                    :max="1000000"
                                    :step="1000"
                                    style="width: 150px"
                                />
                            </el-form-item>
                            
                            <el-form-item label="Outer Loop">
                                <el-input-number 
                                    v-model="autoRunConfig.outer_loop"
                                    :min="1"
                                    :max="100"
                                    :step="1"
                                    style="width: 150px"
                                />
                            </el-form-item>
                        </el-form>
                    </div>
                </el-collapse-item>
            </el-collapse>

            <!-- Button Container -->
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
  inject,
  ref,
  watch,
} from 'vue';

import type { FormInstance } from 'element-plus';
import { ElMessage } from 'element-plus';
import * as THREE from 'three';

import { CenterRoot } from '@/classes/center_root';
import {
  Close,
  Delete,
  Plus,
} from '@element-plus/icons-vue';

const props = defineProps<{
    modelValue: boolean,
    label_dict?: Map<
        string, 
        string | number | string[] | Map<string, string|number> | 
        Map<string, Map<string, Map<string, string | number | number[] | boolean>>>
    >
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const rootObj = inject<CenterRoot>('centerRoot', new CenterRoot(new THREE.Scene()))

// Reactive data
const activeGroups = ref<string[]>([])
const aidPaths = ref<Record<string, Array<{name: string, from_name: string, to_name: string}>>>({})
const directionValues = ref<Record<string, string>>({})
const radiusScaleValues = ref<Record<string, number>>({})
const conflictSettings = ref({
    obstacle_conflict_weight: 1000.0,
    path_conflict_weight: 100.0,
    self_conflict_weight: 100.0
})
const graphConfigs = ref<Record<string, {seg_node_min: number, seg_node_max: number}>>({})
const smoothName = ref({ name: '' })
const autoRunConfig = ref({
    lr: 0.01,
    factor: 0.25,
    patience: 5000,
    cooldown: 5000,
    min_lr: 0.001,
    max_iter: 50000,
    outer_loop: 1
})
const activeCollapseSections = ref<string[]>([])

// Convert GroupPipeDict to a plain object for easier template usage
const GroupPipeDict = computed(() => {
    const GroupPipeDictData = props.label_dict?.get('GroupPipeDict')
    if (GroupPipeDictData && GroupPipeDictData instanceof Map) {
        const obj: Record<string, Array<Record<string, any>>> = {}
        for (const [groupName, pipesMap] of GroupPipeDictData.entries()) {
            obj[groupName] = []
            for (const pipe of (pipesMap as Map<string, Map<string, any>>).values()) {
                obj[groupName].push(Object.fromEntries(pipe.entries()))
            }
        }
        return obj
    }
    return {}
})

// Methods
const handleClose = () => {
    emit('update:modelValue', false)
}

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

const addAidPath = (groupName: string) => {
    if (!aidPaths.value[groupName]) {
        aidPaths.value[groupName] = []
    }
    aidPaths.value[groupName].push({
        name: '',
        from_name: '',
        to_name: ''
    })
}

const removeAidPath = (groupName: string, index: number) => {
    aidPaths.value[groupName].splice(index, 1)
}

const validateName = (row: {name: string, from_name: string, to_name: string}) => {
    if (!row.name.trim()) {
        ElMessage.warning('Name cannot be empty')
    }
}

const validateSmoothName = () => {
    if (!smoothName.value.name.trim()) {
        ElMessage.warning('Smooth name cannot be empty')
    }
}

const handleApply = () => {
    // Validate smooth name
    if (!smoothName.value.name.trim()) {
        ElMessage.error('Smooth name is required')
        return
    }

    // 只校验当前 group 的所有接头都被覆盖
    for (const [groupName, pipes] of Object.entries(GroupPipeDict.value)) {
      const allPipeNames: string[] = []
      if (Array.isArray(pipes)) {
        pipes.forEach((pipe: any) => {
          if (pipe && pipe.name) allPipeNames.push(pipe.name.trim())
        })
      }
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
        ElMessage.error(`分组 ${groupName} 以下接头未被任何路径覆盖: ${missing.join(', ')}`)
        return
      }
    }

    // Validate all aid paths
    let isValid = true

    const allAidPaths: Record<string, Record<string, {from_name: string, to_name: string}>> = {}
    const pipes_config: Record<string, {
        group_name: string,
        radius: number,
        position: number[],
        direction: number[],
        is_input: boolean,
        radius_scale: number
    }> = {}

    for (const [groupName, paths] of Object.entries(aidPaths.value)) {
        allAidPaths[groupName] = {}
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
            allAidPaths[groupName][path.name] = {
                from_name: path.from_name,
                to_name: path.to_name,
            }
        }
        if (!isValid) break
    }

    for (const [groupName, pipes] of Object.entries(GroupPipeDict.value)) {
        for (const pipe of pipes) {
            pipes_config[pipe.name] = {
                group_name: groupName,
                direction: pipe.direction as number[],
                radius_scale: pipe.radius_scale as number,
                radius: pipe.radius as number,
                position: pipe.position as number[],
                is_input: pipe.is_input as boolean
            }
        }
    }
    
    if (isValid) {
        emit('generate', {
            smooth_name: smoothName.value.name.trim(),
            aid_paths: allAidPaths,
            pipes_config: pipes_config,
            graph_configs: graphConfigs.value,
            project: props.label_dict?.get('Project'),
            search_task: props.label_dict?.get('SearchTask'),
            conflict_setting: conflictSettings.value,
            auto_run_config: autoRunConfig.value
        })
        handleClose()
    }
}

// Initialize active groups and pipe configs when component mounts
watch(() => props.modelValue, (newValue) => {
    if (newValue) {
        // Set first group as active by default
        const groupNames = Object.keys(GroupPipeDict.value)
        if (groupNames.length > 0) {
            activeGroups.value = [`pipe-${groupNames[0]}`]
            
            // Initialize direction and radius scale values
            for (const groupName of groupNames) {
                // Initialize graph configs for each group
                graphConfigs.value[groupName] = {
                    seg_node_min: 10,
                    seg_node_max: 50
                }
                
                for (const pipe of GroupPipeDict.value[groupName]) {
                    const direction = pipe.direction as number[]
                    if (direction && Array.isArray(direction)) {
                        directionValues.value[`${groupName}_${pipe.name}`] = `${direction[0]},${direction[1]},${direction[2]}`
                    }
                    radiusScaleValues.value[`${groupName}_${pipe.name}`] = pipe.radius_scale as number || 1.0
                }
            }
        }
    }
}, { immediate: true })
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

/* Update scrollbar styles */
.right-sidebar-content::-webkit-scrollbar {
    width: 4px;
}

.right-sidebar-content::-webkit-scrollbar-track {
    background: transparent;
}

.right-sidebar-content::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #cbd5e1 0%, #94a3b8 100%);
    border-radius: 6px;
    transition: all 0.2s ease;
}

.right-sidebar-content::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #94a3b8 0%, #64748b 100%);
}

.button-container {
    position: fixed;
    right: 0;
    bottom: 0;
    width: 450px;
    padding: 20px;
    border-top: 1px solid rgba(226, 232, 240, 0.6);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
    backdrop-filter: blur(20px);
    z-index: 1100;
}

.button-group {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
}

.button-group .el-button {
    min-width: 90px;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    letter-spacing: 0.025em;
}

.button-group .el-button--primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
}

.button-group .el-button--primary:hover {
    background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
}

.button-group .el-button--default {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(226, 232, 240, 0.6);
    color: #475569;
}

.button-group .el-button--default:hover {
    background: rgba(99, 102, 241, 0.08);
    border-color: rgba(99, 102, 241, 0.2);
    color: #6366f1;
    transform: translateY(-1px);
}

:deep(.el-collapse-item__header) {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(248, 250, 252, 0.8) 100%);
    border-bottom: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 8px 8px 0 0;
    transition: all 0.3s ease;
}

:deep(.el-collapse-item__header:hover) {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
}

:deep(.el-collapse-item__content) {
    padding: 0;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 0 0 8px 8px;
}

:deep(.el-table) {
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

:deep(.el-table th) {
    background: linear-gradient(135deg, rgba(248, 250, 252, 0.8) 0%, rgba(241, 245, 249, 0.8) 100%);
    color: #1e293b;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.025em;
}

:deep(.el-table td) {
    background: rgba(255, 255, 255, 0.8);
    border-bottom: 1px solid rgba(226, 232, 240, 0.3);
}

:deep(.el-table tr:hover td) {
    background: rgba(99, 102, 241, 0.05);
}

:deep(.el-input-number) {
    width: 100%;
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 8px;
    transition: all 0.3s ease;
}

:deep(.el-input-number:hover) {
    border-color: rgba(99, 102, 241, 0.3);
}

:deep(.el-input-number.is-focus) {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

:deep(.el-input-number__decrease),
:deep(.el-input-number__increase) {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(226, 232, 240, 0.6);
    color: #475569;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 6px;
}

:deep(.el-input-number__decrease:hover),
:deep(.el-input-number__increase:hover) {
    background: rgba(99, 102, 241, 0.08);
    border-color: rgba(99, 102, 241, 0.2);
    color: #6366f1;
    transform: scale(1.05);
}

:deep(.el-input-number__input) {
    background: transparent;
    color: #1e293b;
    font-weight: 500;
}

:deep(.el-input__wrapper) {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 8px;
    transition: all 0.3s ease;
}

:deep(.el-input__wrapper:hover) {
    border-color: rgba(99, 102, 241, 0.3);
}

:deep(.el-input__wrapper.is-focus) {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

:deep(.el-form-item__label) {
    color: #1e293b;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.025em;
}

.conflict-setting-section {
    margin-top: 16px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
}

.graph-config-section {
    margin-bottom: 16px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
}

.graph-config-section h4 {
    margin: 0 0 12px 0;
    color: #1e293b;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.025em;
}

.smooth-name-section {
    margin-bottom: 20px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
}

.auto-run-config-section {
    margin-top: 16px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
}

.collapse-title {
    font-weight: 500;
    color: #1e293b;
    font-size: 14px;
    letter-spacing: 0.025em;
}

/* 微动画效果 */
.group-section {
    animation: fadeInUp 0.3s ease-out;
}

.section-header {
    animation: fadeInUp 0.4s ease-out;
}

.smooth-name-section,
.conflict-setting-section,
.auto-run-config-section,
.graph-config-section {
    animation: fadeInUp 0.5s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 悬停效果增强 */
.group-section:hover {
    transform: translateY(-2px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.smooth-name-section:hover,
.conflict-setting-section:hover,
.auto-run-config-section:hover,
.graph-config-section:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
</style>