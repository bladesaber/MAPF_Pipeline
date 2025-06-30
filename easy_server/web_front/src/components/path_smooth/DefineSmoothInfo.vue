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
import { ref, computed, watch, inject } from 'vue'
import { Close, Plus, Delete } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import type { FormInstance } from 'element-plus'
import { CenterRoot } from '@/classes/center_root'
import * as THREE from 'three'

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
    top: 127px;
    width: 450px;
    background: #fff;
    border-left: 1px solid #dcdfe6;
    height: calc(100vh - 127px);
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
    padding-bottom: 64px;
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
    width: 3px;
}

.right-sidebar-content::-webkit-scrollbar-track {
    background: transparent;
}

.right-sidebar-content::-webkit-scrollbar-thumb {
    background: #dcdfe6;
    border-radius: 3px;
}

.right-sidebar-content::-webkit-scrollbar-thumb:hover {
    background: #c0c4cc;
}

.button-container {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 16px;
    border-top: 1px solid #dcdfe6;
    background-color: #f5f7fa;
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

.conflict-setting-section {
    margin-top: 16px;
    padding: 16px;
    border: 1px solid #dcdfe6;
    border-radius: 4px;
}

.graph-config-section {
    margin-bottom: 16px;
    padding: 12px;
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
}

.graph-config-section h4 {
    margin: 0 0 12px 0;
    color: #495057;
    font-size: 14px;
    font-weight: 600;
}

.smooth-name-section {
    margin-bottom: 16px;
    padding: 16px;
    border: 1px solid #dcdfe6;
    border-radius: 4px;
}

.auto-run-config-section {
    margin-top: 16px;
    padding: 16px;
    border: 1px solid #dcdfe6;
    border-radius: 4px;
}

.collapse-title {
    font-weight: 500;
    color: #303133;
}
</style>