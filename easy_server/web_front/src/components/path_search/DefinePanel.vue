<!-- DefinePanel.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>{{ props.mode }}</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon>
          <Close />
        </el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <!-- Group Mode -->
      <template v-if="mode === 'PipeGroup'">
        <div class="section">
          <div class="section-header">
            <span>Name</span>
          </div>
          <div class="section-content">
            <el-input v-model="groupForm.name" placeholder="Enter pipe group name" :maxlength="50" show-word-limit />
          </div>
        </div>

        <div class="section">
          <div class="section-header">
            <span>Color</span>
          </div>
          <div class="section-content">
            <el-color-picker 
              v-model="colorHex"
              :predefine="predefineColors"
              @active-change="handleActiveColorChange"
              @visible-change="handleVisibleChange"
              :show-confirm-button="false"
            />
          </div>
        </div>
      </template>

      <!-- Pipe Mode -->
      <template v-if="mode === 'Pipe'">
        <div class="section">
          <div class="section-header">
            <span>Name</span>
          </div>
          <div class="section-content">
            <el-input v-model="pipeForm.name" placeholder="Enter pipe name" :maxlength="50" show-word-limit />
          </div>
        </div>

        <div class="section">
          <div class="section-header">
            <span>Group</span>
          </div>
          <div class="section-content">
            <el-input v-model="groupValue" :disabled="true" placeholder="Group name from current tag" />
          </div>
        </div>

        <div class="section">
          <div class="section-header">
            <span>Position</span>
          </div>
          <div class="section-content">
            <el-form label-position="top">
              <el-form-item label="Center">
                <div class="vector-input">
                  <el-input-number 
                    v-model="pipeForm.x" 
                    :min="0" 
                    :max="maxSizes.x" 
                    :precision="2" 
                    :step="gridResolution"
                    :step-strictly="true"
                    placeholder="X" />
                  <el-input-number 
                    v-model="pipeForm.y" 
                    :min="0" 
                    :max="maxSizes.y" 
                    :precision="2" 
                    :step="gridResolution"
                    :step-strictly="true"
                    placeholder="Y" />
                  <el-input-number 
                    v-model="pipeForm.z" 
                    :min="0" 
                    :max="maxSizes.z" 
                    :precision="2" 
                    :step="gridResolution"
                    :step-strictly="true"
                    placeholder="Z" />
                </div>
              </el-form-item>
            </el-form>
          </div>
        </div>

        <div class="section">
          <div class="section-header">
            <span>Direction</span>
          </div>
          <div class="section-content">
            <el-form label-position="top">
              <el-form-item label="Vector">
                <div class="vector-input">
                  <el-select v-model="pipeForm.vec_x" style="width: 80px">
                    <el-option :value="1" label="1" />
                    <el-option :value="0" label="0" />
                    <el-option :value="-1" label="-1" />
                  </el-select>
                  <el-select v-model="pipeForm.vec_y" style="width: 80px">
                    <el-option :value="1" label="1" />
                    <el-option :value="0" label="0" />
                    <el-option :value="-1" label="-1" />
                  </el-select>
                  <el-select v-model="pipeForm.vec_z" style="width: 80px">
                    <el-option :value="1" label="1" />
                    <el-option :value="0" label="0" />
                    <el-option :value="-1" label="-1" />
                  </el-select>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </div>

        <div class="section">
          <div class="section-header">
            <span>Size</span>
          </div>
          <div class="section-content">
            <el-form label-position="top">
              <el-form-item label="Radius">
                <el-input-number v-model="pipeForm.radius" :precision="2" :step="0.1" :min="0.1" style="width: 120px" />
              </el-form-item>
            </el-form>
          </div>
        </div>

        <div class="section">
          <div class="section-header">
            <span>Type</span>
          </div>
          <div class="section-content">
            <el-form label-position="top">
              <el-form-item label="Pipe Mode">
                <el-radio-group v-model="pipeForm.pipe_mode">
                  <el-radio label="inlet">Inlet</el-radio>
                  <el-radio label="outlet">Outlet</el-radio>
                  <el-radio label="intermediate">Intermediate</el-radio>
                </el-radio-group>
              </el-form-item>
            </el-form>
          </div>
        </div>
      </template>

      <!-- SearchTask Mode -->
      <template v-if="mode === 'SearchTask'">
        <el-form
          ref="mainTaskFormRef"
          :model="taskForm"
          :rules="taskRules"
          label-width="120px"
        >
          <el-form-item label="Name" prop="name">
            <el-input v-model="taskForm.name" />
          </el-form-item>
        </el-form>
      </template>

      <!-- PriorityTask Mode -->
      <template v-if="mode === 'PriorityTask'">
        <el-form ref="priorityTaskFormRef" :model="priorityTaskForm" :rules="priorityTaskRules" label-width="120px">
          <el-form-item label="Main Task">
            <el-input :model-value="label_dict?.get('SearchTask')" disabled />
          </el-form-item>
          <el-form-item label="Name" prop="name">
            <el-input v-model="priorityTaskForm.name" />
          </el-form-item>
          <el-form-item label="Max Iterations" prop="max_iterations">
            <el-input-number v-model="priorityTaskForm.max_iterations" :min="0" :step="1" style="width: 100%" />
          </el-form-item>
          <el-form-item label="G Weight" prop="g_weight">
            <el-input-number v-model="priorityTaskForm.g_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
          </el-form-item>
          <el-form-item label="H Weight" prop="h_weight">
            <el-input-number v-model="priorityTaskForm.h_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
          </el-form-item>
        </el-form>
      </template>

      <!-- GroupTask Mode -->
      <template v-if="mode === 'GroupTask'">
        <el-form ref="groupTaskFormRef" :model="groupTaskForm" :rules="groupTaskRules" label-width="120px">
          <el-form-item label="Main Task">
            <el-input :model-value="label_dict?.get('SearchTask')" disabled />
          </el-form-item>
          <el-form-item label="Priority Task">
            <el-input :model-value="label_dict?.get('PriorityTask')" disabled />
          </el-form-item>
          <el-form-item label="Name" prop="name">
            <el-select v-model="groupTaskForm.name" placeholder="Select group" style="width: 100%">
              <el-option v-for="group in (label_dict?.get('GroupNames') as string[])" :key="group" :label="group"
                :value="group" />
            </el-select>
          </el-form-item>
          <el-form-item label="Max Iterations" prop="max_iterations">
            <el-input-number v-model="groupTaskForm.max_iterations" :min="0" :step="1" style="width: 100%" />
          </el-form-item>
          <el-form-item label="With Constraint Table" prop="with_constraint_table">
            <el-switch v-model="groupTaskForm.with_constraint_table" active-text="Yes" inactive-text="No" />
          </el-form-item>
          <el-form-item label="Planner Method" prop="planner_method">
            <el-select v-model="groupTaskForm.planner_method" placeholder="Select planner method" style="width: 100%">
              <el-option v-for="method in plannerMethods" :key="method" :label="method" :value="method" />
            </el-select>
          </el-form-item>

          <!-- Weighted A* Search Config -->
          <template v-if="['WeightedAstarSearchBoost', 'WeightedAstarSearchBucket'].includes(groupTaskForm.planner_method)">
            <el-form-item label="G Weight">
              <el-input-number v-model="groupTaskForm.g_weight" :min="0" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="H Weight">
              <el-input-number v-model="groupTaskForm.h_weight" :min="0" :step="0.1" style="width: 100%" />
            </el-form-item>
          </template>

          <!-- Scaled Weighted A* Search Config -->
          <template v-if="['ScaledWeightedAstarSearchBoost', 'ScaledWeightedAstarSearchBucket'].includes(groupTaskForm.planner_method)">
            <el-form-item label="G Weight" prop="g_weight">
              <el-input-number v-model="groupTaskForm.g_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="H Weight" prop="h_weight">
              <el-input-number v-model="groupTaskForm.h_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="Threshold 1">
              <el-input-number v-model="groupTaskForm.threshold1" :min="0" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="Scale Factor 1">
              <el-input-number v-model="groupTaskForm.scale_factor1" :min="1" :step="1" :precision="0" style="width: 100%" />
            </el-form-item>
          </template>

          <!-- Double Scaled Weighted A* Search Config -->
          <template v-if="['DoubleScaledWeightedAstarSearchBoost', 'DoubleScaledWeightedAstarSearchBucket'].includes(groupTaskForm.planner_method)">
            <el-form-item label="G Weight" prop="g_weight">
              <el-input-number v-model="groupTaskForm.g_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="H Weight" prop="h_weight">
              <el-input-number v-model="groupTaskForm.h_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="Threshold 1">
              <el-input-number v-model="groupTaskForm.threshold1" :min="0" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="Scale Factor 1">
              <el-input-number v-model="groupTaskForm.scale_factor1" :min="1" :step="1" :precision="0" style="width: 100%" />
            </el-form-item>
            <el-form-item label="Threshold 2">
              <el-input-number v-model="groupTaskForm.threshold2" :min="0" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="Scale Factor 2">
              <el-input-number v-model="groupTaskForm.scale_factor2" :min="1" :step="1" :precision="0" style="width: 100%" />
            </el-form-item>
          </template>

          <!-- Bi-directional A* Search Config -->
          <template v-if="['BiDirectionalAstarSearchBoost', 'BiDirectionalAstarSearchBucket'].includes(groupTaskForm.planner_method)">
            <el-form-item label="G Weight" prop="g_weight">
              <el-input-number v-model="groupTaskForm.g_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
            </el-form-item>
            <el-form-item label="H Weight" prop="h_weight">
              <el-input-number v-model="groupTaskForm.h_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
            </el-form-item>
          </template>
          
        </el-form>
      </template>

      <!-- CellTask Mode -->
      <template v-if="mode === 'CellTask'">
        <el-form ref="cellTaskFormRef" :model="cellTaskForm" :rules="cellTaskRules" label-width="120px">
          <el-form-item label="Main Task">
            <el-input :model-value="label_dict?.get('SearchTask')" disabled />
          </el-form-item>
          <el-form-item label="Priority Task">
            <el-input :model-value="label_dict?.get('PriorityTask')" disabled />
          </el-form-item>
          <el-form-item label="Group Task">
            <el-input :model-value="label_dict?.get('GroupTask')" disabled />
          </el-form-item>
          <el-form-item label="Name" prop="name">
            <el-input v-model="cellTaskForm.name" />
          </el-form-item>
          <el-form-item label="Start Name" prop="start_name">
            <el-select v-model="cellTaskForm.start_name" placeholder="Select start name" style="width: 100%">
              <el-option v-for="pipe in (label_dict?.get('PipeNames') as string[])"
                :key="pipe" :label="pipe" :value="pipe" />
            </el-select>
          </el-form-item>
          <el-form-item label="End Name" prop="end_name">
            <el-select v-model="cellTaskForm.end_name" placeholder="Select end name" style="width: 100%">
              <el-option v-for="pipe in (label_dict?.get('PipeNames') as string[])"
                :key="pipe" :label="pipe" :value="pipe" />
            </el-select>
          </el-form-item>
          <el-form-item label="Radius" prop="radius">
            <el-input-number v-model="cellTaskForm.radius" :min="0.1" :step="0.1" style="width: 100%" />
          </el-form-item>
          <el-form-item label="Intermediate Point" prop="intermediate_point">
            <el-select v-model="cellTaskForm.intermediate_point" placeholder="Select intermediate point"
              style="width: 100%">
              <el-option v-for="point in (label_dict?.get('IntermediatePointNames') as string[])" :key="point"
                :label="point" :value="point" />
            </el-select>
          </el-form-item>
        </el-form>
      </template>

      <!-- Button Container -->
      <div class="button-container">
        <div class="button-group">
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
  nextTick,
  ref,
  watch,
} from 'vue';

import type { FormInstance } from 'element-plus';
import { ElMessage } from 'element-plus';
import * as THREE from 'three';

import { CenterRoot } from '@/classes/center_root';
import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean,
  mode: 'PipeGroup' | 'Pipe' | 'SearchTask' | 'PriorityTask' | 'GroupTask' | 'CellTask',
  label_dict: Map<string, string | number | string[] | Map<string, string|number>>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const plannerMethods = [
  'AstarSearchBoost', 'AstarSearchBucket',
  'WeightedAstarSearchBoost', 'WeightedAstarSearchBucket',
  'ScaledWeightedAstarSearchBoost', 'ScaledWeightedAstarSearchBucket',
  'DoubleScaledWeightedAstarSearchBoost', 'DoubleScaledWeightedAstarSearchBucket',
  'BiDirectionalAstarSearchBoost', 'BiDirectionalAstarSearchBucket',
]

const rootObj = inject<CenterRoot>('centerRoot', new CenterRoot(new THREE.Scene()))

// Group Form
const groupForm = ref({
  name: '',
  color: [255, 0, 0] // Default to red as RGB array
})

const predefineColors = [
  [255, 0, 0],    // Red
  [0, 255, 0],    // Green
  [0, 0, 255],    // Blue
  [255, 255, 0],  // Yellow
  [255, 0, 255],  // Magenta
  [0, 255, 255],  // Cyan
  [128, 0, 0],    // Dark Red
  [0, 128, 0],    // Dark Green
  [0, 0, 128],    // Dark Blue
  [128, 128, 0],  // Dark Yellow
  [128, 0, 128],  // Dark Magenta
  [0, 128, 128]   // Dark Cyan
]

// Pipe Form
const pipeForm = ref({
  name: '',
  group: '',
  x: 0,
  y: 0,
  z: 0,
  vec_x: 1,
  vec_y: 0,
  vec_z: 0,
  radius: 0.5,
  pipe_mode: 'inlet',
  resolution: 0.5
})

// Task Form
const taskForm = ref({
  name: ''
})
const taskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }]
}

const priorityTaskFormRef = ref<FormInstance>()
const priorityTaskForm = ref({
  name: '',
  max_iterations: 1000,
  g_weight: 1.0,
  h_weight: 1.0
})

// Add watchers for g_weight and h_weight
watch(() => priorityTaskForm.value.g_weight, (newVal) => {
  const sum = newVal + priorityTaskForm.value.h_weight
  if (sum !== 2) {
    priorityTaskForm.value.h_weight = 2 - newVal
  }
})

watch(() => priorityTaskForm.value.h_weight, (newVal) => {
  const sum = priorityTaskForm.value.g_weight + newVal
  if (sum !== 2) {
    priorityTaskForm.value.g_weight = 2 - newVal
  }
})

const priorityTaskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }],
  g_weight: [
    { required: true, message: 'Please input g weight', trigger: 'blur' },
    { validator: (rule: any, value: number, callback: any) => {
      if (value + priorityTaskForm.value.h_weight !== 2) {
        callback(new Error('Sum of g_weight and h_weight must be 2'))
      } else {
        callback()
      }
    }, trigger: 'change' }
  ],
  h_weight: [
    { required: true, message: 'Please input h weight', trigger: 'blur' },
    { validator: (rule: any, value: number, callback: any) => {
      if (priorityTaskForm.value.g_weight + value !== 2) {
        callback(new Error('Sum of g_weight and h_weight must be 2'))
      } else {
        callback()
      }
    }, trigger: 'change' }
  ]
}

const groupTaskFormRef = ref<FormInstance>()
const groupTaskForm = ref({
  name: '',
  planner_method: 'AstarSearchBoost',
  max_iterations: 10000,
  with_constraint_table: false,
  g_weight: 1.0,
  h_weight: 1.0,
  threshold1: 1.0,
  scale_factor1: 1,
  threshold2: 1.0,
  scale_factor2: 1
})

// Add watchers for g_weight and h_weight
watch(() => groupTaskForm.value.g_weight, (newVal) => {
  const sum = newVal + groupTaskForm.value.h_weight
  if (sum !== 2) {
    groupTaskForm.value.h_weight = 2 - newVal
  }
})

watch(() => groupTaskForm.value.h_weight, (newVal) => {
  const sum = groupTaskForm.value.g_weight + newVal
  if (sum !== 2) {
    groupTaskForm.value.g_weight = 2 - newVal
  }
})

const groupTaskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }],
  planner_method: [{ required: true, message: 'Please select planner method', trigger: 'change' }],
  g_weight: [
    { required: true, message: 'Please input g weight', trigger: 'blur' },
    { validator: (rule: any, value: number, callback: any) => {
      if (value + groupTaskForm.value.h_weight !== 2) {
        callback(new Error('Sum of g_weight and h_weight must be 2'))
      } else {
        callback()
      }
    }, trigger: 'change' }
  ],
  h_weight: [
    { required: true, message: 'Please input h weight', trigger: 'blur' },
    { validator: (rule: any, value: number, callback: any) => {
      if (groupTaskForm.value.g_weight + value !== 2) {
        callback(new Error('Sum of g_weight and h_weight must be 2'))
      } else {
        callback()
      }
    }, trigger: 'change' }
  ]
}

const cellTaskFormRef = ref<FormInstance>()
const cellTaskForm = ref({
  name: '',
  start_name: '',
  end_name: '',
  radius: 1.0,
  intermediate_point: '',
})
const cellTaskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }],
  start_name: [{ required: true, message: 'Please select start name', trigger: 'change' }],
  end_name: [{ required: true, message: 'Please select end name', trigger: 'change' }],
  radius: [{ required: true, message: 'Please input radius', trigger: 'blur' }]
}

const groupValue = computed(() => {
  if (!props.label_dict) return ''
  const group = props.label_dict.get('PipeGroup')
  return typeof group === 'string' ? group : (Array.isArray(group) ? group[0] : '')
})

const maxSizes = computed(() => {
  if (!props.label_dict) return { x: 0, y: 0, z: 0 }
  const xSize = props.label_dict.get('XSize') as number
  const ySize = props.label_dict.get('YSize') as number
  const zSize = props.label_dict.get('ZSize') as number
  return {
    x: xSize,
    y: ySize,
    z: zSize,
  }
})

const gridResolution = computed(() => {
  const project = rootObj.proj_dict.get(props.label_dict?.get('Project') as string)
  return project!.grid.resolution
})

// Methods
const handleClose = () => {
  emit('update:modelValue', false)
}

// 新增 colorHex 用于 el-color-picker v-model
const colorHex = ref('#ff0000')

// 新增 lastActiveColor 用于记录最后一次选择的颜色
const lastActiveColor = ref(colorHex.value)

const handleActiveColorChange = (val: string) => {
  lastActiveColor.value = val
  if (val && typeof val === 'string' && val.startsWith('#')) {
    const r = parseInt(val.slice(1, 3), 16)
    const g = parseInt(val.slice(3, 5), 16)
    const b = parseInt(val.slice(5, 7), 16)
    groupForm.value.color = [r, g, b]
  }
}

const handleVisibleChange = (visible: boolean) => {
  if (!visible && lastActiveColor.value !== colorHex.value) {
    colorHex.value = lastActiveColor.value
  }
}

const handleApply = async () => {
  let formRef: FormInstance | undefined
  let formData: any
  let isValid = true

  switch (props.mode) {
    case 'PipeGroup':
      if (!groupForm.value.name.trim()) {
        ElMessage.warning('Please enter a pipe group name')
        isValid = false
      }
      if (!groupForm.value.color || !Array.isArray(groupForm.value.color) || groupForm.value.color.length !== 3) {
        ElMessage.warning('Please select a valid color')
        isValid = false
      }
      formData = { 
        mode: 'PipeGroup', 
        ...groupForm.value,
        color: groupForm.value.color || [255, 0, 0] // Default to red if no color selected
      }
      break

    case 'Pipe':
      if (!pipeForm.value.name.trim()) {
        ElMessage.warning('Please enter a pipe name')
        isValid = false
      }
      if (!groupValue.value) {
        ElMessage.warning('No pipe group is selected')
        isValid = false
      }
      if (pipeForm.value.vec_x === 0 && pipeForm.value.vec_y === 0 && pipeForm.value.vec_z === 0) {
        ElMessage.warning('Direction vector cannot be zero')
        isValid = false
      }
      formData = {
        mode: 'Pipe',
        ...pipeForm.value,
        group: groupValue.value,
        pipe_mode: pipeForm.value.pipe_mode
      }
      break

    case 'SearchTask':
      if (!taskForm.value.name.trim()) {
        ElMessage.warning('Please enter a task name')
        isValid = false
      }
      formData = { mode: 'SearchTask', ...taskForm.value }
      break

    case 'PriorityTask':
      formData = {
        mode: 'PriorityTask',
        ...priorityTaskForm.value,
        main_task_name: props.label_dict?.get('SearchTask'),
        run_cfg: {
          'max_iterations': priorityTaskForm.value.max_iterations,
          'g_weight': priorityTaskForm.value.g_weight,
          'h_weight': priorityTaskForm.value.h_weight,
        }
      }
      break

    case 'GroupTask':
      formData = {
        mode: 'GroupTask',
        ...groupTaskForm.value,
        main_task_name: props.label_dict?.get('SearchTask'),
        priority_task_name: props.label_dict?.get('PriorityTask'),
        run_cfg: {
          'planner_method': groupTaskForm.value.planner_method,
          'max_iterations': groupTaskForm.value.max_iterations,
          'with_constraint_table': groupTaskForm.value.with_constraint_table,
          'g_weight': groupTaskForm.value.g_weight,
          'h_weight': groupTaskForm.value.h_weight,
          'threshold1': groupTaskForm.value.threshold1,
          'scale_factor1': groupTaskForm.value.scale_factor1,
          'threshold2': groupTaskForm.value.threshold2,
          'scale_factor2': groupTaskForm.value.scale_factor2,
        }
      }
      break

    case 'CellTask':
      formData = {
        mode: 'CellTask',
        ...cellTaskForm.value,
        main_task_name: props.label_dict?.get('SearchTask'),
        priority_task_name: props.label_dict?.get('PriorityTask'),
        group_task_name: props.label_dict?.get('GroupTask'),
      }
      break
  }

  if (isValid) {
    formData.label_dict = props.label_dict
    emit('generate', formData)
    emit('update:modelValue', false)
  }
}

watch(() => props.label_dict, (newDict) => {
  if (typeof newDict.get === 'function') {
    pipeForm.value.name = newDict.get('name') as string ?? ''
    pipeForm.value.group = newDict.get('group') as string ?? ''
    pipeForm.value.x = Number(newDict.get('x') ?? 0)
    pipeForm.value.y = Number(newDict.get('y') ?? 0)
    pipeForm.value.z = Number(newDict.get('z') ?? 0)
    pipeForm.value.vec_x = Number(newDict.get('vec_x') ?? 1)
    pipeForm.value.vec_y = Number(newDict.get('vec_y') ?? 0)
    pipeForm.value.vec_z = Number(newDict.get('vec_z') ?? 0)
    pipeForm.value.radius = Number(newDict.get('radius') ?? 0.5)
    pipeForm.value.pipe_mode = newDict.get('pipe_mode') as string ?? 'inlet'
  }
})

watch(() => props.modelValue, (val) => {
  if (!val) {
    // 弹窗关闭时清空
    cellTaskForm.value = {
      name: '',
      start_name: '',
      end_name: '',
      radius: 1.0,
      intermediate_point: '',
    }
  } else {
    // 弹窗打开时用label_dict内容填充
    if (props.label_dict && typeof props.label_dict.get === 'function') {
      cellTaskForm.value.name = props.label_dict.get('name') as string ?? ''
      cellTaskForm.value.start_name = props.label_dict.get('start_name') as string ?? ''
      cellTaskForm.value.end_name = props.label_dict.get('end_name') as string ?? ''
      cellTaskForm.value.radius = Number(props.label_dict.get('radius') ?? 1.0)
      cellTaskForm.value.intermediate_point = props.label_dict.get('intermediate_point') as string ?? ''
    }
  }
})
</script>

<style scoped>
.right-sidebar {
  position: fixed;
  right: 0;
  top: 116px; /* 36px (card-layout) + 36px (menu-bar) + 60px (menu-bar) */
  /* bottom: 150px; */
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
  padding-bottom: 64px;
  min-height: 0;
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

.section {
  margin-bottom: 24px;
}

.section-header {
  font-weight: 500;
  margin-bottom: 16px;
  color: #606266;
}

.section-content {
  padding: 0 8px;
}

.vector-input,
.range-input {
  display: flex;
  align-items: center;
  gap: 8px;
}

.vector-input :deep(.el-input-number),
.range-input :deep(.el-input-number) {
  width: 120px;
}

.range-separator {
  color: #909399;
  font-size: 14px;
}

:deep(.el-form-item) {
  margin-bottom: 16px;
}

:deep(.el-form-item__label) {
  padding-bottom: 4px;
  line-height: 1.5;
}

.button-container {
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
</style>