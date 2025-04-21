<!-- DefinePathSearchCfg.vue -->
<template>
    <div v-if="modelValue" class="right-sidebar">
        <div class="right-sidebar-header">
        <span>{{ getTitle }}</span>
        <el-button 
            link
            class="close-button"
            @click="handleClose"
        >
            <el-icon><Close /></el-icon>
        </el-button>
        </div>
        <div class="right-sidebar-content">
      <!-- MainTask Mode -->
      <template v-if="pathSearchCfgMode === 'MainTask'">
        <el-form
          ref="mainTaskFormRef"
          :model="mainTaskForm"
          :rules="mainTaskRules"
          label-width="120px"
        >
          <el-form-item label="Name" prop="name">
            <el-input v-model="mainTaskForm.name" />
          </el-form-item>
        </el-form>
      </template>

      <!-- PriorityTask Mode -->
      <template v-if="pathSearchCfgMode === 'PriorityTask'">
        <el-form
          ref="priorityTaskFormRef"
          :model="priorityTaskForm"
          :rules="priorityTaskRules"
          label-width="120px"
        >
          <el-form-item label="Main Task">
            <el-input 
              :model-value="currentTagDict.get('MainTask')" 
              disabled
            />
          </el-form-item>
          <el-form-item label="Name" prop="name">
            <el-input v-model="priorityTaskForm.name" />
          </el-form-item>
        </el-form>
      </template>

      <!-- GroupTask Mode -->
      <template v-if="pathSearchCfgMode === 'GroupTask'">
        <el-form
          ref="groupTaskFormRef"
          :model="groupTaskForm"
          :rules="groupTaskRules"
          label-width="120px"
        >
          <el-form-item label="Main Task">
            <el-input 
              :model-value="currentTagDict.get('MainTask')" 
              disabled
            />
          </el-form-item>
          <el-form-item label="Priority Task">
            <el-input 
              :model-value="currentTagDict.get('PriorityTask')" 
              disabled
            />
          </el-form-item>
          <el-form-item label="Name" prop="name">
            <el-select
              v-model="groupTaskForm.name"
              placeholder="Select group"
              style="width: 100%"
            >
              <el-option
                v-for="group in (currentTagDict.get('GroupNames') as string[])"
                :key="group"
                :label="group"
                :value="group"
              />
            </el-select>
          </el-form-item>
        </el-form>
      </template>

      <!-- CellTask Mode -->
      <template v-if="pathSearchCfgMode === 'CellTask'">
        <el-form
          ref="cellTaskFormRef"
          :model="cellTaskForm"
          :rules="cellTaskRules"
          label-width="120px"
        >
          <el-form-item label="Main Task">
            <el-input 
              :model-value="currentTagDict.get('MainTask')" 
              disabled
            />
          </el-form-item>
          <el-form-item label="Priority Task">
            <el-input 
              :model-value="currentTagDict.get('PriorityTask')" 
              disabled
            />
          </el-form-item>
          <el-form-item label="Group Task">
            <el-input 
              :model-value="currentTagDict.get('GroupTask')" 
              disabled
            />
          </el-form-item>
          <el-form-item label="Name" prop="name">
            <el-input v-model="cellTaskForm.name" />
          </el-form-item>
          <el-form-item label="Start Points" prop="start_points">
            <el-select
              v-model="cellTaskForm.start_points"
              multiple
              placeholder="Select start points"
              style="width: 100%"
            >
              <el-option
                v-for="pipe in (currentTagDict.get('GroupPipesRadiusDict') as Map<string, number>).keys()"
                :key="pipe"
                :label="pipe"
                :value="pipe"
              />
            </el-select>
          </el-form-item>
          <el-form-item label="End Points" prop="end_points">
            <el-select
              v-model="cellTaskForm.end_points"
              multiple
              placeholder="Select end points"
              style="width: 100%"
            >
              <el-option
                v-for="pipe in (currentTagDict.get('GroupPipesRadiusDict') as Map<string, number>).keys()"
                :key="pipe"
                :label="pipe"
                :value="pipe"
              />
            </el-select>
          </el-form-item>
          <el-form-item label="Radius" prop="radius">
            <el-input-number 
              v-model="cellTaskForm.radius" 
              :min="0.1" 
              :step="0.1" 
              style="width: 100%"
            />
          </el-form-item>
          <el-form-item label="Intermediate Points">
            <el-select
              v-model="cellTaskForm.intermediate_points"
              multiple
              placeholder="Select intermediate points"
              style="width: 100%"
            >
              <el-option
                v-for="point in (currentTagDict.get('IntermediatePointNames') as string[])"
                :key="point"
                :label="point"
                :value="point"
              />
            </el-select>
          </el-form-item>
        </el-form>
      </template>

      <div class="button-container">
        <div class="button-group">
          <el-button type="primary" @click="handleConfirm">Generate</el-button>
            </div>
        </div>
    </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { FormInstance } from 'element-plus'
import { inject } from 'vue'
import { Close } from '@element-plus/icons-vue'
import type { BlockRoot } from '../../classes/pathSearch_classes/base_root'

const props = defineProps<{
  modelValue: boolean
  pathSearchCfgMode: string
  currentTagDict: Map<string, string | string[] | Map<string, number>>
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'generate', value: any): void
}>()

const blockRoot = inject('blockRoot') as BlockRoot

const handleClose = () => {
  emit('update:modelValue', false)
}

const getTitle = computed(() => {
  switch (props.pathSearchCfgMode) {
    case 'MainTask':
      return 'Add Main Task'
    case 'PriorityTask':
      return 'Add Priority Task'
    case 'GroupTask':
      return 'Add Group Task'
    case 'CellTask':
      return 'Add Cell Task'
    default:
      return 'Add Task'
  }
})

// MainTask Form
const mainTaskFormRef = ref<FormInstance>()
const mainTaskForm = ref({
  name: ''
})
const mainTaskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }]
}

// PriorityTask Form
const priorityTaskFormRef = ref<FormInstance>()
const priorityTaskForm = ref({
  name: ''
})
const priorityTaskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }]
}

// GroupTask Form
const groupTaskFormRef = ref<FormInstance>()
const groupTaskForm = ref({
  name: ''
})
const groupTaskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }]
}

// CellTask Form
const cellTaskFormRef = ref<FormInstance>()
const cellTaskForm = ref({
  name: '',
  start_points: [] as string[],
  end_points: [] as string[],
  radius: 1.0,
  intermediate_points: [] as string[],
})
const cellTaskRules = {
  name: [{ required: true, message: 'Please input name', trigger: 'blur' }],
  start_points: [{ required: true, message: 'Please select start points', trigger: 'change' }],
  end_points: [{ required: true, message: 'Please select end points', trigger: 'change' }],
  radius: [{ required: true, message: 'Please input radius', trigger: 'blur' }]
}

// Get available pipes for the current group
const availablePipes = computed(() => {
  if (props.pathSearchCfgMode !== 'CellTask') return []
  const groupPipesRadiusDict = props.currentTagDict.get('GroupPipesRadiusDict') as Map<string, number>
  return Array.from(groupPipesRadiusDict.keys())
})

const handleConfirm = async () => {
  let formRef: FormInstance | undefined
  let formData: any

  switch (props.pathSearchCfgMode) {
    case 'MainTask':
      formRef = mainTaskFormRef.value
      formData = mainTaskForm.value
      break
    case 'PriorityTask':
      formRef = priorityTaskFormRef.value
      formData = {
        ...priorityTaskForm.value,
        main_task_name: props.currentTagDict.get('MainTask')
      }
      break
    case 'GroupTask':
      formRef = groupTaskFormRef.value
      formData = {
        ...groupTaskForm.value,
        main_task_name: props.currentTagDict.get('MainTask'),
        priority_task_name: props.currentTagDict.get('PriorityTask')
      }
      break
    case 'CellTask':
      formRef = cellTaskFormRef.value
      formData = {
        ...cellTaskForm.value,
        main_task_name: props.currentTagDict.get('MainTask'),
        priority_task_name: props.currentTagDict.get('PriorityTask'),
        group_task_name: props.currentTagDict.get('GroupTask'),
        run_cfg: new Map()
      }
      break
  }

  if (!formRef) return

  await formRef.validate((valid) => {
    if (valid) {
      emit('generate', formData)
      emit('update:modelValue', false)
      // Reset form
      if (formRef) {
        switch (props.pathSearchCfgMode) {
          case 'MainTask':
            mainTaskForm.value.name = ''
            break
          case 'PriorityTask':
            priorityTaskForm.value.name = ''
            break
          case 'GroupTask':
            groupTaskForm.value.name = ''
            break
          case 'CellTask':
            cellTaskForm.value = {
              name: '',
              start_points: [],
              end_points: [],
              radius: 1.0,
              intermediate_points: [],
            }
            break
        }
      }
    }
  })
}
</script>

<style scoped>
.right-sidebar {
  position: fixed;
  top: 0;
  right: 0;
  width: 400px;
  height: 100vh;
  background: white;
  box-shadow: -2px 0 8px rgba(0,0,0,0.15);
  z-index: 1000;
  display: flex;
  flex-direction: column;
}

.right-sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #dcdfe6;
}

.right-sidebar-content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.button-container {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #dcdfe6;
}

.button-group {
  display: flex;
  justify-content: flex-end;
}

.close-button {
  padding: 2px;
}
</style> 