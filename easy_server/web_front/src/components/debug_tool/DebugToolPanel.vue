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
        <el-form :model="form" label-position="top">
          <template v-if="mode === 'PointObstacles'">
            <el-form-item label="Name" required>
              <el-input v-model="form.name" placeholder="Enter obstacle name" :maxlength="50" show-word-limit />
            </el-form-item>
            <el-form-item label="Count">
              <el-input-number v-model="form.count" :min="1" :step="1" />
            </el-form-item>
          </template>

          <template v-if="mode === 'SphereObstacles'">
            <el-form-item label="Name" required>
              <el-input v-model="form.name" placeholder="Enter obstacle name" :maxlength="50" show-word-limit />
            </el-form-item>
            <el-form-item label="Percentage">
              <el-input-number v-model="form.percentage" :min="0" :max="1" :step="0.05" :precision="2" />
            </el-form-item>
            <el-form-item label="Radius Min">
              <el-input-number v-model="form.radius_min" :min="0.1" :step="0.1" :precision="1" />
            </el-form-item>
            <el-form-item label="Radius Max">
              <el-input-number v-model="form.radius_max" :min="0.1" :step="0.1" :precision="1" />
            </el-form-item>
          </template>

          <template v-if="mode === 'SquaresObstacles'">
            <el-form-item label="Name" required>
              <el-input v-model="form.name" placeholder="Enter obstacle name" :maxlength="50" show-word-limit />
            </el-form-item>
            <el-form-item label="Size Min">
              <el-input-number v-model="form.size_min" :min="0.1" :step="0.1" :precision="1" />
            </el-form-item>
            <el-form-item label="Size Max">
              <el-input-number v-model="form.size_max" :min="0.1" :step="0.1" :precision="1" />
            </el-form-item>
            <el-form-item label="Percentage">
              <el-input-number v-model="form.percentage" :min="0" :max="1" :step="0.05" :precision="2" />
            </el-form-item>
          </template>

          <template v-if="mode === 'RandomPipes'">
            <el-form-item label="Group Name" required>
              <el-input v-model="form.group" placeholder="Enter group name" :maxlength="50" show-word-limit />
            </el-form-item>
            <el-form-item label="Count">
              <el-input-number v-model="form.count" :min="2" :step="1" />
            </el-form-item>
            <el-form-item label="Radius">
              <el-input-number v-model="form.radius" :min="0.1" :step="0.1" :precision="1" />
            </el-form-item>
          </template>

          <template v-if="mode === 'RandomTasks'">
            <el-form-item label="Name" required>
              <el-input v-model="form.name" placeholder="Enter task name" :maxlength="50" show-word-limit />
            </el-form-item>
            <el-form-item label="Priority Count" required>
              <el-input-number v-model="form.priority_count" :min="1" :step="1" />
            </el-form-item>
            <section class="priority-run-config">
              <el-form-item label="Max Iterations">
                <el-input-number v-model="form.priority_run_config.max_iterations" :min="1" :step="1" />
              </el-form-item>
              <el-form-item label="H Weight">
                <el-input-number v-model="form.priority_run_config.h_weight" :min="0" :step="0.1" />
              </el-form-item>
              <el-form-item label="G Weight">
                <el-input-number v-model="form.priority_run_config.g_weight" :min="0" :step="0.1" />
              </el-form-item>
            </section>
            <section class="group-run-config">
              <el-form-item label="Max Iterations">
                <el-input-number v-model="form.group_run_config.max_iterations" :min="0" :step="1" style="width: 100%" />
              </el-form-item>
              <el-form-item label="With Constraint Table">
                <el-switch v-model="form.group_run_config.with_constraint_table" active-text="Yes" inactive-text="No" />
              </el-form-item>
              <el-form-item label="Planner Method">
                <el-select v-model="form.group_run_config.planner_method" placeholder="Select planner method" style="width: 100%">
                  <el-option v-for="method in plannerMethods" :key="method" :label="method" :value="method" />
                </el-select>
              </el-form-item>
              <!-- Weighted A* Search Config -->
              <template v-if="['WeightedAstarSearchBoost', 'WeightedAstarSearchBucket'].includes(form.group_run_config.planner_method)">
                <el-form-item label="G Weight">
                  <el-input-number v-model="form.group_run_config.g_weight" :min="0" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="H Weight">
                  <el-input-number v-model="form.group_run_config.h_weight" :min="0" :step="0.1" style="width: 100%" />
                </el-form-item>
              </template>

              <!-- Scaled Weighted A* Search Config -->
              <template v-if="['ScaledWeightedAstarSearchBoost', 'ScaledWeightedAstarSearchBucket'].includes(form.group_run_config.planner_method)">
                <el-form-item label="G Weight">
                  <el-input-number v-model="form.group_run_config.g_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="H Weight">
                  <el-input-number v-model="form.group_run_config.h_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="Threshold 1">
                  <el-input-number v-model="form.group_run_config.threshold1" :min="0" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="Scale Factor 1">
                  <el-input-number v-model="form.group_run_config.scale_factor1" :min="1" :step="1" :precision="0" style="width: 100%" />
                </el-form-item>
              </template>

              <!-- Double Scaled Weighted A* Search Config -->
              <template v-if="['DoubleScaledWeightedAstarSearchBoost', 'DoubleScaledWeightedAstarSearchBucket'].includes(form.group_run_config.planner_method)">
                <el-form-item label="G Weight">
                  <el-input-number v-model="form.group_run_config.g_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="H Weight">
                  <el-input-number v-model="form.group_run_config.h_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="Threshold 1">
                  <el-input-number v-model="form.group_run_config.threshold1" :min="0" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="Scale Factor 1">
                  <el-input-number v-model="form.group_run_config.scale_factor1" :min="1" :step="1" :precision="0" style="width: 100%" />
                </el-form-item>
                <el-form-item label="Threshold 2">
                  <el-input-number v-model="form.group_run_config.threshold2" :min="0" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="Scale Factor 2">
                  <el-input-number v-model="form.group_run_config.scale_factor2" :min="1" :step="1" :precision="0" style="width: 100%" />
                </el-form-item>
              </template>

              <!-- Bi-directional A* Search Config -->
              <template v-if="['BiDirectionalAstarSearchBoost', 'BiDirectionalAstarSearchBucket'].includes(form.group_run_config.planner_method)">
                <el-form-item label="G Weight">
                  <el-input-number v-model="form.group_run_config.g_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
                </el-form-item>
                <el-form-item label="H Weight">
                  <el-input-number v-model="form.group_run_config.h_weight" :min="0" :max="2" :step="0.1" style="width: 100%" />
                </el-form-item>
              </template>
            </section>
          </template>

          <template v-if="mode === 'PointObstacles'">
            <el-form-item label="Name" required>
              <el-input v-model="form.name" placeholder="Enter obstacle name" :maxlength="50" show-word-limit />
            </el-form-item>
            <el-form-item label="Percentage">
              <el-input-number v-model="form.percentage" :min="0" :max="1" :step="0.05" :precision="2" />
            </el-form-item>
          </template>

          <el-form-item>
            <el-button type="primary" @click="handleGenerate">Generate</el-button>
          </el-form-item>
        </el-form>
      </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import type { FormInstance } from 'element-plus'

interface GroupRunConfig {
  max_iterations: number
  with_constraint_table: boolean
  planner_method: string
  g_weight: number
  h_weight: number
  threshold1: number
  scale_factor1: number
  threshold2: number
  scale_factor2: number
}

interface PriorityRunConfig {
  max_iterations: number
  g_weight: number
  h_weight: number
}

interface FormData {
  name: string
  percentage: number
  radius_min: number
  radius_max: number
  size_min: number
  size_max: number
  group: string
  count: number
  radius: number
  priority_count: number
  priority_run_config: PriorityRunConfig
  group_run_config: GroupRunConfig
}

const props = defineProps<{
  modelValue: boolean,
  mode: 'PointObstacles' | 'SphereObstacles' | 'SquaresObstacles' | 'RandomPipes' | 'RandomTasks',
  label_dict: Map<string, string | number | string[] | Map<string, string|number>>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const plannerMethods = [
  'AstarSearchBoost',
  'WeightedAstarSearchBoost',
  'WeightedAstarSearchBucket',
  'ScaledWeightedAstarSearchBoost',
  'ScaledWeightedAstarSearchBucket',
  'DoubleScaledWeightedAstarSearchBoost',
  'DoubleScaledWeightedAstarSearchBucket',
  'BiDirectionalAstarSearchBoost',
  'BiDirectionalAstarSearchBucket'
]

const form = ref<FormData>({
  name: '',
  percentage: 0.25,

  // SphereObstacles fields
  radius_min: 1.0,
  radius_max: 2.0,

  // SquaresObstacles fields
  size_min: 1.0,
  size_max: 2.0,

  // RandomPipes fields
  group: '',
  count: 4,
  radius: 1,
  
  // RandomTasks fields
  priority_count: 1,
  priority_run_config: {
    max_iterations: 1000,
    g_weight: 1.0,
    h_weight: 1.0
  },
  group_run_config: {
    max_iterations: 10000,
    with_constraint_table: false,
    planner_method: 'AstarSearchBoost',
    g_weight: 1.0,
    h_weight: 1.0,
    threshold1: 1.0,
    scale_factor1: 1,
    threshold2: 1.0,
    scale_factor2: 1
  }
})

// Reset form when mode changes
watch(() => props.mode, () => {
  form.value = {
    name: '',
    percentage: 0.1,
    radius_min: 1.0,
    radius_max: 2.0,
    size_min: 1.0,
    size_max: 2.0,
    group: '',
    count: 4,
    radius: 1,
    priority_count: 1,
    priority_run_config: {
      max_iterations: 1000,
      g_weight: 1.0,
      h_weight: 1.0
    },
    group_run_config: {
      max_iterations: 100000,
      with_constraint_table: false,
      planner_method: 'AstarSearchBoost',
      g_weight: 1.0,
      h_weight: 1.0,
      threshold1: 1.0,
      scale_factor1: 1,
      threshold2: 1.0,
      scale_factor2: 1
    }
  }
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  // Validate required fields
  if (props.mode === 'SphereObstacles' && !form.value.name) {
    ElMessage.error('Please enter a name')
    return
  }
  if (props.mode === 'RandomPipes' && !form.value.group) {
    ElMessage.error('Please enter a group name')
    return
  }
  if (props.mode === 'RandomTasks' && !form.value.name) {
    ElMessage.error('Please enter a name')
    return
  }

  let emit_data = {}
  if (props.mode === 'SphereObstacles') {
    emit_data = {
      mode: props.mode,
      name: form.value.name,
      percentage: form.value.percentage,
      radius_min: form.value.radius_min,
      radius_max: form.value.radius_max,
      label_dict: props.label_dict
    }
  } else if (props.mode === 'RandomPipes') {
    emit_data = {
      mode: props.mode,
      group: form.value.group,
      count: form.value.count,
      radius: form.value.radius,
      label_dict: props.label_dict
    }
  } else if (props.mode === 'RandomTasks') {
    emit_data = {
      mode: props.mode,
      name: form.value.name,
      priority_count: form.value.priority_count,
      priority_run_config: form.value.priority_run_config,
      group_run_config: form.value.group_run_config,
      label_dict: props.label_dict
    }
  } else if (props.mode === 'PointObstacles') {
    emit_data = {
      mode: props.mode,
      name: form.value.name,
      percentage: form.value.percentage,
      label_dict: props.label_dict
    }
  } else if (props.mode === 'SquaresObstacles') {
    emit_data = {
      mode: props.mode,
      name: form.value.name,
      size_min: form.value.size_min,
      size_max: form.value.size_max,
      percentage: form.value.percentage,
      label_dict: props.label_dict
    }
  } else if (props.mode === 'PointObstacles') {
    emit_data = {
      mode: props.mode,
      name: form.value.name,
      count: form.value.count,
      label_dict: props.label_dict
    }
  }

  emit('generate', emit_data)
  handleClose()
}
</script>

<style scoped>
.right-sidebar {
  position: fixed;
  right: 0;
  top: 127px; /* 36px (card-layout) + 36px (menu-bar) + 60px (menu-bar) */
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

.priority-task-config {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
</style>