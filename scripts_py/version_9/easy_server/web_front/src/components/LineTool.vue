<!-- LineTool.vue -->
<template>
  <div class="line-tool" v-show="modelValue">
    <div class="tool-header">
      <span>{{ 'Line Tool - ' + lineMode }}</span>
      <el-button link @click="$emit('update:modelValue', false)">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>

    <div v-if="lineMode === 'Line'" class="line-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter line name" />
        </el-form-item>
      </el-form>

      <span>Line Points</span>
      <el-table
        :data="selectedItems"
        style="width: 100%"
        height="200px"
      >
        <el-table-column prop="name" label="Selected" />
        <el-table-column label="Actions" width="100">
          <template #default="{ $index }">
            <el-button
              type="danger"
              size="small"
              @click="removeItem($index)"
            >Delete
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <div class="button-group">
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </div>
    </div>

    <div v-if="lineMode === 'Circle'" class="line-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter line name" />
        </el-form-item>
        <el-form-item label="X">
          <el-input-number 
            v-model="formData.x" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Y">
          <el-input-number 
            v-model="formData.y" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Z">
          <el-input-number 
            v-model="formData.z" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Radius">
          <el-input-number 
            v-model="formData.radius" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector X">
          <el-input-number 
            v-model="formData.vector_x" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector Y">
          <el-input-number 
            v-model="formData.vector_y" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector Z">
          <el-input-number 
            v-model="formData.vector_z" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
      </el-form>
      <div class="button-group">
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </div>
    </div>

    <div v-if="lineMode === 'Arc'" class="line-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter line name" />
        </el-form-item>
        <el-form-item label="X">
          <el-input-number 
            v-model="formData.x" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Y">
          <el-input-number 
            v-model="formData.y" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Z">
          <el-input-number 
            v-model="formData.z" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Radius">
          <el-input-number 
            v-model="formData.radius" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector X">
          <el-input-number 
            v-model="formData.vector_x" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector Y">
          <el-input-number 
            v-model="formData.vector_y" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector Z">
          <el-input-number 
            v-model="formData.vector_z" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Start Angle">
          <el-input-number 
            v-model="formData.start_angle" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="End Angle">
          <el-input-number 
            v-model="formData.end_angle" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
      </el-form>
      <div class="button-group">
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </div>
    </div>

    <div v-if="lineMode === 'Bspline'" class="line-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter line name" />
        </el-form-item>
        <el-form-item label="Degree">
          <el-input-number 
            v-model="formData.degree" 
            :min="1"
            :max="10"
            :step="1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Sample Number">
          <el-input-number 
            v-model="formData.sample_num" 
            :min="10"
            :max="1000"
            :step="10"
            controls-position="right"
          />
        </el-form-item>
      </el-form>
      
      <span>Control Points</span>
      <el-table
        :data="selectedItems"
        style="width: 100%"
        height="200px"
      >
        <el-table-column prop="name" label="Selected" />
        <el-table-column label="Actions" width="100">
          <template #default="{ $index }">
            <el-button
              type="danger"
              size="small"
              @click="removeItem($index)"
            >Delete
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <div class="button-group">
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </div>
    </div>

    <div v-if="lineMode === 'Skeleton'" class="line-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter line name" />
        </el-form-item>
        <el-form-item label="Radius">
          <el-input-number 
            v-model="formData.skeleton_radius" 
            :precision="3"
            :step="0.1"
            :min="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Sample Number">
          <el-input-number 
            v-model="formData.skeleton_sample_num" 
            :min="10"
            :max="1000"
            :step="10"
            controls-position="right"
          />
        </el-form-item>
      </el-form>

      <div class="table-header">
        <span>{{ formData.spline_name }}</span>
        <div class="table-buttons">
          <el-button type="primary" size="small" @click="addSkeletonRow">Add</el-button>
          <el-button type="success" size="small" @click="sampleSkeletonPoints">Sample</el-button>
        </div>
      </div>

      <el-table
        :data="selectedItems"
        style="width: 100%"
        height="200px"
      >
        <el-table-column label="Percentage" width="100">
          <template #default="{ row }">
            <el-input-number 
              v-model="row.percentage"
              :precision="3"
              :step="0.1"
              :min="0"
              :max="1.0"
              size="small"
              controls-position="right"
            />
          </template>
        </el-table-column>
        <el-table-column label="Radius" width="100">
          <template #default="{ row }">
            <el-input-number 
              v-model="row.radius"
              :precision="3"
              :step="0.1"
              :min="0.1"
              size="small"
              controls-position="right"
            />
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="100">
          <template #default="{ $index }">
            <el-button
              type="danger"
              size="small"
              @click="removeItem($index)"
            >Delete
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="button-group">
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </div>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, watch, inject } from 'vue'
import { Close, Delete, Mouse } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { pass } from 'three/tsl';

const props = defineProps<{
  modelValue: boolean,
  lineMode: string
}>()
const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref({
  name: '',
  x: 0,
  y: 0,
  z: 0,
  radius: 0,
  vector_x: 0,
  vector_y: 0,
  vector_z: 0,
  start_angle: 0,
  end_angle: 0,
  degree: 3,
  sample_num: 50,
  skeleton_radius: 1.0,
  skeleton_sample_num: 100,
  spline_name: '',
})

const selectedItems = ref<Array<{ name?: string, radius?: number, percentage?: number }>>([])

const removeItem = (index: number) => {
  selectedItems.value.splice(index, 1)
}

const addItem = ({name, style}: {name: string, style: string}) => {
  if (!selectedItems.value.some(existingItem => existingItem.name === name)) {
    if (props.lineMode === 'Line') {
      if (style === 'point' || style === 'vec_point') {
        selectedItems.value.push({ name: name })
      }
    } 
    else if (props.lineMode === 'Bspline') {
      if (style === 'point' || style === 'vec_point') {
        selectedItems.value.push({ name: name })
      }
    }
    else if (props.lineMode === 'Skeleton') {
      if (style === 'bspline') {
        formData.value.spline_name = name
      }
    }
  } else {
    ElMessage.info('Item already selected')
  }
}

const handleGenerate = () => {
  if (selectedItems.value.length === 0) {
    ElMessage.warning('Please select at least one item')
    return
  }
  if (props.lineMode === 'Line') {
    emit('generate', {
      style: 'Line',
      name: formData.value.name,
      selectedItems: selectedItems.value
    })
  }
  else if (props.lineMode === 'Circle') {
    emit('generate', {
      style: 'Circle',
      name: formData.value.name,
      x: formData.value.x,
      y: formData.value.y,
      z: formData.value.z,
      radius: formData.value.radius,
      vector_x: formData.value.vector_x,
      vector_y: formData.value.vector_y,
      vector_z: formData.value.vector_z,
    })
  }
  else if (props.lineMode === 'Arc') {
    emit('generate', {
      style: 'Arc',
      name: formData.value.name,
      x: formData.value.x,
      y: formData.value.y,
      z: formData.value.z,
      radius: formData.value.radius,
      vector_x: formData.value.vector_x,
      vector_y: formData.value.vector_y,
      vector_z: formData.value.vector_z,
      start_angle: formData.value.start_angle,
      end_angle: formData.value.end_angle,
    })
  }
  else if (props.lineMode === 'Bspline') {
    emit('generate', {
      style: 'Bspline',
      name: formData.value.name,
      degree: formData.value.degree,
      sample_num: formData.value.sample_num,
      selectedItems: selectedItems.value
    })
  }
  else if (props.lineMode === 'Skeleton') {
    emit('generate', {
      style: 'Skeleton',
      name: formData.value.name,
      selectedItems: selectedItems.value
    })
  }
  else {
    ElMessage.warning('Invalid line mode')
  }
  formData.value.name = ''
  selectedItems.value = []
}

const addSkeletonRow = () => {
  if (formData.value.spline_name === '') {
    ElMessage.warning('Please select a line first')
    return
  }
  selectedItems.value.push({
    percentage: 0.0,
    radius: formData.value.skeleton_radius
  })
}

const sampleSkeletonPoints = () => {
  if (formData.value.spline_name === '') {
    ElMessage.warning('Please select a line first')
    return
  }
  const numPoints = formData.value.skeleton_sample_num
  selectedItems.value = []
  
  for (let i = 0; i < numPoints; i++) {
    const percentage = (i / (numPoints - 1)) * 100
    selectedItems.value.push({
      percentage: percentage,
      radius: formData.value.skeleton_radius
    })
  }
}

// Expose methods to parent component
defineExpose({
  addItem
})

</script>

<style scoped>
.line-tool {
  position: fixed;
  top: 60px;
  right: 0;
  width: 400px;
  height: calc(100vh - 60px);
  background: white;
  border-left: 1px solid #dcdfe6;
  display: flex;
  flex-direction: column;
  z-index: 1000;
}

.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: #f5f7fa;
  border-bottom: 1px solid #dcdfe6;
  font-weight: bold;
}

.line-tool-content {
  padding: 15px;
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.table-frame {
  margin: 20px 0;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  height: 200px;
  background-color: #f5f7fa;
  display: flex;
  flex-direction: column;
}

.frame-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid #dcdfe6;
  background: white;
  font-weight: bold;
}

.table-content {
  flex: 1;
  overflow: hidden;
  position: relative;
}

.empty-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #909399;
  gap: 8px;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: white;
}

.empty-state .el-icon {
  font-size: 24px;
}

.button-group {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: auto;
  padding: 8px 0;
}

:deep(.el-form-item) {
  margin-bottom: 12px;
}

:deep(.el-form-item__label) {
  padding-right: 8px;
}

:deep(.el-input-number) {
  width: 100%;
}

:deep(.el-table) {
  --el-table-border-color: #dcdfe6;
  --el-table-header-bg-color: #f5f7fa;
  margin: 8px 0;
}

:deep(.el-table__header) {
  background-color: #f5f7fa;
  padding: 4px 0;
}

:deep(.el-table__row) {
  cursor: pointer;
  padding: 4px 0;
}

:deep(.el-table__row:hover) {
  background-color: #f5f7fa;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.table-buttons {
  display: flex;
  gap: 8px;
}
</style> 