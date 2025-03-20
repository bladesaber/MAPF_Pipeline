<!-- PointTool.vue -->
<template>
  <div class="point-tool" v-show="modelValue">
    <div class="tool-header">
      <span>{{ pointMode === 'VectorPoint' ? 'Vector Point Tool' : 'Point Tool' }}</span>
      <el-button link @click="$emit('update:modelValue', false)">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>

    <div v-if="pointMode === 'Point'" class="point-tool-content">
      <el-form :model="formData" label-width="100px">
        <el-form-item label="Name">
          <el-input 
            v-model="formData.name" 
            controls-position="right"
          />
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
        <el-form-item>
          <el-button type="primary" @click="handleGenerate">Generate</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div v-if="pointMode === 'VectorPoint'" class="point-tool-content">
      <el-form :model="formData" label-width="100px">
        <el-form-item label="Name">
          <el-input 
            v-model="formData.name" 
            controls-position="right"
          />
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
        <el-form-item label="Vector X">
          <el-input-number 
            v-model="formData.vectorX" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector Y">
          <el-input-number 
            v-model="formData.vectorY" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="Vector Z">
          <el-input-number 
            v-model="formData.vectorZ" 
            :precision="3" 
            :step="0.1"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleGenerate">Generate</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div v-if="pointMode === 'PointFromSkeleton'" class="point-tool-content">
      <el-form :model="formData" label-width="100px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter point name" />
        </el-form-item>
      </el-form>

      <span>Line Points</span>
      <el-table
        :data="selectedItems"
        style="width: 100%"
        height="200px"
      >
        <el-table-column prop="name" label="Selected" />
        <el-table-column label="Number" width="100">
          <template #default="{ row }">
            <el-input-number
              v-model="row.sample_num"
              :min="10"
              :max="1000"
              :step="10"
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

    </div>

  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  modelValue: boolean,
  pointMode: string
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = reactive({
  name: '',
  x: 0,
  y: 0,
  z: 0,
  vectorX: 0,
  vectorY: 0,
  vectorZ: 0,
  sample_num: 10
})

const selectedItems = ref<Array<{ name: string, sample_num: number }>>([])

const removeItem = (index: number) => {
  selectedItems.value.splice(index, 1)
}

const addItem = ({name, style}: {name: string, style: string}) => {
  if (!selectedItems.value.some(existingItem => existingItem.name === name)) {
    if (props.pointMode === 'PointFromSkeleton') {
      if (style === 'torus') {
        selectedItems.value.push({ 
          name: name, sample_num: formData.sample_num
        })
      }
    }
  } else {
    ElMessage.info('Item already selected')
  }
}

const handleGenerate = () => {
  if (props.pointMode === 'Point') {
    emit('generate', {
      style: 'Point',
      name: formData.name,
      x: formData.x,
      y: formData.y,
      z: formData.z,
    })
  } 
  else if (props.pointMode === 'VectorPoint') {
    emit('generate', {
      style: 'VectorPoint',
      name: formData.name,
      x: formData.x,
      y: formData.y,
      z: formData.z,
      vectorX: formData.vectorX,
      vectorY: formData.vectorY,
      vectorZ: formData.vectorZ
    })
  } 
  else if (props.pointMode === 'PointFromSkeleton') {
    if (selectedItems.value.length === 0) {
      ElMessage.warning('Please select at least one item')
      return
    }
    emit('generate', {
      style: 'PointFromSkeleton',
      name: formData.name,
      selectedItems: selectedItems.value
    })
  }
  formData.name = ''
  selectedItems.value = []
}

// Expose methods to parent component
defineExpose({
  addItem
})
</script>

<style scoped>
.point-tool {
  position: fixed;
  right: 0;
  top: 60px; /* Adjust based on your top menu height */
  bottom: 0;
  width: 300px;
  padding: 20px;
  background: white;
  border-left: 1px solid #dcdfe6;
  box-shadow: -2px 0 8px rgba(0, 0, 0, 0.1);
  z-index: 100;
}

.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #dcdfe6;
}

:deep(.el-input-number) {
  width: 100%;
}

:deep(.el-select) {
  width: 100%;
}
</style> 