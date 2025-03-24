<!-- SurfaceTool.vue -->
<template>
  <div class="surface-tool" v-show="modelValue">
    <div class="tool-header">
      <span>{{ 'Surface Tool - ' + surfaceMode }}</span>
      <el-button link @click="$emit('update:modelValue', false)">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>

    <div v-if="surfaceMode === 'Disk'" class="surface-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter surface name" />
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
            :min="0.1"
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

    <div v-if="surfaceMode === 'CirclePipeShell'" class="surface-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter surface name" />
        </el-form-item>
      </el-form>

      <span>Selected Lines</span>
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

    <div v-if="surfaceMode === 'BsplineSurface'" class="surface-tool-content">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="formData.name" placeholder="Enter surface name" />
        </el-form-item>
      </el-form>

      <div class="table-section">
        <div class="table-header">
          <span>Boundary Lines</span>
        </div>
        <el-table
          :data="boundaryItems"
          style="width: 100%"
          height="150px"
        >
          <el-table-column prop="name" label="Selected" />
          <el-table-column label="Actions" width="100">
            <template #default="{ $index }">
              <el-button
                type="danger"
                size="small"
                @click="removeBoundaryItem($index)"
              >Delete
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <div class="table-section">
        <div class="table-header">
          <span>Guide Lines</span>
        </div>
        <el-table
          :data="guideItems"
          style="width: 100%"
          height="150px"
        >
          <el-table-column prop="name" label="Selected" />
          <el-table-column label="Actions" width="100">
            <template #default="{ $index }">
              <el-button
                type="danger"
                size="small"
                @click="removeGuideItem($index)"
              >Delete
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
      
      <div class="button-group">
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  modelValue: boolean,
  surfaceMode: string
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref({
  name: '',
  x: 0,
  y: 0,
  z: 0,
  radius: 1.0,
  vector_x: 0,
  vector_y: 0,
  vector_z: 1.0,
})

const selectedItems = ref<Array<{ name: string }>>([])
const boundaryItems = ref<Array<{ name: string }>>([])
const guideItems = ref<Array<{ name: string }>>([])

const removeItem = (index: number) => {
  selectedItems.value.splice(index, 1)
}

const removeBoundaryItem = (index: number) => {
  boundaryItems.value.splice(index, 1)
}

const removeGuideItem = (index: number) => {
  guideItems.value.splice(index, 1)
}

const addItem = ({name, style}: {name: string, style: string}) => {
  if (props.surfaceMode === 'CirclePipeShell') {
    if (style === 'bspline' && !selectedItems.value.some(item => item.name === name)) {
      selectedItems.value.push({ name: name })
    }
  }
  else if (props.surfaceMode === 'BsplineSurface') {
    if (style === 'bspline') {
      if (boundaryItems.value.length < 4 && !boundaryItems.value.some(item => item.name === name)) {
        boundaryItems.value.push({ name: name })
      } else if (!guideItems.value.some(item => item.name === name)) {
        guideItems.value.push({ name: name })
      }
    }
  }
}

const handleGenerate = () => {
  if (formData.value.name === '') {
    ElMessage.error('Name is required')
    return
  }

  if (props.surfaceMode === 'Disk') {
    emit('generate', {
      style: 'Disk',
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
  else if (props.surfaceMode === 'CirclePipeShell') {
    if (selectedItems.value.length === 0) {
      ElMessage.warning('Please select at least one line')
      return
    }
    emit('generate', {
      style: 'CirclePipeShell',
      name: formData.value.name,
      selectedItems: selectedItems.value
    })
  }
  else if (props.surfaceMode === 'BsplineSurface') {
    if (boundaryItems.value.length !== 4) {
      ElMessage.warning('Please select exactly 4 boundary lines')
      return
    }
    emit('generate', {
      style: 'BsplineSurface',
      name: formData.value.name,
      boundaryItems: boundaryItems.value,
      guideItems: guideItems.value
    })
  }
  
  formData.value.name = ''
  selectedItems.value = []
  boundaryItems.value = []
  guideItems.value = []
}

defineExpose({
  addItem
})
</script>

<style scoped>
.surface-tool {
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

.surface-tool-content {
  padding: 15px;
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
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

.button-group {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: auto;
  padding: 8px 0;
}

.table-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
}
</style> 