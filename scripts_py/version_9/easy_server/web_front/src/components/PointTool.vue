<!-- PointTool.vue -->
<template>
  <div class="point-tool" v-show="modelValue">
    <div class="tool-header">
      <span>{{ pointMode === 'VectorPoint' ? 'Vector Point Tool' : 'Point Tool' }}</span>
      <el-button type="text" @click="$emit('update:modelValue', false)">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <el-form :model="formData" label-width="100px">
      <!-- Basic Point Coordinates -->
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

      <!-- Vector Components (shown only when pointMode is VectorPoint) -->
      <template v-if="pointMode === 'VectorPoint'">
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
      </template>

      <!-- Generate Button -->
      <el-form-item>
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { Close } from '@element-plus/icons-vue'

const props = defineProps<{
  modelValue: boolean,
  pointMode: string
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = reactive({
  x: 0,
  y: 0,
  z: 0,
  vectorX: 0,
  vectorY: 0,
  vectorZ: 0
})

const handleGenerate = () => {
  emit('generate', {
    type: props.pointMode === 'VectorPoint' ? 'vector_point' : 'point',
    x: formData.x,
    y: formData.y,
    z: formData.z,
    ...(props.pointMode === 'VectorPoint' && {
      vectorX: formData.vectorX,
      vectorY: formData.vectorY,
      vectorZ: formData.vectorZ
    })
  })
}
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