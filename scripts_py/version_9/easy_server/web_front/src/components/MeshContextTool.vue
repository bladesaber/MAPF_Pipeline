<!-- MeshContextTool.vue -->
<template>
  <div class="mesh-context-tool" v-show="modelValue">
    <div class="tool-header">
      <span>{{ mode === 'Move' ? 'Move Tool' : 'Edit Tool' }}</span>
      <el-button type="text" @click="$emit('update:modelValue', false)">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <el-form :model="formData" label-width="100px">
      <!-- Basic Position Coordinates -->
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

      <!-- Vector Components (shown only when mode is Edit) -->
      <template v-if="mode === 'Edit'">
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

      <!-- Apply Button -->
      <el-form-item>
        <el-button type="primary" @click="handleApply">Apply</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue'
import { Close } from '@element-plus/icons-vue'

const props = defineProps<{
  modelValue: boolean,
  mode: string
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'apply': [data: {
    type: string,
    x: number,
    y: number,
    z: number,
    vectorX?: number,
    vectorY?: number,
    vectorZ?: number
  }]
}>()

const formData = reactive({
  x: 0,
  y: 0,
  z: 0,
  vectorX: 0,
  vectorY: 0,
  vectorZ: 0
})

// Reset form data when mode changes
watch(() => props.mode, () => {
  formData.x = 0
  formData.y = 0
  formData.z = 0
  formData.vectorX = 0
  formData.vectorY = 0
  formData.vectorZ = 0
})

const handleApply = () => {
  emit('apply', {
    type: props.mode,
    x: formData.x,
    y: formData.y,
    z: formData.z,
    ...(props.mode === 'Edit' && {
      vectorX: formData.vectorX,
      vectorY: formData.vectorY,
      vectorZ: formData.vectorZ
    })
  })
  // Close the tool after applying changes
  emit('update:modelValue', false)
}
</script>

<style scoped>
.mesh-context-tool {
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