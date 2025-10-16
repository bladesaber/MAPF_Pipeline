<!-- DefineTaskObstacle.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define Obstacle</span>
      <el-button 
        link
        class="close-button"
        @click="handleClose"
      >
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <!-- Name Section -->
      <div class="section">
        <div class="section-header">
          <span>Name</span>
        </div>
        <div class="section-content">
          <el-input
            v-model="formData.name"
            placeholder="Enter obstacle name"
            :maxlength="50"
            show-word-limit
          />
        </div>
      </div>

      <!-- Obstacle Type Section -->
      <div class="section">
        <div class="section-header">
          <span>Obstacle Type</span>
        </div>
        <div class="section-content">
          <el-select 
            v-model="formData.type"
            placeholder="Select obstacle type"
            style="width: 100%"
          >
            <el-option label="Cylinder" value="CylinderObstacle" />
            <el-option label="Box" value="BoxObstacle" />
            <el-option label="SphereLine" value="SphereLineObstacle" />
          </el-select>
        </div>
      </div>

      <!-- Relax Groups Section -->
      <div class="section">
        <div class="section-header">
          <span>Relax Groups</span>
        </div>
        <div class="section-content">
          <el-select
            v-model="formData.relax_groups"
            multiple
            placeholder="Select relax groups"
            style="width: 100%"
          >
            <el-option
              v-for="group in availableRelaxGroups"
              :key="group"
              :label="group"
              :value="group"
            />
          </el-select>
        </div>
      </div>

      <!-- Dynamic Parameters Section -->
      <div v-if="formData.type" class="section">
        <div class="section-header">
          <span>Parameters</span>
        </div>
        <div class="section-content">
          <el-form label-position="top">
            <!-- Cylinder Parameters -->
            <template v-if="formData.type === 'CylinderObstacle'">
              <el-form-item label="Center Position">
                <div class="vector-input">
                  <el-input-number 
                    v-model="formData.x"
                    :precision="2"
                    :step="0.1"
                    placeholder="X"
                  />
                  <el-input-number 
                    v-model="formData.y"
                    :precision="2"
                    :step="0.1"
                    placeholder="Y"
                  />
                  <el-input-number 
                    v-model="formData.z"
                    :precision="2"
                    :step="0.1"
                    placeholder="Z"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Direction Vector">
                <div class="vector-input">
                  <el-input-number 
                    v-model="formData.vector_x"
                    :precision="2"
                    :step="0.1"
                    placeholder="X"
                  />
                  <el-input-number 
                    v-model="formData.vector_y"
                    :precision="2"
                    :step="0.1"
                    placeholder="Y"
                  />
                  <el-input-number 
                    v-model="formData.vector_z"
                    :precision="2"
                    :step="0.1"
                    placeholder="Z"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Radius">
                <el-input-number 
                  v-model="formData.cylinder_radius"
                  :precision="2"
                  :step="0.1"
                  :min="0.1"
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item label="Height">
                <el-input-number 
                  v-model="formData.height"
                  :precision="2"
                  :step="0.1"
                  :min="0.1"
                  style="width: 120px"
                />
              </el-form-item>
            </template>

            <!-- Box Parameters -->
            <template v-if="formData.type === 'BoxObstacle'">
              <el-form-item label="X Range">
                <div class="range-input">
                  <el-input-number 
                    v-model="formData.xmin"
                    :precision="2"
                    :step="0.1"
                    placeholder="Min"
                  />
                  <span class="range-separator">to</span>
                  <el-input-number 
                    v-model="formData.xmax"
                    :precision="2"
                    :step="0.1"
                    placeholder="Max"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Y Range">
                <div class="range-input">
                  <el-input-number 
                    v-model="formData.ymin"
                    :precision="2"
                    :step="0.1"
                    placeholder="Min"
                  />
                  <span class="range-separator">to</span>
                  <el-input-number 
                    v-model="formData.ymax"
                    :precision="2"
                    :step="0.1"
                    placeholder="Max"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Z Range">
                <div class="range-input">
                  <el-input-number 
                    v-model="formData.zmin"
                    :precision="2"
                    :step="0.1"
                    placeholder="Min"
                  />
                  <span class="range-separator">to</span>
                  <el-input-number 
                    v-model="formData.zmax"
                    :precision="2"
                    :step="0.1"
                    placeholder="Max"
                  />
                </div>
              </el-form-item>
            </template>

            <!-- SphereLine Parameters -->
            <template v-if="formData.type === 'SphereLineObstacle'">
              <el-form-item label="Start Point">
                <div class="vector-input">
                  <el-input-number 
                    v-model="formData.start_x"
                    :precision="2"
                    :step="0.1"
                    placeholder="X"
                  />
                  <el-input-number 
                    v-model="formData.start_y"
                    :precision="2"
                    :step="0.1"
                    placeholder="Y"
                  />
                  <el-input-number 
                    v-model="formData.start_z"
                    :precision="2"
                    :step="0.1"
                    placeholder="Z"
                  />
                </div>
              </el-form-item>
              <el-form-item label="End Point">
                <div class="vector-input">
                  <el-input-number 
                    v-model="formData.end_x"
                    :precision="2"
                    :step="0.1"
                    placeholder="X"
                  />
                  <el-input-number 
                    v-model="formData.end_y"
                    :precision="2"
                    :step="0.1"
                    placeholder="Y"
                  />
                  <el-input-number 
                    v-model="formData.end_z"
                    :precision="2"
                    :step="0.1"
                    placeholder="Z"
                  />
                </div>
              </el-form-item>
              <el-form-item label="Radius">
                <el-input-number 
                  v-model="formData.sphere_radius"
                  :precision="2"
                  :step="0.1"
                  :min="0.1"
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item label="Resolution">
                <el-input-number 
                  v-model="formData.resolution"
                  :precision="2"
                  :step="0.1"
                  :min="0.1"
                  style="width: 120px"
                />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="handleSample">Sample Points</el-button>
              </el-form-item>
              <el-form-item v-if="sampledPoints.length > 0">
                <el-table :data="sampledPoints" style="width: 100%">
                  <el-table-column label="Position">
                    <template #default="{ row }">
                      ({{ row.x.toFixed(2) }}, {{ row.y.toFixed(2) }}, {{ row.z.toFixed(2) }})
                    </template>
                  </el-table-column>
                  <el-table-column label="Radius">
                    <template #default="{ row, $index }">
                      <el-input-number 
                        v-model="row.radius"
                        :precision="2"
                        :step="0.1"
                        :min="0.1"
                        size="small"
                      />
                    </template>
                  </el-table-column>
                  <el-table-column label="Action" width="80">
                    <template #default="{ $index }">
                      <el-button 
                        type="danger" 
                        size="small"
                        @click="handleDeletePoint($index)"
                      >
                        Delete
                      </el-button>
                    </template>
                  </el-table-column>
                </el-table>
              </el-form-item>
            </template>

          </el-form>
        </div>
      </div>

      <!-- Button Container -->
      <div class="button-container">
        <div class="button-group">
          <el-button type="primary" @click="handleGenerate">Generate</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  computed,
  nextTick,
  ref,
  watch,
} from 'vue';

import { ElMessage } from 'element-plus';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean,
  label_dict: Map<string, string | number | string[] | Map<string, string|number>>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const availableRelaxGroups = computed(() => {
  const groups = props.label_dict.get('RelaxGroup')
  return Array.isArray(groups) ? groups : (groups ? [groups] : [])
})

const formData = ref({
  name: '',
  type: '',
  relax_groups: [] as string[],
  // Cylinder parameters
  x: 0,
  y: 0,
  z: 0,
  vector_x: 0,
  vector_y: 0,
  vector_z: 1,
  cylinder_radius: 1,
  height: 1,
  // Box parameters
  xmin: -1,
  xmax: 1,
  ymin: -1,
  ymax: 1,
  zmin: -1,
  zmax: 1,

  // SphereLine parameters
  start_x: 0,
  start_y: 0,
  start_z: 0,
  end_x: 1,
  end_y: 1,
  end_z: 1,
  sphere_radius: 0.5,
  resolution: 0.5
})

const sampledPoints = ref<Array<{x: number, y: number, z: number, radius: number}>>([])

const handleSample = () => {
  const start = {
    x: formData.value.start_x,
    y: formData.value.start_y,
    z: formData.value.start_z
  }
  const end = {
    x: formData.value.end_x,
    y: formData.value.end_y,
    z: formData.value.end_z
  }
  const resolution = formData.value.resolution

  // Calculate distance between points
  const dx = end.x - start.x
  const dy = end.y - start.y
  const dz = end.z - start.z
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

  // Calculate number of points based on resolution
  const numPoints = Math.max(2, Math.ceil(distance / resolution))

  // Generate points
  sampledPoints.value = []
  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1)
    sampledPoints.value.push({
      x: start.x + t * dx,
      y: start.y + t * dy,
      z: start.z + t * dz,
      radius: formData.value.sphere_radius
    })
  }
}

const handleDeletePoint = (index: number) => {
  sampledPoints.value.splice(index, 1)
}

const handleClose = () => {
  emit('update:modelValue', false)
}

const validateForm = () => {
  if (!formData.value.name.trim()) {
    ElMessage.warning('Please enter an obstacle name')
    return false
  }

  if (!formData.value.type) {
    ElMessage.warning('Please select an obstacle type')
    return false
  }

  if (formData.value.type === 'CylinderObstacle') {
    if (formData.value.vector_x === 0 && formData.value.vector_y === 0 && formData.value.vector_z === 0) {
      ElMessage.warning('Direction vector cannot be zero')
      return false
    }
  } else if (formData.value.type === 'BoxObstacle') {
    if (formData.value.xmax <= formData.value.xmin) {
      ElMessage.warning('X maximum must be greater than minimum')
      return false
    }
    if (formData.value.ymax <= formData.value.ymin) {
      ElMessage.warning('Y maximum must be greater than minimum')
      return false
    }
    if (formData.value.zmax <= formData.value.zmin) {
      ElMessage.warning('Z maximum must be greater than minimum')
      return false
    }
  } else if (formData.value.type === 'SphereLineObstacle') {
    if (sampledPoints.value.length === 0) {
      ElMessage.warning('Please sample points first')
      return false
    }
  }

  return true
}

const handleGenerate = () => {
  if (!validateForm()) return

  const outputData = {
    name: formData.value.name,
    type: formData.value.type,
    relax_groups: [...new Set(formData.value.relax_groups)],
    ...(formData.value.type === 'CylinderObstacle' ? {
      x: formData.value.x,
      y: formData.value.y,
      z: formData.value.z,
      vector_x: formData.value.vector_x,
      vector_y: formData.value.vector_y,
      vector_z: formData.value.vector_z,
      radius: formData.value.cylinder_radius,
      height: formData.value.height
    } : formData.value.type === 'BoxObstacle' ? {
      xmin: formData.value.xmin,
      xmax: formData.value.xmax,
      ymin: formData.value.ymin,
      ymax: formData.value.ymax,
      zmin: formData.value.zmin,
      zmax: formData.value.zmax
    } : formData.value.type === 'SphereLineObstacle' ? {
      points: sampledPoints.value.map(p => [p.x, p.y, p.z]),
      radius: sampledPoints.value.map(p => p.radius)
    } : {}),
    label_dict: props.label_dict
  }

  emit('generate', outputData)
  
  // Reset form
  formData.value = {
    name: '',
    type: '',
    relax_groups: [],
    x: 0,
    y: 0,
    z: 0,
    vector_x: 0,
    vector_y: 0,
    vector_z: 1,
    cylinder_radius: 1,
    height: 1,
    xmin: -1,
    xmax: 1,
    ymin: -1,
    ymax: 1,
    zmin: -1,
    zmax: 1,
    start_x: 0,
    start_y: 0,
    start_z: 0,
    end_x: 1,
    end_y: 1,
    end_z: 1,
    sphere_radius: 0.5,
    resolution: 0.5
  }
  sampledPoints.value = []
}

watch(() => props.label_dict, (newDict) => {
  if (typeof newDict.get === 'function') {
    formData.value.name = newDict.get('name') as string ?? ''
    formData.value.type = newDict.get('type') as string ?? ''
    formData.value.relax_groups = Array.isArray(newDict.get('relax_groups')) ? 
      newDict.get('relax_groups') as string[] : 
      (newDict.get('relax_groups') ? [newDict.get('relax_groups') as string] : [])
    
    // CylinderObstacle
    if (formData.value.type === 'CylinderObstacle') {
      formData.value.x = Number(newDict.get('x') ?? 0)
      formData.value.y = Number(newDict.get('y') ?? 0)
      formData.value.z = Number(newDict.get('z') ?? 0)
      formData.value.vector_x = Number(newDict.get('vector_x') ?? 0)
      formData.value.vector_y = Number(newDict.get('vector_y') ?? 0)
      formData.value.vector_z = Number(newDict.get('vector_z') ?? 1)
      formData.value.cylinder_radius = Number(newDict.get('radius') ?? 1)
      formData.value.height = Number(newDict.get('height') ?? 1)
    }
    // BoxObstacle
    else if (formData.value.type === 'BoxObstacle') {
      formData.value.xmin = Number(newDict.get('xmin') ?? -1)
      formData.value.xmax = Number(newDict.get('xmax') ?? 1)
      formData.value.ymin = Number(newDict.get('ymin') ?? -1)
      formData.value.ymax = Number(newDict.get('ymax') ?? 1)
      formData.value.zmin = Number(newDict.get('zmin') ?? -1)
      formData.value.zmax = Number(newDict.get('zmax') ?? 1)
    }
    // SphereLineObstacle
    else if (formData.value.type === 'SphereLineObstacle') {
      let points = newDict.get('points')
      let radius = newDict.get('radius')
      if (typeof points === 'string') points = JSON.parse(points)
      if (typeof radius === 'string') radius = JSON.parse(radius)
      if (Array.isArray(points) && Array.isArray(radius) && points.length === radius.length) {
        sampledPoints.value = points.map((p, i) => ({
          x: Number(p[0]),
          y: Number(p[1]),
          z: Number(p[2]),
          radius: Number(radius[i])
        }))
        if (points.length > 0) {
          formData.value.start_x = Number(points[0][0])
          formData.value.start_y = Number(points[0][1])
          formData.value.start_z = Number(points[0][2])
          formData.value.end_x = Number(points[points.length - 1][0])
          formData.value.end_y = Number(points[points.length - 1][1])
          formData.value.end_z = Number(points[points.length - 1][2])
          formData.value.sphere_radius = Number(radius[0])
        }
      } else {
        sampledPoints.value = []
      }
      formData.value.resolution = Number(newDict.get('resolution') ?? 0.5)
    }
  }
})
</script>

<style scoped>
.right-sidebar {
  position: fixed;
  right: 0;
  top: 124px; /* 36px (card-layout) + 36px (menu-bar) + 60px (menu-bar) */
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
  color: #6366f1;
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