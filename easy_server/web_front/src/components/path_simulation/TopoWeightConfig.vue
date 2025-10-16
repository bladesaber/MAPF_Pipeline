<!-- TopoWeightConfig.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>拓扑权重配置</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>

    <div class="right-sidebar-content">
      <!-- 权重分布配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header">
          <span>权重分布配置</span>
        </div>
        <div class="section-content">
          <el-form label-width="120px">
            <el-form-item label="权重类型">
              <el-select v-model="weightConfig.type" style="width: 100%">
                <el-option label="圆形分布" value="circular" />
                <el-option label="矩形分布" value="rectangular" />
                <el-option label="自定义分布" value="custom" />
              </el-select>
            </el-form-item>
            
            <el-form-item v-if="weightConfig.type === 'circular'" label="中心点">
              <div class="vector-input">
                <el-input-number v-model="weightConfig.center[0]" :precision="2" />
                <el-input-number v-model="weightConfig.center[1]" :precision="2" />
              </div>
            </el-form-item>
            
            <el-form-item v-if="weightConfig.type === 'circular'" label="半径">
              <el-input-number v-model="weightConfig.radius" :min="0" :precision="3" />
            </el-form-item>
            
            <el-form-item v-if="weightConfig.type === 'rectangular'" label="矩形范围">
              <div class="rect-input">
                <div class="rect-row">
                  <span>X: </span>
                  <el-input-number v-model="weightConfig.rect.x_min" :precision="2" />
                  <span> - </span>
                  <el-input-number v-model="weightConfig.rect.x_max" :precision="2" />
                </div>
                <div class="rect-row">
                  <span>Y: </span>
                  <el-input-number v-model="weightConfig.rect.y_min" :precision="2" />
                  <span> - </span>
                  <el-input-number v-model="weightConfig.rect.y_max" :precision="2" />
                </div>
              </div>
            </el-form-item>
            
            <el-form-item label="权重值">
              <el-input-number v-model="weightConfig.weight_value" :min="0" :precision="0" />
            </el-form-item>
            
            <el-form-item label="权重函数">
              <el-select v-model="weightConfig.function" style="width: 100%">
                <el-option label="常数" value="constant" />
                <el-option label="高斯分布" value="gaussian" />
                <el-option label="线性分布" value="linear" />
                <el-option label="指数分布" value="exponential" />
              </el-select>
            </el-form-item>
            
            <el-form-item v-if="weightConfig.function !== 'constant'" label="衰减参数">
              <el-input-number v-model="weightConfig.decay_param" :min="0" :precision="3" />
            </el-form-item>
          </el-form>
        </div>
      </el-card>

      <!-- 可视化预览 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header">
          <span>权重分布预览</span>
        </div>
        <div class="section-content">
          <div class="preview-container" ref="weightPreview">
            <!-- 权重分布可视化 -->
          </div>
          <div class="preview-controls">
            <el-button @click="updatePreview">更新预览</el-button>
            <el-button @click="resetPreview">重置</el-button>
          </div>
        </div>
      </el-card>

      <!-- 高级配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.advanced = !sectionOpen.advanced" style="cursor:pointer;">
          <span>高级配置</span>
          <el-icon :class="{'rotate': sectionOpen.advanced}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.advanced" class="section-content">
            <el-form label-width="120px">
              <el-form-item label="网格分辨率">
                <el-input-number v-model="advancedConfig.resolution" :min="10" :max="100" />
              </el-form-item>
              <el-form-item label="平滑参数">
                <el-input-number v-model="advancedConfig.smoothing" :min="0" :max="1" :precision="2" />
              </el-form-item>
              <el-form-item label="边界处理">
                <el-select v-model="advancedConfig.boundary_handling" style="width: 100%">
                  <el-option label="硬边界" value="hard" />
                  <el-option label="软边界" value="soft" />
                  <el-option label="周期性" value="periodic" />
                </el-select>
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>
    </div>

    <!-- 按钮区域 -->
    <div class="button-container">
      <div class="button-group">
        <el-button @click="handleClose">取消</el-button>
        <el-button type="primary" @click="handleGenerate">应用权重</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  nextTick,
  onMounted,
  reactive,
  ref,
} from 'vue';

import { ElMessage } from 'element-plus';
import * as THREE from 'three';

import {
  ArrowDown,
  Close,
} from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean
  label_dict: Map<string, any>
  simulation_data: any
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

// 折叠状态
const sectionOpen = reactive({
  advanced: false
})

// 权重配置
const weightConfig = ref({
  type: 'circular',
  center: [2.0, 2.0],
  radius: 0.5,
  rect: {
    x_min: 1.0,
    x_max: 3.0,
    y_min: 1.0,
    y_max: 3.0
  },
  weight_value: 1e6,
  function: 'constant',
  decay_param: 1.0
})

// 高级配置
const advancedConfig = ref({
  resolution: 50,
  smoothing: 0.1,
  boundary_handling: 'hard'
})

// 预览容器
const weightPreview = ref<HTMLElement>()
let previewScene: THREE.Scene
let previewRenderer: THREE.WebGLRenderer
let previewCamera: THREE.OrthographicCamera

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  // 验证配置
  if (weightConfig.value.weight_value <= 0) {
    ElMessage.warning('权重值必须大于0')
    return
  }

  if (weightConfig.value.type === 'circular' && weightConfig.value.radius <= 0) {
    ElMessage.warning('半径必须大于0')
    return
  }

  if (weightConfig.value.type === 'rectangular') {
    const rect = weightConfig.value.rect
    if (rect.x_min >= rect.x_max || rect.y_min >= rect.y_max) {
      ElMessage.warning('矩形范围配置无效')
      return
    }
  }

  // 生成配置数据
  const configData = {
    weight: weightConfig.value,
    advanced: advancedConfig.value
  }

  emit('generate', configData)
  emit('update:modelValue', false)
}

// 更新预览
const updatePreview = () => {
  if (!weightPreview.value) return

  // 清除现有场景
  if (previewScene) {
    previewScene.clear()
  }

  // 创建权重分布可视化
  createWeightVisualization()
}

// 重置预览
const resetPreview = () => {
  weightConfig.value = {
    type: 'circular',
    center: [2.0, 2.0],
    radius: 0.5,
    rect: {
      x_min: 1.0,
      x_max: 3.0,
      y_min: 1.0,
      y_max: 3.0
    },
    weight_value: 1e6,
    function: 'constant',
    decay_param: 1.0
  }
  updatePreview()
}

// 创建权重分布可视化
const createWeightVisualization = () => {
  if (!weightPreview.value) return

  // 初始化Three.js场景
  previewScene = new THREE.Scene()
  previewCamera = new THREE.OrthographicCamera(-5, 5, -5, 5, 0.1, 1000)
  previewRenderer = new THREE.WebGLRenderer({ antialias: true })
  previewRenderer.setSize(weightPreview.value.clientWidth, weightPreview.value.clientHeight)
  previewRenderer.setClearColor(0xf8fafc)
  
  weightPreview.value.appendChild(previewRenderer.domElement)

  // 设置相机位置
  previewCamera.position.set(0, 0, 10)
  previewCamera.lookAt(0, 0, 0)

  // 创建权重分布几何体
  const geometry = new THREE.PlaneGeometry(4, 4, 32, 32)
  const material = new THREE.MeshBasicMaterial({ 
    color: 0x6366f1,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide
  })
  
  const weightMesh = new THREE.Mesh(geometry, material)
  previewScene.add(weightMesh)

  // 添加坐标轴
  const axesHelper = new THREE.AxesHelper(2)
  previewScene.add(axesHelper)

  // 渲染场景
  previewRenderer.render(previewScene, previewCamera)
}

onMounted(() => {
  nextTick(() => {
    if (weightPreview.value) {
      createWeightVisualization()
    }
  })
})
</script>

<style scoped>
.right-sidebar {
  position: fixed;
  right: 0;
  top: 124px;
  width: 500px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-left: 1px solid rgba(226, 232, 240, 0.6);
  height: calc(100vh - 124px);
  display: flex;
  flex-direction: column;
  z-index: 1000;
  box-shadow: -8px 0 32px rgba(0, 0, 0, 0.12);
}

.right-sidebar-header {
  padding: 16px 20px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
  flex: 0 0 auto;
}

.right-sidebar-header span {
  font-weight: 600;
  color: #1e293b;
  font-size: 16px;
  letter-spacing: 0.025em;
}

.close-button {
  padding: 10px;
  border-radius: 10px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  color: #64748b;
  background: rgba(255, 255, 255, 0.6);
  border: 1px solid rgba(226, 232, 240, 0.4);
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  font-weight: 600;
}

.close-button:hover {
  color: #ffffff;
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  border-color: #ef4444;
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
}

.right-sidebar-content {
  position: relative;
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 20px;
  padding-bottom: 80px;
  min-height: 0;
  background: rgba(248, 250, 252, 0.3);
}

.section {
  margin-bottom: 28px;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid rgba(226, 232, 240, 0.4);
  transition: all 0.3s ease;
}

.section:hover {
  background: rgba(255, 255, 255, 0.8);
  border-color: rgba(99, 102, 241, 0.2);
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.08);
}

.section-header {
  font-weight: 600;
  margin-bottom: 16px;
  color: #1e293b;
  font-size: 15px;
  letter-spacing: 0.025em;
  border-bottom: 2px solid rgba(99, 102, 241, 0.2);
  padding-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.section-header .el-icon {
  transition: transform 0.3s ease;
}

.section-header .el-icon.rotate {
  transform: rotate(180deg);
}

.section-content {
  padding: 0;
}

.vector-input {
  display: flex;
  align-items: center;
  gap: 8px;
}

.vector-input :deep(.el-input-number) {
  width: 80px;
}

.rect-input {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.rect-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.rect-row span {
  min-width: 20px;
  color: #64748b;
  font-size: 14px;
}

.preview-container {
  height: 200px;
  background: #f8fafc;
  border: 1px solid rgba(226, 232, 240, 0.6);
  border-radius: 8px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
}

.preview-controls {
  display: flex;
  gap: 8px;
  justify-content: center;
}

.button-container {
  position: fixed;
  right: 0;
  bottom: 0;
  width: 500px;
  padding: 20px;
  border-top: 1px solid rgba(226, 232, 240, 0.6);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
  backdrop-filter: blur(20px);
  z-index: 1100;
}

.button-group {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.button-group .el-button {
  min-width: 100px;
  height: 40px;
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.button-group .el-button--primary {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  border: none;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.button-group .el-button--primary:hover {
  background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
}

.button-group .el-button--default {
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(226, 232, 240, 0.6);
  color: #64748b;
}

.button-group .el-button--default:hover {
  background: rgba(255, 255, 255, 1);
  border-color: rgba(99, 102, 241, 0.3);
  color: #6366f1;
  transform: translateY(-1px);
}

:deep(.el-form-item) {
  margin-bottom: 20px;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

:deep(.el-form-item__content) {
  display: flex;
  align-items: center;
  min-height: 32px;
  flex: 1;
  margin-left: 0;
}

:deep(.el-form-item__label) {
  padding-bottom: 6px;
  line-height: 1.4;
  font-weight: 500;
  color: #374151;
  font-size: 13px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  text-align: right;
  word-break: break-word;
  hyphens: auto;
  min-height: 32px;
  padding-right: 16px;
  white-space: normal;
  overflow-wrap: break-word;
  width: 120px;
  flex-shrink: 0;
}

:deep(.el-input) {
  border-radius: 8px;
  transition: all 0.3s ease;
}

:deep(.el-input__wrapper) {
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

:deep(.el-input__wrapper:hover) {
  box-shadow: 0 2px 6px rgba(99, 102, 241, 0.15);
}

:deep(.el-input__wrapper.is-focus) {
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
}

:deep(.el-input-number) {
  width: 100%;
  display: flex;
  align-items: center;
}

:deep(.el-input-number .el-input__wrapper) {
  border-radius: 8px;
}

:deep(.el-input-number .el-input-number__decrease),
:deep(.el-input-number .el-input-number__increase) {
  border-radius: 8px;
  transition: all 0.3s ease;
}

:deep(.el-input-number .el-input-number__decrease:hover),
:deep(.el-input-number .el-input-number__increase:hover) {
  background-color: #6366f1;
  color: white;
}

:deep(.el-select) {
  width: 100%;
}

:deep(.el-select .el-input__wrapper) {
  border-radius: 8px;
}
</style>

