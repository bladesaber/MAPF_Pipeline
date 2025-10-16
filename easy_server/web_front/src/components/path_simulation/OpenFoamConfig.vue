<!-- OpenFoamConfig.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>OpenFOAM 仿真配置</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>

    <div class="right-sidebar-content">
      <!-- 几何配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.geometry = !sectionOpen.geometry" style="cursor:pointer;">
          <span>几何配置</span>
          <el-icon :class="{'rotate': sectionOpen.geometry}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.geometry" class="section-content">
            <el-form label-width="120px">
              <el-form-item label="计算域尺寸">
                <div class="vector-input">
                  <el-input-number v-model="geometryConfig.domainSize[0]" :min="0" :precision="2" />
                  <el-input-number v-model="geometryConfig.domainSize[1]" :min="0" :precision="2" />
                  <el-input-number v-model="geometryConfig.domainSize[2]" :min="0" :precision="2" />
                </div>
              </el-form-item>
              <el-form-item label="管道半径">
                <el-input-number v-model="geometryConfig.pipeRadius" :min="0" :precision="3" />
              </el-form-item>
              <el-form-item label="管道长度">
                <el-input-number v-model="geometryConfig.pipeLength" :min="0" :precision="2" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- 网格配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.mesh = !sectionOpen.mesh" style="cursor:pointer;">
          <span>网格配置</span>
          <el-icon :class="{'rotate': sectionOpen.mesh}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.mesh" class="section-content">
            <el-form label-width="120px">
              <el-form-item label="最大网格尺寸">
                <el-input-number v-model="meshConfig.maxSize" :min="0" :precision="3" />
              </el-form-item>
              <el-form-item label="最小网格尺寸">
                <el-input-number v-model="meshConfig.minSize" :min="0" :precision="3" />
              </el-form-item>
              <el-form-item label="网格算法">
                <el-select v-model="meshConfig.algorithm" style="width: 100%">
                  <el-option label="Frontal-Delaunay" value="6" />
                  <el-option label="Delaunay" value="8" />
                </el-select>
              </el-form-item>
              <el-form-item label="网格质量">
                <el-slider v-model="meshConfig.quality" :min="1" :max="10" show-stops />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- 求解器配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.solver = !sectionOpen.solver" style="cursor:pointer;">
          <span>求解器配置</span>
          <el-icon :class="{'rotate': sectionOpen.solver}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.solver" class="section-content">
            <el-form label-width="120px">
              <el-form-item label="求解器类型">
                <el-select v-model="solverConfig.solver" style="width: 100%">
                  <el-option label="incompressibleFluid" value="incompressibleFluid" />
                  <el-option label="potentialFoam" value="potentialFoam" />
                  <el-option label="topoIncompressibleFluid" value="topoIncompressibleFluid" />
                </el-select>
              </el-form-item>
              <el-form-item label="数值方法">
                <el-select v-model="solverConfig.numericalMethod" style="width: 100%">
                  <el-option label="PIMPLE" value="PIMPLE" />
                  <el-option label="PISO" value="PISO" />
                  <el-option label="SIMPLE" value="SIMPLE" />
                  <el-option label="SIMPLEC" value="SIMPLEC" />
                </el-select>
              </el-form-item>
              <el-form-item label="时间步长">
                <el-input-number v-model="solverConfig.deltaT" :min="0" :precision="6" />
              </el-form-item>
              <el-form-item label="结束时间">
                <el-input-number v-model="solverConfig.endTime" :min="0" :precision="2" />
              </el-form-item>
              <el-form-item label="写入间隔">
                <el-input-number v-model="solverConfig.writeInterval" :min="1" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- 湍流模型配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.turbulence = !sectionOpen.turbulence" style="cursor:pointer;">
          <span>湍流模型配置</span>
          <el-icon :class="{'rotate': sectionOpen.turbulence}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.turbulence" class="section-content">
            <el-form label-width="120px">
              <el-form-item label="湍流模型">
                <el-select v-model="turbulenceConfig.model" style="width: 100%">
                  <el-option label="k-epsilon" value="kEpsilon" />
                  <el-option label="Spalart-Allmaras" value="SpalartAllmaras" />
                </el-select>
              </el-form-item>
              <el-form-item label="参考速度">
                <el-input-number v-model="turbulenceConfig.velocityRef" :min="0" :precision="2" />
              </el-form-item>
              <el-form-item label="参考长度">
                <el-input-number v-model="turbulenceConfig.lengthRef" :min="0" :precision="2" />
              </el-form-item>
              <el-form-item label="湍流强度">
                <el-input-number v-model="turbulenceConfig.turbulenceIntensity" :min="0" :max="1" :precision="3" />
              </el-form-item>
              <el-form-item label="粘度比">
                <el-input-number v-model="turbulenceConfig.viscosityRatio" :min="0" :precision="1" />
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- 边界条件配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.boundary = !sectionOpen.boundary" style="cursor:pointer;">
          <span>边界条件配置</span>
          <el-icon :class="{'rotate': sectionOpen.boundary}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.boundary" class="section-content">
            <el-form label-width="120px">
              <el-form-item label="入口速度">
                <div class="vector-input">
                  <el-input-number v-model="boundaryConfig.inletVelocity[0]" :precision="2" />
                  <el-input-number v-model="boundaryConfig.inletVelocity[1]" :precision="2" />
                  <el-input-number v-model="boundaryConfig.inletVelocity[2]" :precision="2" />
                </div>
              </el-form-item>
              <el-form-item label="出口压力">
                <el-input-number v-model="boundaryConfig.outletPressure" :precision="2" />
              </el-form-item>
              <el-form-item label="壁面条件">
                <el-select v-model="boundaryConfig.wallCondition" style="width: 100%">
                  <el-option label="无滑移" value="noSlip" />
                  <el-option label="滑移" value="slip" />
                </el-select>
              </el-form-item>
              <el-form-item label="入口湍流参数">
                <div class="turbulence-params">
                  <el-form-item label="k值">
                    <el-input-number v-model="boundaryConfig.inletK" :min="0" :precision="6" />
                  </el-form-item>
                  <el-form-item label="epsilon值">
                    <el-input-number v-model="boundaryConfig.inletEpsilon" :min="0" :precision="6" />
                  </el-form-item>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- 物理属性配置 -->
      <el-card shadow="never" class="section" :body-style="{padding: '0'}">
        <div class="section-header" @click="sectionOpen.physics = !sectionOpen.physics" style="cursor:pointer;">
          <span>物理属性配置</span>
          <el-icon :class="{'rotate': sectionOpen.physics}"><arrow-down /></el-icon>
        </div>
        <el-collapse-transition>
          <div v-show="sectionOpen.physics" class="section-content">
            <el-form label-width="120px">
              <el-form-item label="流体密度">
                <el-input-number v-model="physicsConfig.density" :min="0" :precision="3" />
              </el-form-item>
              <el-form-item label="运动粘度">
                <el-input-number v-model="physicsConfig.nu" :min="0" :precision="8" />
              </el-form-item>
              <el-form-item label="重力">
                <div class="vector-input">
                  <el-input-number v-model="physicsConfig.gravity[0]" :precision="2" />
                  <el-input-number v-model="physicsConfig.gravity[1]" :precision="2" />
                  <el-input-number v-model="physicsConfig.gravity[2]" :precision="2" />
                </div>
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
        <el-button type="primary" @click="handleGenerate">生成配置</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  reactive,
  ref,
} from 'vue';

import { ElMessage } from 'element-plus';

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
  geometry: true,
  mesh: false,
  solver: false,
  turbulence: false,
  boundary: false,
  physics: false,
})

// 几何配置
const geometryConfig = ref({
  domainSize: [10.0, 5.0, 5.0], // 计算域尺寸 [x, y, z]
  pipeRadius: 0.5, // 管道半径
  pipeLength: 2.0, // 管道长度
})

// 网格配置
const meshConfig = ref({
  maxSize: 0.1, // 最大网格尺寸
  minSize: 0.05, // 最小网格尺寸
  algorithm: '6', // 网格算法
  quality: 5, // 网格质量 (1-10)
})

// 求解器配置
const solverConfig = ref({
  solver: 'incompressibleFluid', // 求解器类型
  numericalMethod: 'PIMPLE', // 数值方法
  deltaT: 0.005, // 时间步长
  endTime: 10.0, // 结束时间
  writeInterval: 100, // 写入间隔
})

// 湍流模型配置
const turbulenceConfig = ref({
  model: 'kEpsilon', // 湍流模型
  velocityRef: 5.0, // 参考速度
  lengthRef: 1.0, // 参考长度
  turbulenceIntensity: 0.05, // 湍流强度
  viscosityRatio: 10.0, // 粘度比
})

// 边界条件配置
const boundaryConfig = ref({
  inletVelocity: [0.0, 0.0, 5.0], // 入口速度
  outletPressure: 0.0, // 出口压力
  wallCondition: 'noSlip', // 壁面条件
  inletK: 0.375, // 入口k值
  inletEpsilon: 0.375, // 入口epsilon值
})

// 物理属性配置
const physicsConfig = ref({
  density: 1.225, // 流体密度
  nu: 1.5e-05, // 运动粘度
  gravity: [0.0, 0.0, -9.81], // 重力
})

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  // 验证配置
  if (geometryConfig.value.pipeRadius <= 0) {
    ElMessage.warning('管道半径必须大于0')
    return
  }
  
  if (meshConfig.value.minSize >= meshConfig.value.maxSize) {
    ElMessage.warning('最小网格尺寸必须小于最大网格尺寸')
    return
  }

  // 生成配置数据
  const configData = {
    geometry: geometryConfig.value,
    mesh: meshConfig.value,
    solver: solverConfig.value,
    turbulence: turbulenceConfig.value,
    boundary: boundaryConfig.value,
    physics: physicsConfig.value,
  }

  emit('generate', configData)
  emit('update:modelValue', false)
}
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

.right-sidebar-content::-webkit-scrollbar {
  width: 4px;
}

.right-sidebar-content::-webkit-scrollbar-track {
  background: transparent;
}

.right-sidebar-content::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #cbd5e1 0%, #94a3b8 100%);
  border-radius: 6px;
  transition: all 0.2s ease;
}

.right-sidebar-content::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, #94a3b8 0%, #64748b 100%);
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

.turbulence-params {
  padding-left: 20px;
  border-left: 2px solid rgba(99, 102, 241, 0.1);
}

:deep(.el-form-item) {
  margin-bottom: 16px;
}

:deep(.el-form-item__label) {
  padding-bottom: 4px;
  line-height: 1.5;
  font-weight: 500;
  color: #374151;
}

.button-container {
  padding: 16px 20px;
  border-top: 1px solid rgba(226, 232, 240, 0.6);
  background: rgba(248, 250, 252, 0.8);
  flex: 0 0 auto;
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

