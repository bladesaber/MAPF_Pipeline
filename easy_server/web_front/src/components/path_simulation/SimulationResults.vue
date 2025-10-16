<!-- SimulationResults.vue -->
<template>
  <div v-if="modelValue" class="results-modal">
    <div class="modal-overlay" @click="handleClose"></div>
    <div class="modal-content">
      <div class="modal-header">
        <h3>仿真结果 - {{ simulation_name }}</h3>
        <el-button link class="close-button" @click="handleClose">
          <el-icon><Close /></el-icon>
        </el-button>
      </div>

      <div class="modal-body">
        <!-- 结果概览 -->
        <div class="results-overview">
          <el-row :gutter="20">
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ results_data?.convergence?.iterations || 0 }}</div>
                <div class="stat-label">迭代次数</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ results_data?.convergence?.finalResidual || 'N/A' }}</div>
                <div class="stat-label">最终残差</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ results_data?.performance?.executionTime || 'N/A' }}</div>
                <div class="stat-label">执行时间</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ results_data?.mesh?.cellCount || 'N/A' }}</div>
                <div class="stat-label">网格单元数</div>
              </div>
            </el-col>
          </el-row>
        </div>

        <!-- 结果可视化选项 -->
        <div class="visualization-controls">
          <el-tabs v-model="activeTab" type="card">
            <el-tab-pane label="速度场" name="velocity">
              <div class="visualization-panel">
                <div class="control-panel">
                  <el-form :inline="true" label-width="80px">
                    <el-form-item label="显示方式">
                      <el-select v-model="velocityDisplay.mode" style="width: 120px">
                        <el-option label="矢量" value="vector" />
                        <el-option label="等值线" value="contour" />
                        <el-option label="流线" value="streamline" />
                      </el-select>
                    </el-form-item>
                    <el-form-item label="切片方向">
                      <el-select v-model="velocityDisplay.sliceDirection" style="width: 120px">
                        <el-option label="XY平面" value="xy" />
                        <el-option label="XZ平面" value="xz" />
                        <el-option label="YZ平面" value="yz" />
                      </el-select>
                    </el-form-item>
                    <el-form-item label="切片位置">
                      <el-slider v-model="velocityDisplay.slicePosition" :min="0" :max="100" />
                    </el-form-item>
                    <el-form-item>
                      <el-button type="primary" @click="updateVelocityVisualization">更新显示</el-button>
                    </el-form-item>
                  </el-form>
                </div>
                <div class="visualization-area" ref="velocityVisualization">
                  <!-- 3D可视化区域 -->
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane label="压力场" name="pressure">
              <div class="visualization-panel">
                <div class="control-panel">
                  <el-form :inline="true" label-width="80px">
                    <el-form-item label="显示方式">
                      <el-select v-model="pressureDisplay.mode" style="width: 120px">
                        <el-option label="等值面" value="isosurface" />
                        <el-option label="切片" value="slice" />
                        <el-option label="体积渲染" value="volume" />
                      </el-select>
                    </el-form-item>
                    <el-form-item label="数值范围">
                      <el-input-number v-model="pressureDisplay.minValue" :precision="2" />
                      <span class="range-separator">-</span>
                      <el-input-number v-model="pressureDisplay.maxValue" :precision="2" />
                    </el-form-item>
                    <el-form-item>
                      <el-button type="primary" @click="updatePressureVisualization">更新显示</el-button>
                    </el-form-item>
                  </el-form>
                </div>
                <div class="visualization-area" ref="pressureVisualization">
                  <!-- 3D可视化区域 -->
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane label="湍流场" name="turbulence">
              <div class="visualization-panel">
                <div class="control-panel">
                  <el-form :inline="true" label-width="80px">
                    <el-form-item label="湍流参数">
                      <el-select v-model="turbulenceDisplay.parameter" style="width: 120px">
                        <el-option label="湍流动能k" value="k" />
                        <el-option label="湍流耗散率ε" value="epsilon" />
                        <el-option label="湍流粘度" value="nut" />
                      </el-select>
                    </el-form-item>
                    <el-form-item label="对数显示">
                      <el-switch v-model="turbulenceDisplay.logScale" />
                    </el-form-item>
                    <el-form-item>
                      <el-button type="primary" @click="updateTurbulenceVisualization">更新显示</el-button>
                    </el-form-item>
                  </el-form>
                </div>
                <div class="visualization-area" ref="turbulenceVisualization">
                  <!-- 3D可视化区域 -->
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane label="收敛历史" name="convergence">
              <div class="convergence-panel">
                <div class="chart-container" ref="convergenceChart">
                  <!-- 收敛历史图表 -->
                </div>
                <div class="convergence-info">
                  <el-descriptions :column="2" border>
                    <el-descriptions-item label="初始残差">{{ results_data?.convergence?.initialResidual || 'N/A' }}</el-descriptions-item>
                    <el-descriptions-item label="最终残差">{{ results_data?.convergence?.finalResidual || 'N/A' }}</el-descriptions-item>
                    <el-descriptions-item label="收敛率">{{ results_data?.convergence?.convergenceRate || 'N/A' }}</el-descriptions-item>
                    <el-descriptions-item label="总迭代数">{{ results_data?.convergence?.totalIterations || 'N/A' }}</el-descriptions-item>
                  </el-descriptions>
                </div>
              </div>
            </el-tab-pane>
          </el-tabs>
        </div>

        <!-- 数据导出 -->
        <div class="export-section">
          <el-divider>数据导出</el-divider>
          <div class="export-controls">
            <el-button-group>
              <el-button @click="exportVTK">导出VTK</el-button>
              <el-button @click="exportCSV">导出CSV</el-button>
              <el-button @click="exportImages">导出图像</el-button>
              <el-button @click="exportReport">导出报告</el-button>
            </el-button-group>
          </div>
        </div>
      </div>

      <div class="modal-footer">
        <el-button @click="handleClose">关闭</el-button>
        <el-button type="primary" @click="handleExport">导出所有结果</el-button>
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
import * as Plotly from 'plotly.js-dist';
import * as THREE from 'three';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean
  simulation_name: string
  results_data: any
}>()

const emit = defineEmits(['update:modelValue', 'export'])

// 活动标签页
const activeTab = ref('velocity')

// 可视化控制
const velocityDisplay = reactive({
  mode: 'vector',
  sliceDirection: 'xy',
  slicePosition: 50
})

const pressureDisplay = reactive({
  mode: 'isosurface',
  minValue: -1000,
  maxValue: 1000
})

const turbulenceDisplay = reactive({
  parameter: 'k',
  logScale: false
})

// 可视化容器引用
const velocityVisualization = ref<HTMLElement>()
const pressureVisualization = ref<HTMLElement>()
const turbulenceVisualization = ref<HTMLElement>()
const convergenceChart = ref<HTMLElement>()

// 3D场景
let velocityScene: THREE.Scene
let pressureScene: THREE.Scene
let turbulenceScene: THREE.Scene

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleExport = () => {
  emit('export', {
    type: 'all',
    data: props.results_data
  })
}

// 更新速度场可视化
const updateVelocityVisualization = () => {
  console.log('更新速度场可视化:', velocityDisplay)
  // 实现3D速度场可视化
  // 这里需要根据实际的数据格式来实现
}

// 更新压力场可视化
const updatePressureVisualization = () => {
  console.log('更新压力场可视化:', pressureDisplay)
  // 实现3D压力场可视化
}

// 更新湍流场可视化
const updateTurbulenceVisualization = () => {
  console.log('更新湍流场可视化:', turbulenceDisplay)
  // 实现3D湍流场可视化
}

// 导出功能
const exportVTK = () => {
  ElMessage.info('导出VTK格式文件')
  // 实现VTK导出
}

const exportCSV = () => {
  ElMessage.info('导出CSV格式文件')
  // 实现CSV导出
}

const exportImages = () => {
  ElMessage.info('导出图像文件')
  // 实现图像导出
}

const exportReport = () => {
  ElMessage.info('导出仿真报告')
  // 实现报告导出
}

// 初始化收敛历史图表
const initConvergenceChart = () => {
  if (!convergenceChart.value) return

  const data = props.results_data?.convergence?.history || []
  
  const plotData = [{
    x: data.map((_: any, i: number) => i),
    y: data,
    type: 'scatter',
    mode: 'lines+markers',
    name: '残差',
    line: { color: '#6366f1' }
  }]

  const layout = {
    title: '收敛历史',
    xaxis: { title: '迭代次数' },
    yaxis: { title: '残差', type: 'log' },
    margin: { t: 40, r: 40, b: 40, l: 40 }
  }

  Plotly.newPlot(convergenceChart.value, plotData, layout, { responsive: true })
}

onMounted(() => {
  nextTick(() => {
    initConvergenceChart()
  })
})
</script>

<style scoped>
.results-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
}

.modal-content {
  position: relative;
  width: 90vw;
  height: 90vh;
  max-width: 1400px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  padding: 20px 24px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1e293b;
}

.close-button {
  padding: 8px;
  border-radius: 8px;
  transition: all 0.3s ease;
  color: #64748b;
  background: rgba(255, 255, 255, 0.6);
  border: 1px solid rgba(226, 232, 240, 0.4);
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-button:hover {
  color: #ffffff;
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  border-color: #ef4444;
  transform: scale(1.05);
}

.modal-body {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
}

.results-overview {
  margin-bottom: 24px;
}

.stat-card {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
}

.stat-value {
  font-size: 24px;
  font-weight: 700;
  color: #6366f1;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #64748b;
  font-weight: 500;
}

.visualization-controls {
  margin-bottom: 24px;
}

.visualization-panel {
  display: flex;
  flex-direction: column;
  height: 500px;
}

.control-panel {
  padding: 16px;
  background: rgba(248, 250, 252, 0.8);
  border-radius: 8px;
  margin-bottom: 16px;
}

.visualization-area {
  flex: 1;
  background: #f8fafc;
  border: 1px solid rgba(226, 232, 240, 0.6);
  border-radius: 8px;
  position: relative;
  overflow: hidden;
}

.convergence-panel {
  display: flex;
  gap: 24px;
}

.chart-container {
  flex: 1;
  height: 400px;
  background: #f8fafc;
  border: 1px solid rgba(226, 232, 240, 0.6);
  border-radius: 8px;
}

.convergence-info {
  width: 300px;
}

.export-section {
  margin-top: 24px;
}

.export-controls {
  margin-top: 16px;
}

.range-separator {
  margin: 0 8px;
  color: #64748b;
}

.modal-footer {
  padding: 16px 24px;
  border-top: 1px solid rgba(226, 232, 240, 0.6);
  background: rgba(248, 250, 252, 0.8);
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

:deep(.el-tabs__content) {
  padding: 0;
}

:deep(.el-tab-pane) {
  height: 100%;
}

:deep(.el-form-item) {
  margin-bottom: 12px;
}

:deep(.el-form-item__label) {
  font-weight: 500;
  color: #374151;
}
</style>

