<template>
  <div class="report-plotly">
    <el-dialog v-model="visible" title="Report Visualization" width="80%" :before-close="handleClose">
      <div class="graphs-container">
        <!-- Pie Chart Container -->
        <div class="graph-section">
          <h3>Conflict Distribution</h3>
          <div id="pie-chart" class="graph-container"></div>
        </div>
        
        <!-- Line Chart Container -->
        <div class="graph-section">
          <h3>Loss Over Time</h3>
          <div id="line-chart" class="graph-container"></div>
        </div>
      </div>
      
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="handleClose">Close</el-button>
          <el-button type="primary" @click="exportGraphs">Export</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { GraphUtils } from '@/classes/graph_utils'

// Props
const props = defineProps<{
  modelValue: boolean
}>()

// Emits
const emit = defineEmits(['update:modelValue'])

// Reactive data
const visible = ref(false)
const pieData = ref<{ labels: string[], values: number[] } | null>(null)
const lineData = ref<{ x: number[], y: number[] } | null>(null)

// Watch for prop changes
watch(() => props.modelValue, (newValue) => {
  visible.value = newValue
  if (newValue) {
    // Draw graphs when dialog becomes visible
    nextTick(() => {
      if (pieData.value) {
        GraphUtils.drawPieChart(
          'pie-chart',
          pieData.value.labels,
          pieData.value.values,
          'Conflict Distribution',
          ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        )
      }
      if (lineData.value) {
        GraphUtils.drawLineGraph(
          'line-chart',
          lineData.value.x,
          lineData.value.y,
          'Loss Over Time',
          'Iteration',
          'Loss Value',
          '#2c3e50'
        )
      }
    })
  }
})

watch(visible, (newValue) => {
  emit('update:modelValue', newValue)
})

// Methods
const handleClose = () => {
  visible.value = false
}

const updatePieChart = (labels: string[], values: number[]) => {
  console.log('Updating pie chart:', labels, values)
  pieData.value = { labels, values }
  if (visible.value) {
    nextTick(() => {
      try {
        GraphUtils.drawPieChart(
          'pie-chart',
          labels,
          values,
          'Conflict Distribution',
          ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        )
      } catch (error) {
        console.error('Error drawing pie chart:', error)
      }
    })
  }
}

const updateLineChart = (xData: number[], yData: number[]) => {
  console.log('Updating line chart:', xData, yData)
  lineData.value = { x: xData, y: yData }
  if (visible.value) {
    nextTick(() => {
      try {
        GraphUtils.drawLineGraph(
          'line-chart',
          xData,
          yData,
          'Loss Over Time',
          'Iteration',
          'Loss Value',
          '#2c3e50'
        )
      } catch (error) {
        console.error('Error drawing line chart:', error)
      }
    })
  }
}

const exportGraphs = () => {
  if (pieData.value) {
    GraphUtils.exportAsPNG('pie-chart', 'conflict-distribution.png')
  }
  if (lineData.value) {
    GraphUtils.exportAsPNG('line-chart', 'loss-over-time.png')
  }
}

// Expose APIs
defineExpose({
  updatePieChart,
  updateLineChart
})
</script>

<style scoped>
.report-plotly {
  /* Component styles */
}

.graphs-container {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.graph-section {
  flex: 1;
  min-width: 400px;
}

.graph-section h3 {
  margin: 0 0 16px 0;
  color: #333;
  font-size: 16px;
  font-weight: 600;
  text-align: center;
}

.graph-container {
  width: 100%;
  height: 400px;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  background-color: #fff;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

:deep(.el-dialog) {
  max-width: 1200px;
}

:deep(.el-dialog__body) {
  padding: 20px;
}
</style> 