<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Segment</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Segment Configuration</h3>
      </div>
      
      <!-- 不可编辑的重要信息 -->
      <div class="info-section">
        <h4>Basic Information</h4>
        <el-form label-width="120px" size="small">
          <el-form-item label="Segment Name">
            <el-input v-model="segmentData.name" disabled />
          </el-form-item>
          <el-form-item label="Segment Index">
            <el-input-number v-model="segmentData.seg_idx" disabled style="width: 150px" />
          </el-form-item>
          <el-form-item label="Group Index">
            <el-input-number v-model="segmentData.group_idx" disabled style="width: 150px" />
          </el-form-item>
          <el-form-item label="Control Points">
            <el-input v-model="controlPointsText" disabled type="textarea" :rows="2" />
          </el-form-item>
        </el-form>
      </div>

      <!-- B样条参数 -->
      <div class="section-header">
        <h3>B-Spline Parameters</h3>
      </div>
      <el-form :model="segmentData" label-width="120px" size="small">
        <el-form-item label="B-Spline Degree">
          <el-input-number v-model="segmentData.bspline_degree" :min="1" :max="10" :step="1" style="width: 150px" />
        </el-form-item>
        <el-form-item label="B-Spline Density">
          <el-input-number v-model="segmentData.bspline_density" :min="0.1" :max="1.0" :step="0.05" :precision="2" style="width: 150px" />
        </el-form-item>
      </el-form>

      <!-- 重叠和截断参数 -->
      <div class="section-header">
        <h3>Overlap & Intercept Parameters</h3>
      </div>
      <el-form :model="segmentData" label-width="120px" size="small">
        <el-form-item label="Left Overlap Ratio">
          <el-input-number v-model="segmentData.left_overlap_ratio" :min="0.0" :max="1.0" :step="0.05" :precision="2" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Right Overlap Ratio">
          <el-input-number v-model="segmentData.right_overlap_ratio" :min="0.0" :max="1.0" :step="0.05" :precision="2" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Left Intercept Ratio">
          <el-input-number v-model="segmentData.left_intercept_ratio" :min="0.0" :max="1.0" :step="0.05" :precision="2" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Right Intercept Ratio">
          <el-input-number v-model="segmentData.right_intercept_ratio" :min="0.0" :max="1.0" :step="0.05" :precision="2" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Segment Scale">
          <el-input-number v-model="segmentData.segment_scale" :min="0.1" :max="10.0" :step="0.1" :precision="1" style="width: 150px" />
        </el-form-item>
      </el-form>

      <!-- 边界条件 -->
      <div class="section-header">
        <h3>Boundary Conditions</h3>
      </div>
      <el-form :model="segmentData" label-width="120px" size="small">
        <el-form-item label="Left Clamp">
          <el-switch v-model="segmentData.left_clamp" />
        </el-form-item>
        <el-form-item label="Right Clamp">
          <el-switch v-model="segmentData.right_clamp" />
        </el-form-item>
      </el-form>

      <!-- 计算选项 -->
      <div class="section-header">
        <h3>Computation Options</h3>
      </div>
      <el-form :model="segmentData" label-width="120px" size="small">
        <el-form-item label="Analytical Curvature">
          <el-switch v-model="segmentData.with_analytical_curvature_computed" />
        </el-form-item>
        <el-form-item label="Analytical Tangent">
          <el-switch v-model="segmentData.with_analytical_tangent_computed" />
        </el-form-item>
      </el-form>

      <!-- 代价设置 -->
      <div class="section-header">
        <h3>Cost Settings</h3>
      </div>
      <el-form :model="segmentData" label-width="120px" size="small">
        <el-form-item label="Obstacle Thickness">
          <el-input-number v-model="segmentData.cost_setting.obstacle_thickness" :min="0.0" :max="10.0" :step="0.01" :precision="3" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Path Thickness">
          <el-input-number v-model="segmentData.cost_setting.path_thickness" :min="0.0" :max="10.0" :step="0.01" :precision="3" style="width: 150px" />
        </el-form-item>
      </el-form>

      <!-- Cost Configurations -->
      <div class="section-header">
        <h3>Cost Configurations</h3>
        <div style="margin-top: 8px;">
          <el-button type="primary" size="small" @click="addCostConfig">
            <el-icon><Plus /></el-icon>
            Add Cost Config
          </el-button>
        </div>
      </div>
      
      <!-- 当没有Cost Configurations时显示提示 -->
      <div v-if="segmentData.cost_setting.cost_cfgs.length === 0" class="no-cost-configs">
        <el-empty description="No cost configurations" :image-size="60">
          <el-button type="primary" size="small" @click="addCostConfig">
            Add First Cost Config
          </el-button>
        </el-empty>
      </div>
      
      <div v-for="(costConfig, index) in segmentData.cost_setting.cost_cfgs" :key="index" class="cost-config-item">
        <div class="cost-config-header">
          <div class="cost-config-title">
            <el-button 
              type="text" 
              class="collapse-button"
              @click="toggleCostConfig(index)"
            >
              <el-icon :class="{ 'is-expanded': !collapsedCostConfigs[index] }">
                <ArrowRight />
              </el-icon>
            </el-button>
            <h4>Cost Config {{ index + 1 }}</h4>
          </div>
          <el-button type="danger" size="small" @click="removeCostConfig(index)">
            <el-icon><Delete /></el-icon>
          </el-button>
        </div>
        
        <div v-show="!collapsedCostConfigs[index]" class="cost-config-content">
          <el-form :model="costConfig" label-width="120px" size="small">
            <el-form-item label="Cost Type">
              <el-select v-model="costConfig.cost_type" style="width: 200px" @change="onCostTypeChange(costConfig)">
                <el-option label="length" value="length" />
                <el-option label="length_distribution" value="length_distribution" />
                <el-option label="curvature" value="curvature" />
                <el-option label="clamp_vector" value="clamp_vector" />
              </el-select>
            </el-form-item>
            
            <!-- 只有 length 和 length_distribution 类型才显示 Data Type -->
            <el-form-item v-if="costConfig.cost_type === 'length' || costConfig.cost_type === 'length_distribution'" label="Data Type">
              <el-select v-model="costConfig.data_type" style="width: 200px">
                <el-option label="control_points" value="control_points" />
                <el-option label="bspline_points" value="bspline_points" />
              </el-select>
            </el-form-item>
            
            <!-- 所有类型都显示 Method -->
            <el-form-item label="Method">
              <el-select v-model="costConfig.method" style="width: 200px">
                <el-option label="square_sum" value="square_sum" />
                <el-option label="length_variance" value="length_variance" />
                <el-option label="sigmoid_mean" value="sigmoid_mean" />
              </el-select>
            </el-form-item>
            
            <!-- 只有 curvature 类型才显示 Curvature Method -->
            <el-form-item v-if="costConfig.cost_type === 'curvature'" label="Curvature Method">
              <el-select v-model="costConfig.curvature_method" style="width: 200px">
                <el-option label="analytical_curvature" value="analytical_curvature" />
                <el-option label="numerical_curvature" value="numerical_curvature" />
              </el-select>
            </el-form-item>
            
            <!-- 只有 curvature 类型才显示 Curvature Scale -->
            <el-form-item v-if="costConfig.cost_type === 'curvature'" label="Curvature Scale">
              <el-input-number v-model="costConfig.curvature_scale" :min="1" :max="10" :step="1" style="width: 150px" />
            </el-form-item>
            
            <!-- 只有 clamp_vector 类型才显示 Vector 相关字段 -->
            <el-form-item v-if="costConfig.cost_type === 'clamp_vector'" label="Vector X">
              <el-input-number v-model="costConfig.vector![0]" :min="-10" :max="10" :step="0.1" :precision="2" style="width: 150px" />
            </el-form-item>
            
            <el-form-item v-if="costConfig.cost_type === 'clamp_vector'" label="Vector Y">
              <el-input-number v-model="costConfig.vector![1]" :min="-10" :max="10" :step="0.1" :precision="2" style="width: 150px" />
            </el-form-item>
            
            <el-form-item v-if="costConfig.cost_type === 'clamp_vector'" label="Vector Z">
              <el-input-number v-model="costConfig.vector![2]" :min="-10" :max="10" :step="0.1" :precision="2" style="width: 150px" />
            </el-form-item>
            
            <el-form-item v-if="costConfig.cost_type === 'clamp_vector'" label="Vector Length">
              <el-input-number v-model="costConfig.vector_length" :min="0.1" :max="10" :step="0.1" :precision="2" style="width: 150px" />
            </el-form-item>
            
            <!-- 所有类型都显示 Weight -->
            <el-form-item label="Weight">
              <el-input-number v-model="costConfig.weight" :min="0" :max="1000" :step="1" style="width: 150px" />
            </el-form-item>
          </el-form>
        </div>
      </div>

      <div class="button-container">
        <div class="button-group">
          <el-button @click="handleClose">Cancel</el-button>
          <el-button type="primary" @click="handleSave">Save</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  computed,
  defineEmits,
  defineProps,
  ref,
  watch,
} from 'vue';

import {
  ArrowRight,
  Close,
  Delete,
  Plus,
} from '@element-plus/icons-vue';

interface CostConfig {
  cost_type: string;
  data_type?: string;
  method: string;
  curvature_method?: string;
  curvature_scale?: number;
  vector?: number[];
  vector_length?: number;
  weight: number;
}

interface SegmentCostSetting {
  obstacle_thickness: number;
  path_thickness: number;
  cost_cfgs: CostConfig[];
}

interface SegmentDataType {
  name: string;
  seg_idx: number;
  group_idx: number;
  c_idxs: number[];
  bspline_degree: number;
  bspline_density: number;
  left_overlap_ratio: number;
  right_overlap_ratio: number;
  left_intercept_ratio: number;
  right_intercept_ratio: number;
  segment_scale: number;
  with_analytical_curvature_computed: boolean;
  with_analytical_tangent_computed: boolean;
  cost_setting: SegmentCostSetting;
  left_clamp: boolean;
  right_clamp: boolean;
  left_vector: number[];
  right_vector: number[];
}

const props = defineProps<{
  modelValue: boolean,
  segmentData: SegmentDataType
}>()
const emit = defineEmits(['update:modelValue', 'save'])

const segmentData = ref<SegmentDataType>({
  name: '',
  seg_idx: 0,
  group_idx: 0,
  c_idxs: [],
  bspline_degree: 3,
  bspline_density: 0.25,
  left_overlap_ratio: 0.25,
  right_overlap_ratio: 0.25,
  left_intercept_ratio: 0.5,
  right_intercept_ratio: 0.5,
  segment_scale: 1.0,
  with_analytical_curvature_computed: true,
  with_analytical_tangent_computed: false,
  cost_setting: {
    obstacle_thickness: 0.01,
    path_thickness: 0.01,
    cost_cfgs: []
  },
  left_clamp: false,
  right_clamp: false,
  left_vector: [0, 0, 0],
  right_vector: [0, 0, 0]
})

const controlPointsText = computed(() => {
  return props.segmentData?.c_idxs?.join(', ') || '';
})

const collapsedCostConfigs = ref<boolean[]>([])

watch(() => props.modelValue, (v) => {
  if (v && props.segmentData) {
    segmentData.value = { ...props.segmentData }
    
    // 确保cost_cfgs存在，但允许为空数组
    if (!segmentData.value.cost_setting.cost_cfgs) {
      segmentData.value.cost_setting.cost_cfgs = []
    }
    
    // 根据实际的cost_cfgs长度设置collapsedCostConfigs
    collapsedCostConfigs.value = segmentData.value.cost_setting.cost_cfgs.map(() => false)
  }
})

const addCostConfig = () => {
  const defaultConfig: CostConfig = {
    cost_type: 'length',
    data_type: 'control_points',
    method: 'square_sum',
    weight: 1
  }
  
  // 根据cost_type初始化特定属性
  if (defaultConfig.cost_type === 'curvature') {
    defaultConfig.curvature_method = 'analytical_curvature'
    defaultConfig.curvature_scale = 3
  } else if (defaultConfig.cost_type === 'clamp_vector') {
    defaultConfig.vector = [0, 0, 1]
    defaultConfig.vector_length = 1
  }
  
  segmentData.value.cost_setting.cost_cfgs.push(defaultConfig)
  collapsedCostConfigs.value.push(false) // 添加新配置时折叠
}

const removeCostConfig = (index: number) => {
  segmentData.value.cost_setting.cost_cfgs.splice(index, 1)
  collapsedCostConfigs.value.splice(index, 1) // 移除时也移除状态
}

const toggleCostConfig = (index: number) => {
  collapsedCostConfigs.value[index] = !collapsedCostConfigs.value[index]
}

const onCostTypeChange = (costConfig: CostConfig) => {
  // 当 cost_type 改变时，动态控制其他字段的显示
  if (costConfig.cost_type === 'curvature') {
    costConfig.data_type = undefined; // 清除 Data Type
    costConfig.curvature_method = undefined; // 清除 Curvature Method
    costConfig.curvature_scale = undefined; // 清除 Curvature Scale
    costConfig.vector = undefined; // 清除 Vector
    costConfig.vector_length = undefined; // 清除 Vector Length
  } else if (costConfig.cost_type === 'clamp_vector') {
    costConfig.data_type = undefined; // 清除 Data Type
    costConfig.curvature_method = undefined; // 清除 Curvature Method
    costConfig.curvature_scale = undefined; // 清除 Curvature Scale
    costConfig.vector = [0, 0, 1]; // 默认 Vector
    costConfig.vector_length = 1; // 默认 Vector Length
  } else {
    costConfig.curvature_method = undefined; // 清除 Curvature Method
    costConfig.curvature_scale = undefined; // 清除 Curvature Scale
    costConfig.vector = undefined; // 清除 Vector
    costConfig.vector_length = undefined; // 清除 Vector Length
  }
}

const handleClose = () => {
  emit('update:modelValue', false)
}
const handleSave = () => {
  emit('save', { ...segmentData.value })
  emit('update:modelValue', false)
}
</script>

<style scoped>
.right-sidebar {
    position: fixed;
    right: 0;
    top: 124px;
    width: 450px;
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
    padding: 16px;
    padding-bottom: 80px;
    min-height: 0;
}
.section-header {
    margin: 20px 0 12px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    position: relative;
}

.section-header::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 1px;
}

.section-header h3 {
    margin: 0;
    color: #1e293b;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.025em;
}
.info-section {
    margin: 16px 0;
    padding: 12px;
    background-color: #f8f9fa;
    border-radius: 4px;
    border: 1px solid #e9ecef;
}
.info-section h4 {
    margin: 0 0 12px 0;
    color: #495057;
    font-size: 14px;
    font-weight: 600;
}
.cost-config-item {
    margin: 16px 0;
    padding: 16px;
    border: 1px solid #e4e7ed;
    border-radius: 4px;
    background-color: #fafafa;
}
.cost-config-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e4e7ed;
}
.cost-config-header h4 {
    margin: 0;
    color: #303133;
    font-size: 14px;
    font-weight: 600;
}
.cost-config-title {
  display: flex;
  align-items: center;
}
.collapse-button {
  margin-right: 8px;
  transition: transform 0.3s ease;
}
.collapse-button.is-expanded {
  transform: rotate(90deg);
}
.cost-config-content {
  padding-left: 20px; /* Indent for collapsed content */
}
.button-container {
    position: fixed;
    right: 0;
    bottom: 0;
    width: 450px;
    padding: 16px;
    border-top: 1px solid #dcdfe6;
    background-color: #f5f7fa;
    z-index: 1100;
}
.button-group {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
}
.button-group .el-button {
    min-width: 90px;
}
.no-cost-configs {
  padding: 20px;
  text-align: center;
}
</style> 