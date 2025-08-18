<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Edit Connector</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <div class="section-header">
        <h3>Connector Configuration</h3>
      </div>
      
      <!-- 不可编辑的重要信息 -->
      <div class="info-section">
        <h4>Basic Information</h4>
        <el-form label-width="120px" size="small">
          <el-form-item label="Connector Name">
            <el-input v-model="connectorData.name" disabled />
          </el-form-item>
          <el-form-item label="Connector Index">
            <el-input-number v-model="connectorData.con_idx" disabled style="width: 150px" />
          </el-form-item>
          <el-form-item label="Group Index">
            <el-input-number v-model="connectorData.group_idx" disabled style="width: 150px" />
          </el-form-item>
          <el-form-item label="Left Segment Index">
            <el-input-number v-model="connectorData.left_seg_idx" disabled style="width: 150px" />
          </el-form-item>
          <el-form-item label="Right Segment Index">
            <el-input-number v-model="connectorData.right_seg_idx" disabled style="width: 150px" />
          </el-form-item>
        </el-form>
      </div>

      <!-- B样条参数 -->
      <div class="section-header">
        <h3>B-Spline Parameters</h3>
      </div>
      <el-form :model="connectorData" label-width="120px" size="small">
        <el-form-item label="B-Spline Degree">
          <el-input-number v-model="connectorData.bspline_degree" :min="1" :max="10" :step="1" style="width: 150px" />
        </el-form-item>
        <el-form-item label="B-Spline Density">
          <el-input-number v-model="connectorData.bspline_density" :min="0.1" :max="1.0" :step="0.05" :precision="2" style="width: 150px" />
        </el-form-item>
      </el-form>

      <!-- 计算选项 -->
      <div class="section-header">
        <h3>Computation Options</h3>
      </div>
      <el-form :model="connectorData" label-width="120px" size="small">
        <el-form-item label="Analytical Curvature">
          <el-switch v-model="connectorData.with_analytical_curvature_computed" />
        </el-form-item>
        <el-form-item label="Analytical Tangent">
          <el-switch v-model="connectorData.with_analytical_tangent_computed" />
        </el-form-item>
      </el-form>

      <!-- 代价设置 -->
      <div class="section-header">
        <h3>Cost Settings</h3>
      </div>
      <el-form :model="connectorData" label-width="120px" size="small">
        <el-form-item label="Obstacle Thickness">
          <el-input-number v-model="connectorData.cost_setting.obstacle_thickness" :min="0.0" :max="10.0" :step="0.01" :precision="3" style="width: 150px" />
        </el-form-item>
        <el-form-item label="Path Thickness">
          <el-input-number v-model="connectorData.cost_setting.path_thickness" :min="0.0" :max="10.0" :step="0.01" :precision="3" style="width: 150px" />
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
      <div v-if="connectorData.cost_setting.cost_cfgs.length === 0" class="no-cost-configs">
        <el-empty description="No cost configurations" :image-size="60">
          <el-button type="primary" size="small" @click="addCostConfig">
            Add First Cost Config
          </el-button>
        </el-empty>
      </div>
      
      <div v-for="(costConfig, index) in connectorData.cost_setting.cost_cfgs" :key="index" class="cost-config-item">
        <div class="cost-config-header">
          <div class="cost-config-title">
            <el-button 
              type="text"            @click="toggleCostConfig(index)"
              class="collapse-button"
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

interface ConnectorCostSetting {
  obstacle_thickness: number;
  path_thickness: number;
  cost_cfgs: CostConfig[];
}

interface ConnectorDataType {
  name: string;
  con_idx: number;
  group_idx: number;
  left_seg_idx: number;
  right_seg_idx: number;
  bspline_degree: number;
  bspline_density: number;
  with_analytical_curvature_computed: boolean;
  with_analytical_tangent_computed: boolean;
  cost_setting: ConnectorCostSetting;
}

const props = defineProps<{
  modelValue: boolean,
  connectorData: ConnectorDataType
}>()
const emit = defineEmits(['update:modelValue', 'save'])

const connectorData = ref<ConnectorDataType>({
  name: '',
  con_idx: 0,
  group_idx: 0,
  left_seg_idx: 0,
  right_seg_idx: 0,
  bspline_degree: 3,
  bspline_density: 0.25,
  with_analytical_curvature_computed: true,
  with_analytical_tangent_computed: false,
  cost_setting: {
    obstacle_thickness: 0.01,
    path_thickness: 0.01,
    cost_cfgs: []
  }
})

watch(() => props.modelValue, (v) => {
  if (v && props.connectorData) {
    connectorData.value = { ...props.connectorData }
    
    // 确保cost_cfgs存在，但允许为空数组
    if (!connectorData.value.cost_setting.cost_cfgs) {
      connectorData.value.cost_setting.cost_cfgs = []
    }
    
    // 根据实际的cost_cfgs长度设置collapsedCostConfigs
    collapsedCostConfigs.value = connectorData.value.cost_setting.cost_cfgs.map(() => false)
  }
})

const addCostConfig = () => {
  const defaultConfig: CostConfig = {
    cost_type: 'curvature',
    method: 'sigmoid_mean',
    weight: 10
  }
  
  // 根据cost_type初始化特定属性
  if (defaultConfig.cost_type === 'curvature') {
    defaultConfig.curvature_method = 'analytical_curvature'
    defaultConfig.curvature_scale = 3
  } else if (defaultConfig.cost_type === 'clamp_vector') {
    defaultConfig.vector = [0, 0, 1]
    defaultConfig.vector_length = 1
  } else {
    defaultConfig.data_type = 'control_points'
  }
  
  connectorData.value.cost_setting.cost_cfgs.push(defaultConfig)
  collapsedCostConfigs.value.push(false) // 添加新配置时展开
}

const removeCostConfig = (index: number) => {
  connectorData.value.cost_setting.cost_cfgs.splice(index, 1)
  collapsedCostConfigs.value.splice(index, 1)
}

const onCostTypeChange = (costConfig: CostConfig) => {
  // 当 cost_type 改变时，根据新的 cost_type 更新其他字段
  if (costConfig.cost_type === 'length' || costConfig.cost_type === 'length_distribution') {
    costConfig.data_type = 'control_points';
    costConfig.curvature_method = undefined;
    costConfig.curvature_scale = undefined;
    costConfig.vector = undefined;
    costConfig.vector_length = undefined;
  } else if (costConfig.cost_type === 'curvature') {
    costConfig.data_type = undefined;
    costConfig.vector = undefined;
    costConfig.vector_length = undefined;
  } else if (costConfig.cost_type === 'clamp_vector') {
    costConfig.data_type = undefined;
    costConfig.curvature_method = undefined;
    costConfig.curvature_scale = undefined;
  }
}

const handleClose = () => {
  emit('update:modelValue', false)
}
const handleSave = () => {
  emit('save', { ...connectorData.value })
  emit('update:modelValue', false)
}

const collapsedCostConfigs = ref<boolean[]>([])

const toggleCostConfig = (index: number) => {
  collapsedCostConfigs.value[index] = !collapsedCostConfigs.value[index]
}
</script>

<style scoped>
.right-sidebar {
    position: fixed;
    right: 0;
    top: 116px;
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
    color: #409eff;
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
    margin: 16px 0 8px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #409eff;
}
.section-header h3 {
    margin: 0;
    color: #303133;
    font-size: 16px;
    font-weight: 600;
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
.collapse-button {
  transition: transform 0.3s ease;
}
.collapse-button.is-expanded {
  transform: rotate(90deg);
}
.cost-config-title {
  display: flex;
  align-items: center;
  gap: 8px;
}
.cost-config-content {
  padding-left: 20px; /* Indent for collapsed content */
}
.no-cost-configs {
  padding: 20px;
  text-align: center;
}
</style> 