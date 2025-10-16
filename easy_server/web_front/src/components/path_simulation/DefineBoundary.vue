<!-- DefineSimulationTask.vue -->
<template>
  <div v-if="modelValue" class="right-sidebar">
    <div class="right-sidebar-header">
      <span>Define {{ mode }} Boundary</span>
      <el-button link class="close-button" @click="handleClose">
        <el-icon><Close /></el-icon>
      </el-button>
    </div>
    <div class="right-sidebar-content">
      <el-form label-width="120px">
        <el-form-item label="Market">
          <el-select v-model="selectedMarket" style="width: 100%">
            <el-option v-for="market in boundary_list" :key="market" :label="market" :value="market" />
          </el-select>
        </el-form-item>
      </el-form>
      <div v-if="selectedMarket">
        <el-card shadow="never" class="section" :body-style="{padding: '0'}">
          <div class="section-header">
            <span>{{ selectedMarket }}</span>
          </div>
          <div class="section-content">
            <el-form label-width="120px">
              <!-- Inlet -->
              <template v-if="mode === 'Inlet'">
                <el-form-item label="Type">
                  <el-select v-model="formData[selectedMarket].type" style="width: 100%">
                    <el-option label="TOTAL_CONDITIONS" value="TOTAL_CONDITIONS" />
                    <el-option label="MASS_FLOW" value="MASS_FLOW" />
                    <el-option label="VELOCITY_INLET" value="VELOCITY_INLET" />
                    <el-option label="PRESSURE_INLET" value="PRESSURE_INLET" />
                  </el-select>
                </el-form-item>
                <template v-if="formData[selectedMarket].type === 'TOTAL_CONDITIONS'">
                  <el-form-item label="Pressure">
                    <el-input-number v-model="formData[selectedMarket].pressure" />
                  </el-form-item>
                  <el-form-item label="Temperature">
                    <el-input-number v-model="formData[selectedMarket].temperature" />
                  </el-form-item>
                  <el-form-item label="Direction">
                    <div class="vector-input">
                      <el-input-number v-model="formData[selectedMarket].direction[0]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[1]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[2]" :controls="false" />
                    </div>
                  </el-form-item>
                </template>
                <template v-else-if="formData[selectedMarket].type === 'MASS_FLOW'">
                  <el-form-item label="Density">
                    <el-input-number v-model="formData[selectedMarket].density" />
                  </el-form-item>
                  <el-form-item label="Vel Scale">
                    <el-input-number v-model="formData[selectedMarket].vel_scale" />
                  </el-form-item>
                  <el-form-item label="Direction">
                    <div class="vector-input">
                      <el-input-number v-model="formData[selectedMarket].direction[0]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[1]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[2]" :controls="false" />
                    </div>
                  </el-form-item>
                </template>
                <template v-else-if="formData[selectedMarket].type === 'VELOCITY_INLET'">
                  <el-form-item label="Temperature">
                    <el-input-number v-model="formData[selectedMarket].temperature" />
                  </el-form-item>
                  <el-form-item label="Vel Scale">
                    <el-input-number v-model="formData[selectedMarket].vel_scale" />
                  </el-form-item>
                  <el-form-item label="Direction">
                    <div class="vector-input">
                      <el-input-number v-model="formData[selectedMarket].direction[0]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[1]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[2]" :controls="false" />
                    </div>
                  </el-form-item>
                </template>
                <template v-else-if="formData[selectedMarket].type === 'PRESSURE_INLET'">
                  <el-form-item label="Temperature">
                    <el-input-number v-model="formData[selectedMarket].temperature" />
                  </el-form-item>
                  <el-form-item label="Pressure">
                    <el-input-number v-model="formData[selectedMarket].pressure" />
                  </el-form-item>
                  <el-form-item label="Direction">
                    <div class="vector-input">
                      <el-input-number v-model="formData[selectedMarket].direction[0]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[1]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[2]" :controls="false" />
                    </div>
                  </el-form-item>
                </template>
              </template>
              <!-- Outlet -->
              <template v-else-if="mode === 'Outlet'">
                <el-form-item label="Type">
                  <el-select v-model="formData[selectedMarket].type" style="width: 100%">
                    <el-option label="PRESSURE_OUTLET" value="PRESSURE_OUTLET" />
                    <el-option label="MASS_FLOW_OUTLET" value="MASS_FLOW_OUTLET" />
                    <el-option label="VELOCITY_OUTLET" value="VELOCITY_OUTLET" />
                  </el-select>
                </el-form-item>
                <template v-if="formData[selectedMarket].type === 'PRESSURE_OUTLET'">
                  <el-form-item label="Pressure">
                    <el-input-number v-model="formData[selectedMarket].pressure" />
                  </el-form-item>
                </template>
                <template v-else-if="formData[selectedMarket].type === 'MASS_FLOW_OUTLET'">
                  <el-form-item label="Mass Flow">
                    <el-input-number v-model="formData[selectedMarket].mass_flow" />
                  </el-form-item>
                </template>
                <template v-else-if="formData[selectedMarket].type === 'VELOCITY_OUTLET'">
                  <el-form-item label="Temperature">
                    <el-input-number v-model="formData[selectedMarket].temperature" />
                  </el-form-item>
                  <el-form-item label="Vel Scale">
                    <el-input-number v-model="formData[selectedMarket].vel_scale" />
                  </el-form-item>
                  <el-form-item label="Direction">
                    <div class="vector-input">
                      <el-input-number v-model="formData[selectedMarket].direction[0]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[1]" :controls="false" />
                      <el-input-number v-model="formData[selectedMarket].direction[2]" :controls="false" />
                    </div>
                  </el-form-item>
                </template>
              </template>
              <!-- Wall -->
              <template v-else>
                <el-form-item label="Type">
                  <el-select v-model="formData[selectedMarket].type" style="width: 100%" disabled>
                    <el-option label="MARKER_ISOTHERMAL" value="MARKER_ISOTHERMAL" />
                  </el-select>
                </el-form-item>
                <el-form-item label="Temperature">
                  <el-input-number v-model="formData[selectedMarket].temperature" />
                </el-form-item>
              </template>
            </el-form>
          </div>
        </el-card>
      </div>
    </div>
    <div class="button-container">
      <div class="button-group">
        <el-button type="primary" @click="handleGenerate">Generate</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ref,
  watch,
} from 'vue';

import { ElMessage } from 'element-plus';

import { Close } from '@element-plus/icons-vue';

const props = defineProps<{
  modelValue: boolean,
  mode: 'Inlet' | 'Outlet' | 'Wall',
  boundary_list: string[],
  label_dict: Map<string, any>
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const selectedMarket = ref('');

const defaultInlet = () => ({
  type: 'TOTAL_CONDITIONS',
  pressure: 0,
  temperature: 300,
  direction: [1, 0, 0],
  density: 1.225,
  vel_scale: 1,
});
const defaultOutlet = () => ({
  type: 'PRESSURE_OUTLET',
  pressure: 0,
  mass_flow: 0,
  temperature: 300,
  vel_scale: 1,
  direction: [1, 0, 0],
});
const defaultWall = () => ({
  type: 'MARKER_ISOTHERMAL',
  temperature: 300,
});

const formData = ref<{ [market: string]: any }>({});

watch(
  () => props.boundary_list,
  (newList) => {
    const newData: { [market: string]: any } = {};
    for (const market of newList) {
      if (props.mode === 'Inlet') newData[market] = { ...defaultInlet() };
      else if (props.mode === 'Outlet') newData[market] = { ...defaultOutlet() };
      else newData[market] = { ...defaultWall() };
    }
    formData.value = newData;
    // Set default selected market
    if (newList.length > 0) selectedMarket.value = newList[0];
    else selectedMarket.value = '';
  },
  { immediate: true }
);

const handleClose = () => {
  emit('update:modelValue', false)
}

const handleGenerate = () => {
  // Normalize direction vector for the selected market before emit
  const boundaries = JSON.parse(JSON.stringify(formData.value)); // deep copy
  const market = selectedMarket.value;
  if (boundaries[market] && boundaries[market].direction && Array.isArray(boundaries[market].direction)) {
    const dir = boundaries[market].direction;
    const norm = Math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (norm > 0) {
      boundaries[market].direction = dir.map((v: number) => v / norm);
    }
  }
  emit('generate', { mode: props.mode, market, boundary: boundaries[market] })
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
  padding-bottom: 100px;
  min-height: 0;
  background: rgba(248, 250, 252, 0.3);
}

/* 现代化滚动条样式 */
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
  margin-bottom: 24px;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid rgba(226, 232, 240, 0.4);
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.section:hover {
  background: rgba(255, 255, 255, 0.8);
  border-color: rgba(99, 102, 241, 0.2);
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.08);
  transform: translateY(-1px);
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
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  transition: all 0.3s ease;
}

.section-header:hover {
  color: #6366f1;
  border-bottom-color: rgba(99, 102, 241, 0.4);
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
  width: 60px;
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
</style>