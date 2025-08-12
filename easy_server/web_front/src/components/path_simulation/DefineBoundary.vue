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
import { ref, watch } from 'vue';
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
  padding-bottom: 64px;
  min-height: 0;
}

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

.vector-input {
  display: flex;
  align-items: center;
  gap: 8px;
}
.vector-input :deep(.el-input-number) {
  width: 60px;
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