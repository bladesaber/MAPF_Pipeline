<!-- DefineProject.vue -->
<template>
    <div v-if="modelValue" class="right-sidebar">
        <div class="right-sidebar-header">
            <span>Define Project</span>
            <el-button link class="close-button" @click="handleClose">
                <el-icon>
                    <Close />
                </el-icon>
            </el-button>
        </div>
        <div class="right-sidebar-content">
            <!-- Name Section -->
            <div class="section">
                <div class="section-header">
                    <span>Name</span>
                </div>
                <div class="section-content">
                    <el-input v-model="formData.name" placeholder="Enter project name" :maxlength="50"
                        show-word-limit />
                </div>
            </div>

            <!-- Grid Definition Section -->
            <div class="section">
                <div class="section-header">
                    <span>Grid Define</span>
                </div>
                <div class="section-content">
                    <el-form label-position="top">
                        <!-- Grid Size -->
                        <el-form-item label="X Length">
                            <el-input-number 
                                v-model="formData.xlength" 
                                :min="0.1" 
                                :precision="2" 
                                :step="0.1"
                                placeholder="X Length" 
                            />
                        </el-form-item>

                        <el-form-item label="Y Length">
                            <el-input-number 
                                v-model="formData.ylength" 
                                :min="0.1" 
                                :precision="2" 
                                :step="0.1"
                                placeholder="Y Length" 
                            />
                        </el-form-item>

                        <el-form-item label="Z Length">
                            <el-input-number 
                                v-model="formData.zlength" 
                                :min="0.1" 
                                :precision="2" 
                                :step="0.1"
                                placeholder="Z Length" 
                            />
                        </el-form-item>

                        <el-form-item label="Resolution">
                            <el-input-number 
                                v-model="formData.resolution" 
                                :min="0.01" 
                                :max="1" 
                                :precision="2"
                                :step="0.01" 
                                placeholder="Resolution" 
                            />
                        </el-form-item>
                    </el-form>
                </div>
            </div>

            <!-- Button Container -->
            <div class="button-container">
                <div class="button-group">
                    <el-button type="primary" @click="handleApply">Apply</el-button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

interface FormData {
    name: string
    xlength: number
    ylength: number
    zlength: number
    resolution: number
}

const props = defineProps<{
    modelValue: boolean
}>()

const emit = defineEmits(['update:modelValue', 'generate'])

const formData = ref<FormData>({
    name: '',
    xlength: 20.0,
    ylength: 20.0,
    zlength: 20.0,
    resolution: 0.5
})

const handleClose = () => {
    emit('update:modelValue', false)
}

const handleApply = () => {
    if (!formData.value.name.trim()) {
        ElMessage.warning('Please enter a task name')
        return
    }
    if (formData.value.xlength <= 0) {
        ElMessage.warning('X size must be a positive number')
        return
    }
    if (formData.value.ylength <= 0) {
        ElMessage.warning('Y size must be a positive number')
        return
    }
    if (formData.value.zlength <= 0) {
        ElMessage.warning('Z size must be a positive number')
        return
    }
    if (formData.value.resolution <= 0) {
        ElMessage.warning('Resolution must be a positive number')
        return
    }
    if (formData.value.xlength % formData.value.resolution !== 0) {
        ElMessage.warning('X size must be a multiple of resolution')
        return
    }
    if (formData.value.ylength % formData.value.resolution !== 0) {
        ElMessage.warning('Y size must be a multiple of resolution')
        return
    }
    if (formData.value.zlength % formData.value.resolution !== 0) {
        ElMessage.warning('Z size must be a multiple of resolution')
        return
    }

    emit('generate', { ...formData.value })
    emit('update:modelValue', false)
}
</script>

<style scoped>
.right-sidebar {
    position: fixed;
    right: 0;
    top: 127px; /* 36px (card-layout) + 36px (menu-bar) + 60px (menu-bar) */
    width: 450px;
    background: #fff;
    border-left: 1px solid #dcdfe6;
    height: calc(100vh - 127px);
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

:deep(.el-form-item) {
    margin-bottom: 16px;
}

:deep(.el-form-item__label) {
    padding-bottom: 4px;
    line-height: 1.5;
    color: #606266;
}

:deep(.el-input-number) {
    width: 100%;
}

:deep(.el-input-number .el-input__wrapper) {
    padding-right: 0;
}

:deep(.el-input-number .el-input-number__decrease),
:deep(.el-input-number .el-input-number__increase) {
    border: none;
    background: transparent;
}

:deep(.el-input-number .el-input-number__decrease:hover),
:deep(.el-input-number .el-input-number__increase:hover) {
    color: #409eff;
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