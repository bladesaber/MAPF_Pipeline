<template>
  <el-menu mode="horizontal" class="menu-bar">
    <el-dropdown trigger="click" placement="bottom">
      <span class="el-dropdown-link">Help<el-icon class="el-icon--right"><arrow-down /></el-icon></span>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item>Help</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </el-menu>
</template>
<script setup lang="ts">
import { ArrowDown } from '@element-plus/icons-vue';
</script>
<style scoped>
.menu-bar {
  border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
}

.el-dropdown-link {
  cursor: pointer;
  display: flex;
  align-items: center;
  height: 48px;
  padding: 0 24px;
  color: #475569;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-family: inherit;
  border-radius: 8px;
  margin: 0 4px;
  letter-spacing: 0.025em;
}

.el-dropdown-link:hover {
  background: rgba(99, 102, 241, 0.08);
  color: #6366f1;
  transform: translateY(-1px);
}

.el-dropdown-link:hover .el-icon--right {
  color: #6366f1;
  transform: translateY(-1px);
}

.el-dropdown-link + .el-dropdown-link {
  margin-left: 8px;
}

.menu-bar::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(226, 232, 240, 0.6) 50%, transparent 100%);
  z-index: 1;
}
</style> 