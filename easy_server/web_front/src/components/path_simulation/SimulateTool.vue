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
  background: #fff;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
}
.el-dropdown-link {
  cursor: pointer;
  display: flex;
  align-items: center;
  height: 44px;
  padding: 0 20px;
  color: #303133;
  font-size: 15px;
  transition: all 0.3s;
  font-family: inherit;
}
.el-dropdown-link:hover {
  background-color: #f5f7fa;
  color: #409eff;
}
.el-dropdown-link:hover .el-icon--right {
  color: #409eff;
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
  background-color: #dcdfe6;
  z-index: 1;
}
</style> 