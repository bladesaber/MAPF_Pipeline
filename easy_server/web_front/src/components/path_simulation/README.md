# 拓扑优化流体仿真功能设计

## 概述

基于后端`openfoam_simulator.py`中的`test_openfoam_topoIncompressibleFluid`函数，设计并实现了完整的前端拓扑优化流体仿真功能。该功能遵循路径搜索和平滑处理模块的开发模式，提供了6个主要步骤的仿真流程。

## 核心功能

### 1. 仿真步骤流程

按照`test_openfoam_topoIncompressibleFluid`函数的6个步骤实现：

1. **创建场模块** (`CreateFieldModule`)
   - 创建FieldModule实例
   - 设置几何目录和网格参数
   - 初始化OpenFOAM仿真器

2. **计算湍流初始值** (`ComputeInitialValue`)
   - 使用k-epsilon湍流模型
   - 计算湍流动能k和耗散率epsilon
   - 设置参考速度和长度尺度

3. **设置边界条件** (`SetBoundaryConditions`)
   - 创建壁面模块（WallModule）
   - 设置入口管道模块（PipeModule）
   - 设置出口管道模块（PipeModule）
   - 配置各种边界条件参数

4. **生成几何体** (`GenerateGeometry`)
   - 调用`field_module.to_stl()`生成STL文件
   - 创建计算域几何体
   - 进行布尔运算合并几何体

5. **创建OpenFOAM文件** (`CreateOpenFoamFiles`)
   - 创建controlDict、fvSchemes、fvSolution等配置文件
   - 设置求解器参数（topoIncompressibleFluid）
   - 生成网格文件（blockMesh、snappyHexMesh）

6. **设置拓扑权重** (`SetTopoWeight`)
   - 创建topoWeight场文件
   - 设置权重分布（圆形、矩形、自定义）
   - 配置权重函数和参数

### 2. 主要组件

#### SimulateTool.vue
- 主仿真工具组件
- 提供右键菜单操作
- 集成所有仿真步骤
- 处理API通信和进度显示

#### TopoWeightConfig.vue
- 拓扑权重配置组件
- 支持多种权重分布类型
- 提供可视化预览
- 配置权重函数和参数

#### DefineSimulationTask.vue
- 仿真任务定义组件
- 配置仿真参数
- 设置湍流模型和边界条件

#### DefineBoundary.vue
- 边界条件定义组件
- 支持入口、出口、壁面边界
- 配置各种边界条件类型

### 3. API接口设计

基于后端OpenFOAM仿真器，设计了以下API接口：

```
POST /simulate/create_field_module
POST /simulate/compute_initial_value
POST /simulate/set_boundary_conditions
POST /simulate/generate_geometry
POST /simulate/create_openfoam_files
POST /simulate/apply_topo_weight
POST /simulate/run_simulation
POST /simulate/get_results
POST /simulate/export_results
```

### 4. 用户界面设计

#### 右键菜单操作
- 创建仿真任务
- 执行各个仿真步骤
- 运行仿真
- 查看结果

#### 进度显示
- 实时进度条
- 详细日志信息
- 错误处理和提示

#### 配置界面
- 折叠式配置面板
- 参数验证
- 可视化预览

### 5. 技术特点

#### 模块化设计
- 每个仿真步骤独立实现
- 可单独执行和调试
- 支持参数配置

#### 错误处理
- 完整的异常捕获
- 用户友好的错误提示
- 操作回滚机制

#### 可视化支持
- 3D权重分布预览
- 实时进度显示
- 结果可视化

## 使用流程

1. **创建仿真任务**
   - 右键点击GroupCell
   - 选择"Create Simulation Task"
   - 配置仿真参数

2. **执行仿真步骤**
   - 右键点击SimulateTask
   - 按顺序执行6个步骤
   - 每个步骤都有进度显示

3. **配置拓扑权重**
   - 选择"设置拓扑权重"
   - 配置权重分布参数
   - 预览权重效果

4. **运行仿真**
   - 选择"运行仿真"
   - 监控仿真进度
   - 查看仿真结果

## 扩展性

该设计支持以下扩展：

1. **新的湍流模型**
   - 支持Spalart-Allmaras模型
   - 可扩展其他湍流模型

2. **新的边界条件**
   - 支持更多边界条件类型
   - 可自定义边界条件

3. **新的权重分布**
   - 支持复杂权重分布
   - 可导入外部权重数据

4. **结果后处理**
   - 支持多种结果格式
   - 可扩展可视化功能

## 总结

该仿真功能设计完全基于后端OpenFOAM仿真器的实际功能，提供了完整的拓扑优化流体仿真流程。通过模块化设计和用户友好的界面，使得复杂的CFD仿真变得简单易用。同时，该设计具有良好的扩展性，可以支持更多的仿真场景和功能需求。
