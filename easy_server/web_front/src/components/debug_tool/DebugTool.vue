<!-- DebugTool.vue -->
<template>
    <div class="debug-tool" @click="handleClickOutside">
      <el-menu mode="horizontal" class="menu-bar">
        <el-sub-menu index="1">
          <template #title>Help</template>
          <el-menu-item index="1-1">About</el-menu-item>
        </el-sub-menu>
      </el-menu>

      <DebugToolPanel
        v-model="debugToolPanelVisible"
        :mode="debugToolPanelMode"
        :label_dict="label_dict"
        @generate="handleDebugToolPanelGenerate"
        />
      
      <div 
        v-show="gridContextMenuVisible"
        class="context-menu"
        :style="contextMenuStyle"
        @click.stop
      >
        <el-menu>
          <el-menu-item @click="handleTreeContextMenu('Grid/ShowInner')">Show Inner</el-menu-item>
        </el-menu>
      </div>

      <div 
        v-show="obstaclesContextMenuVisible"
        class="context-menu"
        :style="contextMenuStyle"
        @click.stop
        >
        <el-menu>
            <el-menu-item @click="handleTreeContextMenu('Obstacles/PointObstacles')">Random Point Obstacles</el-menu-item>
            <el-menu-item @click="handleTreeContextMenu('Obstacles/SphereObstacles')">Random Sphere Obstacles</el-menu-item>
            <el-menu-item @click="handleTreeContextMenu('Obstacles/SquaresObstacles')">Random Squares Obstacles</el-menu-item>
          </el-menu>
      </div>

      <div 
        v-show="pipesContextMenuVisible"
        class="context-menu"
        :style="contextMenuStyle"
        @click.stop
        >
        <el-menu>
            <el-menu-item @click="handleTreeContextMenu('Pipes/RandomPipes')">Random Pipes</el-menu-item>
        </el-menu>
      </div>

      <div 
        v-show="tasksContextMenuVisible"
        class="context-menu"
        :style="contextMenuStyle"
        @click.stop
        >
        <el-menu>
            <el-menu-item @click="handleTreeContextMenu('Tasks/RandomTasks')">Random Tasks</el-menu-item>
        </el-menu>
      </div>

    </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, defineExpose, inject, defineEmits } from 'vue'
import * as THREE from 'three'
import { ElMessage } from 'element-plus'

import DebugToolPanel from './DebugToolPanel.vue'

import { CenterRoot } from '@/classes/center_root.ts'
import { setVisableDict } from '@/utils/utils.ts'
import { generateSphereObstacles_by_percentage, generateRandomPipes, 
  generateTask, generatePointObstacles_by_percentage, generateSquaresObstacles_by_percentage } from '@/classes/debug_tool/path_debug.ts'
import { PipeMesh } from '@/classes/path_search/pipe_mesh.ts'
import { PointsObstacle, SpheresObstacle, SquaresObstacle } from '@/classes/path_search/obstacle_mesh.ts'
import { SearchTask, PriorityTask, GroupTask, CellTask } from '@/classes/path_search/task_runner.ts'

const emit = defineEmits(['updateTree', 'highlightTreeNode', 'updateLog'])

const rootObj = inject<CenterRoot>('centerRoot', new CenterRoot(new THREE.Scene()))

// ------ Context Menu ------
const contextMenuPosition = reactive({ x: 0, y: 0 })
const contextMenuStyle = computed(() => ({
  position: 'fixed' as const,
  left: `${contextMenuPosition.x}px`,
  top: `${contextMenuPosition.y}px`
}))

// ------ Left Side Bar ------
const label_dict = ref(new Map<string, string | number | string[] | Map<string, string|number>>())

const obstaclesContextMenuVisible = ref(false)
const pipesContextMenuVisible = ref(false)
const tasksContextMenuVisible = ref(false)
const gridContextMenuVisible = ref(false)

const contextMenuVisibleDict = {
  'obstaclesContextMenuVisible': obstaclesContextMenuVisible,
  'pipesContextMenuVisible': pipesContextMenuVisible,
  'tasksContextMenuVisible': tasksContextMenuVisible,
  'gridContextMenuVisible': gridContextMenuVisible
}

const hideContextMenu = () => {
    obstaclesContextMenuVisible.value = false
    pipesContextMenuVisible.value = false
    tasksContextMenuVisible.value = false
    gridContextMenuVisible.value = false
}

const showContextMenu = (data: any) => {
  hideContextMenu()
  contextMenuPosition.x = data.x
  contextMenuPosition.y = data.y

  // Convert data.data to Map
  const newLabelDict = new Map<string, string | number | string[] | Map<string, string|number>>()
  
  // Add basic properties
  if (data.data.label) newLabelDict.set('label', data.data.label)
  if (data.data.style) newLabelDict.set('style', data.data.style)
  
  // Add any additional data properties
  if (data.data.data) {
    Object.entries(data.data.data).forEach(([key, value]) => {
      newLabelDict.set(key, value as string | number | string[] | Map<string, string|number>)
    })
  }

  label_dict.value = newLabelDict

  // Show the appropriate context menu based on style
  const style = data.data.style
  if (style === 'Obstacles') {
    obstaclesContextMenuVisible.value = true
  } else if (style === 'Pipes') {
    pipesContextMenuVisible.value = true
  } else if (style === 'Tasks') {
    tasksContextMenuVisible.value = true
  }
}

// ------ Fragment Visible ------
const debugToolPanelVisible = ref(false)

const fragmentVisibleDict = {
  'debugToolPanelVisible': debugToolPanelVisible,
}

const hideFragmentMenu = () => {
  setVisableDict('debugToolPanelVisible', fragmentVisibleDict, true, false)
}

// ------ Handle Tree Context Menu ------
const debugToolPanelMode = ref<'PointObstacles' | 'SphereObstacles' | 'SquaresObstacles'  | 'RandomPipes' | 'RandomTasks'>('PointObstacles')

const handleTreeContextMenu = (mode: string) => {
  if (mode == 'Obstacles/SphereObstacles'){
    debugToolPanelMode.value = 'SphereObstacles'
    setVisableDict('debugToolPanelVisible', fragmentVisibleDict, false, true)

  } else if (mode == 'Pipes/RandomPipes'){
    debugToolPanelMode.value = 'RandomPipes'
    setVisableDict('debugToolPanelVisible', fragmentVisibleDict, false, true)

  } else if (mode == 'Tasks/RandomTasks'){
    debugToolPanelMode.value = 'RandomTasks'
    setVisableDict('debugToolPanelVisible', fragmentVisibleDict, false, true)

  } else if (mode == 'Obstacles/PointObstacles'){
    debugToolPanelMode.value = 'PointObstacles'
    setVisableDict('debugToolPanelVisible', fragmentVisibleDict, false, true)
    
  } else if (mode == 'Obstacles/SquaresObstacles'){
    debugToolPanelMode.value = 'SquaresObstacles'
    setVisableDict('debugToolPanelVisible', fragmentVisibleDict, false, true)
  }
  hideContextMenu()
}

const handleDebugToolPanelGenerate = (data: any) => {
  const current_label_dict = data.label_dict
  const project_manager = rootObj.proj_dict.get(current_label_dict.get('Project')!)

  if (data.mode == 'SphereObstacles'){
    const pipes_dict: {x: number, y: number, z: number, radius: number}[] = []
    project_manager!.pipe_dict.forEach((pipes, group_name) => {
      pipes.forEach((pipe) => {
        pipes_dict.push({x: pipe.x, y: pipe.y, z: pipe.z, radius: pipe.radius})
      })
    })
    
    const random_points_data = generateSphereObstacles_by_percentage(
      project_manager!.grid.xlength!, project_manager!.grid.ylength!, project_manager!.grid.zlength!,
      data.percentage, data.radius_min, data.radius_max,
      pipes_dict
    )

    const obstacle_points = []
    const obstacle_radii = []
    for (const point of random_points_data){
      obstacle_points.push([point.x, point.y, point.z])
      obstacle_radii.push(point.radius)
    }

    const obstacle_mesh = new SpheresObstacle(
      data.name, [255, 0, 0],
      obstacle_points,
      obstacle_radii, 
      [], rootObj.get_scene()
    );
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.obstacle_dict.set(obstacle_mesh.name, obstacle_mesh)

  } else if (data.mode == 'RandomPipes'){
    if (project_manager!.pipe_dict.has(data.group)){
      ElMessage.error('Pipe group already exists')
      return
    }
    project_manager!.pipe_dict.set(data.group, new Map<string, PipeMesh>())
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.group_color_dict.set(
      data.group, [Math.random() * 255, Math.random() * 255, Math.random() * 255]
    )

    const other_pipes_dict = []
    for (const pipe of project_manager!.pipe_dict.get(data.group)!.values()){
      other_pipes_dict.push({
        x: project_manager!.grid.value_to_grid(pipe.x), 
        y: project_manager!.grid.value_to_grid(pipe.y), 
        z: project_manager!.grid.value_to_grid(pipe.z), 
        radius: project_manager!.grid.value_to_grid_scaled(pipe.radius)  // Be careful with the unit
      })
    }

    const random_pipes = generateRandomPipes(
      project_manager!.grid.xsize!, project_manager!.grid.ysize!, project_manager!.grid.zsize!,
      project_manager!.grid.value_to_grid_scaled(data.radius), data.count, other_pipes_dict, [], 
      project_manager!.grid.resolution
    ) // TODO: Add obstacle dict

    for (let i = 0; i < random_pipes.length; i++){
      const pipe = random_pipes[i]
      const pipe_mesh = new PipeMesh(
        `${data.group}_${i}`, 
        pipe.x, pipe.y, pipe.z, data.radius,
        pipe.vector_x, pipe.vector_y, pipe.vector_z,
        rootObj.proj_dict.get(current_label_dict.get('Project')!)!.group_color_dict.get(data.group)!,
        data.group,
        pipe.pipe_mode,
        rootObj.get_scene()
      )

      rootObj.proj_dict.get(current_label_dict.get('Project')!)!.pipe_dict.get(data.group)!.set(
        pipe_mesh.name, pipe_mesh
      )
    }
  
  } else if (data.mode == 'RandomTasks'){
    if (project_manager?.search_task_dict.has(data.name)){
      ElMessage.error('Search task already exists')
      return
    }

    const search_task = new SearchTask(data.name)
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.search_task_dict.set(
      search_task.name, search_task
    )

    for (let i = 0; i < data.priority_count; i++){
      const priority_run_config = new Map<string, string | number | boolean>()
      Object.entries(data.priority_run_config).forEach(([key, value]) => {
        priority_run_config.set(key, value as string | number | boolean)
      })
      const priority_task = new PriorityTask(`priority_${i}`, priority_run_config)
      search_task.priority_tasks.push(priority_task)
    }

    project_manager!.pipe_dict.forEach((pipes, group_name) => {
      const pipe_names = Array.from(pipes.keys())
      const min_radius = Math.min(...Array.from(pipes.values()).map((pipe) => pipe.radius))
      const random_priority_tasks = generateTask(pipe_names, min_radius, data.priority_count)
      
      for (let i = 0; i < data.priority_count; i++){
        const cell_tasks = random_priority_tasks[i]
        const priority_task = search_task.priority_tasks[i]
        
        const group_run_config = new Map<string, string | number | boolean>()
        Object.entries(data.group_run_config).forEach(([key, value]) => {
          group_run_config.set(key, value as string | number | boolean)
        })

        const group_task = new GroupTask(group_name, group_run_config)
        priority_task.group_tasks.set(group_task.name, group_task)

        for (let j = 0; j < cell_tasks.length; j++){
          const cell_task = new CellTask(
            `${cell_tasks[j].start_name}_${cell_tasks[j].end_name}`, 
            cell_tasks[j].start_name, cell_tasks[j].end_name,
            cell_tasks[j].radius, 
            rootObj.proj_dict.get(current_label_dict.get('Project')!)!.group_color_dict.get(group_name)!
          )
          group_task.cell_tasks.push(cell_task)
        }
      }
    })
  } else if (data.mode == 'PointObstacles'){
    const pipes_dict: {x: number, y: number, z: number, radius: number}[] = []
    project_manager!.pipe_dict.forEach((pipes, group_name) => {
      pipes.forEach((pipe) => {
        pipes_dict.push({
          x: project_manager!.grid.value_to_grid(pipe.x), 
          y: project_manager!.grid.value_to_grid(pipe.y), 
          z: project_manager!.grid.value_to_grid(pipe.z), 
          radius: project_manager!.grid.value_to_grid(pipe.radius)  // Be careful with the unit
        })
      })
    })

    const random_points_data = generatePointObstacles_by_percentage(
      project_manager!.grid.xsize!, project_manager!.grid.ysize!, project_manager!.grid.zsize!, 
      data.percentage, pipes_dict, 
      project_manager!.grid.resolution
    )

    const obstacle_points = []
    for (const point of random_points_data){
      obstacle_points.push([
        project_manager!.grid.grid_to_value(point[0]),
        project_manager!.grid.grid_to_value(point[1]),
        project_manager!.grid.grid_to_value(point[2])
      ])
    }

    const obstacle_mesh = new PointsObstacle(
      data.name, [255, 0, 0],
      obstacle_points,
      [], rootObj.get_scene()
    );
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.obstacle_dict.set(obstacle_mesh.name, obstacle_mesh)
  
  } else if (data.mode == 'SquaresObstacles'){
    const pipes_dict: {x: number, y: number, z: number, radius: number}[] = []
    project_manager!.pipe_dict.forEach((pipes, group_name) => {
      pipes.forEach((pipe) => {
        pipes_dict.push({x: pipe.x, y: pipe.y, z: pipe.z, radius: pipe.radius})
      })
    })

    const random_squares_data = generateSquaresObstacles_by_percentage(
      project_manager!.grid.xlength!, project_manager!.grid.ylength!, project_manager!.grid.zlength!, 
      data.size_min, data.size_max, data.percentage, pipes_dict
    )

    const obstacle_points = []
    const obstacle_sizes = []
    for (const square of random_squares_data){
      obstacle_points.push([square.x, square.y, square.z])
      obstacle_sizes.push(square.size)
    }

    const obstacle_mesh = new SquaresObstacle(
      data.name, [255, 0, 0],
      obstacle_points,
      obstacle_sizes,
      [], rootObj.get_scene()
    );
    rootObj.proj_dict.get(current_label_dict.get('Project')!)!.obstacle_dict.set(obstacle_mesh.name, obstacle_mesh)
  }

  emit('updateTree')
}

// ------ Click Outside Handler ------
const handleClickOutside = (event: MouseEvent) => {
  // Check if click is outside of any context menu
  const contextMenus = document.querySelectorAll('.context-menu')
  let clickedInsideMenu = false
  
  contextMenus.forEach(menu => {
    if (menu.contains(event.target as Node)) {
      clickedInsideMenu = true
    }
  })
  
  if (!clickedInsideMenu) {
    hideContextMenu()
  }
}

// ------ Expose ------
defineExpose({
  hideContextMenu,
  showContextMenu
})

</script>

<style scoped>
.debug-tool {
  width: 100%;
  position: relative;
}

.menu-bar {
  border-bottom: 1px solid #dcdfe6;
}

.context-menu {
  position: fixed;
  background: white;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  z-index: 3000;
}

.context-menu :deep(.el-menu) {
  border-right: none;
}

.context-menu :deep(.el-menu-item) {
  height: 32px;
  line-height: 32px;
  padding: 0 16px;
}

.context-menu :deep(.el-menu-item:hover) {
  background-color: #f5f7fa;
}

/* Add styles for click outside handler */
.click-outside-handler {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2999;
  background: transparent;
}
</style>