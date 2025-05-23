<template>
  <div 
    v-if="modelValue" 
    class="search-dialog" 
    :class="{ minimized }"
    :style="dialogStyle"
    @mousedown="startDrag"
  >
    <div class="dialog-header">
      <span>Path Search Progress</span>
      <div class="dialog-controls">
        <el-button 
          link
          @click="toggleMinimize"
        >
          <el-icon>
            <component :is="minimized ? 'FullScreen' : 'Minus'" />
          </el-icon>
        </el-button>
        <el-button 
          link
          @click="handleClose"
        >
          <el-icon><Close /></el-icon>
        </el-button>
      </div>
    </div>

    <div class="dialog-content" v-show="!minimized">
      <div class="log-window">
        <div v-for="(message, index) in messages" :key="index" class="log-entry" :class="{ 'success': message.is_success }">
          <span class="log-time">[{{ message.time }}]</span>
          <span class="log-type">[{{ message.type }}]</span>
          <span class="log-message">{{ message.message }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, inject } from 'vue'
import { Close, FullScreen, Minus } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { BlockRoot } from '../../classes/pathSearch_classes/base_root.ts'
import { ObstacleBlock } from '../../classes/pathSearch_classes/obstacle_block.ts'
import { BaseBlock } from '../../classes/pathSearch_classes/base_block.ts'
import { MainTask } from '../../classes/pathSearch_classes/base_task.ts'
import { el } from 'element-plus/es/locales.mjs'
const rootObj = inject<BlockRoot>('blockRoot')

interface LayoutData {
  name: string
  status: string
  progress: number
}

interface Position {
  x: number
  y: number
}

interface LogMessage {
  type: string
  is_success: boolean
  message: string
  time: string
  time_cost?: string | number
}

const props = defineProps<{
  modelValue: boolean
  currentTagDict: Map<string, string | string[] | Map<string, string|number> | number>
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'close'): void
  (e: 'lockLeftSideBarTree'): void
  (e: 'unlockLeftSideBarTree'): void
  (e: 'parseDiscretizeGrid', data: any): void
  (e: 'parseSearchPreprocess', data: any): void
  (e: 'parseRunPrioritySearch', data: any): void
  (e: 'updateRunPrioritySearch', data: any): void
}>()

// State
const minimized = ref(false)
const position = ref<Position>({
  x: window.innerWidth - 600,
  y: 100
})
const lastPosition = ref<Position | null>(null)
const isDragging = ref(false)
const dragOffset = ref<Position>({ x: 0, y: 0 })
const layoutData = ref<LayoutData[]>([])
const socket = ref<WebSocket | null>(null)
const messages = ref<LogMessage[]>([])

// Computed
const dialogStyle = computed(() => {
  if (minimized.value) {
    return {
      left: '20px',
      bottom: '20px',
      top: 'auto',
      transform: 'none'
    }
  }
  return {
    left: `${position.value.x}px`,
    top: `${position.value.y}px`
  }
})

// Methods
const connectWebSocket = () => {
  if (socket.value) return

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//192.168.110.54:9998/path_search/run`
  
  socket.value = new WebSocket(wsUrl)
  
  socket.value.onopen = () => {
    send_msg({ run_type: "HandShake" })
    messages.value.unshift({
      type: 'System',
      is_success: true,
      message: 'WebSocket connected',
      time: new Date().toLocaleTimeString()
    })
  }

  socket.value.onmessage = (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data)
      // console.log('Received WebSocket message:', data)

      if (data.run_type === 'prepareGrid') {
        response_discretize_grid(data)
      }
      else if (data.run_type === 'searchPreprocess') {
        response_search_preprocess(data)
      }
      else if (data.run_type === 'runPrioritySearch') {
        console.log('runPrioritySearch', data)
        response_run_priority_search(data)
      }

    } catch (error) {
      console.error('Error parsing WebSocket message:', error)
      messages.value.unshift({
        type: 'Error',
        is_success: false,
        message: `Error parsing WebSocket message: ${String(error)}`,
        time: new Date().toLocaleTimeString()
      })
    }
  }
  
  socket.value.onclose = () => {
    messages.value.unshift({
      type: 'System',
      is_success: false,
      message: 'WebSocket connection closed',
      time: new Date().toLocaleTimeString()
    })
    socket.value = null
  }
  
  socket.value.onerror = (error: Event) => {
    messages.value.unshift({
      type: 'Error',
      is_success: false,
      message: `WebSocket error: ${String(error)}`,
      time: new Date().toLocaleTimeString()
    })
  }
}

const send_msg = (data: any) => {
  if (socket.value && socket.value.readyState === WebSocket.OPEN) {
    socket.value.send(JSON.stringify(data))
  }
}

const disconnectWebSocket = () => {
  if (socket.value) {
    socket.value.close()
    socket.value = null
  }
}

const startDrag = (event: MouseEvent) => {
  const target = event.target as HTMLElement
  if (target.closest('.dialog-controls')) return
  if (!minimized.value) {
    isDragging.value = true
    dragOffset.value = {
      x: event.clientX - position.value.x,
      y: event.clientY - position.value.y
    }
  }
}

const onDrag = (event: MouseEvent) => {
  if (!isDragging.value || minimized.value) return
  position.value = {
    x: event.clientX - dragOffset.value.x,
    y: event.clientY - dragOffset.value.y
  }
}

const stopDrag = () => {
  isDragging.value = false
}

const toggleMinimize = () => {
  if (!minimized.value) {
    lastPosition.value = { ...position.value }
  } else if (lastPosition.value) {
    position.value = { ...lastPosition.value }
  }
  minimized.value = !minimized.value
}

const handleClose = () => {
  emit('update:modelValue', false)
  disconnectWebSocket()
  emit('close')
}

// Watch for visibility changes
watch(() => props.modelValue, (newVal) => {
  if (newVal) {
    connectWebSocket()
  } else {
    disconnectWebSocket()
  }
})

let current_manager_name = '';
let current_main_task_name = '';
let current_priority_task_name = '';
let current_group_task_name = '';
let current_cell_task_name = '';

// ------ Discretize Grid Request -------
const discretizeGrid_data = {
  grid_data: [] as number[][],
  distance_to_obstacle: [] as number[],
}
const prepare_discretize_grid = (task_manager_name: string) => {
  current_manager_name = task_manager_name;
  discretizeGrid_data.grid_data.length = 0
  discretizeGrid_data.distance_to_obstacle.length = 0
}

const request_discretize_grid = (task_manager_name: string) => {
  const task_manager = rootObj!.get_task_manager(task_manager_name)

  let is_valid = true;
  task_manager!.obstacle_blocks.forEach(
    (block: BaseBlock) => {
      if (!(block as ObstacleBlock).has_discretize_mesh) {
        ElMessage.error(`Obstacle ${block.name} has no discretize mesh`)
        is_valid = false;
      }
    }
  )
  if (!is_valid) {
    emit('unlockLeftSideBarTree')
    return
  }

  const obstacle_points: number[][] = []
  task_manager!.obstacle_blocks.forEach(
    (block: BaseBlock) => {
      const obstacle_block = block as ObstacleBlock
      obstacle_points.push(...obstacle_block.discretize_points)
    }
  )
  
  prepare_discretize_grid(task_manager!.name)
  send_msg({ 
    run_type: "prepareGrid", 
    x_size: task_manager!.get_xsize(),
    y_size: task_manager!.get_ysize(),
    z_size: task_manager!.get_zsize(),
    resolution: task_manager!.get_resolution(),
    obstacle_points_data: obstacle_points 
  })
}

const response_discretize_grid = async (data: any) => {
  if (!data.total_finished) {
    const messageDetails = [
      `Run Type: ${data.run_type}`,
      `Name: ${data.name}`,
      `Progress: ${data.progress}%`,
      `is_finished: ${data.is_finished}`,
    ].join(' | ')
    messages.value.unshift({
      type: data.run_type || 'Info',
      is_success: data.is_finished || false,
      message: messageDetails,
      time: new Date().toLocaleTimeString()
    })

    if (data.name === 'return_validate_data') {
      (data.data as number[][]).forEach(point => {
        discretizeGrid_data.grid_data.push([point[0], point[1], point[2]])
        discretizeGrid_data.distance_to_obstacle.push(point[3])
      })
    }

  } else {
    const messageDetails = [
      `Run Type: ${data.run_type}`,
      `Name: ${data.name}`,
      `Progress: ${data.progress}%`,
      `Total Finished: ${data.total_finished}`,
    ].join(' | ')
    messages.value.unshift({
      type: data.run_type || 'Info',
      is_success: data.is_finished || false,
      message: messageDetails,
      time: new Date().toLocaleTimeString()
    })
    emit('parseDiscretizeGrid', {
      name: current_manager_name,
      grid_data: discretizeGrid_data.grid_data,
      distance_to_obstacle: discretizeGrid_data.distance_to_obstacle
    })
  }
}

// ------ Search Preprocess -------
const request_search_preprocess = (manager_name: string, solution_name: string) => {
  const task_manager = rootObj!.get_task_manager(manager_name)
  if (!task_manager) {
    ElMessage.error('Task not found')
    emit('unlockLeftSideBarTree')
    return
  }

  task_manager.main_tasks.forEach((main_task: MainTask) => {
    if (main_task.name === solution_name) {
      main_task.clear_has_preprocess()
    }
  })

  const main_task = task_manager.main_tasks.get(solution_name)
  if (!main_task) {
    ElMessage.error('Main task not found')
    emit('unlockLeftSideBarTree')
    return
  }

  current_manager_name = manager_name;
  current_main_task_name = solution_name;

  // ------ assemble data ------
  const data: any = {}
  task_manager.group_cfg.forEach((group_cfg: any) => {
    data[group_cfg.name] = {}
    group_cfg.pipes.forEach((pipe: string) => {
      const pipe_obj = task_manager.get_pipe(pipe)
      data[group_cfg.name][pipe] = {
        x: pipe_obj!.discrete_x,
        y: pipe_obj!.discrete_y,
        z: pipe_obj!.discrete_z,
        vector_x: pipe_obj!.vector_x,
        vector_y: pipe_obj!.vector_y,
        vector_z: pipe_obj!.vector_z,
        radius: pipe_obj!.discrete_radius,
      }
    })
  })

  send_msg({ run_type: "searchPreprocess", data: data })
}

const response_search_preprocess = (data: any) => {
  messages.value.unshift({
    type: data.run_type || 'Info',
    is_success: data.is_finished || false,
    message: data.msg,
    time: new Date().toLocaleTimeString()
  })
  if (!data.is_finished) {
    ElMessage.success(data.msg)
  }
  const task_manager = rootObj!.get_task_manager(current_manager_name)
  const main_task = task_manager!.main_tasks.get(current_main_task_name)
  main_task!.set_has_preprocess();
  emit('parseSearchPreprocess', {})
}

// ------ Run Priority Search -------
const request_run_priority_search = (manager_name: string, solution_name: string, priority_task_name: string) => {
  const task_manager = rootObj!.get_task_manager(manager_name)
  if (!task_manager!.has_discretized_grid()) {
    ElMessage.error(`Environment block ${task_manager!.name} has no discretize mesh`)
    emit('unlockLeftSideBarTree')
    return
  }

  const main_task = task_manager!.main_tasks.get(solution_name)
  if (!main_task!.has_preprocess) {
    ElMessage.error(`Main task ${solution_name} hasn't preprocess`)
    emit('unlockLeftSideBarTree')
    return
  }
  const priority_task = main_task!.priority_tasks.find(priority_task => priority_task.name === priority_task_name)

  current_manager_name = manager_name;
  current_main_task_name = solution_name;
  current_priority_task_name = priority_task_name;

  // ------ assemble task set ------
  const data: any = {task_set:{}, group_pipes:{}}

  priority_task!.group_tasks.forEach(group_task => {
    data.task_set[group_task.name] = []
    group_task.cell_tasks.forEach(cell_task => {
      data.task_set[group_task.name].push({
        task_name: cell_task.name,
        start_name: cell_task.start_point,
        end_name: cell_task.end_point,
        radius: cell_task.discretize_radius,
      })
    })
  })

  task_manager!.group_cfg.forEach((group_cfg: any) => {
    data.group_pipes[group_cfg.name] = group_cfg.pipes
  })

  send_msg({ run_type: "runPrioritySearch", data: data })
}

const response_run_priority_search = (data: any) => {
  messages.value.unshift({
    type: data.run_type || 'Info',
    is_success: data.is_finished || false,
    message: data.msg,
    time: new Date().toLocaleTimeString()
  })
  if (data.total_finished) {
    if (!data.data.is_success) {
      ElMessage.error(data.msg)
    }
    emit('parseRunPrioritySearch', {
      name: current_manager_name,
      main_task_name: current_main_task_name,
      priority_task_name: current_priority_task_name,
      is_success: data.data.is_success,
      path_set_group: data.data.path_set_group
    })
  } else {
    if (data.progress_data){
      emit('updateRunPrioritySearch', {
        name: current_manager_name,
        main_task_name: current_main_task_name,
        priority_task_name: current_priority_task_name,
        is_success: data.progress_data.is_success,
        path_set_group: data.progress_data.path_set_group
      })
    }
  }
}

// ------ Task Request -------

defineExpose({
  request_discretize_grid,
  request_search_preprocess,
  request_run_priority_search
})

</script>

<style scoped>
.search-dialog {
  position: fixed;
  width: 600px;
  background: white;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.15);
  z-index: 2000;
}

.search-dialog.minimized {
  width: 200px;
  height: auto;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  cursor: move;
}

.dialog-header span {
  font-weight: 500;
}

.dialog-controls {
  display: flex;
  gap: 8px;
}

.dialog-controls .el-button {
  padding: 4px;
}

.dialog-controls .el-button:hover {
  color: #409eff;
}

.dialog-content {
  padding: 16px;
  height: 400px;
  display: flex;
  flex-direction: column;
}

.log-window {
  flex: 1;
  overflow-y: auto;
  background: #1e1e1e;
  color: #fff;
  font-family: monospace;
  padding: 8px;
  border-radius: 4px;
}

.log-entry {
  padding: 4px 8px;
  border-bottom: 1px solid #333;
  white-space: pre-wrap;
  word-break: break-all;
}

.log-entry.success {
  color: #4caf50;
}

.log-time {
  color: #888;
  margin-right: 8px;
}

.log-type {
  color: #2196f3;
  margin-right: 8px;
}

.log-message {
  color: inherit;
}

.minimized .dialog-content {
  display: none;
}
</style> 