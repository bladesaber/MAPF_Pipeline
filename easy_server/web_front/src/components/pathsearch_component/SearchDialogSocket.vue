<template>
  <div 
    v-show="isVisible"
    class="search-dialog" 
    :class="{ minimized: isMinimized }"
    :style="dialogStyle"
    @mousedown="startDrag"
  >
    <div class="dialog-header">
      <span>Path Search</span>
      <div class="dialog-controls">
        <el-button 
          type="text" 
          @click="toggleMinimize"
          :icon="isMinimized ? 'el-icon-full-screen' : 'el-icon-minus'"
        />
        <el-button 
          type="text" 
          @click="closeDialog"
          icon="el-icon-close"
        />
      </div>
    </div>

    <div class="dialog-content" v-show="!isMinimized">
      <el-table
        :data="layoutData"
        style="width: 100%"
        height="300"
        border
      >
        <el-table-column
          prop="name"
          label="Layout Name"
          width="180"
        />
        <el-table-column
          prop="status"
          label="Status"
          width="100"
        />
        <el-table-column
          prop="progress"
          label="Progress"
        >
          <template #default="scope">
            <el-progress 
              :percentage="scope.row.progress"
              :status="scope.row.progress === 100 ? 'success' : ''"
            />
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SearchDialogSocket',
  data() {
    return {
      isVisible: true,
      isMinimized: false,
      position: {
        x: window.innerWidth - 600,
        y: 100
      },
      lastPosition: null,
      isDragging: false,
      dragOffset: { x: 0, y: 0 },
      layoutData: [],
      socket: null
    }
  },
  computed: {
    dialogStyle() {
      if (this.isMinimized) {
        return {
          left: '20px',
          bottom: '20px',
          top: 'auto',
          transform: 'none'
        }
      }
      return {
        left: `${this.position.x}px`,
        top: `${this.position.y}px`
      }
    }
  },
  mounted() {
    this.connectWebSocket()
    window.addEventListener('mousemove', this.onDrag)
    window.addEventListener('mouseup', this.stopDrag)
  },
  beforeDestroy() {
    this.disconnectWebSocket()
    window.removeEventListener('mousemove', this.onDrag)
    window.removeEventListener('mouseup', this.stopDrag)
  },
  methods: {
    connectWebSocket() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/path_search/run`
      
      this.socket = new WebSocket(wsUrl)
      
      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.handleWebSocketMessage(data)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
      
      this.socket.onclose = () => {
        console.log('WebSocket connection closed')
      }
      
      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    },
    
    disconnectWebSocket() {
      if (this.socket) {
        this.socket.close()
        this.socket = null
      }
    },
    
    handleWebSocketMessage(data) {
      // Update layout data based on received message
      const existingLayout = this.layoutData.find(l => l.name === data.name)
      if (existingLayout) {
        Object.assign(existingLayout, data)
      } else {
        this.layoutData.push({
          name: data.name,
          status: data.status,
          progress: data.progress || 0
        })
      }
    },
    
    startDrag(event) {
      if (event.target.closest('.dialog-controls')) return
      
      this.isDragging = true
      this.dragOffset = {
        x: event.clientX - this.position.x,
        y: event.clientY - this.position.y
      }
    },
    
    onDrag(event) {
      if (!this.isDragging || this.isMinimized) return
      
      this.position = {
        x: event.clientX - this.dragOffset.x,
        y: event.clientY - this.dragOffset.y
      }
    },
    
    stopDrag() {
      this.isDragging = false
    },
    
    toggleMinimize() {
      if (!this.isMinimized) {
        this.lastPosition = { ...this.position }
      } else if (this.lastPosition) {
        this.position = { ...this.lastPosition }
      }
      this.isMinimized = !this.isMinimized
    },
    
    closeDialog() {
      this.isVisible = false
      this.disconnectWebSocket()
      this.$emit('close')
    }
  }
}
</script>

<style scoped>
.search-dialog {
  position: fixed;
  width: 600px;
  background: white;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
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
  padding: 10px 15px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  cursor: move;
}

.dialog-controls {
  display: flex;
  gap: 5px;
}

.dialog-content {
  padding: 15px;
}

.minimized .dialog-content {
  display: none;
}
</style> 