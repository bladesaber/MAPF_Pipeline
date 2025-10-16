<!-- PostViewer.vue -->
<template>
  <div ref="container" class="viewer-container"></div>
</template>

<script setup lang="ts">
import {
  computed,
  inject,
  onBeforeUnmount,
  onMounted,
  ref,
} from 'vue';

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { CenterRoot } from '../classes/center_root.ts';

// 不再修改材质颜色，采用外围wireframe高亮

const container = ref<HTMLElement | null>(null)
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls
let raycaster: THREE.Raycaster
let mouse: THREE.Vector2
let clickEventX: number = 0
let clickEventY: number = 0

const emit = defineEmits([
  'leftClick-select-object', 'rightClick-select-object', 'rightClick-control-point', 'check-highlight-status'
])
const rootObj = inject<CenterRoot>('centerRoot', new CenterRoot(new THREE.Scene()))

const init = () => {
  if (!container.value || !rootObj) return

  // Scene setup
  rootObj.get_scene().background = new THREE.Color(0x333333)

  // Camera setup
  camera = new THREE.PerspectiveCamera(
    50,
    container.value.clientWidth / container.value.clientHeight,
    1,
    1000
  )
  camera.position.set(50, 50, 50)

  // Renderer setup
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(container.value.clientWidth, container.value.clientHeight)
  renderer.setPixelRatio(window.devicePixelRatio)
  container.value.appendChild(renderer.domElement)

  // Raycaster setup
  raycaster = new THREE.Raycaster()
  mouse = new THREE.Vector2()

  // Controls
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: undefined,
    RIGHT: THREE.MOUSE.PAN
  }
  // controls.enableZoom = true
  // controls.enablePan = true
  // controls.enableRotate = true
  // controls.zoomSpeed = 1.0

  // Add click event listener
  renderer.domElement.addEventListener('click', onMouseLeftClick)
  renderer.domElement.addEventListener('contextmenu', onMouseRightClick)
  
  // Lighting
  const ambientLight = new THREE.AmbientLight(0x404040, 1.5)
  rootObj.get_scene().add(ambientLight)

  const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
  directionalLight.position.set(1, 1, 1).normalize()
  rootObj.get_scene().add(directionalLight)

  // Grid helper
  const gridHelper = new THREE.GridHelper(100, 10)
  rootObj.get_scene().add(gridHelper)

  // Axes helper
  const axesHelper = new THREE.AxesHelper(50)
  rootObj.get_scene().add(axesHelper)

  // Handle window resize
  window.addEventListener('resize', onWindowResize)
  // Initial resize to ensure correct size
  onWindowResize()
}

// ---------------- handle selected mesh ----------------
let selectedMesh: THREE.Mesh | null = null
// 不再修改材质颜色，采用外围wireframe高亮

const handleMeshSelection = (mesh: THREE.Mesh, eventType: 'left' | 'right') => {
  // 仅记录被选中的mesh，不改变其材质颜色
  selectedMesh = mesh

  // Emit appropriate event
  if (eventType === 'left') {
    emit('leftClick-select-object', selectedMesh.name)
  } else {
    // 判断是否为控制点
    if (selectedMesh.userData && selectedMesh.userData.type === 'ControlPoint') {
      emit('rightClick-control-point', selectedMesh.userData, clickEventX, clickEventY)
    } else {
      emit('rightClick-select-object', selectedMesh.name, clickEventX, clickEventY)
    }
  }
}

const onMouseLeftClick = (event: MouseEvent) => {
  if (!container.value) return

  // Calculate mouse position in normalized device coordinates (-1 to +1)
  const rect = container.value.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / container.value.clientWidth) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / container.value.clientHeight) * 2 + 1

  // Update the picking ray with the camera and mouse position
  raycaster.setFromCamera(mouse, camera)

  // Calculate objects intersecting the picking ray
  const intersects = raycaster.intersectObjects(rootObj.get_scene().children, true)

  if (intersects.length > 0) {
    // Skip the first intersection (outer box) and find the first valid mesh
    const mesh = intersects.find(obj => 
      obj.object instanceof THREE.Mesh && 
      !(obj.object.parent instanceof THREE.GridHelper) &&
      !(obj.object.parent instanceof THREE.AxesHelper) &&
      !obj.object.name.includes('_discretize') &&  // Skip discretize meshes
      // Allow selection of grid edges but not the main grid mesh
      (obj.object.name.includes('_edges') || !obj.object.name.includes('grid')) &&
      (obj.object.visible === true)
    )

    if (mesh) {
      handleMeshSelection(mesh.object as THREE.Mesh, 'left')
    } else {
      // 点击其它对象但不是有效mesh，通知清除
      emit('leftClick-select-object', '')
    }
  } else {
    // 点击空白处，通知清除
    emit('leftClick-select-object', '')
  }
}

let rightClickStartTime = 0
const onMouseRightClick = (event: MouseEvent) => {
  rightClickStartTime = Date.now()
  if (!container.value) return

  // Calculate mouse position in normalized device coordinates (-1 to +1)
  const rect = container.value.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / container.value.clientWidth) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / container.value.clientHeight) * 2 + 1

  // Update the picking ray with the camera and mouse position
  raycaster.setFromCamera(mouse, camera)

  // Calculate objects intersecting the picking ray
  const intersects = raycaster.intersectObjects(rootObj.get_scene().children, true)

  if (intersects.length > 0) {
    // 优先找控制点
    const mesh = intersects.find(obj =>
      obj.object.userData && obj.object.userData.type === 'ControlPoint'
    );
    if (mesh) {
      clickEventX = event.clientX;
      clickEventY = event.clientY;
      handleMeshSelection(mesh.object as THREE.Mesh, 'right');
      document.addEventListener('mouseup', onRightMouseUp);
      return;
    }
    // 否则按原有逻辑找普通mesh
    // Skip the first intersection (outer box) and find the first valid mesh
    const fallback = intersects.slice(1).find(obj => 
      obj.object instanceof THREE.Mesh && 
      !(obj.object.parent instanceof THREE.GridHelper) &&
      !(obj.object.parent instanceof THREE.AxesHelper) &&
      !obj.object.name.includes('_discretize') &&  // Skip discretize meshes
      // Allow selection of grid edges but not the main grid mesh
      (obj.object.name.includes('_edges') || !obj.object.name.includes('grid'))
    )
    if (fallback) {
      clickEventX = event.clientX
      clickEventY = event.clientY
      handleMeshSelection(fallback.object as THREE.Mesh, 'right')
      document.addEventListener('mouseup', onRightMouseUp)
    }
  }
}

const onRightMouseUp = () => {
  const rightClickDuration = Date.now() - rightClickStartTime
  if (rightClickDuration < 300) {
    // Right click handling is now done in handleMeshSelection
  }
  document.removeEventListener('mouseup', onRightMouseUp)
}

const onContextMenu = (event: MouseEvent) => {
  event.preventDefault()
}

const animate = () => {
  requestAnimationFrame(animate)
  controls.update()
  // 触发高亮mesh状态检查
  emit('check-highlight-status')
  renderer.render(rootObj.get_scene(), camera)
}

const onWindowResize = () => {
  if (!container.value) return
  
  // Get the actual container size
  const width = container.value.clientWidth
  const height = container.value.clientHeight

  // Update camera
  camera.aspect = width / height
  camera.updateProjectionMatrix()

  // Update renderer
  renderer.setSize(width, height)
}

onMounted(() => {
  init()
  animate()
  renderer.domElement.addEventListener('mousedown', onMouseRightClick)
  renderer.domElement.addEventListener('mouseup', onRightMouseUp)
  renderer.domElement.addEventListener('contextmenu', onContextMenu)

  // Add ResizeObserver for container size changes
  const resizeObserver = new ResizeObserver(() => {
    onWindowResize()
  })
  
  if (container.value) {
    resizeObserver.observe(container.value)
  }

  onBeforeUnmount(() => {
    resizeObserver.disconnect()
    window.removeEventListener('resize', onWindowResize)
    renderer.domElement.removeEventListener('click', onMouseLeftClick)
    renderer.domElement.removeEventListener('mousedown', onMouseRightClick)
    renderer.domElement.removeEventListener('mouseup', onRightMouseUp)
    renderer.domElement.removeEventListener('contextmenu', onContextMenu)
    renderer.dispose()
  })
})
</script>

<style scoped>
.viewer-container {
  width: 100%;
  height: 100%;
  min-height: 0;
  position: relative;
  display: flex;
}

:deep(canvas) {
  width: 100% !important;
  height: 100% !important;
}
</style> 