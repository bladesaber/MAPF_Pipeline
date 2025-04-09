<!-- PathSearchViewer.vue -->
<template>
  <div ref="container" class="viewer-container"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, inject, computed } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { BlockRoot } from '../../classes/pathSearch_classes/base_root.ts'
import { setMeshColor } from '../../utils/visulize_utils.ts'

const container = ref<HTMLElement | null>(null)
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls
let raycaster: THREE.Raycaster
let mouse: THREE.Vector2
let clickEventX: number = 0
let clickEventY: number = 0

const emit = defineEmits([
  'leftClick-select-object', 'rightClick-select-object'
])
const rootObj = inject<BlockRoot>('blockRoot', new BlockRoot(new THREE.Scene()))

const init = () => {
  if (!container.value || !rootObj) return

  // Scene setup
  rootObj.get_scene().background = new THREE.Color(0xdddddd)

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
let selectedMesh_left: THREE.Mesh | null = null
let selectedMesh_right: THREE.Mesh | null = null

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
    // Find the first mesh that's not a helper
    const mesh = intersects.find(obj => 
      obj.object instanceof THREE.Mesh && 
      !(obj.object.parent instanceof THREE.GridHelper) &&
      !(obj.object.parent instanceof THREE.AxesHelper)
    )

    if (mesh) {
      if (selectedMesh_left) {
        setMeshColor(selectedMesh_left, 0xff00ff)
      }
      if (selectedMesh_right) {
        setMeshColor(selectedMesh_right, 0xff00ff)
        selectedMesh_right = null
      }

      selectedMesh_left = mesh.object as THREE.Mesh
      setMeshColor(selectedMesh_left, 0xffff00)
      emit('leftClick-select-object', selectedMesh_left.name)
    }
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
    const mesh = intersects.find(obj => 
      obj.object instanceof THREE.Mesh && 
      !(obj.object.parent instanceof THREE.GridHelper) &&
      !(obj.object.parent instanceof THREE.AxesHelper)
    )
    if (mesh) {
      selectedMesh_right = mesh.object as THREE.Mesh
      clickEventX = event.clientX
      clickEventY = event.clientY
      document.addEventListener('mouseup', onRightMouseUp)
    }
  }
}

const onRightMouseUp = () => {
  const rightClickDuration = Date.now() - rightClickStartTime
  if (rightClickDuration < 300) {
    if (selectedMesh_right) {
      if (selectedMesh_left) {
        setMeshColor(selectedMesh_left, 0xff00ff)
        selectedMesh_left = null
      }
      setMeshColor(selectedMesh_right, 0xffff00)
      emit('rightClick-select-object', selectedMesh_right.name, clickEventX, clickEventY)
    }
  }
  document.removeEventListener('mouseup', onRightMouseUp)
}

const onContextMenu = (event: MouseEvent) => {
  event.preventDefault()
}

const animate = () => {
  requestAnimationFrame(animate)
  controls.update()
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