<!-- ThreeViewer.vue -->
<template>
  <div ref="container" class="viewer-container"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, inject } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RootObj } from '../classes/store_obj.ts'

const container = ref<HTMLElement | null>(null)
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls
let raycaster: THREE.Raycaster
let mouse: THREE.Vector2
let selectedMesh: THREE.Mesh | null = null

const emit = defineEmits(['select-object'])
const rootObj = inject<RootObj>('storeObj', new RootObj(new THREE.Scene()))

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
      // Reset previous selection
      if (selectedMesh) {
        if (selectedMesh.material instanceof THREE.MeshStandardMaterial) {
          selectedMesh.material.color.setHex(0xffffff)
        }
        else if (selectedMesh.material instanceof THREE.MeshPhysicalMaterial) {
          selectedMesh.material.color.setHex(0xffffff)
        } 
        else if (selectedMesh.material instanceof THREE.MeshLambertMaterial) {
          selectedMesh.material.color.setHex(0xffffff)
        }
        else if (selectedMesh.material instanceof THREE.MeshBasicMaterial) {
          selectedMesh.material.color.setHex(0xffffff)
        }
        else if (selectedMesh.material instanceof THREE.MeshPhongMaterial) {
          selectedMesh.material.color.setHex(0xffffff)
        }
        else if (selectedMesh.material instanceof THREE.MeshToonMaterial) {
          selectedMesh.material.color.setHex(0xffffff)
        }
        else {
          throw new Error('Unknown material type: ' + selectedMesh.material)
        }
      }

      // Highlight new selection
      selectedMesh = mesh.object as THREE.Mesh
      if (selectedMesh.material instanceof THREE.MeshStandardMaterial) {
        selectedMesh.material.color.setHex(0xfff000)
      }
      else if (selectedMesh.material instanceof THREE.MeshPhysicalMaterial) {
        selectedMesh.material.color.setHex(0xfff000)
      } 
      else if (selectedMesh.material instanceof THREE.MeshLambertMaterial) {
        selectedMesh.material.color.setHex(0xfff000)
      }
      else if (selectedMesh.material instanceof THREE.MeshBasicMaterial) {
        selectedMesh.material.color.setHex(0xfff000)
      }
      else if (selectedMesh.material instanceof THREE.MeshPhongMaterial) {
        selectedMesh.material.color.setHex(0xfff000)
      }
      else if (selectedMesh.material instanceof THREE.MeshToonMaterial) {
        selectedMesh.material.color.setHex(0xfff000)
      }
      else {
        throw new Error('Unknown material type: ' + selectedMesh.material)
      }

      // Emit selection event with the mesh name
      emit('select-object', selectedMesh.name)
    }
  }
}

const onMouseRightClick = (event: MouseEvent) => {
  console.log('Right click detected')
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
    renderer.domElement.removeEventListener('contextmenu', onMouseRightClick)
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