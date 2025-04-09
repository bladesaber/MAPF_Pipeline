import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import SurfacePostprocessing from '../views/SurfacePostprocessing.vue'
import SimulateOptimize from '../views/SimulateOptimize.vue'
import PathSearch from '../views/PathSearch.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/surface-postprocessing',
      name: 'surface-postprocessing',
      component: SurfacePostprocessing,
    },
    {
      path: '/simulate-optimize',
      name: 'simulate-optimize',
      component: SimulateOptimize,
    },
    {
      path: '/path-search',
      name: 'path-search',
      component: PathSearch,
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('../views/AboutView.vue'),
    },
  ],
})

export default router
