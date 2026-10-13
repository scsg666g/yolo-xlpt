import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    // The chart bundle is route-loaded and stays out of the dashboard entry.
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('/echarts/') || id.includes('/zrender/') || id.includes('/echarts-for-react/')) return 'charts'
          if (id.includes('/@xyflow/') || id.includes('/d3-')) return 'canvas'
          return undefined
        },
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:18088',
    },
  },
})
