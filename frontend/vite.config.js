import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend port: default to 7860 (HF Spaces), but can override with VITE_API_PORT env var
const apiPort = process.env.VITE_API_PORT || 7860

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
