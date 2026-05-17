import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // Point Vite's public directory to our project's data/ folder.
  // This means the React app can fetch '/latest_intelligence.json'
  // directly from the Python pipeline's output — no copying needed.
  // When the pipeline regenerates the JSON, the UI sees fresh data
  // on the next page reload automatically.
  publicDir: '../data',
  server: {
    port: 5173,
    strictPort: true,
  },
})
