import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  server: {
    host: "0.0.0.0",
    port: "3000",
    // allowedHosts: ["xrgqoemoedit.sealoshzh.site"],
    // proxy: {
    //   // 将所有请求重定向到 VuePress 服务
    //   '/docs': {
    //     target: 'http://localhost:3001',
    //     changeOrigin: true,
    //     rewrite: (path) => path.replace(/^\/docs/, '')
    //   }
    // }
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
