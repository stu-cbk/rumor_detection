import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

// https://vitejs.dev/config/
export default defineConfig({
  server:{
    open:true,
    port:5173,
    host:'127.0.0.1',
    proxy: {
      '/api/v1': {
        target: 'http://127.0.0.1:5005',	//实际请求地址
        changeOrigin: true,
        //去掉接口中的 '/api'以便和后端接口匹配
        rewrite: (path) => path.replace(/^\/api\/v1/, ""),
      },
      '/api/v2': {
        target: 'http://127.0.0.1:8080',	//实际请求地址
        changeOrigin: true,
        //去掉接口中的 '/api'以便和后端接口匹配
        rewrite: (path) => path.replace(/^\/api\/v2/, ""),
      },
    }
  },
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
