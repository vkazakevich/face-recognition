import path from 'node:path'

import { defineConfig } from 'vite'

import { viteStaticCopy } from 'vite-plugin-static-copy'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/@mediapipe/tasks-vision/wasm',
          dest: 'mediapipe-wasm'
        },
        {
          src: 'models/face_landmarker.task',
          dest: 'models'
        }
      ]
    })
  ],
  base: '/face-recognition/'
})
