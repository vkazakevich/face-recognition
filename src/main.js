import './style.css'

import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from '@mediapipe/tasks-vision'

// Elements
const video = document.getElementById('video')
const canvasElement = document.getElementById('canvas')
const canvasCtx = canvasElement.getContext('2d')

const drawingUtils = new DrawingUtils(canvasCtx)

// Constants
const RECOGNITION_THRESHOLD = 0.2
const videoWidth = 640

// Init variables
let faceLandmarker

let knownFacesDB = []
let vectorBuffer = []

let isEnrolling = false
let lastVideoTime = -1

let showPoints = false
let showConnectors = false
let showNumbers = false

// Init mediapipe lib
async function initFaceLandmark() {
  const vision = await FilesetResolver.forVisionTasks(
    // 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    '/mediapipe-wasm/wasm'
  )

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      // modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      modelAssetPath: `../models/face_landmarker.task`,
      delegate: 'GPU'
    },
    outputFaceBlendshapes: true,
    numFaces: 1,
    runningMode: 'VIDEO'
  })
}

async function initWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false
  })

  video.srcObject = stream

  video.addEventListener('loadeddata', () => {
    const radio = video.videoHeight / video.videoWidth
    video.style.width = videoWidth + 'px'
    video.style.height = videoWidth * radio + 'px'
    canvasElement.style.width = videoWidth + 'px'
    canvasElement.style.height = videoWidth * radio + 'px'
    canvasElement.width = video.videoWidth
    canvasElement.height = video.videoHeight

    predictWebcam()
  })
}

// Utils
function euclideanDistance(a, b) {
  return Math.hypot(...Object.keys(a).map((k) => b[k] - a[k]))
}

// Landmarks utils
function calculateAverageVector(vectorBuffer) {
  if (!vectorBuffer || vectorBuffer.length === 0) return null

  const vectorSize = vectorBuffer[0].length
  const sumVector = new Array(vectorSize).fill(0)

  for (const vector of vectorBuffer) {
    for (let i = 0; i < vectorSize; i++) {
      sumVector[i] += vector[i]
    }
  }

  return sumVector.map((sum) => sum / vectorBuffer.length)
}

const createFeatureVector = (landmarks) => {
  // Eye Distance
  const scaleIndex = euclideanDistance(landmarks[468], landmarks[473])

  const features = []

  features.push(euclideanDistance(landmarks[33], landmarks[263])) // Between eyes
  features.push(euclideanDistance(landmarks[133], landmarks[362])) // Inner eye corners
  features.push(euclideanDistance(landmarks[362], landmarks[263])) // Right eye width
  features.push(euclideanDistance(landmarks[159], landmarks[145])) // Left eye height
  features.push(euclideanDistance(landmarks[386], landmarks[374])) // Right eye height
  features.push(euclideanDistance(landmarks[135], landmarks[364])) // Nose width
  features.push(euclideanDistance(landmarks[1], landmarks[168])) // Nose length
  features.push(euclideanDistance(landmarks[10], landmarks[1])) // Nose bridge
  features.push(euclideanDistance(landmarks[61], landmarks[291])) // Mouth width
  features.push(euclideanDistance(landmarks[0], landmarks[17])) // Mouth height
  features.push(euclideanDistance(landmarks[55], landmarks[285])) // Between eyebrows
  features.push(euclideanDistance(landmarks[105], landmarks[334])) // Brow width
  features.push(euclideanDistance(landmarks[107], landmarks[336])) // Outer brow distance
  features.push(euclideanDistance(landmarks[172], landmarks[397])) // Face width
  features.push(euclideanDistance(landmarks[149], landmarks[378])) // Lower jaw width
  features.push(euclideanDistance(landmarks[152], landmarks[17])) // Chin to mouth

  return features.map((feature) => feature / scaleIndex)
}

function recognizeFace(landmarks) {
  landmarks.map(({ x, y, z }) => [x, y, z])

  const currentVector = createFeatureVector(landmarks)
  if (!currentVector) return

  if (isEnrolling) {
    vectorBuffer.push(currentVector)
    return `Enrolling...`
  }

  let bestMatch = {
    name: 'Unknown',
    distance: Infinity
  }

  for (const knownFace of knownFacesDB) {
    const distance = euclideanDistance(currentVector, knownFace.vector)

    if (distance < bestMatch.distance) {
      bestMatch = { name: knownFace.name, distance: distance }
    }
  }

  if (bestMatch.distance < RECOGNITION_THRESHOLD) {
    return bestMatch.name
  }

  return 'Unknown'
}

function getBoundingBox(landmarks) {
  if (!landmarks || landmarks.length === 0) {
    return null
  }

  let minX = landmarks[0].x
  let minY = landmarks[0].y
  let maxX = landmarks[0].x
  let maxY = landmarks[0].y

  for (const landmark of landmarks) {
    minX = Math.min(minX, landmark.x)
    minY = Math.min(minY, landmark.y)
    maxX = Math.max(maxX, landmark.x)
    maxY = Math.max(maxY, landmark.y)
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  }
}

async function predictWebcam() {
  let results = null

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime
    results = faceLandmarker.detectForVideo(video, performance.now())
  }

  if (results && results.faceLandmarks) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)

    if (showPoints) {
      for (const landmarks of results.faceLandmarks) {
        drawingUtils.drawLandmarks(landmarks, {
          color: 'lime',
          lineWidth: 0,
          radius: 1
        })
      }
    }

    if (showNumbers) {
      const canvasCtx = canvasElement.getContext('2d')
      canvasCtx.fillStyle = 'lime'
      canvasCtx.font = '8px Arial'

      for (const landmarks of results.faceLandmarks) {
        landmarks.forEach((point, index) => {
          const x = point.x * canvasElement.width
          const y = point.y * canvasElement.height
          canvasCtx.fillText(index, x, y)
        })
      }
    }

    if (showConnectors) {
      for (const landmarks of results.faceLandmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          { color: '#C0C0C070', lineWidth: 1 }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
          { color: '#FF3030' }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
          { color: '#FF3030' }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
          { color: '#30FF30' }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
          { color: '#30FF30' }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
          { color: '#E0E0E0' }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
          { color: '#E0E0E0' }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
          { color: '#FF3030' }
        )
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
          { color: '#30FF30' }
        )
      }
    }

    for (const landmarks of results.faceLandmarks) {
      canvasCtx.fillStyle = 'red'
      canvasCtx.font = '16px Arial'
      canvasCtx.lineWidth = 1
      canvasCtx.strokeStyle = 'red'

      const boundingBox = getBoundingBox(landmarks)

      if (boundingBox) {
        canvasCtx.strokeRect(
          boundingBox.x * canvasElement.width,
          boundingBox.y * canvasElement.height,
          boundingBox.width * canvasElement.width,
          boundingBox.height * canvasElement.height
        )
      }

      const username = recognizeFace(landmarks)

      const point = landmarks[10]

      const x = boundingBox.x * canvasElement.width
      const y = point.y * canvasElement.height

      canvasCtx.fillText(username, x, y - 5)
    }
  }

  window.requestAnimationFrame(predictWebcam)
}

function refreshUserList() {
  const ul = document.getElementById('userList')

  ul.innerHTML = ''

  knownFacesDB.forEach((face, index) => {
    const li = document.createElement('li')
    li.className = 'bg-gray-100 rounded-md p-2 mb-2'
    li.textContent = face.name
    ul.appendChild(li)
  })
}

// Init App
document.addEventListener('DOMContentLoaded', async () => {
  const savedFaces = localStorage.getItem('knownFacesDB')

  if (savedFaces) {
    knownFacesDB = JSON.parse(savedFaces)
  }

  refreshUserList()

  await initFaceLandmark()
  await initWebcam()
})

const buttonElements = document.querySelectorAll('.control_button_js')
buttonElements.forEach((el) => {
  el.addEventListener('click', (e) => {
    e.preventDefault()

    e.target.classList.toggle('bg-purple-200')

    const { control } = e.target.dataset

    switch (control) {
      case 'points':
        showPoints = !showPoints
        break

      case 'connectors':
        showConnectors = !showConnectors
        break

      case 'numbers':
        showNumbers = !showNumbers
        break
    }
  })
})

// Registration
const enrollingForm = document.getElementById('enrollingForm')
enrollingForm.addEventListener('submit', (e) => {
  e.preventDefault()

  const formData = new FormData(e.target)
  const { username } = Object.fromEntries(formData)

  const name = username.trim()
  if (name == '') return false

  isEnrolling = true
  vectorBuffer = []

  setTimeout(() => {
    isEnrolling = false

    document.getElementById('usernameField').value = ''

    const vector = calculateAverageVector(vectorBuffer)

    knownFacesDB = [...knownFacesDB, { name, vector }]
    localStorage.setItem('knownFacesDB', JSON.stringify(knownFacesDB))

    refreshUserList()
  }, 3000)
})

// Clean database
const clearDbButton = document.getElementById('clearDB')

clearDbButton.addEventListener('click', (e) => {
  e.preventDefault()
  localStorage.removeItem('knownFacesDB')
  knownFacesDB = []
  refreshUserList()
})
