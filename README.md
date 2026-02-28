# AI Smart Focus & Dynamic Subject Tracking System

### **[>>> LIVE DEMO — Click Here to Try <<<](https://lakshmanan272.github.io/Pragyan_khel/)**

---

> Real-time AI-powered object detection, multi-object tracking, and intelligent selective focus — runs entirely in the browser, no server needed.

---

## How to Use

1. **Open the [Live Demo](https://lakshmanan272.github.io/Pragyan_khel/)**
2. Wait for the YOLOv11 model to load (~3-5 seconds)
3. **Upload a video** or click **Webcam**
4. **Click on any detected object** to lock focus — background blurs instantly
5. Click a different object to switch focus
6. Adjust the **Blur slider** to control intensity
7. Click **Clear Focus** to remove focus lock

---

## Tech Stack & Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Detection** | YOLOv11-nano (ONNX Runtime Web) | Real-time object detection at 640x640 |
| **Tracking** | ByteTrack | Persistent multi-object tracking with velocity prediction |
| **Rendering** | WebGL Fragment Shader | Gaussian blur compositing at 60 FPS |
| **Inference** | ONNX Runtime WASM | On-device neural network execution |
| **UI** | Vanilla HTML/CSS/JS | Premium glassmorphism dark theme |

---

## AI Pipeline

```
Video Frame → YOLOv11-nano Detection (every 3rd frame)
                    ↓
            ByteTrack Association
         (3-round matching: high-conf → low-conf → lost tracks)
                    ↓
            Persistent Track IDs + Velocity Prediction (EMA α=0.4)
                    ↓
         Click-to-Focus → Elliptical Mask Generation
                    ↓
         WebGL Gaussian Blur Shader (13-tap kernel)
                    ↓
         2D Canvas Overlay (bounding boxes + labels)
                    ↓
                  Display
```

---

## Key Features

- **Click-to-Focus** — Select any detected object to keep it sharp while blurring the background
- **Real-time Tracking** — ByteTrack maintains persistent IDs across frames even through occlusion
- **On-Device AI** — All inference runs in the browser via WASM, no cloud API needed
- **WebGL Rendering** — Hardware-accelerated Gaussian blur at 60 FPS
- **Multiple Input Sources** — Upload MP4/WebM video files or use live webcam
- **Adaptive Detection** — Handles fast motion, multiple subjects, and varying lighting
- **80 Object Classes** — Detects people, vehicles, animals, electronics, furniture, and more (COCO dataset)

---

## Project Structure

```
├── index.html              # Complete standalone app (UI + AI engine)
├── models/
│   └── yolov11n.onnx       # YOLOv11-nano model (10.2 MB)
├── ARCHITECTURE.md          # Detailed technical documentation
└── README.md                # This file
```

---

## Run Locally

```bash
git clone https://github.com/lakshmanan272/Pragyan_khel.git
cd Pragyan_khel
npx http-server -p 8080 --cors
# Open http://127.0.0.1:8080
```

---

## Technical Highlights

- **Letterbox preprocessing** — Maintains aspect ratio with gray padding for accurate detection
- **Non-Maximum Suppression (NMS)** — Eliminates duplicate detections per class (IoU threshold: 0.45)
- **3-round ByteTrack association** — High-confidence → Low-confidence → Lost track recovery
- **Greedy IoU matching** — Fast approximation of Hungarian algorithm for real-time performance
- **Elliptical gradient mask** — Smooth focus-to-blur transition around selected subject
- **WebGL GLSL shader** — 13×13 Gaussian kernel with adaptive step size for variable blur radius

---

**Pragyan Hackathon — NIT Trichy**
