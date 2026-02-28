# AI Smart Auto-Focus & Dynamic Subject Tracking System — V2

## Production Architecture Blueprint

---

## System Overview

A **real-time, client-side** AI system that runs entirely in the browser using **WebGPU + WebAssembly**. The user selects any object in a video, and the system maintains sharp focus on it while dynamically blurring the background — all at 60 FPS.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (Client-Side)                       │
│                                                                     │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌────────────┐  │
│  │  Video    │───▶│  YOLOv11n │───▶│ ByteTrack │───▶│  User      │  │
│  │  Input    │    │  (ONNX/   │    │ (Persistent│   │  Tap/Click │  │
│  │  (MP4 /   │    │  WebGPU)  │    │  IDs)     │    │  → Lock ID │  │
│  │  Webcam)  │    └───────────┘    └─────┬─────┘    └──────┬─────┘  │
│  └──────────┘                            │                  │       │
│                                          ▼                  ▼       │
│                                   ┌─────────────┐   ┌────────────┐  │
│                                   │ MobileSAM   │   │  target_id │  │
│                                   │ Encoder +   │◀──│  locked    │  │
│                                   │ Decoder     │   └────────────┘  │
│                                   │ (Sparse     │                   │
│                                   │  Crop)      │                   │
│                                   └──────┬──────┘                   │
│                                          │                          │
│                                          ▼                          │
│                                   ┌─────────────┐                   │
│                                   │ WebGL Blur  │                   │
│                                   │ Fragment    │──▶ Display        │
│                                   │ Shader      │                   │
│                                   └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer              | Technology                        | Purpose                           |
|--------------------|-----------------------------------|-----------------------------------|
| **Frontend**       | Next.js 16 / React 19 / TypeScript | App framework + SSR               |
| **Styling**        | Tailwind CSS 4                    | UI components                     |
| **Inference**      | ONNX Runtime Web (WebGPU/WASM)   | AI model execution on-device      |
| **Detection**      | YOLOv11-nano (ONNX)              | 80-class object detection at 640² |
| **Tracking**       | ByteTrack (pure TypeScript)       | Persistent ID assignment          |
| **Segmentation**   | MobileSAM (Encoder+Decoder ONNX) | Pixel-perfect subject masks       |
| **Rendering**      | WebGL 1 + Fragment Shaders        | Gaussian blur compositing         |
| **Math**           | Hungarian Algorithm + IoU         | Optimal track-detection matching  |

---

## The 60 FPS Pipeline ("Secret Sauce")

### Frame Budget: 16.67ms per frame

```
Frame N:
├── [EVERY FRAME]     Render Loop (WebGL)                    ~2ms
│   ├── Upload video texture to GPU
│   ├── Upload mask texture (cached)
│   └── Execute blur fragment shader → display
│
├── [EVERY 3rd FRAME] Detection (YOLOv11n via ONNX/WebGPU)  ~8ms
│   ├── Capture frame to OffscreenCanvas
│   ├── Preprocess: letterbox to 640×640, normalize [0,1]
│   ├── Run ONNX inference (MobileNetV2 backbone)
│   ├── Parse [1,84,8400] output tensor
│   └── NMS filtering → Detection[]
│
├── [EVERY 3rd FRAME] Tracking (ByteTrack)                   ~0.5ms
│   ├── Split detections → high-conf / low-conf
│   ├── Predict track positions via velocity
│   ├── Hungarian matching (3 rounds)
│   └── Create/update/remove tracks → TrackedObject[]
│
├── [EVERY 6th FRAME] Segmentation (MobileSAM)               ~12ms
│   ├── Sparse crop: expand target bbox 20%, pad to square
│   ├── Resize crop to 1024×1024, ImageNet normalize
│   ├── SAM Encoder → image embeddings (cached)
│   ├── SAM Decoder + center-point prompt → 256×256 mask
│   └── Project mask back to full-frame coordinates
│
└── [ON USER CLICK]   Target Selection                        ~0.1ms
    ├── Map (x,y) to nearest bounding box
    └── Lock ByteTrack ID as target_id
```

---

## Thread Architecture

```
┌─────────────────────────────────┐
│         MAIN THREAD             │
│                                 │
│  ┌─────────────┐               │
│  │ React UI    │               │
│  │ Components  │               │
│  └──────┬──────┘               │
│         │                      │
│  ┌──────▼──────┐               │
│  │ WebGL       │               │
│  │ Render Loop │  60 FPS       │
│  │ (rAF)       │               │
│  └──────┬──────┘               │
│         │                      │
│  ┌──────▼──────┐               │
│  │ Inference   │               │
│  │ Pipeline    │  Throttled    │
│  │ (YOLO +     │  to every     │
│  │  ByteTrack  │  3rd frame    │
│  │  + SAM)     │               │
│  └─────────────┘               │
│                                 │
│  ONNX Runtime Web handles      │
│  GPU scheduling internally     │
│  via WebGPU/WASM threads       │
└─────────────────────────────────┘
```

ONNX Runtime Web with WebGPU execution provider manages GPU compute scheduling internally. The detection runs every 3rd frame, and segmentation every 6th frame, keeping the main thread free for 60 FPS rendering.

---

## Key Technical Details

### 1. Sparse Crop Optimization (The Performance Secret)

Instead of running MobileSAM on the full 1920×1080 frame, we:

```
Full frame (1920×1080) = 2,073,600 pixels
Cropped bbox (200×300) → padded square (360×360) = 129,600 pixels
                                                     ↑
                                             94% fewer pixels!
```

**Math: Crop Projection**
```
Given:
  target_bbox = [bx, by, bw, bh]     // from ByteTrack
  expand = 0.20                        // 20% context padding

Expand bbox:
  ex = bw × expand, ey = bh × expand
  crop = [bx - ex, by - ey, bw + 2ex, bh + 2ey]

Pad to square:
  max_side = max(crop_w, crop_h)
  crop_x = cx - max_side/2
  crop_y = cy - max_side/2

Encode: resize crop to 1024×1024 → SAM encoder

Decode: center prompt at:
  px = ((target_center_x - crop_x) / crop_w) × 1024
  py = ((target_center_y - crop_y) / crop_h) × 1024

Project mask back:
  for each mask pixel (mx, my):
    frame_x = crop_x + mx × (crop_w / mask_w)
    frame_y = crop_y + my × (crop_h / mask_h)
```

### 2. ByteTrack Algorithm

Three-round association with Hungarian matching:

```
Round 1: High-confidence detections (>0.5) vs Active tracks    → IoU matching
Round 2: Low-confidence detections (0.1-0.5) vs Remaining      → Strict IoU
Round 3: Unmatched high-conf detections vs Lost tracks          → Re-identify
```

Velocity estimation with EMA:
```
v_new = α × (center_new - center_old) + (1 - α) × v_old
```

### 3. WebGL Blur Shader

13-tap separable Gaussian with adaptive step size:

```glsl
// For each pixel:
mask = texture2D(u_maskTexture, v_texCoord).r;

if (mask > 0.95) {
    // Subject pixel: output sharp
    gl_FragColor = texture2D(u_videoTexture, v_texCoord);
} else {
    // Background: apply Gaussian blur
    for y in [-6..6]:
        for x in [-6..6]:
            weight = exp(-(x²+y²) / (2σ²))
            color += sample(offset) × weight

    // Smooth transition at mask boundary
    alpha = smoothstep(0.0, edge_width, mask)
    final = mix(blurred, sharp, alpha)
}
```

---

## File Structure

```
smart-focus-v2/
├── public/
│   └── models/                    # ONNX model files (download separately)
│       ├── yolov11n.onnx          # YOLOv11-nano (~6MB)
│       ├── mobile_sam_encoder.onnx # MobileSAM encoder (~5MB)
│       └── mobile_sam_decoder.onnx # MobileSAM decoder (~4MB)
│
├── src/
│   ├── app/
│   │   ├── layout.tsx             # Root layout with metadata
│   │   ├── page.tsx               # Main page — orchestrates everything
│   │   └── globals.css            # Tailwind + custom styles
│   │
│   ├── components/
│   │   ├── VideoCanvas.tsx        # WebGL canvas + overlay + click handler
│   │   ├── ControlPanel.tsx       # Upload/webcam/settings controls
│   │   ├── InfoPanel.tsx          # Status, tracked objects, metrics
│   │   └── LoadingOverlay.tsx     # Model loading progress
│   │
│   ├── engine/
│   │   ├── yolo-detector.ts       # YOLOv11n ONNX wrapper (detection)
│   │   ├── bytetrack.ts           # ByteTrack multi-object tracker
│   │   ├── mobilesam.ts           # MobileSAM encoder+decoder (segmentation)
│   │   └── pipeline.ts            # Orchestrator: YOLO → ByteTrack → SAM
│   │
│   ├── renderer/
│   │   ├── webgl-renderer.ts      # WebGL blur shader + 2D overlay
│   │   └── shaders/
│   │       ├── vertex.glsl        # Fullscreen quad vertex shader
│   │       └── blur-fragment.glsl # Gaussian selective blur shader
│   │
│   ├── hooks/
│   │   ├── useVideoCapture.ts     # Video/webcam lifecycle management
│   │   ├── useInferencePipeline.ts # Pipeline loading + state management
│   │   └── useWebGLRenderer.ts    # WebGL lifecycle + render calls
│   │
│   └── utils/
│       ├── math.ts                # BBox, IoU, Hungarian algorithm, types
│       └── image-processing.ts    # YOLO/SAM preprocessing + mask projection
│
├── next.config.ts                 # WASM + ONNX + COEP/COOP headers
├── tsconfig.json                  # TypeScript config with path aliases
├── postcss.config.mjs             # Tailwind PostCSS config
├── package.json                   # Dependencies
└── ARCHITECTURE.md                # This document
```

---

## Model Downloads

Place these files in `public/models/`:

### YOLOv11-nano (Required)
```bash
# Option 1: Export from ultralytics
pip install ultralytics
yolo export model=yolo11n.pt format=onnx imgsz=640 simplify=True

# Option 2: Download pre-exported
# https://github.com/ultralytics/assets/releases
```

### MobileSAM (Optional — enables pixel-perfect masks)
```bash
# Clone and export
git clone https://github.com/ChaoningZhang/MobileSAM
cd MobileSAM
python scripts/export_onnx_model.py \
  --checkpoint ./weights/mobile_sam.pt \
  --output ./mobile_sam \
  --return-single-mask
```

---

## How to Run

```bash
# Install dependencies
cd smart-focus-v2
npm install

# Place ONNX models
cp yolov11n.onnx public/models/
cp mobile_sam_encoder.onnx public/models/   # optional
cp mobile_sam_decoder.onnx public/models/   # optional

# Start development server
npm run dev

# Open http://localhost:3000
```

### Usage:
1. Wait for YOLO model to load (~3-5 seconds)
2. Click **Upload Video** or **Webcam**
3. Purple bounding boxes appear around detected objects
4. **Click any object** to lock focus — background blurs instantly
5. Click a different object to **switch focus** in real-time
6. Adjust blur strength and detection FPS with sliders

---

## Performance Targets

| Metric                | Target    | Achieved Via                      |
|-----------------------|-----------|-----------------------------------|
| Render FPS            | 60 fps    | WebGL shader (GPU-accelerated)    |
| Detection latency     | <15ms     | YOLOv11n via WebGPU              |
| Tracking latency      | <1ms      | Pure JS ByteTrack + Hungarian     |
| Segmentation latency  | <20ms     | Sparse crop + MobileSAM           |
| Model load time       | <5s       | ONNX graph optimization           |
| Memory usage          | <300MB    | Efficient tensor management       |

---

## Handling Edge Cases

| Challenge            | Solution                                              |
|----------------------|-------------------------------------------------------|
| **Fast motion**      | ByteTrack velocity prediction + expanded IoU search   |
| **Occlusion**        | 30-frame lost tolerance + velocity-based re-acquisition|
| **Multiple subjects**| All detected with persistent IDs; user selects one    |
| **Low light**        | MobileNetV2 backbone handles varied illumination      |
| **Focus switch**     | Instant: click sets new target_id, next frame updates |
| **No WebGPU**        | Automatic fallback to WASM execution provider         |
| **No SAM models**    | Fallback to bbox-based elliptical gradient mask        |
