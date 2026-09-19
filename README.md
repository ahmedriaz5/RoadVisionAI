# RoadVisionAI
Real-time AI traffic detection and vehicle tracking system using YOLO11, OpenCV, ByteTrack, TensorRT, CUDA, and FastAPI.
# 🚦 RoadVisionAI

### Real-Time AI Traffic Detection & Vehicle Tracking

RoadVisionAI is a real-time computer vision system designed to detect, track, and analyze vehicles from traffic video streams.

The system uses **YOLO11**, **ByteTrack**, **OpenCV**, **PyTorch**, and **NVIDIA TensorRT** to create a GPU-accelerated traffic analysis pipeline with a **FastAPI** backend.

---

## ✨ Features

* 🚗 Real-time vehicle detection
* 🎯 Persistent vehicle tracking with ByteTrack
* 🏷️ Vehicle class identification
* 📊 Active and total vehicle statistics
* 🆔 Unique tracking IDs
* 📈 Real-time FPS monitoring
* 🎥 Video stream processing
* 🖼️ Real-time annotated video output
* ⚡ NVIDIA GPU acceleration
* 🚀 TensorRT FP16 inference
* 🔌 FastAPI backend
* 🧠 Local AI inference without external AI APIs

### Supported Vehicle Classes

* 🚗 Car
* 🏍️ Motorcycle
* 🚌 Bus
* 🚛 Truck
* 🚲 Bicycle

---

## 🧠 System Architecture

```text
Video Source
     │
     ▼
 OpenCV Video Capture
     │
     ▼
 YOLO11 Object Detection
     │
     ▼
 TensorRT FP16 Inference
     │
     ▼
 ByteTrack Object Tracking
     │
     ├───────────────┐
     ▼               ▼
Vehicle Statistics   Annotated Video
     │               │
     └───────┬───────┘
             ▼
        FastAPI Backend
             │
             ▼
        Real-Time UI/API
```

---

## 🛠️ Technology Stack

| Technology  | Purpose                            |
| ----------- | ---------------------------------- |
| Python      | Core programming language          |
| YOLO11      | Object detection                   |
| Ultralytics | YOLO model framework               |
| OpenCV      | Video processing and visualization |
| ByteTrack   | Multi-object tracking              |
| PyTorch     | Deep learning framework            |
| TensorRT    | GPU inference optimization         |
| CUDA        | NVIDIA GPU acceleration            |
| FastAPI     | Backend API                        |
| Uvicorn     | ASGI server                        |

---

## ⚡ GPU Optimization

A major part of the project was optimizing the inference pipeline for NVIDIA GPUs.

The original YOLO11n PyTorch inference pipeline was benchmarked and then converted into a **TensorRT FP16 engine**.

### Optimization workflow

```text
YOLO11n PyTorch Model
        │
        ▼
      ONNX
        │
        ▼
TensorRT FP16 Engine
        │
        ▼
NVIDIA GPU Inference
```

The TensorRT engine allows the application to take advantage of optimized NVIDIA inference execution rather than relying only on standard PyTorch inference.

---

## 🖥️ Hardware

The system was developed and tested on an NVIDIA laptop GPU environment.

**GPU:**

* NVIDIA GeForce RTX 4050 Laptop GPU
* 6 GB VRAM

**CPU:**

* AMD Ryzen 5 8645HS

**RAM:**

* 16 GB

---

## 📊 Model

**Model:** YOLO11n

The project uses the lightweight YOLO11n model to balance detection accuracy and real-time inference performance.

The TensorRT version uses an FP16 engine:

```text
yolo11n.engine
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RoadVisionAI.git
cd RoadVisionAI
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🎥 Add a Video

Place your test video inside:

```text
data/videos/test.mp4
```

The default configuration expects:

```text
data/videos/test.mp4
```

You can also configure another source through the application settings.

---

## ▶️ Run the Application

Start the FastAPI server:

```powershell
python run.py
```

The backend will start at:

```text
http://127.0.0.1:8000
```

---

## 📁 Project Structure

```text
RoadVisionAI/
│
├── apps/
│   └── backend/
│       ├── config.py
│       ├── main.py
│       ├── vision.py
│       └── schemas.py
│
├── data/
│   └── videos/
│       └── test.mp4
│
├── yolo11n.pt
├── yolo11n.engine
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔍 Detection Pipeline

For every incoming frame, the system performs:

1. Video frame capture
2. YOLO11 vehicle detection
3. Confidence filtering
4. Vehicle class filtering
5. ByteTrack tracking
6. Track ID assignment
7. Vehicle statistics calculation
8. Bounding-box visualization
9. FPS calculation
10. JPEG encoding
11. Backend delivery

---

## 📊 Real-Time Statistics

RoadVisionAI maintains:

* Total tracked vehicles
* Currently active vehicles
* Cars
* Motorcycles
* Buses
* Trucks
* Bicycles
* Current processing FPS

---

## 🔧 Configuration

Important settings include:

```text
VIDEO_SOURCE
MODEL_NAME
CONFIDENCE
IOU
DEVICE
TRACKER
MAX_FPS
```

The application can be configured for different video sources, detection thresholds, tracking configuration, and inference devices.

---

## 🎯 Project Goals

The project was developed to explore practical applications of:

* Computer Vision
* Deep Learning
* Real-Time Object Detection
* Multi-Object Tracking
* GPU Computing
* AI Inference Optimization
* Video Analytics
* Backend AI Integration

---

## 🚀 Future Improvements

Planned improvements include:

* [ ] Traffic density estimation
* [ ] Vehicle counting by direction
* [ ] Lane detection
* [ ] Speed estimation
* [ ] License plate recognition
* [ ] Traffic violation detection
* [ ] Accident detection
* [ ] Multi-camera support
* [ ] WebSocket-based live streaming
* [ ] Dashboard analytics
* [ ] Edge deployment
* [ ] Further TensorRT optimization
* [ ] Higher-throughput inference pipeline

---

## ⚠️ Performance Note

Performance depends on GPU hardware, input resolution, model size, video resolution, tracking configuration, and video encoding overhead.

The project was specifically optimized and benchmarked on an **NVIDIA RTX 4050 Laptop GPU** using TensorRT FP16 inference.

---

## 🔐 Privacy

The system is designed for local video processing and does not require external AI APIs for its core detection pipeline.

Do not upload private, personal, or sensitive video footage to a public repository.

---

## 📜 License

This project is intended as a portfolio and educational computer vision project.

Add an appropriate open-source license before distributing the repository publicly.
