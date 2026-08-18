# Real-Time YOLO Pose Estimation & Object Detection System
**Final Year Diploma Computer Engineering Project**

This project delivers a high-performance, lag-free real-time Computer Vision system combining **YOLOv8/v11 Object Detection** and **17-Keypoint Human Pose Estimation** using OpenCV and Python.

---

## 🚀 Key Features
1. **Multi-Task Computer Vision**: Runs object detection and human skeleton pose estimation concurrently.
2. **Asynchronous Threaded Frame Reader**: Eliminates OpenCV buffer lag and camera latency.
3. **Hardware Acceleration**: Automatic GPU (CUDA) acceleration with CPU fallback.
4. **Customizable Resolution & Frame Skipping**: Achieve 30+ FPS even on basic laptop hardware.
5. **Interactive Controls**: Live toggles for Pose, Objects, Screenshot capture, and HUD stats.

---

## 🛠️ Step-by-Step Installation

### Step 1: Install Python
Ensure Python 3.9, 3.10, or 3.11 is installed on your computer.
Check in terminal:
```bash
python --version
```

### Step 2: Set up Virtual Environment (Recommended)
```bash
# Open terminal in project folder
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Required Libraries
```bash
pip install -r requirements.txt
```

*Note: For GPU acceleration with NVIDIA graphics cards, install CUDA PyTorch:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎬 Running the System

Execute the main script:
```bash
python main.py
```

### ⌨️ Keyboard Shortcuts During Execution
- `p` : Toggle Pose Estimation ON / OFF
- `o` : Toggle Object Detection ON / OFF
- `s` : Save current frame screenshot to disk
- `q` : Safely exit program

---

## ⚡ How to Fix Lag on Slower Laptops
If your video is lagging on CPU:
1. Open `config.py`
2. Change `FRAME_WIDTH = 320` and `FRAME_HEIGHT = 240`
3. Set `FRAME_SKIP = 1` (processes alternate frames for 2x speedup)
4. Use Nano model `yolov8n.pt`
