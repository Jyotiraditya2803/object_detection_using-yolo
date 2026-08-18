# config.py - Central Configuration for YOLO Local Project
import torch

class Config:
    # Model Weights (Ultralytics auto-downloads these on first run)
    OBJECT_MODEL_PATH = "yolov8n.pt"        # Nano object detection model (fastest)
    POSE_MODEL_PATH = "yolov8n-pose.pt"     # Nano pose estimation model (17 keypoints)
    
    # Confidence and NMS Filters
    CONFIDENCE_THRESHOLD = 0.45             # Filter out detections with confidence below 45%
    IOU_THRESHOLD = 0.45                    # NMS Overlap Threshold
    
    # Camera & Video Stream Settings
    CAMERA_INDEX = 0                        # Default web camera (change to 1 for external USB camera)
    FRAME_WIDTH = 640                       # Capture width (lower = higher FPS)
    FRAME_HEIGHT = 480                      # Capture height
    FRAME_SKIP = 0                          # Skip frames on slow CPUs (0 = process every frame)
    
    # Hardware Acceleration Selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Visualization Customization
    SKELETON_COLOR = (0, 255, 0)            # Green color (BGR) for skeleton bones
    JOINT_COLOR = (0, 0, 255)               # Red color (BGR) for joint keypoint dots
    BOX_COLOR = (255, 128, 0)               # Orange color (BGR) for bounding boxes
    BOX_THICKNESS = 2
