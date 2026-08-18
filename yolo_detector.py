# yolo_detector.py - Modular Object Detection Wrapper
from ultralytics import YOLO
import torch

class YOLODetector:
    """Class to handle YOLO object detection on image frames."""
    
    def __init__(self, model_path="yolov8n.pt", device="cpu", conf=0.45, iou=0.45):
        self.device = device
        self.conf = conf
        self.iou = iou
        print(f"[INFO] Initializing Object Detector: {model_path} on {device}...")
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Processes frame and returns list of detected object dicts:
        [{ 'label': 'laptop', 'confidence': 0.88, 'box': (x1, y1, x2, y2), 'class_id': 63 }]
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[cls_id]
                
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                    "class_id": cls_id
                })
        return detections
