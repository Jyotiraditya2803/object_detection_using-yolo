# yolo_pose.py - Human Pose Estimation Engine
from ultralytics import YOLO
import torch

class YOLOPoseEstimator:
    """
    Human Pose Estimator extracting 17 COCO keypoints:
    0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear,
    5:L_Shoulder, 6:R_Shoulder, 7:L_Elbow, 8:R_Elbow,
    9:L_Wrist, 10:R_Wrist, 11:L_Hip, 12:R_Hip,
    13:L_Knee, 14:R_Knee, 15:L_Ankle, 16:R_Ankle
    """
    
    # 17 Keypoint COCO Bone Skeleton Connection Pairs
    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),             # Facial Features
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),     # Shoulders & Arms
        (5, 11), (6, 12), (11, 12),                  # Torso & Hips
        (11, 13), (13, 15), (12, 14), (14, 16)       # Legs & Ankles
    ]

    KEYPOINT_NAMES = [
        "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
        "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
    ]

    def __init__(self, model_path="yolov8n-pose.pt", device="cpu", conf=0.45):
        self.device = device
        self.conf = conf
        print(f"[INFO] Initializing Pose Estimator: {model_path} on {device}...")
        self.model = YOLO(model_path)

    def estimate(self, frame):
        """
        Detects human poses in image.
        Returns list of pose objects with bounding box and 17 keypoint dicts.
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf,
            verbose=False
        )
        
        poses = []
        for r in results:
            if r.keypoints is None:
                continue
                
            boxes = r.boxes
            keypoints_data = r.keypoints.data.cpu().numpy() # Shape: (N, 17, 3) -> [x, y, conf]
            
            for idx, kpts in enumerate(keypoints_data):
                box = map(int, boxes[idx].xyxy[0]) if len(boxes) > idx else (0, 0, 0, 0)
                pose_keypoints = []
                for kp_idx, (x, y, kp_conf) in enumerate(kpts):
                    pose_keypoints.append({
                        "id": kp_idx,
                        "name": self.KEYPOINT_NAMES[kp_idx],
                        "x": int(x),
                        "y": int(y),
                        "confidence": float(kp_conf)
                    })
                poses.append({
                    "box": tuple(box),
                    "keypoints": pose_keypoints,
                    "confidence": float(boxes[idx].conf[0]) if len(boxes) > idx else 0.0
                })
        return poses
