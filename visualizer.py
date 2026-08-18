# visualizer.py - OpenCV Graphic Overlay Rendering
import cv2
import numpy as np

class Visualizer:
    """Utility class to render bounding boxes, skeletons, and HUD overlays."""
    
    @staticmethod
    def draw_detections(frame, detections, box_color=(255, 128, 0)):
        """Draws object detection bounding boxes with text banners."""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['label']} {det['confidence']*100:.1f}%"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw text background pill
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + w + 6, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_poses(frame, poses, skeleton_pairs, min_conf=0.35):
        """Draws human pose keypoints and connecting bones."""
        for pose in poses:
            kpts = pose["keypoints"]
            
            # Draw connecting bone lines
            for p1, p2 in skeleton_pairs:
                kp1, kp2 = kpts[p1], kpts[p2]
                if kp1["confidence"] >= min_conf and kp2["confidence"] >= min_conf:
                    cv2.line(frame, (kp1["x"], kp1["y"]), (kp2["x"], kp2["y"]), (0, 255, 0), 2, cv2.LINE_AA)
                    
            # Draw keypoint joint circles
            for kp in kpts:
                if kp["confidence"] >= min_conf:
                    cv2.circle(frame, (kp["x"], kp["y"]), 5, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (kp["x"], kp["y"]), 6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_hud(frame, fps, latency_ms, device, pose_count, obj_count):
        """Draws real-time heads-up status display overlay."""
        # Semi-transparent dark banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (320, 115), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (12, 12), (320, 115), (59, 130, 246), 1)
        
        cv2.putText(frame, f"FPS: {fps:.1f} ({latency_ms:.1f} ms)", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Hardware: {device.upper()}", (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Persons Tracked: {pose_count}", (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 204, 21), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Objects Detected: {obj_count}", (22, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
        return frame
