# main.py - Main Real-Time YOLO Pose & Object Detection System
# Final Year Diploma Computer Engineering Project
import cv2
import sys
import time
from config import Config
from video_stream import VideoStream
from fps_calculator import FPSCalculator
from yolo_detector import YOLODetector
from yolo_pose import YOLOPoseEstimator
from visualizer import Visualizer

def main():
    print("=" * 60)
    print("  YOLOv8/v11 Real-Time Pose & Object Detection System")
    print("  Final Year Diploma Project - Local High-FPS Deployment")
    print("=" * 60)
    print(f"[INFO] Processing Device: {Config.DEVICE.upper()}")
    print("[INFO] Controls:")
    print("  - Press 'q' : Quit Application")
    print("  - Press 'p' : Toggle Human Pose Estimation")
    print("  - Press 'o' : Toggle Object Detection")
    print("  - Press 's' : Save Current Frame Screenshot")
    print("-" * 60)

    # 1. Initialize Threaded Video Capture (Prevents OpenCV frame buffer lag)
    video = VideoStream(
        src=Config.CAMERA_INDEX,
        width=Config.FRAME_WIDTH,
        height=Config.FRAME_HEIGHT
    ).start()
    time.sleep(1.0) # Warm up camera sensor

    # 2. Initialize AI Models
    detector = YOLODetector(
        model_path=Config.OBJECT_MODEL_PATH,
        device=Config.DEVICE,
        conf=Config.CONFIDENCE_THRESHOLD
    )
    pose_estimator = YOLOPoseEstimator(
        model_path=Config.POSE_MODEL_PATH,
        device=Config.DEVICE,
        conf=Config.CONFIDENCE_THRESHOLD
    )
    
    fps_calc = FPSCalculator()

    # Active state flags
    show_pose = True
    show_objects = True
    frame_counter = 0

    try:
        while True:
            grabbed, frame = video.read()
            if not grabbed or frame is None:
                print("[WARNING] Failed to grab frame from camera. Retrying...")
                time.sleep(0.01)
                continue

            frame_counter += 1
            
            # Optional frame skipping for lower-end CPUs
            if Config.FRAME_SKIP > 0 and (frame_counter % (Config.FRAME_SKIP + 1)) != 0:
                cv2.imshow("YOLO Real-Time Detection System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Run Inferences
            objects = []
            poses = []

            if show_objects:
                objects = detector.detect(frame)
            
            if show_pose:
                poses = pose_estimator.estimate(frame)

            # Draw Overlays
            if show_objects:
                frame = Visualizer.draw_detections(frame, objects)

            if show_pose:
                frame = Visualizer.draw_poses(frame, poses, YOLOPoseEstimator.SKELETON_PAIRS)

            # Calculate FPS & render HUD
            fps, latency = fps_calc.update()
            frame = Visualizer.draw_hud(frame, fps, latency, Config.DEVICE, len(poses), len(objects))

            # Display output frame
            cv2.imshow("YOLO Real-Time Detection System", frame)

            # Key bindings
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quitting application...")
                break
            elif key == ord('p'):
                show_pose = not show_pose
                print(f"[TOGGLE] Pose Estimation: {'ON' if show_pose else 'OFF'}")
            elif key == ord('o'):
                show_objects = not show_objects
                print(f"[TOGGLE] Object Detection: {'ON' if show_objects else 'OFF'}")
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"[SAVED] Screenshot saved to {filename}")

    except KeyboardInterrupt:
        print("[INFO] Program stopped by keyboard interrupt.")
    finally:
        video.stop()
        cv2.destroyAllWindows()
        print("[INFO] Camera stream closed and windows destroyed.")

if __name__ == "__main__":
    main()
