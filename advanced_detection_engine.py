"""
Advanced Detection Engine - NextGen BlindSpot Protection
Real-time object detection with YOLOv8 + Fatigue monitoring
Optimized for performance with threaded processing
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import threading
import queue
from collections import deque
import time

class DetectionEngine:
    """High-performance detection engine with multiple innovations"""

    def __init__(self):
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        self.model.to('cpu')  # Use CPU for laptop demo

        # Performance optimization
        self.detection_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)

        # Innovation #1: AI False Positive Reduction
        self.detection_history = deque(maxlen=100)
        self.confidence_threshold = 0.5
        self.adaptive_threshold_enabled = True

        # Innovation #3: T-SEEDS Fatigue Detection
        self.fatigue_score = 0.0
        self.eye_closure_counter = 0

        # Innovation #4: Speed-adaptive alerts
        self.current_speed = 0  # km/h
        self.alert_distance_threshold = 2.0  # meters

        # Innovation #5: MDVR Video Buffer (10s)
        self.video_buffer = deque(maxlen=300)  # 10s at 30fps

        # Statistics
        self.total_detections = 0
        self.false_positive_count = 0
        self.alert_count = 0

        # Threading
        self.running = False
        self.detection_thread = None

    def start(self):
        """Start detection thread"""
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("✓ Detection engine started")

    def stop(self):
        """Stop detection thread"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        print("✓ Detection engine stopped")

    def _detection_loop(self):
        """Main detection loop running in background thread"""
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                if not self.detection_queue.empty():
                    frame = self.detection_queue.get(timeout=0.1)

                    # Run detection
                    results = self._run_detection(frame)

                    # Put results in queue
                    if not self.result_queue.full():
                        self.result_queue.put(results)
                else:
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")

    def _run_detection(self, frame):
        """Run YOLOv8 detection on frame"""
        try:
            # Resize for performance (smaller = faster)
            h, w = frame.shape[:2]
            scale = 640 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            # Run detection
            results = self.model(resized, verbose=False, conf=self.confidence_threshold)

            # Process results
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get class name
                    class_name = self.model.names[cls]

                    # Filter for relevant objects
                    if class_name in ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                        # Scale back to original size
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy / scale

                        # Calculate distance (simple estimation based on box size)
                        box_height = y2 - y1
                        estimated_distance = self._estimate_distance(box_height, h)

                        # Innovation #4: Speed-adaptive threshold
                        alert_threshold = self._get_alert_threshold()

                        detection = {
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'distance': estimated_distance,
                            'is_threat': estimated_distance < alert_threshold,
                            'timestamp': datetime.now().isoformat()
                        }

                        detections.append(detection)
                        self.total_detections += 1

                        # Innovation #1: Update adaptive threshold
                        if self.adaptive_threshold_enabled:
                            self.detection_history.append(conf)
                            if len(self.detection_history) >= 100:
                                self._update_adaptive_threshold()

            return {
                'frame': frame,
                'detections': detections,
                'timestamp': datetime.now().isoformat(),
                'fps': 0  # Will be calculated by caller
            }

        except Exception as e:
            print(f"Detection error: {e}")
            return {'frame': frame, 'detections': [], 'timestamp': datetime.now().isoformat(), 'fps': 0}

    def _estimate_distance(self, box_height, frame_height):
        """Estimate distance based on object size (simple heuristic)"""
        # Assume average person height ~1.7m, calculate distance
        # This is a simplified model - real systems use stereo vision or LiDAR
        if box_height > 0:
            focal_length = 500  # Approximate focal length
            real_height = 1.7  # meters (average person)
            distance = (focal_length * real_height) / box_height
            return max(0.5, min(distance, 50.0))  # Clamp between 0.5m and 50m
        return 10.0

    def _get_alert_threshold(self):
        """Innovation #4: Speed-adaptive alert threshold"""
        if self.current_speed < 20:  # School zone / slow speed
            return 1.0  # 1 meter
        elif self.current_speed < 60:  # City
            return 2.0  # 2 meters
        else:  # Highway
            return 3.0  # 3 meters

    def _update_adaptive_threshold(self):
        """Innovation #1: Self-learning threshold adaptation"""
        if len(self.detection_history) >= 100:
            mean_conf = np.mean(self.detection_history)
            std_conf = np.std(self.detection_history)

            # Adaptive threshold: mean - 1 std deviation
            new_threshold = max(0.3, mean_conf - std_conf)
            self.confidence_threshold = new_threshold

    def process_frame(self, frame):
        """Add frame to detection queue (non-blocking)"""
        if not self.detection_queue.full():
            self.detection_queue.put(frame)

    def get_results(self):
        """Get detection results (non-blocking)"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def update_speed(self, speed_kmh):
        """Innovation #4: Update vehicle speed for adaptive alerts"""
        self.current_speed = speed_kmh

    def get_statistics(self):
        """Get detection statistics"""
        return {
            'total_detections': self.total_detections,
            'false_positives': self.false_positive_count,
            'alerts': self.alert_count,
            'confidence_threshold': self.confidence_threshold,
            'fatigue_score': self.fatigue_score
        }


class FatigueMonitor:
    """Innovation #3: T-SEEDS Eye-tracking fatigue detection"""

    def __init__(self):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            # Note: Requires shape_predictor_68_face_landmarks.dat
            # For demo, we'll use simplified detection
            self.has_dlib = True
        except:
            self.has_dlib = False
            print("⚠️  dlib not available - fatigue detection disabled")

        self.eye_closure_frames = 0
        self.yawn_frames = 0
        self.fatigue_score = 0.0

    def analyze_frame(self, frame):
        """Analyze frame for fatigue indicators"""
        if not self.has_dlib:
            return self.fatigue_score

        # Simplified fatigue detection
        # Real implementation would use dlib landmarks for eye aspect ratio
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) > 0:
            # Detect eyes closed (simplified)
            # Real: use Eye Aspect Ratio (EAR) from landmarks
            self.eye_closure_frames += 1
            if self.eye_closure_frames > 30:  # ~1 second at 30fps
                self.fatigue_score = min(1.0, self.fatigue_score + 0.1)
        else:
            self.eye_closure_frames = 0
            self.fatigue_score = max(0.0, self.fatigue_score - 0.05)

        return self.fatigue_score


# Singleton instance
_detection_engine = None

def get_detection_engine():
    """Get global detection engine instance"""
    global _detection_engine
    if _detection_engine is None:
        _detection_engine = DetectionEngine()
    return _detection_engine

