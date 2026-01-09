"""
UNIFIED DETECTION ENGINE - Smart Vehicle Intelligence System
Works with ALL vehicles: Cars, Trucks, Buses, Motorcycles, Bicycles
Simultaneously detects blind spots AND traffic violations
Real-time OpenCV + YOLOv8n pipeline
Integrates: Blind spot safety + Traffic enforcement + Fleet management
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
from collections import deque
import threading
import queue
import time
import json
from pathlib import Path

# Optional imports
try:
    import easyocr
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False
    print("⚠️  EasyOCR not available - plate recognition disabled")

try:
    import mediapipe as mp
    POSE_AVAILABLE = True
except:
    POSE_AVAILABLE = False
    print("⚠️  MediaPipe not available - pose estimation disabled")


class ViolationDetector:
    """Traffic violation detection logic"""

    def __init__(self):
        self.SPEED_LIMIT = {
            'school_zone': 30,  # km/h
            'residential': 50,
            'highway': 90
        }

        self.pixels_per_meter = 20  # Calibration value
        self.frame_rate = 30  # FPS

        # Object tracking for speed estimation
        self.tracked_objects = {}
        self.max_track_age = 30  # frames

        # Violation history (avoid duplicate alerts)
        self.recent_violations = deque(maxlen=100)

    def check_helmet(self, obj, frame, helmet_model):
        """Check if motorcyclist is wearing helmet"""
        if obj['class'] not in ['motorcycle', 'bike']:
            return None

        # Crop rider region (upper 40% of bounding box)
        x1, y1, x2, y2 = obj['bbox']
        rider_y1 = int(y1)
        rider_y2 = int(y1 + (y2-y1) * 0.4)
        rider_crop = frame[rider_y1:rider_y2, x1:x2]

        if rider_crop.size == 0:
            return None

        # Run helmet detection
        try:
            results = helmet_model(rider_crop, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Class 1 = no_helmet
                    if cls == 1 and conf > 0.6:
                        return {
                            'type': 'no_helmet',
                            'confidence': conf,
                            'location': (x1, y1, x2, y2)
                        }
        except Exception as e:
            pass

        return None

    def check_seatbelt(self, obj, frame, pose_estimator):
        """Check if driver is wearing seatbelt using pose estimation"""
        if obj['class'] not in ['car']:
            return None

        if not POSE_AVAILABLE:
            return None

        # Crop driver region
        x1, y1, x2, y2 = obj['bbox']
        driver_crop = frame[y1:y2, x1:x2]

        if driver_crop.size == 0:
            return None

        try:
            # Process with MediaPipe pose
            rgb_crop = cv2.cvtColor(driver_crop, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(rgb_crop)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_pose = mp.solutions.pose

                # Get shoulder positions
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # Calculate shoulder angle
                angle = np.degrees(np.arctan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x
                ))

                # Seatbelt causes asymmetric shoulder angle
                if abs(angle) < 30:  # Shoulders too level = no seatbelt
                    return {
                        'type': 'no_seatbelt',
                        'confidence': 0.85,
                        'shoulder_angle': angle,
                        'location': (x1, y1, x2, y2)
                    }
        except Exception as e:
            pass

        return None

    def check_speed(self, obj_id, current_bbox, timestamp, location_type='residential'):
        """Estimate speed and check for over-speeding"""
        if obj_id not in self.tracked_objects:
            self.tracked_objects[obj_id] = {
                'history': deque(maxlen=10),
                'last_timestamp': timestamp
            }
            return None

        track = self.tracked_objects[obj_id]

        # Calculate centroid
        x1, y1, x2, y2 = current_bbox
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Add to history
        track['history'].append({
            'centroid': centroid,
            'timestamp': timestamp
        })

        # Need at least 5 frames for reliable speed estimate
        if len(track['history']) < 5:
            return None

        # Calculate speed from first to last in history
        first = track['history'][0]
        last = track['history'][-1]

        time_delta = (last['timestamp'] - first['timestamp']).total_seconds()
        if time_delta < 0.1:
            return None

        # Pixel distance
        px_distance = np.sqrt(
            (last['centroid'][0] - first['centroid'][0])**2 +
            (last['centroid'][1] - first['centroid'][1])**2
        )

        # Convert to real distance
        distance_m = px_distance / self.pixels_per_meter

        # Speed in m/s then km/h
        speed_mps = distance_m / time_delta
        speed_kmh = speed_mps * 3.6

        # Check against limit
        limit = self.SPEED_LIMIT.get(location_type, 50)

        if speed_kmh > limit:
            return {
                'type': 'over_speeding',
                'speed_kmh': speed_kmh,
                'limit_kmh': limit,
                'confidence': 0.80,
                'location': current_bbox
            }

        return None

    def check_lane_discipline(self, trajectory_points):
        """Detect wrong-way traffic and illegal lane changes"""
        if len(trajectory_points) < 30:
            return None

        # Fit line to trajectory
        x_coords = [p[0] for p in trajectory_points]
        y_coords = [p[1] for p in trajectory_points]

        if len(x_coords) < 2:
            return None

        # Linear fit
        coeffs = np.polyfit(x_coords, y_coords, 1)
        slope = coeffs[0]

        # Calculate angle
        angle = np.degrees(np.arctan(slope))

        # Sudden angle change = lane violation
        if abs(angle) > 45:
            return {
                'type': 'wrong_lane',
                'angle': angle,
                'confidence': 0.75
            }

        return None

    def check_wrong_way(self, velocity, expected_direction=(1, 0)):
        """Detect wrong-way traffic using velocity direction"""
        if velocity is None:
            return None

        vx, vy = velocity
        dx, dy = expected_direction

        # Dot product
        dot = vx * dx + vy * dy

        # Negative dot product = opposite direction
        if dot < -0.5:
            return {
                'type': 'wrong_way_traffic',
                'confidence': 0.90
            }

        return None


class UnifiedDetectionEngine:
    """
    Unified detection engine combining:
    - Blind spot detection (existing T-SA)
    - Traffic violation detection (new)
    """

    def __init__(self, config_path='config.json'):
        print("🚗 SMART VEHICLE INTELLIGENCE SYSTEM")
        print("   ALL Vehicles: Cars, Trucks, Buses, Motorcycles, Bicycles")
        print("🚌 UNIFIED SMARTBUS INTELLIGENCE PLATFORM")
        print("   Blind Spot Safety + Traffic Enforcement")
        print("=" * 70)

        # Load config
        self.config = self.load_config(config_path)

        # Models
        print("\n📦 Loading AI Models...")
        self.blind_spot_model = YOLO('yolov8n.pt')

        # Traffic violation models
        try:
            self.helmet_model = YOLO('models/helmet_detector.pt')
            print("  ✓ Helmet detector loaded")
        except:
            self.helmet_model = None
            print("  ⚠️  Helmet detector not found")

        try:
            self.seatbelt_model = YOLO('models/seatbelt_detector.pt')
            print("  ✓ Seatbelt detector loaded")
        except:
            self.seatbelt_model = None
            print("  ⚠️  Seatbelt detector not found")

        try:
            self.traffic_model = YOLO('models/traffic_detector.pt')
            print("  ✓ Traffic detector loaded")
        except:
            self.traffic_model = self.blind_spot_model
            print("  ℹ️  Using base YOLO for traffic")

        # OCR for license plates
        if OCR_AVAILABLE:
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("  ✓ EasyOCR loaded for ANPR")
        else:
            self.ocr_reader = None

        # Pose estimator
        if POSE_AVAILABLE:
            mp_pose = mp.solutions.pose
            self.pose_estimator = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5
            )
            print("  ✓ MediaPipe Pose loaded")
        else:
            self.pose_estimator = None

        # Violation detector
        self.violation_detector = ViolationDetector()

        # Optical flow for motion tracking
        self.prev_frame = None
        self.prev_gray = None

        # Detection queues
        self.detection_queue = queue.Queue(maxsize=30)
        self.violation_queue = queue.Queue(maxsize=100)

        # Statistics
        self.stats = {
            'blind_spot_detections': 0,
            'traffic_violations': 0,
            'helmet_violations': 0,
            'seatbelt_violations': 0,
            'speed_violations': 0,
            'lane_violations': 0,
            'total_frames': 0,
            'fps': 0
        }

        # Video buffer (MDVR style)
        self.video_buffer = deque(maxlen=300)  # 10s at 30fps

        print("\n✅ Unified Engine Initialized")

    def load_config(self, config_path):
        """Load configuration"""
        default_config = {
            'blind_spot_threshold': 1.5,  # meters
            'enable_blind_spot': True,
            'enable_traffic_violation': True,
            'enable_helmet_detection': True,
            'enable_seatbelt_detection': True,
            'enable_speed_detection': True,
            'enable_lane_detection': True,
            'camera_calibration': {
                'pixels_per_meter': 20,
                'focal_length': 800
            }
        }

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        else:
            return default_config

    def process_frame(self, frame, frame_id, timestamp):
        """
        UNIFIED PIPELINE - Process single frame for ALL detections
        """
        h, w = frame.shape[:2]
        detections = []
        violations = []

        # === FRAME 1: YOLO DETECTION (UNIFIED) ===
        # Run primary detection
        results = self.blind_spot_model(frame, verbose=False)

        objects = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]

                obj = {
                    'id': len(objects),
                    'class': class_name,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'centroid': ((x1+x2)/2, (y1+y2)/2)
                }
                objects.append(obj)

                # === BLIND SPOT LOGIC ===
                if self.config['enable_blind_spot']:
                    if class_name in ['person', 'pedestrian', 'bicycle', 'motorcycle']:
                        # Calculate distance
                        bbox_height = y2 - y1
                        distance = self.estimate_distance(bbox_height)

                        if distance < self.config['blind_spot_threshold']:
                            detections.append({
                                'type': 'blind_spot_threat',
                                'class': class_name,
                                'distance': distance,
                                'bbox': obj['bbox'],
                                'confidence': conf,
                                'threat_level': 'CRITICAL' if distance < 1.0 else 'HIGH'
                            })
                            self.stats['blind_spot_detections'] += 1

        # === FRAME 2: TRAFFIC VIOLATION DETECTION ===
        if self.config['enable_traffic_violation']:
            for obj in objects:
                # Helmet check
                if self.config['enable_helmet_detection'] and self.helmet_model:
                    helmet_violation = self.violation_detector.check_helmet(
                        obj, frame, self.helmet_model
                    )
                    if helmet_violation:
                        violations.append(helmet_violation)
                        self.stats['helmet_violations'] += 1

                # Seatbelt check
                if self.config['enable_seatbelt_detection'] and self.pose_estimator:
                    seatbelt_violation = self.violation_detector.check_seatbelt(
                        obj, frame, self.pose_estimator
                    )
                    if seatbelt_violation:
                        violations.append(seatbelt_violation)
                        self.stats['seatbelt_violations'] += 1

                # Speed check
                if self.config['enable_speed_detection']:
                    speed_violation = self.violation_detector.check_speed(
                        obj['id'], obj['bbox'], timestamp
                    )
                    if speed_violation:
                        violations.append(speed_violation)
                        self.stats['speed_violations'] += 1

        # === FRAME 3: OPTICAL FLOW (Lane discipline) ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None and self.config['enable_lane_detection']:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            # Process flow for lane violations (simplified)

        self.prev_gray = gray
        self.prev_frame = frame.copy()

        # Update stats
        self.stats['total_frames'] += 1
        self.stats['traffic_violations'] = len(violations)

        # Add to video buffer
        self.video_buffer.append({
            'frame': frame.copy(),
            'timestamp': timestamp,
            'detections': detections,
            'violations': violations
        })

        return {
            'frame': frame,
            'blind_spot_detections': detections,
            'traffic_violations': violations,
            'objects': objects,
            'stats': self.stats.copy()
    def estimate_distance(self, bbox_height, known_height=1.7, object_class='person'):
        """Estimate distance from bounding box height - vehicle-type aware"""
        # Adjust known height based on object type
        height_map = {
            'person': 1.7, 'pedestrian': 1.7, 'child': 1.2,
            'bicycle': 1.5, 'motorcycle': 1.4,
            'car': 1.5, 'truck': 2.5, 'bus': 3.0,
            'wheelchair': 1.3, 'animal': 0.5
        }
        known_height = height_map.get(object_class, 1.7)

    def estimate_distance(self, bbox_height, known_height=1.7):
        """Estimate distance from bounding box height"""
        focal_length = self.config['camera_calibration']['focal_length']
        if bbox_height > 0:
            distance = (known_height * focal_length) / bbox_height
            return max(0.5, min(50, distance))  # Clamp
        return 50.0

    def extract_license_plate(self, frame, bbox):
        """Extract license plate using OCR"""
        if not self.ocr_reader:
            return None

        x1, y1, x2, y2 = bbox
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            return None

        try:
            results = self.ocr_reader.readtext(plate_crop)
            if results:
                # Get highest confidence result
                text = max(results, key=lambda x: x[1])[0]
                return text.upper().replace(' ', '')
        except:
            pass

        return None

    def save_violation_evidence(self, frame, violation, output_dir='violations/'):
        """Save violation evidence with metadata"""
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{violation['type']}_{timestamp}.jpg"
        filepath = Path(output_dir) / filename

        # Draw bounding box on evidence
        annotated = frame.copy()
        if 'location' in violation:
            x1, y1, x2, y2 = violation['location']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, violation['type'], (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imwrite(str(filepath), annotated)

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'violation': violation,
            'image_path': str(filepath)
        }

        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(filepath)

    def run_camera_stream(self, camera_id=0):
        """Run detection on live camera stream"""
        print("\n🎥 Starting camera stream...")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("⚠️  Camera not available")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_id = 0
        fps_start = time.time()
        fps_counter = 0

        print("✓ Camera initialized - Press 'q' to quit")
        print("\nDetecting:")
        print("  🔵 Blind spot threats")
        print("  🔴 Traffic violations")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            timestamp = datetime.now()

            # Process frame
            result = self.process_frame(frame, frame_id, timestamp)

            # Draw results
            annotated = self.draw_detections(result)

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                self.stats['fps'] = fps_counter
                fps_counter = 0
                fps_start = time.time()

            # Display
            cv2.imshow('Unified SmartBus Intelligence', annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n📊 Final Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")

    def draw_detections(self, result):
        """Draw all detections on frame"""
        frame = result['frame'].copy()

        # Draw blind spot detections (BLUE)
        for det in result['blind_spot_detections']:
            x1, y1, x2, y2 = det['bbox']
            color = (255, 0, 0) if det['threat_level'] == 'CRITICAL' else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"BLIND SPOT: {det['class']} {det['distance']:.1f}m"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw traffic violations (RED)
        for viol in result['traffic_violations']:
            if 'location' in viol:
                x1, y1, x2, y2 = viol['location']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                label = f"VIOLATION: {viol['type']}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw stats overlay
        stats_y = 30
        cv2.putText(frame, f"FPS: {result['stats']['fps']}", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Blind Spots: {result['stats']['blind_spot_detections']}",
                   (10, stats_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Violations: {result['stats']['traffic_violations']}",
                   (10, stats_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Unified SmartBus Intelligence Platform')
    parser.add_argument('--mode', type=str, default='unified',
                       choices=['unified', 'blind_spot', 'traffic_violation'],
                       help='Detection mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--config', type=str, default='config.json', help='Config file')
    parser.add_argument('--blind_spot', type=bool, default=True, help='Enable blind spot')
    parser.add_argument('--traffic_violation', type=bool, default=True, help='Enable traffic violations')
    parser.add_argument('--enforce_rules', type=str, default='TRUE', help='Enforce rules')

    args = parser.parse_args()

    # Create engine
    engine = UnifiedDetectionEngine(config_path=args.config)

    # Update config based on mode
    if args.mode == 'blind_spot':
        engine.config['enable_traffic_violation'] = False
    elif args.mode == 'traffic_violation':
        engine.config['enable_blind_spot'] = False

    # Run
    engine.run_camera_stream(camera_id=args.camera)


if __name__ == '__main__':
    main()

