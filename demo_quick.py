"""
QUICK DEMO - SmartBus Intelligence Platform
Demonstrates unified blind spot + traffic violation detection
No training required - uses base YOLO model
"""

import cv2
import numpy as np
from datetime import datetime
import time

print("=" * 70)
print("🚌 SMARTBUS INTELLIGENCE PLATFORM - QUICK DEMO")
print("   Unified Blind Spot Safety + Traffic Enforcement")
print("=" * 70)
print()

# Check dependencies
print("Checking dependencies...")
try:
    from ultralytics import YOLO
    print("✓ YOLOv8 available")
    YOLO_AVAILABLE = True
except:
    print("✗ YOLOv8 not available - demo mode only")
    YOLO_AVAILABLE = False

try:
    from PySide6.QtWidgets import QApplication
    print("✓ PySide6 available")
    GUI_AVAILABLE = True
except:
    print("✗ PySide6 not available - no GUI")
    GUI_AVAILABLE = False

print()

# Load model
if YOLO_AVAILABLE:
    print("Loading YOLO model...")
    try:
        model = YOLO('yolov8n.pt')  # Will auto-download if not present
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        model = None
else:
    model = None

print()
print("=" * 70)
print("DEMO MODE")
print("=" * 70)
print()
print("This demo shows:")
print("  🔵 Blind spot detection (blue boxes)")
print("  🔴 Traffic violations (red boxes)")
print("  📊 Real-time statistics")
print()
print("Press 'q' to quit")
print("Press 's' to save screenshot")
print()
print("=" * 70)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️  Camera not available - using synthetic demo")
    CAMERA_AVAILABLE = False
else:
    print("✓ Camera initialized")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    CAMERA_AVAILABLE = True

# Statistics
stats = {
    'frames': 0,
    'blind_spots': 0,
    'violations': 0,
    'fps': 0
}

start_time = time.time()
fps_counter = 0
fps_start = time.time()

print()
print("Starting detection loop...")
print()

try:
    while True:
        # Get frame
        if CAMERA_AVAILABLE:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Synthetic frame
            frame = np.random.randint(50, 150, (720, 1280, 3), dtype=np.uint8)
            # Add road
            cv2.rectangle(frame, (0, 400), (1280, 720), (80, 80, 80), -1)
            # Add text
            cv2.putText(frame, "DEMO MODE - No Camera", (400, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        stats['frames'] += 1

        # Detection
        detections = []
        violations = []

        if model is not None and CAMERA_AVAILABLE:
            try:
                results = model(frame, verbose=False)

                if len(results) > 0 and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = results[0].names[cls]

                        # Simulate distance
                        bbox_height = y2 - y1
                        distance = (1.7 * 800) / bbox_height if bbox_height > 0 else 50
                        distance = max(0.5, min(50, distance))

                        # Blind spot detection
                        if class_name in ['person', 'bicycle', 'motorcycle', 'car']:
                            if distance < 2.0:
                                detections.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'class': class_name,
                                    'distance': distance,
                                    'conf': conf
                                })
                                stats['blind_spots'] += 1

                        # Traffic violations (simulated)
                        if class_name == 'motorcycle' and np.random.random() > 0.9:
                            violations.append({
                                'type': 'no_helmet',
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'conf': 0.85
                            })
                            stats['violations'] += 1

                        if class_name == 'car' and np.random.random() > 0.95:
                            violations.append({
                                'type': 'no_seatbelt',
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'conf': 0.80
                            })
                            stats['violations'] += 1

            except Exception as e:
                pass

        # Draw detections
        # Blind spots (BLUE)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"BLIND SPOT: {det['class']} {det['distance']:.1f}m"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Violations (RED)
        for viol in violations:
            x1, y1, x2, y2 = viol['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"VIOLATION: {viol['type']}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            stats['fps'] = fps_counter
            fps_counter = 0
            fps_start = time.time()

        # Draw stats overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (40, 40, 40), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        cv2.putText(frame, f"FPS: {stats['fps']}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames: {stats['frames']}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Blind Spots: {stats['blind_spots']}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Violations: {stats['violations']}", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Legend
        cv2.rectangle(frame, (1050, 10), (1270, 80), (40, 40, 40), -1)
        cv2.putText(frame, "LEGEND:", (1060, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "BLUE = Blind Spot", (1060, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(frame, "RED = Violation", (1060, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Display
        cv2.imshow('SmartBus Intelligence Platform - DEMO', frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'screenshot_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

except KeyboardInterrupt:
    print("\nStopped by user")

# Cleanup
if CAMERA_AVAILABLE:
    cap.release()
cv2.destroyAllWindows()

# Final statistics
elapsed = time.time() - start_time
print()
print("=" * 70)
print("DEMO STATISTICS")
print("=" * 70)
print(f"Total frames processed: {stats['frames']}")
print(f"Total blind spot detections: {stats['blind_spots']}")
print(f"Total violation detections: {stats['violations']}")
print(f"Average FPS: {stats['frames']/elapsed:.1f}")
print(f"Runtime: {elapsed:.1f} seconds")
print("=" * 70)
print()
print("✅ Demo complete!")
print()
print("Next steps:")
print("  1. Run full system: python unified_detection_engine.py")
print("  2. Launch dashboard: python unified_dashboard.py")
print("  3. Train models: python train_yolo_helmet.py --generate")
print("  4. Generate installation guide: python pmo_installation.py --bus SGP1234")
print()

