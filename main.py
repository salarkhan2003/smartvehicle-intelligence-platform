import sys, cv2, numpy as np, sqlite3, os, time, winsound, random
from collections import deque
from datetime import datetime
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install: pip install ultralytics")

# YOLO COCO Classes
YOLO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl'
}

# Camera Enumeration
def find_cameras():
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                backend = cap.getBackendName()
                available.append({'index': i, 'name': f"Camera {i} ({backend})"})
            cap.release()
    return available

def select_camera():
    cameras = find_cameras()
    if not cameras:
        QMessageBox.warning(None, "No Camera", "No cameras detected! Using test pattern.")
        return -1
    if len(cameras) == 1:
        return cameras[0]['index']
    
    usb_cameras = [c for c in cameras if c['index'] > 0]
    default_cam = usb_cameras[0]['index'] if usb_cameras else cameras[0]['index']
    
    items = [f"{c['name']}" for c in cameras]
    selected, ok = QInputDialog.getItem(None, "Select Camera", 
                                        f"Found {len(cameras)} cameras. Select USB camera:",
                                        items, default_cam if default_cam < len(cameras) else 0, False)
    if ok and selected:
        idx = items.index(selected)
        return cameras[idx]['index']
    return default_cam

# Database Setup
def init_db():
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, violation_type TEXT, 
                  details TEXT, severity TEXT)''')
    c.execute("SELECT COUNT(*) FROM violations")
    if c.fetchone()[0] == 0:
        samples = [
            ("2026-01-10 10:30:15", "Overspeeding", "Speed: 75 km/h", "HIGH"),
            ("2026-01-10 11:15:22", "No Helmet", "Motorcycle rider", "CRITICAL"),
            ("2026-01-10 11:45:10", "Overspeeding", "Speed: 82 km/h", "HIGH")
        ]
        for ts, vtype, details, severity in samples:
            c.execute("INSERT INTO violations (timestamp, violation_type, details, severity) VALUES (?, ?, ?, ?)",
                     (ts, vtype, details, severity))
    conn.commit()
    conn.close()

# Camera Worker Thread
class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray, dict)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.prev_frame = None
        self.speed_kmh = 0
        self.ear_value = 0.3
        self.blind_spot_left = False
        self.blind_spot_right = False
        
        # Initialize YOLO model
        try:
            self.yolo = YOLO('yolov8n.pt')
            print("✓ YOLOv8n model loaded successfully")
        except Exception as e:
            print(f"✗ YOLO init error: {e}")
            self.yolo = None
    
    def calculate_mock_ear(self):
        # Mock EAR calculation (0.25-0.35, lower = more drowsy)
        self.ear_value = max(0.15, min(0.35, self.ear_value + random.uniform(-0.02, 0.02)))
        return self.ear_value
    
    def run(self):
        cap = None
        if self.camera_index >= 0:
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                print(f"✓ Camera {self.camera_index} opened successfully")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                print(f"✗ Failed to open camera {self.camera_index}")
                cap = None
        
        while self.running:
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA READ FAILED", (50, 240), 
                               cv2.FONT_HERSHEY_BOLD, 1, (0, 0, 255), 2)
            else:
                # Test pattern
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO CAMERA - TEST MODE", (50, 240), 
                           cv2.FONT_HERSHEY_BOLD, 1, (0, 0, 255), 2)
            
            frame = cv2.resize(frame, (640, 480))
            
            data = {
                'detections': [],
                'total_count': 0,
                'speed': 0,
                'ear': 0.3,
                'zone': 'Normal',
                'turn': 'STRAIGHT',
                'threat': 0,
                'helmet_status': 'N/A',
                'violations': [],
                'blind_spot_left': False,
                'blind_spot_right': False
            }
            
            # YOLO Detection
            if self.yolo:
                try:
                    results = self.yolo(frame, verbose=False, conf=0.5)
                    person_on_bike = False
                    helmet_detected = False
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            class_name = YOLO_CLASSES.get(cls_id, f'class_{cls_id}').upper()
                            
                            # Distance estimation
                            bbox_height = y2 - y1
                            dist = max(0.5, 3.5 - (bbox_height / 80))
                            
                            # Threat calculation
                            threat = 0
                            if dist < 1.0:
                                threat = 95
                            elif dist < 2.0:
                                threat = 70
                            elif dist < 3.0:
                                threat = 40
                            else:
                                threat = 15
                            
                            # Color based on threat
                            if threat > 70:
                                color = (0, 0, 255)  # Red
                            elif threat > 40:
                                color = (0, 165, 255)  # Orange
                            else:
                                color = (0, 255, 0)  # Green
                            
                            # Draw detection
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{class_name} {conf*100:.0f}%"
                            cv2.putText(frame, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"{dist:.1f}m", (x1, y2+20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            
                            # Track detections
                            data['detections'].append({
                                'class': class_name,
                                'conf': conf,
                                'dist': dist,
                                'threat': threat
                            })
                            data['total_count'] += 1
                            data['threat'] = max(data['threat'], threat)
                            
                            # Blind Spot Detection (check if vehicle in side zones)
                            frame_width = frame.shape[1]
                            center_x = (x1 + x2) // 2
                            
                            if cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                                if center_x < frame_width * 0.3:  # Left 30%
                                    data['blind_spot_left'] = True
                                    cv2.putText(frame, "BLIND SPOT LEFT!", (10, 90),
                                              cv2.FONT_HERSHEY_BOLD, 0.7, (0, 0, 255), 2)
                                elif center_x > frame_width * 0.7:  # Right 70%
                                    data['blind_spot_right'] = True
                                    cv2.putText(frame, "BLIND SPOT RIGHT!", (450, 90),
                                              cv2.FONT_HERSHEY_BOLD, 0.7, (0, 0, 255), 2)
                            
                            # Helmet detection logic
                            if cls_id == 0:  # Person
                                person_on_bike = True
                            if cls_id in [1, 3]:  # Bicycle or Motorcycle
                                person_on_bike = True
                            
                            # Mock helmet detection (in production, use helmet YOLO model)
                            if person_on_bike and cls_id == 0:
                                # Check if person's head region has helmet-like features (simplified)
                                head_region_height = int(bbox_height * 0.25)
                                head_y1 = max(0, y1)
                                head_y2 = min(frame.shape[0], y1 + head_region_height)
                                
                                # Mock: Random helmet detection for demo
                                helmet_detected = random.random() > 0.3
                                
                                if not helmet_detected:
                                    data['helmet_status'] = 'NO HELMET'
                                    data['violations'].append({
                                        'type': 'No Helmet',
                                        'details': f'{class_name} on vehicle',
                                        'severity': 'CRITICAL'
                                    })
                                    cv2.putText(frame, "NO HELMET!", (x1, y1-30),
                                              cv2.FONT_HERSHEY_BOLD, 0.6, (0, 0, 255), 2)
                                else:
                                    data['helmet_status'] = 'HELMET OK'
                
                except Exception as e:
                    print(f"YOLO error: {e}")
            
            # Optical Flow Speed Estimation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_frame is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(self.prev_frame, gray, None,
                                                        0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_val = np.mean(mag)
                    self.speed_kmh = flow_val * 0.5  # Adjusted calibration
                    data['speed'] = self.speed_kmh
                    
                    # Speed violation check
                    if self.speed_kmh > 60:
                        data['violations'].append({
                            'type': 'Overspeeding',
                            'details': f'Speed: {self.speed_kmh:.1f} km/h',
                            'severity': 'HIGH'
                        })
                except:
                    pass
            self.prev_frame = gray.copy()
            
            # Mock EAR for fatigue
            data['ear'] = self.calculate_mock_ear()
            drowsiness = max(0, min(100, (0.3 - data['ear']) * 400))
            
            if drowsiness > 70:
                data['violations'].append({
                    'type': 'Driver Fatigue',
                    'details': f'Drowsiness: {drowsiness:.0f}%',
                    'severity': 'CRITICAL'
                })
            
            # Mock GPS Geofencing
            mock_in_zone = random.random() < 0.2
            data['zone'] = 'SCHOOL ZONE' if mock_in_zone else 'Normal'
            
            # Mock CAN Bus Turn Signal
            turn_options = ['STRAIGHT', 'LEFT', 'RIGHT']
            data['turn'] = turn_options[int(time.time() * 0.3) % 3]
            
            # Display stats on frame
            cv2.putText(frame, f"Speed: {data['speed']:.1f} km/h", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Detections: {data['total_count']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            self.frame_ready.emit(frame, data)
            self.msleep(33)  # ~30 FPS
        
        if cap is not None:
            cap.release()

# Main Application
class SmartVehicleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Select camera
        self.camera_index = select_camera()
        camera_info = f"Camera {self.camera_index}" if self.camera_index >= 0 else "Test Pattern"
        self.setWindowTitle(f"SmartVehicle Intelligence v2.0 - {camera_info}")
        self.setGeometry(50, 50, 1400, 800)
        self.setStyleSheet("""
            QMainWindow {background: #1a1a1a;} 
            QLabel {color: #fff;} 
            QPushButton {background: #2a5; color: #fff; padding: 8px; border-radius: 4px;}
            QPushButton:hover {background: #3b6;}
            QProgressBar {border: 2px solid #555; border-radius: 5px; text-align: center;}
            QProgressBar::chunk {background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f00, stop:1 #f80);}
        """)
        
        init_db()
        
        self.stats = {'detections': 0, 'alerts': 0, 'violations': 0}
        self.event_logs = []
        self.alert_active = False
        
        self.setup_ui()
        
        # Camera Worker
        self.worker = CameraWorker(camera_index=self.camera_index)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.start()
        
        self.add_log(f"✓ System initialized - Camera {self.camera_index}")
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # LEFT PANEL - Video Feed
        left_panel = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 3px solid #0f0; background: #000;")
        left_panel.addWidget(self.video_label)
        
        # Control Buttons
        btn_layout = QHBoxLayout()
        self.test_alert_btn = QPushButton("⚠ TEST ALERT")
        self.test_alert_btn.clicked.connect(self.trigger_test_alert)
        self.export_btn = QPushButton("📄 Export Logs")
        self.export_btn.clicked.connect(self.export_logs)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_camera)
        btn_layout.addWidget(self.test_alert_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.stop_btn)
        left_panel.addLayout(btn_layout)
        
        # RIGHT PANEL - Telemetry
        right_panel = QVBoxLayout()
        
        # Stats Grid
        grid = QGridLayout()
        
        # Row 0: Speed
        self.speed_label = QLabel("Speed: 0 km/h")
        self.speed_label.setStyleSheet("font: bold 18pt; color: #0ff;")
        grid.addWidget(self.speed_label, 0, 0, 1, 2)
        
        # Row 1: Threat
        grid.addWidget(QLabel("Threat Level:"), 1, 0)
        self.threat_bar = QProgressBar()
        self.threat_bar.setMaximum(100)
        grid.addWidget(self.threat_bar, 1, 1)
        
        # Row 2: Fatigue
        grid.addWidget(QLabel("Fatigue Level:"), 2, 0)
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setMaximum(100)
        self.fatigue_bar.setStyleSheet("QProgressBar::chunk {background: #f80;}")
        grid.addWidget(self.fatigue_bar, 2, 1)
        
        # Row 3: Detections
        self.detection_label = QLabel("Detections: 0")
        self.detection_label.setStyleSheet("font: bold 14pt; color: #0f0;")
        grid.addWidget(self.detection_label, 3, 0, 1, 2)
        
        # Row 4: Zone
        self.zone_label = QLabel("Zone: Normal")
        self.zone_label.setStyleSheet("font: bold 12pt; color: #0f0;")
        grid.addWidget(self.zone_label, 4, 0, 1, 2)
        
        # Row 5: Turn Signal
        self.turn_label = QLabel("Turn: STRAIGHT")
        self.turn_label.setStyleSheet("font: 11pt; color: #ff0;")
        grid.addWidget(self.turn_label, 5, 0, 1, 2)
        
        # Row 6: Helmet Status
        self.helmet_label = QLabel("Helmet: N/A")
        self.helmet_label.setStyleSheet("font: bold 12pt; color: #fff;")
        grid.addWidget(self.helmet_label, 6, 0, 1, 2)
        
        # Row 7: Blind Spot Indicators
        blind_spot_layout = QHBoxLayout()
        self.blind_left_label = QLabel("⬅ LEFT: ✓")
        self.blind_left_label.setStyleSheet("font: bold 11pt; color: #0f0;")
        self.blind_right_label = QLabel("RIGHT: ✓ ➡")
        self.blind_right_label.setStyleSheet("font: bold 11pt; color: #0f0;")
        blind_spot_layout.addWidget(self.blind_left_label)
        blind_spot_layout.addStretch()
        blind_spot_layout.addWidget(self.blind_right_label)
        grid.addLayout(blind_spot_layout, 7, 0, 1, 2)
        
        # Row 8: Alert Status
        self.alert_label = QLabel("⚫ Status: Normal")
        self.alert_label.setStyleSheet("font: bold 14pt; color: #0f0; padding: 10px; border: 2px solid #0f0;")
        grid.addWidget(self.alert_label, 8, 0, 1, 2)
        
        # Row 9: Stats
        self.stats_label = QLabel("Detections: 0 | Alerts: 0 | Violations: 0")
        self.stats_label.setStyleSheet("font: 10pt; color: #ff0;")
        grid.addWidget(self.stats_label, 9, 0, 1, 2)
        
        right_panel.addLayout(grid)
        
        # Event Logs
        right_panel.addWidget(QLabel("📋 Live Event Logs:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #222; color: #0f0; font: 9pt Consolas;")
        self.log_text.setMaximumHeight(150)
        right_panel.addWidget(self.log_text)
        
        # Violations Table
        right_panel.addWidget(QLabel("⚠ Violations Table:"))
        self.violations_table = QTableWidget()
        self.violations_table.setColumnCount(4)
        self.violations_table.setHorizontalHeaderLabels(["Time", "Type", "Details", "Severity"])
        self.violations_table.horizontalHeader().setStretchLastSection(True)
        self.violations_table.setStyleSheet("background: #222; color: #fff;")
        self.violations_table.setMaximumHeight(150)
        self.load_violations_table()
        right_panel.addWidget(self.violations_table)
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
    
    def update_frame(self, frame, data):
        # Update video
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        
        # Update telemetry
        self.speed_label.setText(f"Speed: {data['speed']:.1f} km/h")
        
        # Threat
        self.threat_bar.setValue(int(data['threat']))
        if data['threat'] > 70:
            threat_text = "CRITICAL"
            self.threat_bar.setStyleSheet("QProgressBar::chunk {background: #f00;}")
        elif data['threat'] > 40:
            threat_text = "HIGH"
            self.threat_bar.setStyleSheet("QProgressBar::chunk {background: #f80;}")
        else:
            threat_text = "LOW"
            self.threat_bar.setStyleSheet("QProgressBar::chunk {background: #0f0;}")
        self.threat_bar.setFormat(f"{threat_text} ({data['threat']}%)")
        
        # Fatigue
        drowsiness = max(0, min(100, (0.3 - data['ear']) * 400))
        self.fatigue_bar.setValue(int(drowsiness))
        self.fatigue_bar.setFormat(f"{drowsiness:.0f}%")
        
        # Detections
        self.stats['detections'] += data['total_count']
        self.detection_label.setText(f"Detections: {data['total_count']} (Total: {self.stats['detections']})")
        
        # Zone
        self.zone_label.setText(f"Zone: {data['zone']}")
        self.zone_label.setStyleSheet(f"font: bold 12pt; color: {'#f00' if 'SCHOOL' in data['zone'] else '#0f0'};")
        
        # Turn
        self.turn_label.setText(f"Turn: {data['turn']} | Side Scan: {'ON' if data['turn'] != 'STRAIGHT' else 'OFF'}")
        
        # Helmet
        self.helmet_label.setText(f"Helmet: {data['helmet_status']}")
        if data['helmet_status'] == 'NO HELMET':
            self.helmet_label.setStyleSheet("font: bold 12pt; color: #f00;")
        elif data['helmet_status'] == 'HELMET OK':
            self.helmet_label.setStyleSheet("font: bold 12pt; color: #0f0;")
        else:
            self.helmet_label.setStyleSheet("font: bold 12pt; color: #fff;")
        
        # Blind Spot Indicators
        if data['blind_spot_left']:
            self.blind_left_label.setText("⬅ LEFT: ⚠ VEHICLE!")
            self.blind_left_label.setStyleSheet("font: bold 11pt; color: #f00; background: #500;")
            # Beep warning
            try:
                winsound.Beep(800, 200)
            except:
                pass
        else:
            self.blind_left_label.setText("⬅ LEFT: ✓")
            self.blind_left_label.setStyleSheet("font: bold 11pt; color: #0f0;")
        
        if data['blind_spot_right']:
            self.blind_right_label.setText("RIGHT: ⚠ VEHICLE! ➡")
            self.blind_right_label.setStyleSheet("font: bold 11pt; color: #f00; background: #500;")
            # Beep warning
            try:
                winsound.Beep(800, 200)
            except:
                pass
        else:
            self.blind_right_label.setText("RIGHT: ✓ ➡")
            self.blind_right_label.setStyleSheet("font: bold 11pt; color: #0f0;")
        
        # Process violations
        for violation in data['violations']:
            self.log_violation(violation['type'], violation['details'], violation['severity'])
            self.add_log(f"⚠ {violation['type']}: {violation['details']}")
        
        # Trigger alert if needed
        if data['threat'] > 75 or drowsiness > 70:
            self.trigger_alert()
            self.add_log(f"🚨 ALERT: Threat {data['threat']}% | Drowsiness {drowsiness:.0f}%")
        
        # Update detection logs
        for det in data['detections']:
            if det['threat'] > 60:
                self.add_log(f"🎯 {det['class']}: {det['dist']:.1f}m (Threat: {det['threat']}%)")
        
        self.stats_label.setText(f"Detections: {self.stats['detections']} | Alerts: {self.stats['alerts']} | Violations: {self.stats['violations']}")
    
    def trigger_alert(self):
        if self.alert_active:
            return
        
        self.alert_active = True
        self.stats['alerts'] += 1
        
        # Visual
        self.alert_label.setText("🔴 CRITICAL ALERT!")
        self.alert_label.setStyleSheet("font: bold 14pt; color: #f00; padding: 10px; border: 2px solid #f00; background: #500;")
        
        # Audio
        try:
            winsound.Beep(1000, 500)
        except:
            pass
        
        self.add_log("� Critical alert triggered!")
        
        QTimer.singleShot(3000, self.reset_alert)
    
    def reset_alert(self):
        self.alert_active = False
        self.alert_label.setText("⚫ Status: Normal")
        self.alert_label.setStyleSheet("font: bold 14pt; color: #0f0; padding: 10px; border: 2px solid #0f0;")
    
    def trigger_test_alert(self):
        self.add_log("🧪 Manual test alert triggered")
        self.trigger_alert()
    
    def log_violation(self, vtype, details, severity):
        self.stats['violations'] += 1
        try:
            conn = sqlite3.connect('violations.db')
            c = conn.cursor()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO violations (timestamp, violation_type, details, severity) VALUES (?, ?, ?, ?)",
                     (ts, vtype, details, severity))
            conn.commit()
            conn.close()
            self.load_violations_table()
        except Exception as e:
            print(f"DB error: {e}")
    
    def load_violations_table(self):
        try:
            conn = sqlite3.connect('violations.db')
            c = conn.cursor()
            c.execute("SELECT timestamp, violation_type, details, severity FROM violations ORDER BY id DESC LIMIT 10")
            rows = c.fetchall()
            conn.close()
            
            self.violations_table.setRowCount(len(rows))
            for row_idx, row in enumerate(rows):
                for col_idx, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    if col_idx == 3:  # Severity column
                        if value == 'CRITICAL':
                            item.setForeground(QColor(255, 0, 0))
                        elif value == 'HIGH':
                            item.setForeground(QColor(255, 165, 0))
                    self.violations_table.setItem(row_idx, col_idx, item)
        except Exception as e:
            print(f"Table load error: {e}")
    
    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.event_logs.append(log_entry)
        if len(self.event_logs) > 20:
            self.event_logs.pop(0)
        self.log_text.setPlainText("\n".join(self.event_logs))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def export_logs(self):
        try:
            conn = sqlite3.connect('violations.db')
            c = conn.cursor()
            c.execute("SELECT * FROM violations")
            rows = c.fetchall()
            conn.close()
            
            filename = f"violations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w') as f:
                f.write("ID,Timestamp,Type,Details,Severity\n")
                for row in rows:
                    f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
            
            self.add_log(f"✅ Exported {len(rows)} violations to {filename}")
            QMessageBox.information(self, "Export Complete", f"Exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def stop_camera(self):
        self.worker.running = False
        self.worker.wait()
        self.add_log("Camera stopped")
        self.stop_btn.setEnabled(False)
    
    def closeEvent(self, event):
        self.worker.running = False
        self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartVehicleApp()
    window.show()
    sys.exit(app.exec())
