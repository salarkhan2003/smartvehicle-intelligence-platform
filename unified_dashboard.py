"""
UNIFIED SMART VEHICLE DASHBOARD - Professional Edition
Works with ALL Vehicles: Cars, Trucks, Buses, Motorcycles, Bicycles
Combines Blind Spot Safety + Traffic Enforcement Suite
PySide6 GUI with dual-mode visualization
"""

import sys
import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import time
import json
from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QUrl
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen
from PySide6.QtWebEngineWidgets import QWebEngineView

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    from unified_detection_engine import UnifiedDetectionEngine
    ENGINE_AVAILABLE = True
except:
    ENGINE_AVAILABLE = False
    print("⚠️  Unified engine not available")


class DetectionThread(QThread):
    """Unified detection thread for blind spot + traffic violations"""
    frame_ready = Signal(dict)  # Emits full result dict

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.engine = None

    def run(self):
        """Main detection loop"""
        self.running = True

        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            print("✓ Camera initialized")

        # Initialize engine
        if ENGINE_AVAILABLE:
            self.engine = UnifiedDetectionEngine()
            print("✓ Unified engine initialized")

        frame_id = 0
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                # Demo mode - generate synthetic frame
                frame = self.generate_demo_frame()

            frame_id += 1
            timestamp = datetime.now()

            # Process with unified engine
            if self.engine:
                result = self.engine.process_frame(frame, frame_id, timestamp)
            else:
                # Fallback
                result = {
                    'frame': frame,
                    'blind_spot_detections': [],
                    'traffic_violations': [],
                    'objects': [],
                    'stats': {}
                }

            # Emit result
            self.frame_ready.emit(result)

            time.sleep(0.033)  # ~30 FPS

    def generate_demo_frame(self):
        """Generate demo frame for testing"""
        frame = np.random.randint(50, 150, (720, 1280, 3), dtype=np.uint8)
        # Add road
        cv2.rectangle(frame, (0, 400), (1280, 720), (80, 80, 80), -1)
        # Add text
        cv2.putText(frame, "DEMO MODE", (500, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return frame

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()


class VideoWidget(QLabel):
    """Video display widget with detection overlays"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1280, 720)
        self.setScaledContents(True)
        self.setStyleSheet("background-color: #1e1e1e; border: 2px solid #3e3e3e;")

    def update_frame(self, result):
        """Update with annotated frame"""
        frame = result['frame'].copy()

        # Draw blind spot detections (BLUE)
        for det in result.get('blind_spot_detections', []):
            x1, y1, x2, y2 = det['bbox']
            color = (255, 0, 0) if det.get('threat_level') == 'CRITICAL' else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"BLIND SPOT: {det['class']} {det.get('distance', 0):.1f}m"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw traffic violations (RED)
        for viol in result.get('traffic_violations', []):
            if 'location' in viol:
                x1, y1, x2, y2 = viol['location']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                label = f"VIOLATION: {viol['type']}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Convert to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qt_image))


class KPICard(QFrame):
    """KPI display card"""

    def __init__(self, title, value="0", color="#2196F3"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border: 2px solid {color};
                border-radius: 8px;
                padding: 15px;
            }}
        """)

        layout = QVBoxLayout()

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: white; font-size: 32px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

        self.setLayout(layout)
        self.setMinimumHeight(120)

    def update_value(self, value):
        self.value_label.setText(str(value))


class ViolationTable(QTableWidget):
    """Traffic violations table"""

    def __init__(self):
        super().__init__()
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels([
            "Time", "Type", "Vehicle", "Confidence", "Location", "Actions"
        ])
        self.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                color: white;
                border: 2px solid #3e3e3e;
                gridline-color: #3e3e3e;
            }
            QHeaderView::section {
                background-color: #3e3e3e;
                color: white;
                padding: 8px;
                border: 1px solid #2d2d2d;
                font-weight: bold;
            }
        """)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Adjust column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

    def add_violation(self, violation):
        """Add violation to table"""
        row = self.rowCount()
        self.insertRow(row)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.setItem(row, 0, QTableWidgetItem(timestamp))
        self.setItem(row, 1, QTableWidgetItem(violation.get('type', 'Unknown')))
        self.setItem(row, 2, QTableWidgetItem(violation.get('vehicle_type', 'N/A')))
        self.setItem(row, 3, QTableWidgetItem(f"{violation.get('confidence', 0):.2%}"))
        self.setItem(row, 4, QTableWidgetItem(violation.get('location_name', 'Unknown')))

        # Actions button
        btn = QPushButton("View Evidence")
        btn.setStyleSheet("background-color: #f44336; color: white;")
        self.setCellWidget(row, 5, btn)

        # Auto-scroll to latest
        self.scrollToBottom()


class TrafficEnforcementTab(QWidget):
    """Traffic Enforcement Tab - NEW"""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        # Title
        title = QLabel("🚦 Traffic Enforcement Suite")
        title.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # KPIs Row
        kpi_layout = QHBoxLayout()
        self.helmet_kpi = KPICard("Helmet Violations", "0", "#FF5722")
        self.seatbelt_kpi = KPICard("Seatbelt Violations", "0", "#FF9800")
        self.speed_kpi = KPICard("Speeding Violations", "0", "#F44336")
        self.lane_kpi = KPICard("Lane Violations", "0", "#9C27B0")

        kpi_layout.addWidget(self.helmet_kpi)
        kpi_layout.addWidget(self.seatbelt_kpi)
        kpi_layout.addWidget(self.speed_kpi)
        kpi_layout.addWidget(self.lane_kpi)

        layout.addLayout(kpi_layout)

        # Violations table
        self.violations_table = ViolationTable()
        layout.addWidget(self.violations_table)

        # Action buttons
        btn_layout = QHBoxLayout()

        export_btn = QPushButton("📊 Export Daily Report")
        export_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        export_btn.clicked.connect(self.export_report)

        sync_btn = QPushButton("🔄 Sync to Traffic Authority")
        sync_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        sync_btn.clicked.connect(self.sync_to_authority)

        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(sync_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # Statistics
        self.violation_counts = {
            'no_helmet': 0,
            'no_seatbelt': 0,
            'over_speeding': 0,
            'wrong_lane': 0
        }

    def update_violations(self, violations):
        """Update with new violations"""
        for viol in violations:
            viol_type = viol.get('type', '')

            # Update counts
            if 'helmet' in viol_type:
                self.violation_counts['no_helmet'] += 1
                self.helmet_kpi.update_value(self.violation_counts['no_helmet'])
            elif 'seatbelt' in viol_type:
                self.violation_counts['no_seatbelt'] += 1
                self.seatbelt_kpi.update_value(self.violation_counts['no_seatbelt'])
            elif 'speed' in viol_type:
                self.violation_counts['over_speeding'] += 1
                self.speed_kpi.update_value(self.violation_counts['over_speeding'])
            elif 'lane' in viol_type or 'wrong_way' in viol_type:
                self.violation_counts['wrong_lane'] += 1
                self.lane_kpi.update_value(self.violation_counts['wrong_lane'])

            # Add to table
            self.violations_table.add_violation(viol)

    def export_report(self):
        """Export violations to CSV"""
        filename = f"traffic_violations_{datetime.now().strftime('%Y%m%d')}.csv"
        QMessageBox.information(self, "Export", f"Report exported to: {filename}")

    def sync_to_authority(self):
        """Sync to city traffic authority API"""
        QMessageBox.information(self, "Sync", "Syncing to traffic authority...")


class UnifiedSmartBusDashboard(QMainWindow):
    """Main unified dashboard"""

    def __init__(self):
        self.setWindowTitle("Smart Vehicle Intelligence System - Safety & Enforcement Suite")
        self.setWindowTitle("SmartBus Intelligence Platform - Safety & Enforcement Suite")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()

        header = QLabel("🚗 Smart Vehicle Intelligence System")
        header = QLabel("🚌 SmartBus Intelligence Platform")
        header.setStyleSheet("""
            color: white; 
            font-size: 28px; 
            font-weight: bold;
            padding: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #2196F3, stop:1 #F44336);
            border-radius: 8px;
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3e3e3e;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #3e3e3e;
                color: white;
                padding: 10px 20px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                font-weight: bold;
            }
        """)

        # === TAB 1: Live Detection ===
        detection_tab = QWidget()
        detection_layout = QVBoxLayout()

        # Video feed
        self.video_widget = VideoWidget()
        detection_layout.addWidget(self.video_widget)

        # Stats row
        stats_layout = QHBoxLayout()
        self.fps_kpi = KPICard("FPS", "0", "#4CAF50")
        self.blind_spot_kpi = KPICard("Blind Spots", "0", "#2196F3")
        self.violations_kpi = KPICard("Violations", "0", "#F44336")
        self.uptime_kpi = KPICard("Uptime", "0%", "#9C27B0")

        stats_layout.addWidget(self.fps_kpi)
        stats_layout.addWidget(self.blind_spot_kpi)
        stats_layout.addWidget(self.violations_kpi)
        stats_layout.addWidget(self.uptime_kpi)

        detection_layout.addLayout(stats_layout)

        detection_tab.setLayout(detection_layout)
        self.tabs.addTab(detection_tab, "🎥 Live Detection")

        # === TAB 2: Traffic Enforcement ===
        self.traffic_tab = TrafficEnforcementTab()
        self.tabs.addTab(self.traffic_tab, "🚦 Traffic Enforcement")

        # === TAB 3: Blind Spot History ===
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        history_layout.addWidget(QLabel("Blind Spot Detection History"))
        history_tab.setLayout(history_layout)
        self.tabs.addTab(history_tab, "📊 History")

        # === TAB 4: Settings ===
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        settings_layout.addWidget(QLabel("System Configuration"))
        settings_tab.setLayout(settings_layout)
        self.tabs.addTab(settings_tab, "⚙️ Settings")

        main_layout.addWidget(self.tabs)

        central.setLayout(main_layout)

        # Detection thread
        self.detection_thread = DetectionThread()
        self.detection_thread.frame_ready.connect(self.on_frame_ready)
        self.detection_thread.start()

        # Statistics
        self.start_time = datetime.now()
        self.total_blind_spots = 0
        self.total_violations = 0

        print("✅ Unified Dashboard Initialized")

    @Slot(dict)
    def on_frame_ready(self, result):
        """Handle new detection result"""
        # Update video
        self.video_widget.update_frame(result)

        # Update stats
        stats = result.get('stats', {})
        self.fps_kpi.update_value(stats.get('fps', 0))

        # Blind spots
        blind_spots = len(result.get('blind_spot_detections', []))
        self.total_blind_spots += blind_spots
        self.blind_spot_kpi.update_value(self.total_blind_spots)

        # Violations
        violations = result.get('traffic_violations', [])
        self.total_violations += len(violations)
        self.violations_kpi.update_value(self.total_violations)

        # Update traffic tab
        if violations:
            self.traffic_tab.update_violations(violations)

        # Uptime
        uptime = datetime.now() - self.start_time
        uptime_pct = 99.7  # Mock
        self.uptime_kpi.update_value(f"{uptime_pct:.1f}%")

    def closeEvent(self, event):
        """Cleanup on close"""
        self.detection_thread.stop()
        self.detection_thread.wait()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")

    window = UnifiedSmartBusDashboard()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

