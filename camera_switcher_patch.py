"""
Camera Switcher Patch for SmartVehicle v3.0
Add these methods to your SmartVehicleApp_v3 class
"""

# METHOD 1: Add to SmartVehicleApp_v3 class
def _switch_camera(self):
    """Switch to selected camera - Real-time camera switching"""
    new_camera_index = self.camera_combo.currentData()
    
    if new_camera_index == self.camera_index:
        self.add_log("⚠ Already using this camera")
        return
    
    self.add_log(f"🔄 Switching to camera {new_camera_index}...")
    
    # Stop current worker thread
    self.worker.running = False
    self.worker.wait(1000)  # Wait max 1 second
    
    # Update camera index
    self.camera_index = new_camera_index
    
    # Create and start new worker with new camera
    self.worker = EnhancedCameraWorker(camera_index=self.camera_index, config=self.config)
    self.worker.frame_ready.connect(self.update_frame)
    self.worker.metrics_ready.connect(self.update_metrics)
    self.worker.start()
    
    # Update UI
    cam_name = self.camera_combo.currentText()
    self.add_log(f"✓ Successfully switched to {cam_name}")
    self.setWindowTitle(f"SmartVehicle Intelligence v3.0 Enterprise - {cam_name}")


# UI ADDITION: Add to setup_ui() method after export button
"""
# Camera Switcher UI (add after self.export_btn)
from PySide6.QtWidgets import QComboBox

# Create horizontal layout for camera selector
camera_layout = QHBoxLayout()

# Label
camera_label = QLabel("📹 Camera:")
camera_label.setStyleSheet("color: #0ff; font-weight: bold;")
camera_layout.addWidget(camera_label)

# Dropdown
self.camera_combo = QComboBox()
self.camera_combo.setStyleSheet('''
    QComboBox {
        background: #333;
        color: #fff;
        padding: 5px;
        border: 1px solid #0ff;
        border-radius: 3px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #0ff;
    }
''')

# Populate with cameras
cameras = find_cameras()
for cam in cameras:
    self.camera_combo.addItem(cam['name'], cam['index'])

# Select current camera
current_idx = self.camera_combo.findData(self.camera_index)
if current_idx >= 0:
    self.camera_combo.setCurrentIndex(current_idx)

camera_layout.addWidget(self.camera_combo)

# Switch button
self.switch_camera_btn = QPushButton("🔄 Switch")
self.switch_camera_btn.clicked.connect(self._switch_camera)
self.switch_camera_btn.setStyleSheet('''
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #07a, stop:1 #055);
        padding: 5px 15px;
        border-radius: 3px;
    }
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #09c, stop:1 #067);
    }
''')
camera_layout.addWidget(self.switch_camera_btn)

# Add to button layout
btn_layout.addLayout(camera_layout, 2, 0, 1, 3)

# Move stop button to row 3
# Change stop button line:
# btn_layout.addWidget(self.stop_btn, 3, 1, 1, 2)
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║  Camera Switcher Patch - Installation Instructions          ║
╚══════════════════════════════════════════════════════════════╝

STEP 1: Add _switch_camera() method
-------------------------------------
Copy the _switch_camera() method from above and add it to
your SmartVehicleApp_v3 class (around line 1100).

STEP 2: Add UI elements to setup_ui()
---------------------------------------
In the setup_ui() method, after the line:
    btn_layout.addWidget(self.export_btn, 1, 0)

Add the camera switcher UI code from the docstring above.

STEP 3: Test it!
-----------------
1. Restart the app: python main_v3.py
2. You'll see a camera dropdown and "Switch" button
3. Select different camera and click Switch
4. Camera feed will change in real-time!

FEATURES:
✅ Real-time camera switching (no restart needed)
✅ Shows all available cameras
✅ Smooth transition
✅ Logs camera changes
✅ Updates window title

""")
