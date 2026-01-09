"""
PMO Installation & Deployment Suite
Complete installation guide for SBS Transit/SMRT bus deployment
Matches TNT Surveillance PMO-Engineer JD requirements
"""

import os
from datetime import datetime
from pathlib import Path
import json


class PMOInstallationSuite:
    """
    Installation suite for SmartBus Intelligence Platform
    Generates comprehensive installation documentation
    """
    
    def __init__(self, bus_id, depot, installer_name=""):
        self.bus_id = bus_id
        self.depot = depot
        self.installer_name = installer_name
        self.timestamp = datetime.now()
        
        print("=" * 70)
        print("🔧 PMO INSTALLATION SUITE")
        print("=" * 70)
        print(f"Bus ID: {bus_id}")
        print(f"Depot: {depot}")
        print(f"Installer: {installer_name}")
        print(f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)
        
    def generate_bom(self):
        """Generate Bill of Materials"""
        print("\n📋 BILL OF MATERIALS (BOM)")
        print("-" * 70)
        
        bom = {
            "Hardware": {
                "Cameras": [
                    {"item": "USB Camera 1080P 30FPS", "qty": 4, "location": "Front/Rear/Left/Right"},
                    {"item": "Camera Mounts (Adhesive)", "qty": 4, "cost_sgd": 50}
                ],
                "Processing": [
                    {"item": "Raspberry Pi 4 Model B (8GB)", "qty": 1, "cost_sgd": 120},
                    {"item": "Power Supply 5V 3A USB-C", "qty": 1, "cost_sgd": 15},
                    {"item": "MicroSD Card 128GB", "qty": 1, "cost_sgd": 30}
                ],
                "Alerting": [
                    {"item": "GPIO Buzzer Module", "qty": 1, "cost_sgd": 5},
                    {"item": "LED Indicator (Red)", "qty": 3, "cost_sgd": 10},
                    {"item": "Vibration Motor", "qty": 1, "cost_sgd": 5}
                ],
                "Wiring": [
                    {"item": "USB Extension Cables 5m", "qty": 4, "cost_sgd": 40},
                    {"item": "GPIO Jumper Wires", "qty": 1, "cost_sgd": 10},
                    {"item": "Cable Ties & Clips", "qty": 1, "cost_sgd": 15}
                ],
                "Networking": [
                    {"item": "4G LTE Modem", "qty": 1, "cost_sgd": 80},
                    {"item": "SIM Card (Data Plan)", "qty": 1, "cost_sgd": 30}
                ],
                "Optional": [
                    {"item": "GPS Module", "qty": 1, "cost_sgd": 25},
                    {"item": "CAN Bus Interface", "qty": 1, "cost_sgd": 50}
                ]
            },
            "Software": {
                "OS": "Raspberry Pi OS 64-bit (Bullseye)",
                "Python": "3.10+",
                "Libraries": "See requirements.txt"
            }
        }
        
        total_cost = 0
        for category, items in bom["Hardware"].items():
            print(f"\n{category}:")
            for item in items:
                if "cost_sgd" in item:
                    total_cost += item["cost_sgd"] * item.get("qty", 1)
                    print(f"  [{item['qty']}x] {item['item']} - SGD ${item['cost_sgd']}")
                else:
                    print(f"  [{item['qty']}x] {item['item']}")
        
        print(f"\n{'=' * 70}")
        print(f"ESTIMATED TOTAL COST PER BUS: SGD ${total_cost:.2f}")
        print(f"{'=' * 70}")
        
        return bom
        
    def generate_wiring_diagram(self):
        """Generate wiring diagram instructions"""
        print("\n⚡ WIRING DIAGRAM")
        print("-" * 70)
        
        wiring = {
            "GPIO Pin Assignments": {
                "GPIO 18 (Pin 12)": "Buzzer Module (+)",
                "GPIO 19 (Pin 35)": "LED Strobe",
                "GPIO 20 (Pin 38)": "Vibration Motor",
                "GND (Pin 6)": "Common Ground",
                "5V (Pin 2)": "Power to Modules"
            },
            "Camera Connections": {
                "USB Port 1": "Front Camera (1080P)",
                "USB Port 2": "Rear Camera (1080P)",
                "USB Port 3": "Left Side Camera (1080P)",
                "USB Port 4": "Right Side Camera (1080P)"
            },
            "Power": {
                "Bus 12V DC": "→ DC-DC Converter (12V to 5V 3A) → Raspberry Pi USB-C",
                "Fuse": "5A Inline Fuse on 12V line",
                "Ground": "Connect to bus chassis ground"
            },
            "Network": {
                "Ethernet": "LTE Modem → Raspberry Pi Ethernet Port",
                "WiFi": "Raspberry Pi built-in WiFi for local debugging"
            }
        }
        
        for section, connections in wiring.items():
            print(f"\n{section}:")
            for pin, function in connections.items():
                print(f"  {pin:20} → {function}")
        
        print("\n⚠️  SAFETY WARNINGS:")
        print("  1. Disconnect bus battery before installation")
        print("  2. Use proper insulation on all connections")
        print("  3. Secure all cables away from moving parts")
        print("  4. Test all connections before final assembly")
        
        return wiring
        
    def generate_installation_steps(self):
        """Step-by-step installation procedure"""
        print("\n📝 INSTALLATION PROCEDURE")
        print("-" * 70)
        
        steps = [
            {
                "step": 1,
                "task": "Pre-Installation Checks",
                "details": [
                    "Verify bus power is OFF",
                    "Collect all tools and components",
                    "Review wiring diagram",
                    "Prepare mounting locations"
                ],
                "duration": "15 min"
            },
            {
                "step": 2,
                "task": "Mount Cameras",
                "details": [
                    "Clean camera mounting surfaces with alcohol",
                    "Front camera: Above windshield, center",
                    "Rear camera: Above rear window, center",
                    "Side cameras: On mirrors or pillars at 1.8m height",
                    "Secure with adhesive mounts + cable ties"
                ],
                "duration": "30 min"
            },
            {
                "step": 3,
                "task": "Route Camera Cables",
                "details": [
                    "Run USB cables through ceiling channels",
                    "Avoid pinch points and moving parts",
                    "Secure every 30cm with cable ties",
                    "Leave 20cm service loop at each end"
                ],
                "duration": "45 min"
            },
            {
                "step": 4,
                "task": "Install Raspberry Pi",
                "details": [
                    "Mount Pi in driver cabin control box",
                    "Ensure ventilation clearance (5cm)",
                    "Connect USB cameras to Pi",
                    "Install GPIO modules (buzzer, LED, motor)"
                ],
                "duration": "20 min"
            },
            {
                "step": 5,
                "task": "Wire GPIO Alerts",
                "details": [
                    "Connect GPIO 18 → Buzzer (+)",
                    "Connect GPIO 19 → LED (+)",
                    "Connect GPIO 20 → Vibration Motor (+)",
                    "Connect all (-) to GND pin",
                    "Test each output with multimeter (3.3V)"
                ],
                "duration": "25 min"
            },
            {
                "step": 6,
                "task": "Power Connection",
                "details": [
                    "Install DC-DC converter near Pi",
                    "Connect bus 12V → Converter input (with 5A fuse)",
                    "Connect converter 5V output → Pi USB-C",
                    "Verify voltage: 5.1V ± 0.1V at Pi input"
                ],
                "duration": "20 min"
            },
            {
                "step": 7,
                "task": "Network Setup",
                "details": [
                    "Install LTE modem, insert SIM card",
                    "Connect modem to Pi Ethernet port",
                    "Configure APN settings in Pi",
                    "Test internet: ping google.com"
                ],
                "duration": "15 min"
            },
            {
                "step": 8,
                "task": "Software Installation",
                "details": [
                    "Boot Raspberry Pi",
                    "Install Raspberry Pi OS (if not pre-loaded)",
                    "Clone TNT repository: git clone ...",
                    "Install dependencies: pip install -r requirements.txt",
                    "Configure bus_id and depot in config.json",
                    "Start detection engine: python unified_detection_engine.py"
                ],
                "duration": "30 min"
            },
            {
                "step": 9,
                "task": "System Commissioning",
                "details": [
                    "Run commissioning tests (see pmo_testing.py)",
                    "Test all 4 cameras: verify 1080P 30FPS",
                    "Test blind spot detection: walk in front",
                    "Test traffic violation detection: simulate scenarios",
                    "Test GPIO alerts: verify buzzer, LED, vibration",
                    "Test network: verify OC dashboard connection",
                    "Calibrate distance: measure known distances"
                ],
                "duration": "45 min"
            },
            {
                "step": 10,
                "task": "Final Inspection & Handover",
                "details": [
                    "Verify all cables secured and labeled",
                    "Check no loose wires or connections",
                    "Clean installation area",
                    "Generate test report (PDF)",
                    "Get driver signature on handover form",
                    "Upload to fleet management system"
                ],
                "duration": "20 min"
            }
        ]
        
        total_time = 0
        for step in steps:
            print(f"\nStep {step['step']}: {step['task']} ({step['duration']})")
            for detail in step["details"]:
                print(f"  • {detail}")
            
            # Parse duration
            duration_min = int(step['duration'].split()[0])
            total_time += duration_min
        
        print(f"\n{'=' * 70}")
        print(f"ESTIMATED TOTAL INSTALLATION TIME: {total_time} minutes ({total_time/60:.1f} hours)")
        print(f"{'=' * 70}")
        
        return steps
        
    def generate_testing_checklist(self):
        """Generate commissioning test checklist"""
        print("\n✅ COMMISSIONING TEST CHECKLIST")
        print("-" * 70)
        
        tests = {
            "Camera Tests": [
                "Front camera: 1080P @ 30FPS - PASS/FAIL",
                "Rear camera: 1080P @ 30FPS - PASS/FAIL",
                "Left camera: 1080P @ 30FPS - PASS/FAIL",
                "Right camera: 1080P @ 30FPS - PASS/FAIL",
                "All cameras: No distortion or dead pixels - PASS/FAIL"
            ],
            "Detection Tests": [
                "Blind spot detection: Person at 1.5m - PASS/FAIL",
                "Distance accuracy: ±5cm at known distance - PASS/FAIL",
                "Helmet detection: Identify no-helmet - PASS/FAIL",
                "Seatbelt detection: Identify no-seatbelt - PASS/FAIL",
                "Speed estimation: ±2 km/h accuracy - PASS/FAIL",
                "Lane detection: Wrong-way traffic - PASS/FAIL",
                "ANPR: License plate recognition - PASS/FAIL"
            ],
            "Alert Tests": [
                "GPIO Buzzer: Sounds on threat - PASS/FAIL",
                "GPIO LED: Flashes on threat - PASS/FAIL",
                "GPIO Vibration: Pulses on threat - PASS/FAIL",
                "Voice alert: Clear audio output - PASS/FAIL"
            ],
            "Network Tests": [
                "LTE connection: Internet accessible - PASS/FAIL",
                "OC dashboard: WebSocket streaming - PASS/FAIL",
                "Database logging: Detections recorded - PASS/FAIL",
                "Video upload: MDVR clips saved - PASS/FAIL"
            ],
            "Integration Tests": [
                "CAN bus: Speed reading correct - PASS/FAIL",
                "GPS: Location accuracy ±5m - PASS/FAIL",
                "Geofencing: School zone threshold - PASS/FAIL",
                "Fatigue: T-SEEDS prediction working - PASS/FAIL"
            ],
            "Performance Tests": [
                "FPS: Sustained 30 FPS for 5 min - PASS/FAIL",
                "Latency: Detection to alert <100ms - PASS/FAIL",
                "CPU: <80% utilization - PASS/FAIL",
                "Memory: <70% utilization - PASS/FAIL",
                "Temperature: Pi CPU <70°C - PASS/FAIL"
            ]
        }
        
        for category, test_items in tests.items():
            print(f"\n{category}:")
            for test in test_items:
                print(f"  [ ] {test}")
        
        print(f"\n{'=' * 70}")
        print("TESTER SIGNATURE: ________________  DATE: ________")
        print("SUPERVISOR APPROVAL: ________________  DATE: ________")
        print(f"{'=' * 70}")
        
        return tests
        
    def generate_troubleshooting_guide(self):
        """Generate troubleshooting guide"""
        print("\n🔧 TROUBLESHOOTING GUIDE")
        print("-" * 70)
        
        issues = {
            "Camera Not Detected": [
                "Check USB cable connection",
                "Try different USB port",
                "Test camera on PC: lsusb or Device Manager",
                "Replace camera if defective"
            ],
            "No Detection Output": [
                "Verify YOLO model loaded: check models/ folder",
                "Check Python dependencies: pip list",
                "Review logs: /var/log/tnt_detection.log",
                "Restart service: sudo systemctl restart tnt-detection"
            ],
            "GPIO Alerts Not Working": [
                "Check GPIO pin connections",
                "Test with Python: RPi.GPIO.setmode(BCR.BCM)",
                "Verify voltage: 3.3V on GPIO pins",
                "Check module power supply"
            ],
            "Network Connection Lost": [
                "Check LTE modem status LED",
                "Verify SIM card inserted correctly",
                "Test with: ping 8.8.8.8",
                "Check APN settings in modem config"
            ],
            "High CPU Usage": [
                "Check FPS settings: reduce to 15 FPS",
                "Use YOLOv8n (nano) model, not larger",
                "Disable unused features in config.json",
                "Check for runaway processes: htop"
            ],
            "Inaccurate Distance": [
                "Recalibrate camera: measure known distances",
                "Update pixels_per_meter in config",
                "Check camera lens for dirt/scratches",
                "Verify camera angle: should be level"
            ]
        }
        
        for issue, solutions in issues.items():
            print(f"\n⚠️  {issue}:")
            for i, solution in enumerate(solutions, 1):
                print(f"  {i}. {solution}")
        
        print(f"\n{'=' * 70}")
        print("For additional support:")
        print("  Email: support@tntsurveillance.com")
        print("  Phone: +65 6xxx xxxx")
        print("  Emergency: +65 9xxx xxxx (24/7)")
        print(f"{'=' * 70}")
        
        return issues
        
    def generate_pdf_report(self, filename=None):
        """Generate PDF installation report"""
        if filename is None:
            filename = f"Installation_Report_{self.bus_id}_{self.timestamp.strftime('%Y%m%d')}.pdf"
        
        print(f"\n📄 Generating PDF report: {filename}")
        
        # Here you would use reportlab to generate actual PDF
        # For now, generate text version
        
        report = {
            "bus_id": self.bus_id,
            "depot": self.depot,
            "installer": self.installer_name,
            "timestamp": self.timestamp.isoformat(),
            "bom": self.generate_bom(),
            "wiring": self.generate_wiring_diagram(),
            "steps": self.generate_installation_steps(),
            "tests": self.generate_testing_checklist(),
            "troubleshooting": self.generate_troubleshooting_guide()
        }
        
        # Save as JSON
        json_filename = filename.replace('.pdf', '.json')
        with open(json_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Report saved: {json_filename}")
        print(f"  (PDF generation requires reportlab library)")
        
        return json_filename
        
    def run_full_suite(self):
        """Execute complete installation suite"""
        print("\n" + "=" * 70)
        print("EXECUTING FULL INSTALLATION SUITE")
        print("=" * 70)
        
        self.generate_bom()
        self.generate_wiring_diagram()
        self.generate_installation_steps()
        self.generate_testing_checklist()
        self.generate_troubleshooting_guide()
        
        report_file = self.generate_pdf_report()
        
        print("\n" + "=" * 70)
        print("✅ INSTALLATION SUITE COMPLETE")
        print("=" * 70)
        print(f"Installation report: {report_file}")
        print(f"Ready for bus {self.bus_id} deployment at {self.depot} depot")
        print("=" * 70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PMO Installation Suite')
    parser.add_argument('--bus', type=str, required=True, help='Bus ID (e.g., SGP1234)')
    parser.add_argument('--depot', type=str, required=True, help='Depot location')
    parser.add_argument('--installer', type=str, default='', help='Installer name')
    parser.add_argument('--generate-pdf', action='store_true', help='Generate PDF report')
    
    args = parser.parse_args()
    
    suite = PMOInstallationSuite(
        bus_id=args.bus,
        depot=args.depot,
        installer_name=args.installer
    )
    
    suite.run_full_suite()


if __name__ == '__main__':
    main()

