#!/usr/bin/env python3
"""
SmartVehicle Intelligence System v3.0 - Dependency Installer
Automatically installs all required packages for the system
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_package(package):
    """Install a single package"""
    print(f"📦 Installing {package}...")
    success, output = run_command(f"pip install {package}")
    if success:
        print(f"✅ {package} installed successfully")
        return True
    else:
        print(f"❌ Failed to install {package}: {output}")
        return False

def install_requirements():
    """Install all requirements from requirements.txt"""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    print("📋 Installing from requirements.txt...")
    success, output = run_command("pip install -r requirements.txt")
    if success:
        print("✅ All requirements installed successfully")
        return True
    else:
        print(f"❌ Failed to install requirements: {output}")
        return False

def verify_installations():
    """Verify that key packages are installed correctly"""
    
    print("\n🔍 Verifying installations...")
    
    packages_to_test = [
        ('cv2', 'OpenCV'),
        ('ultralytics', 'YOLOv8'),
        ('easyocr', 'EasyOCR'),
        ('mediapipe', 'MediaPipe'),
        ('PySide6', 'PySide6'),
        ('psutil', 'psutil'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy')
    ]
    
    all_good = True
    
    for package, name in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - FAILED")
            all_good = False
    
    # Test optional packages
    optional_packages = [
        ('pyttsx3', 'Text-to-Speech'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib')
    ]
    
    print("\n🔧 Optional packages:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name} - Available")
        except ImportError:
            print(f"⚠️  {name} - Not available (optional)")
    
    return all_good

def download_yolo_model():
    """Download YOLOv8 model if not present"""
    print("\n🤖 Checking YOLO model...")
    
    if os.path.exists('yolov8n.pt'):
        print("✅ YOLOv8n model already exists")
        return True
    
    print("📥 Downloading YOLOv8n model (first time only)...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download the model
        print("✅ YOLOv8n model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'data',
        'data/recordings',
        'data/snapshots',
        'data/plates',
        'data/logs',
        'data/test_plates',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def main():
    """Main installation process"""
    
    print("🚗 SmartVehicle Intelligence System v3.0")
    print("🔧 Dependency Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip first
    print("\n📦 Upgrading pip...")
    run_command("python -m pip install --upgrade pip")
    
    # Install requirements
    print("\n📋 Installing dependencies...")
    if not install_requirements():
        print("\n❌ Installation failed!")
        print("Try installing manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify installations
    if not verify_installations():
        print("\n⚠️  Some packages failed to install properly")
        print("The system may still work with reduced functionality")
    
    # Download YOLO model
    download_yolo_model()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("\n🚀 Next steps:")
    print("   1. Run the system: python main_v3.py")
    print("   2. Test OCR: python test_ocr_anpr.py")
    print("   3. Test animals: python test_animal_detection.py")
    print("\n📚 Documentation:")
    print("   - README.md - System overview")
    print("   - QUICKSTART_v3.md - Quick start guide")
    print("   - PROJECT_REPORT.md - Detailed features")
    
    print("\n✨ SmartVehicle Intelligence System is ready!")

if __name__ == '__main__':
    main()