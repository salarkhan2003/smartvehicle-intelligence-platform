"""
SmartVehicle v3.0 Setup Script
Creates __init__.py files and validates installation
"""

import os
import sys

def create_init_files():
    """Create __init__.py files in all module directories"""
    
    directories = [
        'core',
        'ai_models',
        'features',
        'database',
        'utils',
        'config'
    ]
    
    for dir_name in directories:
        init_file = os.path.join(dir_name, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""{dir_name.upper()} module for SmartVehicle v3.0"""\n')
            print(f"✓ Created {init_file}")
        else:
            print(f"  {init_file} already exists")

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("\n=== Checking Dependencies ===\n")
    
    dependencies = [
        ('PySide6', 'PySide6'),
        ('OpenCV', 'cv2'),
        ('NumPy', 'numpy'),
        ('Ultralytics (YOLOv8)', 'ultralytics'),
        ('MediaPipe', 'mediapipe'),
        ('EasyOCR', 'easyocr'),
        ('PyTorch', 'torch'),
        ('pyttsx3 (Text-to-Speech)', 'pyttsx3'),
        ('SciPy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('Pillow', 'PIL')
    ]
    
    missing = []
    
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ WARNING: {len(missing)} dependencies missing!")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def check_models():
    """Check if required model files exist"""
    
    print("\n=== Checking Model Files ===\n")
    
    models = [
        ('yolov8n.pt', 'YOLOv8 Nano model', True),
        ('models/helmet_detector.pt', 'Custom Helmet Detector', False),
        ('models/fatigue_model.pkl', 'Fatigue Prediction Model', False)
    ]
    
    for filepath, name, required in models:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✓ {name} - {size:.1f} MB")
        elif required:
            print(f"⚠ {name} - Will be downloaded on first run")
        else:
            print(f"  {name} - Optional (not found)")

def check_config():
    """Check if configuration files exist"""
    
    print("\n=== Checking Configuration ===\n")
    
    configs = [
        'config/settings.json',
        'config/zones.json'
    ]
    
    for config in configs:
        if os.path.exists(config):
            print(f"✓ {config}")
        else:
            print(f"✗ {config} - MISSING!")

def check_directories():
    """Check if required directories exist"""
    
    print("\n=== Checking Directories ===\n")
    
    dirs = [
        'data/recordings',
        'data/snapshots',
        'data/logs',
        'models',
        'config',
        'core',
        'ai_models',
        'utils'
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"  Creating {dir_path}...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ {dir_path} created")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  SmartVehicle Intelligence System v3.0 - Setup Script        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("Initializing project structure...\n")
    
    # Create __init__.py files
    print("=== Creating Module Init Files ===\n")
    create_init_files()
    
    # Check directories
    check_directories()
    
    # Check configuration
    check_config()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check models
    check_models()
    
    print("\n" + "="*60)
    
    if deps_ok:
        print("\n✓ SETUP COMPLETE!")
        print("\nYou can now run:")
        print("  python main_v3.py")
        print("\nOr use the launcher:")
        print("  run_v3.bat  (Windows)")
        print("\n")
        return 0
    else:
        print("\n⚠ SETUP INCOMPLETE")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nThen run this script again.")
        print("\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
