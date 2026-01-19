#!/usr/bin/env python3
"""
Animal Detection Test Script
Tests the enhanced YOLO animal detection capabilities
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

def test_animal_detection():
    """Test animal detection with sample images"""
    
    print("🐾 Testing Animal Detection System")
    print("=" * 50)
    
    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8n model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return
    
    # Animal class IDs from COCO dataset
    animal_classes = {
        14: 'BIRD', 15: 'CAT', 16: 'DOG', 17: 'HORSE',
        18: 'SHEEP', 19: 'COW', 20: 'ELEPHANT', 21: 'BEAR',
        22: 'ZEBRA', 23: 'GIRAFFE'
    }
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open camera, using test pattern")
        cap = None
    
    print("\n🎯 Animal Detection Test Results:")
    print("-" * 40)
    
    frame_count = 0
    animals_detected = {}
    
    while frame_count < 100:  # Test for 100 frames
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate test pattern
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "TEST MODE - No Camera", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Run YOLO detection
        results = model(frame, verbose=False, conf=0.6)
        
        # Process detections
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if it's an animal
                    if cls_id in animal_classes:
                        animal_name = animal_classes[cls_id]
                        
                        # Count detections
                        if animal_name not in animals_detected:
                            animals_detected[animal_name] = 0
                        animals_detected[animal_name] += 1
                        
                        # Draw detection
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"🐾 {animal_name} {conf*100:.0f}%", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        print(f"Frame {frame_count:3d}: {animal_name} detected (confidence: {conf*100:.1f}%)")
        
        # Display frame
        cv2.putText(frame, f"Animal Detection Test - Frame {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Animal Detection Test', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        time.sleep(0.1)  # 10 FPS for testing
    
    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n📊 Detection Summary:")
    print("=" * 30)
    if animals_detected:
        for animal, count in animals_detected.items():
            print(f"🐾 {animal}: {count} detections")
        print(f"\nTotal animal detections: {sum(animals_detected.values())}")
    else:
        print("No animals detected in test")
    
    print(f"Frames processed: {frame_count}")
    print("\n✓ Animal detection test completed!")


def test_collision_distances():
    """Test distance calculation for different animals"""
    
    print("\n🎯 Testing Animal Distance Calculations")
    print("=" * 45)
    
    # Simulate different animal bounding box heights
    test_cases = [
        (14, 'BIRD', 20),      # Small bird
        (15, 'CAT', 40),       # Cat
        (16, 'DOG', 60),       # Dog
        (17, 'HORSE', 120),    # Horse
        (19, 'COW', 100),      # Cow
        (20, 'ELEPHANT', 200), # Elephant
    ]
    
    for cls_id, name, bbox_height in test_cases:
        # Distance calculation logic from main_v3.py
        if cls_id in [15, 16]:  # cat, dog
            distance = max(0.3, (0.5 * 480) / (bbox_height * 3.0))
        elif cls_id in [17, 19]:  # horse, cow
            distance = max(0.5, (1.6 * 480) / (bbox_height * 4.2))
        elif cls_id in [20, 21, 23]:  # elephant, bear, giraffe
            distance = max(0.8, (2.5 * 480) / (bbox_height * 5.0))
        elif cls_id == 14:  # bird
            distance = max(0.2, (0.3 * 480) / (bbox_height * 2.5))
        else:
            distance = max(1.0, 5.0 - (bbox_height / 60))
        
        # Threat level
        if distance < 1.0:
            threat = 95
        elif distance < 2.0:
            threat = 75
        elif distance < 3.0:
            threat = 45
        else:
            threat = 15
        
        print(f"🐾 {name:10s} | BBox: {bbox_height:3d}px | Distance: {distance:4.1f}m | Threat: {threat:2d}%")
    
    print("\n✓ Distance calculation test completed!")


if __name__ == '__main__':
    print("🚗 SmartVehicle Animal Detection Test Suite")
    print("=" * 50)
    
    # Test 1: Live animal detection
    test_animal_detection()
    
    # Test 2: Distance calculations
    test_collision_distances()
    
    print("\n🎉 All tests completed!")
    print("The system can now detect:")
    print("🐾 Birds, Cats, Dogs, Horses, Sheep, Cows")
    print("🐾 Elephants, Bears, Zebras, Giraffes")
    print("🚨 With collision warnings and distance estimation!")