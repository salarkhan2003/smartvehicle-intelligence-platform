#!/usr/bin/env python3
"""
OCR and ANPR Test Script
Tests the enhanced license plate recognition with EasyOCR
"""

import cv2
import numpy as np
import os
from ai_models.anpr_engine import ANPREngine
import time

def create_test_license_plates():
    """Create synthetic license plate images for testing"""
    
    print("🔧 Creating test license plates...")
    
    # Create test directory
    test_dir = 'data/test_plates'
    os.makedirs(test_dir, exist_ok=True)
    
    # Test plate texts
    test_plates = [
        'ABC1234',
        'XYZ5678', 
        'DEF9012',
        'GHI3456',
        'JKL7890'
    ]
    
    created_plates = []
    
    for i, plate_text in enumerate(test_plates):
        # Create a synthetic license plate image
        img = np.ones((60, 200, 3), dtype=np.uint8) * 255  # White background
        
        # Add border
        cv2.rectangle(img, (2, 2), (197, 57), (0, 0, 0), 2)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        # Calculate text size for centering
        (text_width, text_height), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
        x = (img.shape[1] - text_width) // 2
        y = (img.shape[0] + text_height) // 2
        
        # Draw text
        cv2.putText(img, plate_text, (x, y), font, font_scale, (0, 0, 0), thickness)
        
        # Save image
        filename = f'test_plate_{i+1}_{plate_text}.jpg'
        filepath = os.path.join(test_dir, filename)
        cv2.imwrite(filepath, img)
        
        created_plates.append({
            'filepath': filepath,
            'expected_text': plate_text,
            'image': img
        })
        
        print(f"✓ Created test plate: {plate_text}")
    
    return created_plates

def test_ocr_accuracy():
    """Test OCR accuracy on synthetic license plates"""
    
    print("\n🎯 Testing OCR Accuracy")
    print("=" * 40)
    
    # Initialize ANPR engine
    anpr = ANPREngine()
    
    if not anpr.reader:
        print("✗ EasyOCR not available - Cannot test OCR")
        return
    
    # Create test plates
    test_plates = create_test_license_plates()
    
    correct_reads = 0
    total_tests = len(test_plates)
    
    print(f"\n📊 Testing {total_tests} synthetic license plates...")
    print("-" * 50)
    
    for i, plate_data in enumerate(test_plates):
        expected = plate_data['expected_text']
        image = plate_data['image']
        
        # Extract text using OCR
        ocr_result = anpr.extract_text_from_plate(image)
        detected = ocr_result['text']
        confidence = ocr_result['confidence']
        
        # Check accuracy
        is_correct = detected == expected
        if is_correct:
            correct_reads += 1
        
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"Test {i+1:2d}: Expected '{expected}' | Got '{detected}' | Conf: {confidence:.2f} | {status}")
    
    # Calculate accuracy
    accuracy = (correct_reads / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"📈 OCR Accuracy Results:")
    print(f"   Correct: {correct_reads}/{total_tests}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("🎉 EXCELLENT - OCR is working well!")
    elif accuracy >= 60:
        print("👍 GOOD - OCR is functional")
    elif accuracy >= 40:
        print("⚠ FAIR - OCR needs improvement")
    else:
        print("❌ POOR - OCR may have issues")
    
    return accuracy

def test_live_anpr():
    """Test ANPR with live camera feed"""
    
    print("\n📹 Testing Live ANPR")
    print("=" * 30)
    
    # Initialize ANPR
    anpr = ANPREngine()
    
    if not anpr.reader:
        print("✗ EasyOCR not available - Cannot test live ANPR")
        return
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open camera - Using test mode")
        cap = None
    
    print("🎯 Live ANPR Test Instructions:")
    print("   - Show license plates to camera")
    print("   - Press 's' to save detected plate")
    print("   - Press 'q' to quit")
    print("   - Green box = valid format, Yellow = invalid format")
    
    frame_count = 0
    plates_detected = 0
    
    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate test pattern
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "NO CAMERA - TEST MODE", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        frame_count += 1
        
        # Process every 5th frame to reduce load
        if frame_count % 5 == 0:
            # Detect license plates
            plates = anpr.process_frame(frame)
            
            # Draw detected plates
            for plate in plates:
                x1, y1, x2, y2 = plate['bbox']
                
                # Color based on validity
                color = (0, 255, 0) if plate['valid_format'] else (0, 255, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw plate text and confidence
                label = f"{plate['plate_number']}"
                conf_label = f"Conf: {plate['confidence']:.2f}"
                
                cv2.putText(frame, label, (x1, y1-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, conf_label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                plates_detected += 1
        
        # Show statistics
        stats = anpr.get_statistics()
        cv2.putText(frame, f"Plates Detected: {plates_detected}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Success Rate: {stats['success_rate_percent']:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Live ANPR Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'anpr_test_{timestamp}.jpg'
            cv2.imwrite(f'data/{filename}', frame)
            print(f"📸 Saved frame: {filename}")
    
    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Live ANPR test completed")
    print(f"   Frames processed: {frame_count}")
    print(f"   Plates detected: {plates_detected}")

def test_plate_preprocessing():
    """Test license plate preprocessing pipeline"""
    
    print("\n🔧 Testing Plate Preprocessing")
    print("=" * 35)
    
    anpr = ANPREngine()
    
    # Create a test plate with noise
    test_plate = np.ones((60, 200, 3), dtype=np.uint8) * 200
    
    # Add noise
    noise = np.random.randint(-50, 50, test_plate.shape, dtype=np.int16)
    test_plate = np.clip(test_plate.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add text
    cv2.putText(test_plate, 'ABC123', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Test preprocessing
    processed = anpr.preprocess_plate_region(test_plate)
    
    if processed is not None:
        print("✓ Preprocessing successful")
        
        # Save comparison
        comparison = np.hstack([test_plate, cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite('data/preprocessing_test.jpg', comparison)
        print("📸 Saved preprocessing comparison: data/preprocessing_test.jpg")
    else:
        print("✗ Preprocessing failed")

def main():
    """Main test function"""
    
    print("🚗 SmartVehicle OCR & ANPR Test Suite")
    print("=" * 50)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Test 1: OCR Accuracy
    accuracy = test_ocr_accuracy()
    
    # Test 2: Preprocessing
    test_plate_preprocessing()
    
    # Test 3: Live ANPR (optional)
    if accuracy and accuracy > 50:  # Only if OCR is working reasonably
        response = input("\n🎥 Test live ANPR with camera? (y/n): ").lower()
        if response == 'y':
            test_live_anpr()
    
    print("\n🎉 All OCR/ANPR tests completed!")
    print("\nTo install EasyOCR if not available:")
    print("   pip install easyocr")
    print("\nTo test with real license plates:")
    print("   python test_ocr_anpr.py")

if __name__ == '__main__':
    main()