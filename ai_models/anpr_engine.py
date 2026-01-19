"""
TIER 3: ANPR Engine (Automatic Number Plate Recognition) - ENHANCED WITH OCR
License plate detection and text recognition using EasyOCR and OpenCV
"""

import cv2
import numpy as np
import re
import json
from datetime import datetime
import os

try:
    import easyocr
    OCR_AVAILABLE = True
    print("✓ EasyOCR available for ANPR")
except ImportError:
    OCR_AVAILABLE = False
    print("⚠ EasyOCR not available - Install with: pip install easyocr")


class ANPREngine:
    """
    Enhanced Automatic Number Plate Recognition System with OCR
    - License plate detection using contour analysis
    - Text extraction using EasyOCR
    - Pattern matching for various license plate formats
    - Confidence scoring and validation
    """
    
    def __init__(self, config_path='config/settings.json'):
        """Initialize ANPR engine with OCR support"""
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.config = config.get('anpr', {})
        except:
            self.config = {
                'enabled': True,
                'min_confidence': 0.7,
                'languages': ['en'],
                'plate_patterns': [
                    r'^[A-Z]{2,3}\d{3,4}[A-Z]?$',  # Standard format
                    r'^\d{1,3}[A-Z]{1,3}\d{1,4}$',  # Alternative format
                    r'^[A-Z]{1,2}\d{2}[A-Z]{2}\d{4}$'  # Long format
                ]
            }
        
        # Initialize EasyOCR reader
        self.reader = None
        if OCR_AVAILABLE and self.config.get('enabled', True):
            try:
                languages = self.config.get('languages', ['en'])
                self.reader = easyocr.Reader(languages, gpu=False)  # CPU mode for compatibility
                print(f"✓ EasyOCR initialized with languages: {languages}")
            except Exception as e:
                print(f"✗ EasyOCR initialization failed: {e}")
                self.reader = None
        
        # License plate patterns
        self.plate_patterns = [re.compile(pattern) for pattern in self.config.get('plate_patterns', [])]
        self.confidence_threshold = self.config.get('min_confidence', 0.7)
        
        # Recent plates cache (avoid duplicates)
        self.recent_plates = {}
        self.plate_timeout = 5  # seconds
        
        # Statistics
        self.total_detections = 0
        self.successful_reads = 0
        
        print(f"✓ ANPR Engine initialized (OCR: {'Enabled' if self.reader else 'Disabled'})")
    
    def preprocess_plate_region(self, plate_roi):
        """
        Preprocess license plate region for better OCR
        
        Args:
            plate_roi: Cropped license plate region
            
        Returns:
            Preprocessed image for OCR
        """
        if plate_roi is None or plate_roi.size == 0:
            return None
        
        # Resize for better OCR (minimum 200px width)
        height, width = plate_roi.shape[:2]
        if width < 200:
            scale = 200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            plate_roi = cv2.resize(plate_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(plate_roi.shape) == 3:
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_roi
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_license_plates(self, frame, vehicle_detections=None):
        """
        Detect potential license plate regions in frame
        
        Args:
            frame: Input frame
            vehicle_detections: List of vehicle detection bounding boxes
            
        Returns:
            List of potential license plate regions
        """
        plate_regions = []
        
        if vehicle_detections is None or len(vehicle_detections) == 0:
            # Search entire frame if no vehicle detections
            search_regions = [{'bbox': (0, 0, frame.shape[1], frame.shape[0]), 'vehicle_id': -1}]
        else:
            # Search within vehicle bounding boxes
            search_regions = []
            for i, detection in enumerate(vehicle_detections):
                if detection['class_id'] in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    search_regions.append({
                        'bbox': detection['bbox'],
                        'vehicle_id': detection['id']
                    })
        
        for region in search_regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Extract region of interest
            roi = frame[max(0, y1):min(frame.shape[0], y2), 
                       max(0, x1):min(frame.shape[1], x2)]
            
            if roi.size == 0:
                continue
            
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Edge detection
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area < 500:  # Too small
                    continue
                
                # Get bounding rectangle
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                
                # License plate aspect ratio check (typically 2:1 to 5:1)
                aspect_ratio = rect_w / rect_h if rect_h > 0 else 0
                if not (1.5 <= aspect_ratio <= 6.0):
                    continue
                
                # Size constraints
                if rect_w < 60 or rect_h < 15:
                    continue
                
                # Extract potential plate region
                plate_x1 = max(0, x1 + rect_x - 5)
                plate_y1 = max(0, y1 + rect_y - 5)
                plate_x2 = min(frame.shape[1], x1 + rect_x + rect_w + 5)
                plate_y2 = min(frame.shape[0], y1 + rect_y + rect_h + 5)
                
                plate_roi = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                
                if plate_roi.size > 0:
                    plate_regions.append({
                        'roi': plate_roi,
                        'bbox': (plate_x1, plate_y1, plate_x2, plate_y2),
                        'vehicle_id': region['vehicle_id'],
                        'aspect_ratio': aspect_ratio,
                        'area': area
                    })
        
        return plate_regions
    
    def extract_text_from_plate(self, plate_roi):
        """
        Extract text from license plate using OCR
        
        Args:
            plate_roi: License plate region image
            
        Returns:
            dict: OCR results with text and confidence
        """
        if not self.reader or plate_roi is None or plate_roi.size == 0:
            return {'text': '', 'confidence': 0.0, 'valid': False}
        
        try:
            # Preprocess the plate region
            processed_roi = self.preprocess_plate_region(plate_roi)
            if processed_roi is None:
                return {'text': '', 'confidence': 0.0, 'valid': False}
            
            # Run OCR on both original and processed images
            results_original = self.reader.readtext(plate_roi, detail=1)
            results_processed = self.reader.readtext(processed_roi, detail=1)
            
            # Combine results and find best match
            all_results = results_original + results_processed
            
            best_result = {'text': '', 'confidence': 0.0, 'valid': False}
            
            for (bbox, text, confidence) in all_results:
                # Clean up text
                cleaned_text = self.clean_plate_text(text)
                
                # Validate against patterns
                is_valid = self.validate_plate_text(cleaned_text)
                
                # Update best result if this is better
                if confidence > best_result['confidence'] and len(cleaned_text) >= 4:
                    best_result = {
                        'text': cleaned_text,
                        'confidence': confidence,
                        'valid': is_valid,
                        'bbox': bbox
                    }
            
            return best_result
            
        except Exception as e:
            print(f"OCR error: {e}")
            return {'text': '', 'confidence': 0.0, 'valid': False}
    
    def clean_plate_text(self, text):
        """
        Clean and normalize license plate text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ''
        
        # Remove spaces and special characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common OCR corrections
        corrections = {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2',
            'B': '8', 'G': '6', 'Q': '0'
        }
        
        # Apply corrections contextually
        result = ''
        for i, char in enumerate(cleaned):
            if char in corrections:
                # Use context to decide correction
                if i < len(cleaned) // 2:  # First half - likely letters
                    result += char
                else:  # Second half - likely numbers
                    result += corrections.get(char, char)
            else:
                result += char
        
        return result
    
    def validate_plate_text(self, text):
        """
        Validate license plate text against known patterns
        
        Args:
            text: License plate text
            
        Returns:
            bool: True if valid format
        """
        if not text or len(text) < 4:
            return False
        
        # Check against configured patterns
        for pattern in self.plate_patterns:
            if pattern.match(text):
                return True
        
        # Basic validation - mix of letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter and has_number and 4 <= len(text) <= 10
    
    def process_frame(self, frame, vehicle_detections=None):
        """
        Process frame for license plate recognition
        
        Args:
            frame: Input frame
            vehicle_detections: List of vehicle detections
            
        Returns:
            List of detected license plates
        """
        if not self.reader:
            return []
        
        self.total_detections += 1
        detected_plates = []
        current_time = datetime.now()
        
        # Clean up old plates from cache
        expired_plates = [plate for plate, timestamp in self.recent_plates.items() 
                         if (current_time - timestamp).total_seconds() > self.plate_timeout]
        for plate in expired_plates:
            del self.recent_plates[plate]
        
        # Detect potential license plate regions
        plate_regions = self.detect_license_plates(frame, vehicle_detections)
        
        for region in plate_regions:
            # Extract text using OCR
            ocr_result = self.extract_text_from_plate(region['roi'])
            
            if (ocr_result['confidence'] >= self.confidence_threshold and 
                len(ocr_result['text']) >= 4):
                
                plate_text = ocr_result['text']
                
                # Check if we've seen this plate recently
                if plate_text in self.recent_plates:
                    continue
                
                # Add to recent plates
                self.recent_plates[plate_text] = current_time
                self.successful_reads += 1
                
                detected_plates.append({
                    'plate_number': plate_text,
                    'confidence': ocr_result['confidence'],
                    'valid_format': ocr_result['valid'],
                    'bbox': region['bbox'],
                    'vehicle_id': region['vehicle_id'],
                    'timestamp': current_time,
                    'ocr_bbox': ocr_result.get('bbox', None)
                })
                
                print(f"📋 License plate detected: {plate_text} (confidence: {ocr_result['confidence']:.2f})")
        
        return detected_plates
    
    def get_statistics(self):
        """Get ANPR performance statistics"""
        success_rate = (self.successful_reads / self.total_detections * 100) if self.total_detections > 0 else 0
        
        return {
            'total_detections': self.total_detections,
            'successful_reads': self.successful_reads,
            'success_rate_percent': round(success_rate, 1),
            'recent_plates_count': len(self.recent_plates),
            'ocr_enabled': self.reader is not None
        }
    
    def save_plate_image(self, plate_roi, plate_text, output_dir='data/plates'):
        """Save detected license plate image"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{plate_text}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, plate_roi)
        return filepath


class SpeedEnforcement:
    """
    Speed estimation using optical flow
    """
    
    def __init__(self, config_path='config/settings.json'):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.config = config['speed']
        except:
            self.config = {
                'calibration_multiplier': 0.5,
                'overspeed_threshold': 60,
                'school_zone_limit': 40
            }
        
        self.prev_gray = None
        self.calibration = self.config['calibration_multiplier']
        self.threshold = self.config['overspeed_threshold']
        
        print("✓ Speed Enforcement initialized")
    
    def estimate_speed(self, frame, zone_type='default'):
        """
        Estimate speed using optical flow
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return {'speed_kmh': 0, 'violation': False, 'limit': self.threshold}
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        speed_raw = np.mean(mag)
        
        # Apply calibration
        speed_kmh = speed_raw * self.calibration
        
        self.prev_gray = gray
        
        # Get zone limit
        zone_limits = {
            'school': self.config.get('school_zone_limit', 40),
            'residential': self.config.get('residential_limit', 50),
            'highway': self.config.get('highway_limit', 90),
            'default': self.threshold
        }
        
        limit = zone_limits.get(zone_type, self.threshold)
        violation = speed_kmh > limit
        
        return {
            'speed_kmh': round(speed_kmh, 1),
            'violation': violation,
            'limit': limit,
            'zone_type': zone_type
        }


# Example usage and testing
if __name__ == '__main__':
    import time
    
    print("🔍 Testing ANPR Engine with OCR")
    
    # Initialize ANPR
    anpr = ANPREngine()
    
    if anpr.reader:
        print("✓ OCR is available - Testing with webcam")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("📹 Press 'q' to quit, 's' to save plate image")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                plates = anpr.process_frame(frame)
                
                # Draw detected plates
                for plate in plates:
                    x1, y1, x2, y2 = plate['bbox']
                    
                    # Draw bounding box
                    color = (0, 255, 0) if plate['valid_format'] else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw plate text
                    label = f"{plate['plate_number']} ({plate['confidence']:.2f})"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show statistics
                stats = anpr.get_statistics()
                cv2.putText(frame, f"Plates: {stats['successful_reads']}/{stats['total_detections']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('ANPR Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and plates:
                    # Save first detected plate
                    plate_roi = frame[plates[0]['bbox'][1]:plates[0]['bbox'][3],
                                     plates[0]['bbox'][0]:plates[0]['bbox'][2]]
                    anpr.save_plate_image(plate_roi, plates[0]['plate_number'])
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("✗ Cannot open webcam")
    else:
        print("⚠ OCR not available - Install EasyOCR: pip install easyocr")
    
    print("✓ ANPR test completed")