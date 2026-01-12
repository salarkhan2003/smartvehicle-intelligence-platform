"""
TIER 3: ANPR Engine (Automatic Number Plate Recognition) - STUB FOR MISSING EASYOCR
License plate detection - Currently disabled due to missing EasyOCR
"""

import cv2
import numpy as np
import re
import json
from datetime import datetime


class ANPREngine:
    """
    Automatic Number Plate Recognition System - STUB VERSION
    """
    
    def __init__(self, config_path='config/settings.json'):
        """Initialize ANPR engine stub"""
        print("⚠ ANPR Engine stub - EasyOCR not available")
        self.reader = None
        self.config = {'enabled': False}
        self.plate_pattern = re.compile(r'^[A-Z]{3}\d{4}[A-Z]$')
        self.confidence_threshold = 0.8
        self.recent_plates = {}
        self.plate_timeout = 5
    
    def process_frame(self, frame, vehicle_detections=None):
        """Return empty list - ANPR disabled"""
        return []


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
