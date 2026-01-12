"""
TIER 2: Driver Monitoring System (T-SEEDS Product)
Real MediaPipe-based driver state analysis
- Face Detection
- Eye Aspect Ratio (EAR) for drowsiness
- Yawn Detection (MAR - Mouth Aspect Ratio)
- Head Pose Tracking
- Fatigue Prediction
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
import time
import json

class DriverMonitor:
    """
    Complete driver monitoring system using MediaPipe Face Mesh
    Implements T-SEEDS (TNT - Scalable Enhanced Eye Drowsiness System)
    """
    
    def __init__(self, config_path='config/settings.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.config = config['driver_monitoring']
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks (MediaPipe  indices)
        # Left eye: 362, 385, 387, 263, 373, 380
        # Right eye: 33, 160, 158, 133, 153, 144
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Mouth landmarks for yawn detection
        # Outer mouth: 61, 291, 0, 17, 78, 308, 402, 14
        self.MOUTH_OUTER = [61, 291, 199, 0, 17, 39, 181, 78, 82, 13, 312, 308, 402, 14, 178]
        
        # Head pose landmarks
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EYE_LEFT = 263
        self.RIGHT_EYE_RIGHT = 33
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291
        
        # Thresholds from config
        self.EAR_THRESHOLD = self.config['ear_threshold']
        self.EAR_CONSEC_FRAMES = self.config['ear_consecutive_frames']
        self.YAWN_THRESHOLD = self.config['yawn_threshold']
        self.YAWN_CONSEC_FRAMES = self.config['yawn_consecutive_frames']
        self.HEAD_POSE_THRESHOLD = self.config['head_pose_threshold']
        self.FATIGUE_THRESHOLD = self.config['fatigue_threshold']
        
        # State tracking
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)
        self.blink_counter = 0
        self.yawn_counter = 0
        self.drowsy_frames = 0
        self.yawn_frames = 0
        
        # Fatigue calculation
        self.fatigue_score = 0
        self.fatigue_history = deque(maxlen=100)
        
        print("✓ Driver Monitor initialized with MediaPipe Face Mesh (468 landmarks)")
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: Array of 6 (x,y) coordinates for eye
            
        Returns:
            float: EAR value
        """
        # Vertical distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR calculation
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection
        
        Args:
            mouth_landmarks: Array of mouth coordinates
            
        Returns:
            float: MAR value
        """
        # Vertical distances (top to bottom)
        A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])  # Center
        B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])   # Left
        C = dist.euclidean(mouth_landmarks[6], mouth_landmarks[12])  # Right
        
        # Horizontal distance
        D = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])
        
        # MAR calculation
        mar = (A + B + C) / (3.0 * D)
        return mar
    
    def get_head_pose(self, landmarks, frame_shape):
        """
        Estimate head pose angles (pitch, yaw, roll)
        
        Args:
            landmarks: Face mesh landmarks
            frame_shape: Image shape
            
        Returns:
            dict: pitch, yaw, roll angles in degrees
        """
        img_h, img_w = frame_shape[:2]
        
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera internals
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        # 2D image points from landmarks
        image_points = np.array([
            (landmarks[self.NOSE_TIP].x * img_w, landmarks[self.NOSE_TIP].y * img_h),
            (landmarks[self.CHIN].x * img_w, landmarks[self.CHIN].y * img_h),
            (landmarks[self.LEFT_EYE_LEFT].x * img_w, landmarks[self.LEFT_EYE_LEFT].y * img_h),
            (landmarks[self.RIGHT_EYE_RIGHT].x * img_w, landmarks[self.RIGHT_EYE_RIGHT].y * img_h),
            (landmarks[self.LEFT_MOUTH].x * img_w, landmarks[self.LEFT_MOUTH].y * img_h),
            (landmarks[self.RIGHT_MOUTH].x * img_w, landmarks[self.RIGHT_MOUTH].y * img_h)
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = euler_angles.flatten()[:3]
        
        return {
            'pitch': pitch,   # Up/down
            'yaw': yaw,       # Left/right
            'roll': roll      # Tilt
        }
    
    def analyze_frame(self, frame):
        """
        Analyze frame for driver state
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            dict: Driver monitoring results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        # Default result
        analysis = {
            'face_detected': False,
            'ear': 0.0,
            'mar': 0.0,
            'drowsy': False,
            'yawning': False,
            'distracted': False,
            'fatigue_score': 0,
            'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0},
            'alert': False,
            'status': 'NO_FACE'
        }
        
        if not results.multi_face_landmarks:
            return analysis
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        analysis['face_detected'] = True
        
        # Get image dimensions
        img_h, img_w = frame.shape[:2]
        
        # Extract eye landmarks
        left_eye = np.array([(landmarks[pt].x * img_w, landmarks[pt].y * img_h) 
                             for pt in self.LEFT_EYE])
        right_eye = np.array([(landmarks[pt].x * img_w, landmarks[pt].y * img_h) 
                              for pt in self.RIGHT_EYE])
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(ear)
        analysis['ear'] = round(ear, 3)
        
        # Drowsiness detection
        if ear < self.EAR_THRESHOLD:
            self.drowsy_frames += 1
        else:
            if self.drowsy_frames >= self.EAR_CONSEC_FRAMES:
                self.blink_counter += 1
            self.drowsy_frames = 0
        
        analysis['drowsy'] = self.drowsy_frames >= self.EAR_CONSEC_FRAMES
        
        # Extract mouth landmarks
        mouth_points = np.array([(landmarks[pt].x * img_w, landmarks[pt].y * img_h) 
                                 for pt in self.MOUTH_OUTER])
        
        # Calculate MAR
        mar = self.calculate_mar(mouth_points)
        self.mar_history.append(mar)
        analysis['mar'] = round(mar, 3)
        
        # Yawn detection
        if mar > self.YAWN_THRESHOLD:
            self.yawn_frames += 1
        else:
            if self.yawn_frames >= self.YAWN_CONSEC_FRAMES:
                self.yawn_counter += 1
            self.yawn_frames = 0
        
        analysis['yawning'] = self.yawn_frames >= self.YAWN_CONSEC_FRAMES
        
        # Head pose estimation
        head_pose = self.get_head_pose(landmarks, frame.shape)
        analysis['head_pose'] = {k: round(v, 1) for k, v in head_pose.items()}
        
        # Distraction detection (head turned away)
        if abs(head_pose['yaw']) > self.HEAD_POSE_THRESHOLD or \
           abs(head_pose['pitch']) > self.HEAD_POSE_THRESHOLD:
            analysis['distracted'] = True
        
        # Calculate fatigue score (0-100)
        fatigue_components = {
            'ear': max(0, (self.EAR_THRESHOLD - ear) / self.EAR_THRESHOLD) * 40,  # 40% weight
            'yawn': (self.yawn_counter / 10) * 30 if self.yawn_counter < 10 else 30,  # 30% weight
            'blinks': (self.blink_counter / 20) * 20 if self.blink_counter < 20 else 20,  # 20% weight
            'distraction': 10 if analysis['distracted'] else 0  # 10% weight
        }
        
        self.fatigue_score = min(100, sum(fatigue_components.values()))
        self.fatigue_history.append(self.fatigue_score)
        analysis['fatigue_score'] = round(self.fatigue_score, 1)
        
        # Overall status
        if self.fatigue_score >= self.FATIGUE_THRESHOLD:
            analysis['status'] = 'CRITICAL_FATIGUE'
            analysis['alert'] = True
        elif analysis['drowsy'] or self.drowsy_frames > 5:
            analysis['status'] = 'DROWSY'
            analysis['alert'] = True
        elif analysis['yawning']:
            analysis['status'] = 'YAWNING'
        elif analysis['distracted']:
            analysis['status'] = 'DISTRACTED'
            analysis['alert'] = True
        else:
            analysis['status'] = 'ALERT'
        
        return analysis
    
    def draw_analysis(self, frame, analysis):
        """
        Draw driver monitoring visualization on frame
        
        Args:
            frame: OpenCV frame
            analysis: Analysis results from analyze_frame
            
        Returns:
            Annotated frame
        """
        if not analysis['face_detected']:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Status color
        status_colors = {
            'ALERT': (0, 255, 0),
            'YAWNING': (0, 255, 255),
            'DROWSY': (0, 165, 255),
            'DISTRACTED': (0, 140, 255),
            'CRITICAL_FATIGUE': (0, 0, 255)
        }
        color = status_colors.get(analysis['status'], (255, 255, 255))
        
        # Draw status
        cv2.putText(frame, f"Status: {analysis['status']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw metrics
        y_offset = 60
        metrics = [
            f"EAR: {analysis['ear']:.3f}",
            f"MAR: {analysis['mar']:.3f}",
            f"Fatigue: {analysis['fatigue_score']:.0f}%",
            f"Yaw: {analysis['head_pose']['yaw']:.1f}°",
            f"Pitch: {analysis['head_pose']['pitch']:.1f}°"
        ]
        
        for metric in metrics:
            cv2.putText(frame, metric, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Alert indicator
        if analysis['alert']:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, "⚠ DRIVER ALERT ⚠", (frame.shape[1]//4, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return frame
    
    def reset(self):
        """Reset all counters"""
        self.blink_counter = 0
        self.yawn_counter = 0
        self.drowsy_frames = 0
        self.yawn_frames = 0
        self.fatigue_score = 0
        print("Driver monitor reset")


# Example usage
if __name__ == '__main__':
    # Test with webcam
    cap = cv2.VideoCapture(0)
    monitor = DriverMonitor()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze driver state
        analysis = monitor.analyze_frame(frame)
        
        # Draw visualization
        annotated = monitor.draw_analysis(frame, analysis)
        
        # Display
        cv2.imshow('Driver Monitoring - T-SEEDS', annotated)
        
        # Print analysis every 30 frames
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 30 == 0:
            print(f"Status: {analysis['status']}, Fatigue: {analysis['fatigue_score']}%")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
