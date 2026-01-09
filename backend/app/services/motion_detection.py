"""
Motion Detection Service using OpenCV and YOLOv8.
Real-time motion detection, object detection, and alert triggering.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from ultralytics import YOLO
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from motion/object detection."""
    motion_detected: bool
    motion_percentage: float
    confidence_score: float
    objects: List[Dict]
    bounding_boxes: List[Tuple[int, int, int, int]]
    timestamp: datetime
    frame_data: Optional[np.ndarray] = None


class MotionDetectionService:
    """
    High-performance motion detection using OpenCV.
    Implements background subtraction and contour analysis.
    """

    def __init__(self):
        """Initialize motion detection parameters."""
        self.motion_threshold = settings.MOTION_THRESHOLD
        self.min_contour_area = settings.MIN_CONTOUR_AREA

        # Initialize background subtractor
        if settings.BACKGROUND_SUBTRACTOR == "MOG2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=True
            )

        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        logger.info(f"Motion detection initialized with {settings.BACKGROUND_SUBTRACTOR}")

    def detect_motion(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect motion in a single frame.

        Args:
            frame: Input frame (BGR format from cv2)

        Returns:
            DetectionResult with motion analysis
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                motion_detected=False,
                motion_percentage=0.0,
                confidence_score=0.0,
                objects=[],
                bounding_boxes=[],
                timestamp=datetime.utcnow()
            )

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (value 127 in MOG2)
        fg_mask[fg_mask == 127] = 0

        # Morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        significant_contours = [
            c for c in contours
            if cv2.contourArea(c) > self.min_contour_area
        ]

        # Calculate motion percentage
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_percentage = (motion_pixels / total_pixels) * 100

        # Extract bounding boxes
        bounding_boxes = []
        for contour in significant_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        # Determine if motion detected
        motion_detected = (
            motion_percentage > (self.motion_threshold / 10) and
            len(significant_contours) > 0
        )

        # Confidence based on contour strength
        confidence = min(motion_percentage / 20, 1.0) if motion_detected else 0.0

        return DetectionResult(
            motion_detected=motion_detected,
            motion_percentage=motion_percentage,
            confidence_score=confidence,
            objects=[],
            bounding_boxes=bounding_boxes,
            timestamp=datetime.utcnow(),
            frame_data=frame
        )

    def draw_detection(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw bounding boxes and info on frame.

        Args:
            frame: Original frame
            result: Detection result

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw bounding boxes
        for (x, y, w, h) in result.bounding_boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw motion info
        status_text = "MOTION DETECTED" if result.motion_detected else "No Motion"
        status_color = (0, 0, 255) if result.motion_detected else (0, 255, 0)

        cv2.putText(
            annotated,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )

        cv2.putText(
            annotated,
            f"Motion: {result.motion_percentage:.1f}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return annotated


class ObjectDetectionService:
    """
    Object detection using YOLOv8.
    Detects persons, vehicles, and other objects.
    """

    def __init__(self):
        """Initialize YOLO model."""
        try:
            self.model = YOLO(settings.YOLO_MODEL)
            self.confidence_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
            self.iou_threshold = settings.YOLO_IOU_THRESHOLD
            self.target_classes = settings.yolo_classes_list
            logger.info(f"YOLO model loaded: {settings.YOLO_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame using YOLO.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detected objects with class, confidence, and bbox
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            detected_objects = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract detection info
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    # Filter by target classes
                    if class_name.lower() in [c.lower() for c in self.target_classes]:
                        detected_objects.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox,  # [x1, y1, x2, y2]
                            'class_id': class_id
                        })

            logger.debug(f"Detected {len(detected_objects)} objects")
            return detected_objects

        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """
        Draw YOLO detections on frame.

        Args:
            frame: Original frame
            objects: List of detected objects

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for obj in objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Color based on class
            if obj['class'].lower() == 'person':
                color = (0, 0, 255)  # Red for persons
            else:
                color = (255, 0, 0)  # Blue for vehicles

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{obj['class']} {obj['confidence']:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return annotated


class CombinedDetectionService:
    """
    Combined motion and object detection pipeline.
    Integrates both services for comprehensive analysis.
    """

    def __init__(self):
        """Initialize both detection services."""
        self.motion_detector = MotionDetectionService()
        self.object_detector = ObjectDetectionService()
        logger.info("Combined detection service initialized")

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process frame through both motion and object detection.

        Args:
            frame: Input frame

        Returns:
            Combined detection result
        """
        # First detect motion
        motion_result = self.motion_detector.detect_motion(frame)

        # If motion detected, run object detection
        if motion_result.motion_detected:
            objects = self.object_detector.detect_objects(frame)
            motion_result.objects = objects

            # Boost confidence if objects detected
            if objects:
                motion_result.confidence_score = min(
                    motion_result.confidence_score + 0.3,
                    1.0
                )

        return motion_result

    def annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw all detections on frame.

        Args:
            frame: Original frame
            result: Detection result

        Returns:
            Fully annotated frame
        """
        # Draw motion detection
        annotated = self.motion_detector.draw_detection(frame, result)

        # Draw object detection
        if result.objects:
            annotated = self.object_detector.draw_detections(annotated, result.objects)

        # Add timestamp
        timestamp_text = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            annotated,
            timestamp_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return annotated

