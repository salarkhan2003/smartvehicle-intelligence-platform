"""
MDVR (Mobile Digital Video Recorder) Module
Implements 10-second pre-event buffer for LTA compliance
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os
import json
from threading import Thread, Lock

class MDVRRecorder:
    """
    Mobile Digital Video Recorder with circular buffer
    - 10-second pre-event recording
    - 5-second post-event recording
    - H.264 encoding
    - MP4 container format
    """
    
    def __init__(self, config_path='config/settings.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.mdvr_config = config['mdvr']
        self.buffer_seconds = self.mdvr_config['buffer_seconds']
        self.pre_event_seconds = self.mdvr_config['pre_event_seconds']
        self.post_event_seconds = self.mdvr_config['post_event_seconds']
        self.fps = self.mdvr_config['fps']
        self.codec = self.mdvr_config['codec']
        
        # Calculate buffer size (frames)
        self.buffer_size = self.buffer_seconds * self.fps  # 300 frames @ 30 FPS
        
        # Circular buffer for frames
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = Lock()
        
        # Recording state
        self.is_recording = False
        self.current_writer = None
        self.recording_path = None
        self.post_event_frames = 0
        self.post_event_limit = self.post_event_seconds * self.fps
        
        # Output directory
        self.output_dir = 'data/recordings'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ MDVR initialized: {self.buffer_seconds}s buffer ({self.buffer_size} frames)")
    
    def add_frame(self, frame, timestamp=None):
        """
        Add frame to circular buffer
        
        Args:
            frame: OpenCV frame (BGR)
            timestamp: Optional timestamp for the frame
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.buffer_lock:
            # Store frame with metadata
            frame_data = {
                'frame': frame.copy(),
                'timestamp': timestamp,
                'shape': frame.shape
            }
            self.frame_buffer.append(frame_data)
            
            # If recording, write frame
            if self.is_recording and self.current_writer:
                self.current_writer.write(frame)
                self.post_event_frames += 1
                
                # Check if post-event recording is complete
                if self.post_event_frames >= self.post_event_limit:
                    self.stop_recording()
    
    def trigger_event(self, event_type='alert', metadata=None):
        """
        Trigger event recording (pre + post event)
        
        Args:
            event_type: Type of event (alert, violation, collision, etc.)
            metadata: Additional metadata to save with the recording
        """
        if self.is_recording:
            print("⚠ Already recording, ignoring new trigger")
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{event_type}_{timestamp}.mp4"
        self.recording_path = os.path.join(self.output_dir, filename)
        
        # Get frame dimensions from buffer
        if len(self.frame_buffer) == 0:
            print("✗ Buffer empty, cannot start recording")
            return None
        
        frame_shape = self.frame_buffer[0]['shape']
        height, width = frame_shape[0], frame_shape[1]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.current_writer = cv2.VideoWriter(
            self.recording_path,
            fourcc,
            self.fps,
            (width, height)
        )
        
        if not self.current_writer.isOpened():
            print(f"✗ Failed to open video writer: {self.recording_path}")
            return None
        
        # Write buffered frames (pre-event)
        with self.buffer_lock:
            buffered_frames = list(self.frame_buffer)
        
        print(f"📹 Writing {len(buffered_frames)} pre-event frames...")
        for frame_data in buffered_frames:
            self.current_writer.write(frame_data['frame'])
        
        # Start post-event recording
        self.is_recording = True
        self.post_event_frames = 0
        
        # Save metadata
        if metadata:
            metadata_path = self.recording_path.replace('.mp4', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'event_type': event_type,
                    'timestamp': timestamp,
                    'pre_event_frames': len(buffered_frames),
                    'post_event_frames': self.post_event_limit,
                    **metadata
                }, f, indent=2)
        
        print(f"✓ Event recording started: {filename}")
        return self.recording_path
    
    def stop_recording(self):
        """Stop current recording and release resources"""
        if self.current_writer:
            self.current_writer.release()
            self.current_writer = None
        
        if self.is_recording:
            print(f"✓ Recording saved: {self.recording_path}")
            self.is_recording = False
            self.recording_path = None
            self.post_event_frames = 0
    
    def get_buffer_status(self):
        """Get current buffer status"""
        return {
            'buffer_size': len(self.frame_buffer),
            'buffer_capacity': self.buffer_size,
            'buffer_percentage': (len(self.frame_buffer) / self.buffer_size) * 100 if self.buffer_size > 0 else 0,
            'is_recording': self.is_recording,
            'post_event_frames': self.post_event_frames if self.is_recording else 0
        }
    
    def clear_buffer(self):
        """Clear the frame buffer (emergency stop)"""
        with self.buffer_lock:
            self.frame_buffer.clear()
        print("⚠ Buffer cleared")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_recording()


class SnapshotManager:
    """
    Manages alert snapshots
    Captures high-quality images on events
    """
    
    def __init__(self, output_dir='data/snapshots'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Snapshot manager initialized: {self.output_dir}")
    
    def capture(self, frame, event_type='alert', metadata=None):
        """
        Capture snapshot with metadata overlay
        
        Args:
            frame: OpenCV frame
            event_type: Event type label
            metadata: Dictionary with detection info
        
        Returns:
            Path to saved snapshot
        """
        timestamp = datetime.now()
        filename = f"{event_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create annotated frame
        annotated = frame.copy()
        
        # Add timestamp overlay
        time_text = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated, time_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add event type
        cv2.putText(annotated, f"EVENT: {event_type.upper()}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add metadata if provided
        if metadata:
            y_offset = 90
            for key, value in metadata.items():
                text = f"{key}: {value}"
                cv2.putText(annotated, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # Save image
        cv2.imwrite(filepath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"📸 Snapshot saved: {filename}")
        return filepath


# Example usage
if __name__ == '__main__':
    # Test MDVR
    mdvr = MDVRRecorder()
    
    # Simulate frames
    for i in range(500):  # 500 frames = 16.7 seconds @ 30 FPS
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, f"Frame {i}", (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        mdvr.add_frame(test_frame)
        
        # Trigger event at frame 300
        if i == 300:
            mdvr.trigger_event('test_alert', {'threat_level': 90})
    
    print("✓ MDVR test complete")
    print(f"Buffer status: {mdvr.get_buffer_status()}")
