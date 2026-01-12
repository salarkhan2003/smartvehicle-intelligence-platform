"""
Performance Monitoring Module
Real-time FPS, latency, and system health tracking
"""

import time
import psutil
import os
from collections import deque
from datetime import datetime
import json

class PerformanceMonitor:
    """
    Monitors system performance metrics in real-time
    - FPS (Frames Per Second)
    - Latency (processing time per frame)
    - CPU usage
    - Memory usage
    - Disk I/O
    """
    
    def __init__(self, window_size=30):
        self.window_size = window_size  # Moving average window
        
        # Timing metrics
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        
        # Counters
        self.total_frames = 0
        self.start_time = time.time()
        
        # System metrics
        self.process = psutil.Process(os.getpid())
        
        # Performance logs
        self.logs = []
        self.log_interval = 10  # Log every 10 seconds
        self.last_log_time = time.time()
        
        print("✓ Performance Monitor initialized")
    
    def start_frame(self):
        """Mark start of frame processing"""
        return time.time()
    
    def end_frame(self, start_time):
        """
        Mark end of frame processing and calculate metrics
        
        Args:
            start_time: Time when frame processing started
            
        Returns:
            dict: Current performance metrics
        """
        current_time = time.time()
        
        # Calculate processing time
        processing_time = current_time - start_time
        self.processing_times.append(processing_time)
        
        # Calculate frame time (time between frames)
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        self.total_frames += 1
        
        # Get current metrics
        metrics = self.get_metrics()
        
        # Periodic logging
        if current_time - self.last_log_time >= self.log_interval:
            self.log_metrics(metrics)
            self.last_log_time = current_time
        
        return metrics
    
    def get_metrics(self):
        """
        Get current performance metrics
        
        Returns:
            dict: Performance metrics
        """
        # FPS calculation
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0
        
        # Latency calculation (processing time)
        if len(self.processing_times) > 0:
            avg_latency = sum(self.processing_times) / len(self.processing_times)
            latency_ms = avg_latency * 1000
        else:
            latency_ms = 0
        
        # System metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
        
        # Overall runtime
        runtime = time.time() - self.start_time
        
        return {
            'fps': round(fps, 2),
            'latency_ms': round(latency_ms, 2),
            'cpu_percent': round(cpu_percent, 1),
            'memory_mb': round(memory_mb, 1),
            'total_frames': self.total_frames,
            'runtime_seconds': round(runtime, 1),
            'avg_fps': round(self.total_frames / runtime, 2) if runtime > 0 else 0
        }
    
    def get_performance_grade(self):
        """
        Get performance grade based on FPS and latency
        
        Returns:
            str: Grade (EXCELLENT, GOOD, FAIR, POOR)
        """
        metrics = self.get_metrics()
        fps = metrics['fps']
        latency = metrics['latency_ms']
        
        if fps >= 28 and latency < 50:
            return "EXCELLENT"
        elif fps >= 20 and latency < 100:
            return "GOOD"
        elif fps >= 15 and latency < 150:
            return "FAIR"
        else:
            return "POOR"
    
    def log_metrics(self, metrics):
        """Log metrics to history"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **metrics,
            'grade': self.get_performance_grade()
        }
        self.logs.append(log_entry)
        
        # Keep only last hour of logs (360 entries @ 10s interval)
        if len(self.logs) > 360:
            self.logs = self.logs[-360:]
    
    def export_logs(self, filepath='data/logs/performance.json'):
        """Export performance logs to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'export_time': datetime.now().isoformat(),
                'total_logs': len(self.logs),
                'logs': self.logs
            }, f, indent=2)
        
        print(f"✓ Performance logs exported: {filepath}")
        return filepath
    
    def get_summary(self):
        """Get performance summary"""
        if len(self.logs) == 0:
            return "No performance data available"
        
        metrics = self.get_metrics()
        
        summary = f"""
╔════════════════════════════════════════════════╗
║        PERFORMANCE SUMMARY                     ║
╠════════════════════════════════════════════════╣
║ FPS:           {metrics['fps']:>6.2f} frames/sec         ║
║ Latency:       {metrics['latency_ms']:>6.2f} ms                 ║
║ CPU Usage:     {metrics['cpu_percent']:>6.1f} %                  ║
║ Memory:        {metrics['memory_mb']:>6.1f} MB                 ║
║ Total Frames:  {metrics['total_frames']:>6d}                    ║
║ Runtime:       {metrics['runtime_seconds']:>6.1f} seconds            ║
║ Grade:         {self.get_performance_grade():>10s}              ║
╚════════════════════════════════════════════════╝
        """
        return summary


class CameraHealthMonitor:
    """
    Monitors camera health and diagnostics
    - Frame drop detection
    - Resolution changes
    - Connection stability
    - Image quality metrics
    """
    
    def __init__(self):
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_check_time = time.time()
        self.health_status = "UNKNOWN"
        self.issues = []
        
        # Quality metrics
        self.brightness_history = deque(maxlen=30)
        self.contrast_history = deque(maxlen=30)
        
        print("✓ Camera Health Monitor initialized")
    
    def check_frame(self, frame, expected_shape=(480, 640, 3)):
        """
        Check frame health
        
        Args:
            frame: OpenCV frame to check
            expected_shape: Expected frame shape
            
        Returns:
            dict: Health report
        """
        self.total_frames += 1
        self.issues = []
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            self.dropped_frames += 1
            self.issues.append("NULL_FRAME")
            self.health_status = "CRITICAL"
            return self.get_health_report()
        
        # Check resolution
        if frame.shape != expected_shape:
            self.issues.append(f"RESOLUTION_MISMATCH: Expected {expected_shape}, got {frame.shape}")
            self.health_status = "WARNING"
        
        # Check brightness
        gray = frame.mean()
        self.brightness_history.append(gray)
        
        if gray < 20:
            self.issues.append("TOO_DARK")
            self.health_status = "WARNING"
        elif gray > 235:
            self.issues.append("TOO_BRIGHT")
            self.health_status = "WARNING"
        
        # Check contrast (std deviation)
        contrast = frame.std()
        self.contrast_history.append(contrast)
        
        if contrast < 10:
            self.issues.append("LOW_CONTRAST")
            self.health_status = "WARNING"
        
        # If no issues, status is healthy
        if not self.issues:
            self.health_status = "HEALTHY"
        
        return self.get_health_report()
    
    def get_health_report(self):
        """Get camera health report"""
        drop_rate = (self.dropped_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        avg_brightness = sum(self.brightness_history) / len(self.brightness_history) if self.brightness_history else 0
        avg_contrast = sum(self.contrast_history) / len(self.contrast_history) if self.contrast_history else 0
        
        return {
            'status': self.health_status,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'drop_rate_percent': round(drop_rate, 2),
            'avg_brightness': round(avg_brightness, 1),
            'avg_contrast': round(avg_contrast, 1),
            'issues': self.issues,
            'uptime_seconds': round(time.time() - self.last_check_time, 1)
        }
    
    def reset(self):
        """Reset health monitor"""
        self.total_frames = 0
        self.dropped_frames = 0
        self.issues = []
        self.brightness_history.clear()
        self.contrast_history.clear()
        self.last_check_time = time.time()
        print("Health monitor reset")


# Example usage
if __name__ == '__main__':
    import numpy as np
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    
    for i in range(100):
        start = monitor.start_frame()
        
        # Simulate processing
        time.sleep(0.033)  # ~30 FPS
        
        metrics = monitor.end_frame(start)
        
        if i % 30 == 0:
            print(f"Frame {i}: FPS={metrics['fps']}, Latency={metrics['latency_ms']}ms")
    
    print(monitor.get_summary())
    
    # Test camera health
    health = CameraHealthMonitor()
    
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    report = health.check_frame(test_frame)
    print(f"Camera Health: {report}")
