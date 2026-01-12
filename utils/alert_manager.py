"""
TIER 5: Multi-Modal Alert System (T-DA Product)
Visual, Audio, and Voice alerts for driver awareness
"""

import winsound
import threading
import time
import json
from datetime import datetime
from PySide6.QtCore import QObject, Signal

try:
    import pyttsx3
    VOICE_AVAILABLE = True
except:
    VOICE_AVAILABLE = False
    print("⚠ pyttsx3 not available, voice alerts disabled")


class AlertManager(QObject):
    """
    Complete multi-modal alert system
    - Visual: UI flash animations
    - Audio: Frequency-based b eeps
    - Voice: Text-to-speech warnings
    """
    
    # Qt signals for UI updates
    visual_alert = Signal(str, str)  # (severity, message)
    
    def __init__(self, config_path='config/settings.json'):
        super().__init__()
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.config = config['alerts']
        except:
            self.config = {
                'visual_enabled': True,
                'audio_enabled': True,
                'voice_enabled': True,
                'cooldown_seconds': 3,
                'flash_duration_ms': 500,
                'audio_frequencies': {
                    'low': 500,
                    'medium': 800,
                    'high': 1000,
                    'critical': 1500
                }
            }
        
        # Alert state
        self.last_alert_time = {}
        self.alert_cooldown = self.config['cooldown_seconds']
        
        # Voice engine (text-to-speech)
        self.voice_engine = None
        if VOICE_AVAILABLE and self.config['voice_enabled']:
            try:
                self.voice_engine = pyttsx3.init()
                # Configure voice
                self.voice_engine.setProperty('rate', 150)  # Speed
                self.voice_engine.setProperty('volume', 0.9)  # Volume
                
                # Use female voice if available
                voices = self.voice_engine.getProperty('voices')
                if len(voices) > 1:
                    self.voice_engine.setProperty('voice', voices[1].id)
                
                print("✓ Voice alerts enabled (pyttsx3)")
            except Exception as e:
                print(f"Voice engine init failed: {e}")
                self.voice_engine = None
        
        # Audio frequencies
        self.frequencies = self.config['audio_frequencies']
        
        print("✓ Alert Manager initialized")
    
    def trigger(self, alert_type, message, severity='medium', force=False):
        """
        Trigger multi-modal alert
        
        Args:
            alert_type: Alert identifier (drowsiness, collision, etc.)
            message: Human-readable message
            severity: low, medium, high, critical
            force: Bypass cooldown
            
        Returns:
            bool: Alert triggered or not
        """
        # Check cooldown
        if not force and not self._check_cooldown(alert_type):
            return False
        
        # Update cooldown
        self.last_alert_time[alert_type] = datetime.now()
        
        # Trigger all modalities
        if self.config['visual_enabled']:
            self._trigger_visual(severity, message)
        
        if self.config['audio_enabled']:
            self._trigger_audio(severity)
        
        if self.config['voice_enabled']:
            self._trigger_voice(message)
        
        # Log alert
        self._log_alert(alert_type, message, severity)
        
        return True
    
    def _check_cooldown(self, alert_type):
        """Check if alert is within cooldown period"""
        if alert_type not in self.last_alert_time:
            return True
        
        time_diff = (datetime.now() - self.last_alert_time[alert_type]).total_seconds()
        return time_diff >= self.alert_cooldown
    
    def _trigger_visual(self, severity, message):
        """Trigger visual alert (Qt signal for UI)"""
        self.visual_alert.emit(severity, message)
    
    def _trigger_audio(self, severity):
        """Trigger audio beep based on severity"""
        # Run in separate thread to not block
        thread = threading.Thread(target=self._play_beep, args=(severity,))
        thread.daemon = True
        thread.start()
    
    def _play_beep(self, severity):
        """Play beep sound (Windows only)"""
        try:
            freq = self.frequencies.get(severity, 800)
            duration_ms = self.config['flash_duration_ms']
            
            # Beep pattern based on severity
            if severity == 'critical':
                # Rapid beeps
                for _ in range(3):
                    winsound.Beep(freq, 200)
                    time.sleep(0.1)
            elif severity == 'high':
                # Two beeps
                winsound.Beep(freq, duration_ms)
                time.sleep(0.1)
                winsound.Beep(freq, duration_ms)
            else:
                # Single beep
                winsound.Beep(freq, duration_ms)
        except Exception as e:
            print(f"Audio alert error: {e}")
    
    def _trigger_voice(self, message):
        """Trigger voice alert (text-to-speech)"""
        if not self.voice_engine:
            return
        
        # Run in separate thread
        thread = threading.Thread(target=self._speak, args=(message,))
        thread.daemon = True
        thread.start()
    
    def _speak(self, message):
        """Speak message using TTS"""
        try:
            # Stop any ongoing speech
            self.voice_engine.stop()
            
            # Speak new message
            self.voice_engine.say(message)
            self.voice_engine.runAndWait()
        except Exception as e:
            print(f"Voice alert error: {e}")
    
    def _log_alert(self, alert_type, message, severity):
        """Log alert to file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        # Append to log file
        try:
            import os
            os.makedirs('data/logs', exist_ok=True)
            
            with open('data/logs/alerts.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Alert logging error: {e}")
    
    # Predefined alerts for common scenarios
    def alert_drowsiness(self, fatigue_score):
        """Drowsiness alert"""
        severity = 'high' if fatigue_score > 80 else 'medium'
        message = f"Driver fatigue detected. Please take a break."
        return self.trigger('drowsiness', message, severity)
    
    def alert_collision_warning(self, distance, ttc):
        """Collision warning"""
        severity = 'critical' if ttc < 1.5 else 'high'
        message = f"Collision warning! Object at {distance:.1f} meters."
        return self.trigger('collision', message, severity)
    
    def alert_overspeed(self, speed, limit):
        """Overspeed alert"""
        message = f"Speeding! {speed:.0f} km/h in {limit:.0f} km/h zone."
        return self.trigger('overspeed', message, 'medium')
    
    def alert_blind_spot(self, side):
        """Blind spot alert"""
        message = f"Vehicle in {side} blind spot!"
        return self.trigger('blind_spot', message, 'high')
    
    def alert_helmet_violation(self, plate=None):
        """Helmet violation"""
        msg = f"Helmet violation detected"
        if plate:
            msg += f" - {plate}"
        return self.trigger('helmet', msg, 'low')
    
    def alert_seatbelt_violation(self):
        """Seatbelt violation"""
        message = "Seatbelt not detected!"
        return self.trigger('seatbelt', message, 'medium')
    
    def alert_pedestrian_crossing(self):
        """Pedestrian crossing alert"""
        message = "Pedestrian crossing detected. Slow down."
        return self.trigger('pedestrian', message, 'high')
    
    def alert_school_zone(self):
        """School zone alert"""
        message = "Entering school zone. Reduce speed to 40 km/h."
        return self.trigger('school_zone', message, 'low')
    
    def alert_distraction(self):
        """Driver distraction"""
        message = "Eyes on the road, please."
        return self.trigger('distraction', message, 'medium')
    
    def test_all_alerts(self):
        """Test all alert modalities"""
        print("\n=== Testing Alert System ===")
        
        # Test severities
        severities = ['low', 'medium', 'high', 'critical']
        
        for sev in severities:
            print(f"\nTesting {sev} severity...")
            self.trigger(
                alert_type=f'test_{sev}',
                message=f"This is a {sev} severity test alert",
                severity=sev,
                force=True
            )
            time.sleep(2)
        
        print("\n✓ Alert system test complete")


class VisualAlertWidget:
    """
    Helper class for visual alert rendering
    Can be inherited by Qt widgets
    """
    
    @staticmethod
    def get_severity_color(severity):
        """Get RGB color for severity level"""
        colors = {
            'low': (0, 255, 0),       # Green
            'medium': (255, 165, 0),  # Orange
            'high': (255, 69, 0),     # Red-Orange
            'critical': (255, 0, 0)   # Red
        }
        return colors.get(severity, (255, 255, 255))
    
    @staticmethod
    def get_severity_style(severity):
        """Get Qt StyleSheet for severity"""
        colors = {
            'low': '#00FF00',
            'medium': '#FFA500',
            'high': '#FF4500',
            'critical': '#FF0000'
        }
        color = colors.get(severity, '#FFFFFF')
        
        return f"""
            QLabel {{
                background-color: {color};
                color: white;
                font-weight: bold;
                font-size: 18px;
                padding: 10px;
                border-radius: 5px;
            }}
        """


# Example usage
if __name__ == '__main__':
    # Test alert manager
    alert_mgr = AlertManager()
    
    # Test predefined alerts
    print("\n=== Testing Predefined Alerts ===\n")
    
    time.sleep(1)
    alert_mgr.alert_drowsiness(85)
    time.sleep(2)
    
    alert_mgr.alert_collision_warning(1.2, 1.0)
    time.sleep(2)
    
    alert_mgr.alert_overspeed(75, 60)
    time.sleep(2)
    
    alert_mgr.alert_blind_spot('LEFT')
    time.sleep(2)
    
    alert_mgr.alert_school_zone()
    time.sleep(2)
    
    # Test all severities
    alert_mgr.test_all_alerts()
    
    print("\n✓ All tests complete")
