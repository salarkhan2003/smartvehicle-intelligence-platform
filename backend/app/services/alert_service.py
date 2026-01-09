"""
Alert Management Service.
Handles alert creation, cooldown logic, severity escalation, and acknowledgment.
"""

from typing import Optional, List, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.models import Alert, AlertSeverity, Vehicle, Detection
from app.core.config import settings
from app.db.session import get_redis
import logging
import json

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert lifecycle including creation, cooldown, and acknowledgment.
    Implements intelligent alert rules to prevent spam and ensure critical alerts reach operators.
    """

    def __init__(self):
        """Initialize alert manager with configuration."""
        self.cooldown_seconds = settings.ALERT_COOLDOWN_SECONDS
        self.alert_duration = settings.ALERT_DURATION_SECONDS
        self.max_alerts_per_hour = settings.MAX_ALERTS_PER_VEHICLE_PER_HOUR
        self.redis_client = get_redis()
        logger.info("Alert Manager initialized")

    def _get_cooldown_key(self, vehicle_id: int) -> str:
        """Get Redis key for vehicle cooldown tracking."""
        return f"alert:cooldown:vehicle:{vehicle_id}"

    def _get_alert_count_key(self, vehicle_id: int) -> str:
        """Get Redis key for alert count tracking."""
        return f"alert:count:vehicle:{vehicle_id}"

    def _get_last_alert_key(self, vehicle_id: int) -> str:
        """Get Redis key for last alert data."""
        return f"alert:last:vehicle:{vehicle_id}"

    def is_in_cooldown(self, vehicle_id: int) -> bool:
        """
        Check if vehicle is in alert cooldown period.

        Args:
            vehicle_id: Vehicle ID to check

        Returns:
            True if in cooldown, False otherwise
        """
        key = self._get_cooldown_key(vehicle_id)
        return self.redis_client.exists(key) > 0

    def set_cooldown(self, vehicle_id: int):
        """
        Set cooldown period for vehicle.

        Args:
            vehicle_id: Vehicle ID
        """
        key = self._get_cooldown_key(vehicle_id)
        self.redis_client.setex(
            key,
            self.cooldown_seconds,
            "1"
        )
        logger.debug(f"Cooldown set for vehicle {vehicle_id} for {self.cooldown_seconds}s")

    def get_alert_count(self, vehicle_id: int) -> int:
        """
        Get alert count for vehicle in the last hour.

        Args:
            vehicle_id: Vehicle ID

        Returns:
            Number of alerts in last hour
        """
        key = self._get_alert_count_key(vehicle_id)
        count = self.redis_client.get(key)
        return int(count) if count else 0

    def increment_alert_count(self, vehicle_id: int):
        """
        Increment alert count for vehicle (expires after 1 hour).

        Args:
            vehicle_id: Vehicle ID
        """
        key = self._get_alert_count_key(vehicle_id)

        if self.redis_client.exists(key):
            self.redis_client.incr(key)
        else:
            self.redis_client.setex(key, 3600, 1)  # 1 hour expiry

    def determine_severity(
        self,
        vehicle_id: int,
        motion_percentage: float,
        objects_detected: List[Dict],
        vehicle_speed: Optional[float] = None
    ) -> AlertSeverity:
        """
        Determine alert severity based on multiple factors.

        Args:
            vehicle_id: Vehicle ID
            motion_percentage: Motion intensity (0-100)
            objects_detected: List of detected objects
            vehicle_speed: Vehicle speed in km/h

        Returns:
            Alert severity level
        """
        # Get recent alert history
        alert_count = self.get_alert_count(vehicle_id)

        # Base severity on motion intensity
        if motion_percentage > 80:
            severity = AlertSeverity.HIGH
        elif motion_percentage > 50:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        # Escalate if persons detected
        person_detected = any(obj['class'].lower() == 'person' for obj in objects_detected)
        if person_detected:
            if severity == AlertSeverity.LOW:
                severity = AlertSeverity.MEDIUM
            elif severity == AlertSeverity.MEDIUM:
                severity = AlertSeverity.HIGH

        # Escalate if vehicle is moving
        if vehicle_speed and vehicle_speed > 10:  # Moving faster than 10 km/h
            if severity == AlertSeverity.MEDIUM:
                severity = AlertSeverity.HIGH
            elif severity == AlertSeverity.HIGH:
                severity = AlertSeverity.CRITICAL

        # Escalate if frequent alerts (potential incident)
        if alert_count > 5:
            if severity != AlertSeverity.CRITICAL:
                severity = AlertSeverity.HIGH

        return severity

    def create_alert(
        self,
        db: Session,
        vehicle_id: int,
        detection_id: Optional[int],
        severity: AlertSeverity,
        message: str,
        alert_type: str = "motion_detected"
    ) -> Optional[Alert]:
        """
        Create a new alert with cooldown and rate limit checks.

        Args:
            db: Database session
            vehicle_id: Vehicle ID
            detection_id: Related detection ID
            severity: Alert severity
            message: Alert message
            alert_type: Type of alert

        Returns:
            Created Alert object or None if suppressed
        """
        # Check cooldown
        if self.is_in_cooldown(vehicle_id):
            logger.debug(f"Alert suppressed for vehicle {vehicle_id} - in cooldown")
            return None

        # Check rate limit
        alert_count = self.get_alert_count(vehicle_id)
        if alert_count >= self.max_alerts_per_hour:
            logger.warning(
                f"Alert rate limit reached for vehicle {vehicle_id} "
                f"({alert_count} alerts in last hour)"
            )
            # Still allow CRITICAL alerts
            if severity != AlertSeverity.CRITICAL:
                return None

        # Create alert
        alert = Alert(
            vehicle_id=vehicle_id,
            detection_id=detection_id,
            severity=severity,
            message=message,
            alert_type=alert_type,
            triggered_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=self.alert_duration)
        )

        db.add(alert)
        db.commit()
        db.refresh(alert)

        # Set cooldown and increment count
        self.set_cooldown(vehicle_id)
        self.increment_alert_count(vehicle_id)

        # Store last alert info in Redis
        self._store_last_alert(vehicle_id, alert)

        # Publish to Redis pub/sub for real-time updates
        self._publish_alert(alert)

        logger.info(
            f"Alert created: ID={alert.id}, Vehicle={vehicle_id}, "
            f"Severity={severity.value}, Type={alert_type}"
        )

        return alert

    def _store_last_alert(self, vehicle_id: int, alert: Alert):
        """Store last alert data in Redis for quick access."""
        key = self._get_last_alert_key(vehicle_id)
        data = {
            'id': alert.id,
            'severity': alert.severity.value,
            'message': alert.message,
            'triggered_at': alert.triggered_at.isoformat()
        }
        self.redis_client.setex(
            key,
            3600,  # Keep for 1 hour
            json.dumps(data)
        )

    def _publish_alert(self, alert: Alert):
        """Publish alert to Redis pub/sub for WebSocket broadcasting."""
        try:
            channel = f"alerts:vehicle:{alert.vehicle_id}"
            message = {
                'id': alert.id,
                'vehicle_id': alert.vehicle_id,
                'severity': alert.severity.value,
                'message': alert.message,
                'alert_type': alert.alert_type,
                'triggered_at': alert.triggered_at.isoformat()
            }
            self.redis_client.publish(channel, json.dumps(message))

            # Also publish to general alerts channel
            self.redis_client.publish("alerts:all", json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")

    def acknowledge_alert(
        self,
        db: Session,
        alert_id: int,
        user_id: int,
        notes: Optional[str] = None
    ) -> Optional[Alert]:
        """
        Acknowledge an alert.

        Args:
            db: Database session
            alert_id: Alert ID
            user_id: User acknowledging the alert
            notes: Optional notes

        Returns:
            Updated Alert object or None if not found
        """
        alert = db.query(Alert).filter(Alert.id == alert_id).first()

        if not alert:
            logger.warning(f"Alert {alert_id} not found")
            return None

        if alert.is_acknowledged:
            logger.warning(f"Alert {alert_id} already acknowledged")
            return alert

        alert.is_acknowledged = True
        alert.acknowledged_by = user_id
        alert.acknowledged_at = datetime.utcnow()
        alert.notes = notes

        db.commit()
        db.refresh(alert)

        logger.info(f"Alert {alert_id} acknowledged by user {user_id}")

        return alert

    def get_active_alerts(
        self,
        db: Session,
        vehicle_id: Optional[int] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            db: Database session
            vehicle_id: Filter by vehicle ID
            severity: Filter by severity
            limit: Maximum number of alerts

        Returns:
            List of Alert objects
        """
        query = db.query(Alert).filter(Alert.is_acknowledged == False)

        if vehicle_id:
            query = query.filter(Alert.vehicle_id == vehicle_id)

        if severity:
            query = query.filter(Alert.severity == severity)

        alerts = query.order_by(Alert.triggered_at.desc()).limit(limit).all()

        return alerts

    def get_alert_statistics(
        self,
        db: Session,
        vehicle_id: Optional[int] = None,
        hours: int = 24
    ) -> Dict:
        """
        Get alert statistics for dashboard.

        Args:
            db: Database session
            vehicle_id: Filter by vehicle ID
            hours: Time window in hours

        Returns:
            Dictionary with statistics
        """
        from sqlalchemy import func

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        query = db.query(Alert).filter(Alert.triggered_at >= cutoff_time)

        if vehicle_id:
            query = query.filter(Alert.vehicle_id == vehicle_id)

        # Total alerts
        total_alerts = query.count()

        # By severity
        severity_counts = (
            query.with_entities(Alert.severity, func.count(Alert.id))
            .group_by(Alert.severity)
            .all()
        )

        # Acknowledged vs unacknowledged
        acknowledged = query.filter(Alert.is_acknowledged == True).count()
        unacknowledged = total_alerts - acknowledged

        # Average response time (time to acknowledge)
        avg_response = (
            query.filter(Alert.is_acknowledged == True)
            .with_entities(
                func.avg(
                    func.extract('epoch', Alert.acknowledged_at - Alert.triggered_at)
                )
            )
            .scalar()
        )

        return {
            'total_alerts': total_alerts,
            'acknowledged': acknowledged,
            'unacknowledged': unacknowledged,
            'by_severity': {
                severity.value: count for severity, count in severity_counts
            },
            'avg_response_time_seconds': float(avg_response) if avg_response else 0.0,
            'time_window_hours': hours
        }

