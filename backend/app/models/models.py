"""
Database models for TNT Vehicle Intelligence Platform.
SQLAlchemy ORM models for PostgreSQL database.
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, Enum as SQLEnum, Index, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum


Base = declarative_base()


class UserRole(str, enum.Enum):
    """User access roles."""
    ADMIN = "admin"
    TECHNICIAN = "technician"
    DRIVER = "driver"
    OBSERVER = "observer"


class VehicleStatus(str, enum.Enum):
    """Vehicle online/offline status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ALERT = "alert"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class User(Base):
    """User accounts with role-based access."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(SQLEnum(UserRole), default=UserRole.OBSERVER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)

    # Relationships
    audit_logs = relationship("AuditLog", back_populates="user")
    acknowledged_alerts = relationship("Alert", back_populates="acknowledged_by_user")


class Fleet(Base):
    """Fleet grouping for vehicles."""
    __tablename__ = "fleets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    vehicles = relationship("Vehicle", back_populates="fleet")


class Vehicle(Base):
    """Vehicle information and current state."""
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(100), unique=True, nullable=False, index=True)
    vehicle_number = Column(String(50), nullable=False, index=True)
    fleet_id = Column(Integer, ForeignKey("fleets.id"))

    # Vehicle details
    make = Column(String(50))
    model = Column(String(50))
    year = Column(Integer)
    vin = Column(String(17))

    # Status
    status = Column(SQLEnum(VehicleStatus), default=VehicleStatus.OFFLINE, nullable=False)
    last_heartbeat = Column(DateTime)

    # Location
    latitude = Column(Float)
    longitude = Column(Float)

    # Telemetry (current values)
    speed = Column(Float, default=0.0)
    rpm = Column(Integer, default=0)
    brake_status = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    fleet = relationship("Fleet", back_populates="vehicles")
    detections = relationship("Detection", back_populates="vehicle")
    alerts = relationship("Alert", back_populates="vehicle")
    telemetry_readings = relationship("TelemetryReading", back_populates="vehicle")

    # Indexes
    __table_args__ = (
        Index('idx_vehicle_status_heartbeat', 'status', 'last_heartbeat'),
    )


class Detection(Base):
    """Motion detection events from vehicles."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)

    # Detection data
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    motion_percentage = Column(Float)  # 0-100
    confidence_score = Column(Float)  # 0-1

    # Objects detected (JSON array)
    objects_detected = Column(JSON)  # [{class: "person", confidence: 0.95, bbox: [x,y,w,h]}]

    # Image reference
    image_path = Column(String(255))
    thumbnail_path = Column(String(255))

    # Telemetry at detection time
    vehicle_speed = Column(Float)
    vehicle_location = Column(JSON)  # {lat, lon}

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="detections")
    alerts = relationship("Alert", back_populates="detection")

    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_detection_vehicle_timestamp', 'vehicle_id', 'timestamp'),
        Index('idx_detection_timestamp', 'timestamp'),
    )


class Alert(Base):
    """Alerts generated from detections."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), index=True)

    # Alert details
    severity = Column(SQLEnum(AlertSeverity), default=AlertSeverity.MEDIUM, nullable=False)
    message = Column(Text, nullable=False)
    alert_type = Column(String(50))  # "motion_detected", "object_detected", "speed_violation"

    # Status
    is_acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"))
    acknowledged_at = Column(DateTime)
    notes = Column(Text)

    # Timestamps
    triggered_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="alerts")
    detection = relationship("Detection", back_populates="alerts")
    acknowledged_by_user = relationship("User", back_populates="acknowledged_alerts")

    # Indexes
    __table_args__ = (
        Index('idx_alert_vehicle_triggered', 'vehicle_id', 'triggered_at'),
        Index('idx_alert_severity', 'severity'),
        Index('idx_alert_acknowledged', 'is_acknowledged'),
    )


class TelemetryReading(Base):
    """Time-series vehicle telemetry data."""
    __tablename__ = "telemetry_readings"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # CAN bus data
    speed = Column(Float)  # km/h
    rpm = Column(Integer)
    engine_temp = Column(Float)  # Celsius
    fuel_level = Column(Float)  # percentage
    brake_status = Column(Boolean)

    # GPS
    latitude = Column(Float)
    longitude = Column(Float)
    altitude = Column(Float)
    heading = Column(Float)  # degrees

    # Additional data (JSON for flexibility)
    extra_data = Column(JSON)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="telemetry_readings")

    # Partitioning strategy: partition by month
    __table_args__ = (
        Index('idx_telemetry_vehicle_timestamp', 'vehicle_id', 'timestamp'),
    )


class AuditLog(Base):
    """Audit trail for all user actions."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)

    action = Column(String(100), nullable=False)  # "login", "acknowledge_alert", "modify_config"
    resource = Column(String(100))  # "alert:123", "vehicle:456"
    details = Column(JSON)  # Additional context

    ip_address = Column(String(45))
    user_agent = Column(String(255))

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")


class SystemConfig(Base):
    """System-wide configuration parameters."""
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    value_type = Column(String(20))  # "string", "int", "float", "bool", "json"
    description = Column(Text)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(Integer, ForeignKey("users.id"))

