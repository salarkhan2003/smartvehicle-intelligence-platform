"""
Core configuration management for TNT Vehicle Intelligence Platform.
Loads environment variables and provides centralized settings access.
"""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "TNT Vehicle Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database (SQLite for demo - no PostgreSQL required)
    DATABASE_URL: str = "sqlite:///./tnt_platform.db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # Redis (Optional - not required for demo)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SESSION_DB: int = 1
    REDIS_CACHE_DB: int = 2
    REDIS_QUEUE_DB: int = 3

    # JWT Authentication
    SECRET_KEY: str = "tnt_surveillance_secret_key_change_in_production_2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # Security
    API_RATE_LIMIT: int = 100
    PASSWORD_MIN_LENGTH: int = 8

    # Motion Detection
    MOTION_DETECTION_FPS: int = 30
    MOTION_THRESHOLD: int = 50
    BACKGROUND_SUBTRACTOR: str = "MOG2"
    MIN_CONTOUR_AREA: int = 500

    # YOLO
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.4
    YOLO_CLASSES: str = "person,car,truck,bus,motorcycle,bicycle"

    @property
    def yolo_classes_list(self) -> List[str]:
        return [cls.strip() for cls in self.YOLO_CLASSES.split(",")]

    # Alerts
    ALERT_COOLDOWN_SECONDS: int = 3
    ALERT_DURATION_SECONDS: int = 5
    ALERT_SEVERITY_LEVELS: str = "LOW,MEDIUM,HIGH,CRITICAL"
    MAX_ALERTS_PER_VEHICLE_PER_HOUR: int = 100

    # Telemetry
    TELEMETRY_UPDATE_INTERVAL: int = 1
    MAX_VEHICLE_SPEED: int = 120
    SIMULATE_CAN_BUS: bool = True

    # Data Retention
    DETECTION_RETENTION_DAYS: int = 90
    ALERT_RETENTION_DAYS: int = 365
    LOG_RETENTION_DAYS: int = 30

    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_IMAGE_TYPES: str = "jpg,jpeg,png,bmp"

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # WebSocket
    WS_PING_INTERVAL: int = 30
    WS_PING_TIMEOUT: int = 10
    WS_MAX_MESSAGE_SIZE: int = 10485760

    # Email (optional)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "TNT Surveillance <alerts@tntsurveillance.com>"

    # Deployment
    WORKERS: int = 4
    RELOAD: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid reading .env file on every call.
    """
    return Settings()


# Global settings instance
settings = get_settings()

