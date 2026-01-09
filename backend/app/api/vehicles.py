"""
Vehicle API endpoints.
Handles fleet management, vehicle status, and live monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json
import asyncio
import logging

from app.db.session import get_db, get_redis
from app.models.models import Vehicle, VehicleStatus, Fleet
from app.services.telemetry_service import telemetry_service
from app.services.motion_detection import CombinedDetectionService
import cv2
import numpy as np
import base64

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic schemas for request/response
from pydantic import BaseModel

class VehicleResponse(BaseModel):
    id: int
    device_id: str
    vehicle_number: str
    status: str
    last_heartbeat: Optional[datetime]
    latitude: Optional[float]
    longitude: Optional[float]
    speed: Optional[float]

    class Config:
        from_attributes = True


class VehicleCreate(BaseModel):
    device_id: str
    vehicle_number: str
    fleet_id: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None


class VehicleUpdate(BaseModel):
    vehicle_number: Optional[str] = None
    status: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@router.get("/", response_model=List[VehicleResponse])
async def get_vehicles(
    status: Optional[VehicleStatus] = None,
    fleet_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get list of vehicles with optional filtering.

    Query Parameters:
        - status: Filter by vehicle status (online, offline, maintenance, alert)
        - fleet_id: Filter by fleet ID
        - skip: Pagination offset
        - limit: Maximum results
    """
    query = db.query(Vehicle)

    if status:
        query = query.filter(Vehicle.status == status)

    if fleet_id:
        query = query.filter(Vehicle.fleet_id == fleet_id)

    vehicles = query.offset(skip).limit(limit).all()

    return vehicles


@router.post("/", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED)
async def create_vehicle(
    vehicle: VehicleCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new vehicle in the system.
    """
    # Check if device_id already exists
    existing = db.query(Vehicle).filter(Vehicle.device_id == vehicle.device_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Vehicle with device_id {vehicle.device_id} already exists"
        )

    # Create vehicle
    db_vehicle = Vehicle(**vehicle.dict())
    db.add(db_vehicle)
    db.commit()
    db.refresh(db_vehicle)

    # Register with telemetry service
    telemetry_service.register_vehicle(db_vehicle.id)

    logger.info(f"Vehicle created: {db_vehicle.id} - {db_vehicle.vehicle_number}")

    return db_vehicle


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(
    vehicle_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific vehicle.
    """
    vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()

    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle {vehicle_id} not found"
        )

    return vehicle


@router.put("/{vehicle_id}", response_model=VehicleResponse)
async def update_vehicle(
    vehicle_id: int,
    vehicle_update: VehicleUpdate,
    db: Session = Depends(get_db)
):
    """
    Update vehicle information.
    """
    vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()

    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle {vehicle_id} not found"
        )

    # Update fields
    for field, value in vehicle_update.dict(exclude_unset=True).items():
        setattr(vehicle, field, value)

    vehicle.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(vehicle)

    return vehicle


@router.get("/{vehicle_id}/live")
async def get_vehicle_live_data(
    vehicle_id: int,
    db: Session = Depends(get_db)
):
    """
    Get current live data for a vehicle (telemetry + status).
    """
    vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()

    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle {vehicle_id} not found"
        )

    # Get telemetry
    telemetry = telemetry_service.get_telemetry(vehicle_id)

    # Get last alert from Redis
    redis_client = get_redis()
    last_alert_key = f"alert:last:vehicle:{vehicle_id}"
    last_alert_data = redis_client.get(last_alert_key)
    last_alert = json.loads(last_alert_data) if last_alert_data else None

    return {
        "vehicle": {
            "id": vehicle.id,
            "device_id": vehicle.device_id,
            "vehicle_number": vehicle.vehicle_number,
            "status": vehicle.status.value,
            "last_heartbeat": vehicle.last_heartbeat.isoformat() if vehicle.last_heartbeat else None
        },
        "telemetry": telemetry.get_state_summary() if telemetry else None,
        "last_alert": last_alert
    }


@router.websocket("/{vehicle_id}/stream")
async def vehicle_stream(
    websocket: WebSocket,
    vehicle_id: int
):
    """
    WebSocket endpoint for real-time vehicle data streaming.
    Streams telemetry, motion detection, and alerts.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for vehicle {vehicle_id}")

    # Initialize detection service
    detector = CombinedDetectionService()

    # Simulated camera (in production, this would connect to actual camera)
    cap = cv2.VideoCapture(0)  # Use webcam for demo, or video file

    try:
        while True:
            # Get telemetry
            telemetry = telemetry_service.get_telemetry(vehicle_id)

            if telemetry:
                # Capture frame (simulated)
                ret, frame = cap.read()

                if ret:
                    # Run detection
                    result = detector.process_frame(frame)

                    # Annotate frame
                    annotated_frame = detector.annotate_frame(frame, result)

                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Prepare message
                    message = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "telemetry": {
                            "speed": telemetry.speed,
                            "rpm": telemetry.rpm,
                            "location": {
                                "lat": telemetry.latitude,
                                "lon": telemetry.longitude
                            }
                        },
                        "detection": {
                            "motion_detected": result.motion_detected,
                            "motion_percentage": result.motion_percentage,
                            "confidence": result.confidence_score,
                            "objects": result.objects
                        },
                        "frame": frame_base64
                    }

                    await websocket.send_json(message)

                await asyncio.sleep(1.0 / 10)  # 10 FPS
            else:
                await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for vehicle {vehicle_id}")
    except Exception as e:
        logger.error(f"WebSocket error for vehicle {vehicle_id}: {e}")
    finally:
        cap.release()


@router.get("/{vehicle_id}/statistics")
async def get_vehicle_statistics(
    vehicle_id: int,
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """
    Get statistics for a specific vehicle.
    """
    from datetime import timedelta
    from sqlalchemy import func
    from app.models.models import Detection, Alert

    vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    cutoff = datetime.utcnow() - timedelta(hours=hours)

    # Detection count
    detection_count = db.query(func.count(Detection.id)).filter(
        Detection.vehicle_id == vehicle_id,
        Detection.timestamp >= cutoff
    ).scalar()

    # Alert count
    alert_count = db.query(func.count(Alert.id)).filter(
        Alert.vehicle_id == vehicle_id,
        Alert.triggered_at >= cutoff
    ).scalar()

    # Average motion percentage
    avg_motion = db.query(func.avg(Detection.motion_percentage)).filter(
        Detection.vehicle_id == vehicle_id,
        Detection.timestamp >= cutoff
    ).scalar()

    return {
        "vehicle_id": vehicle_id,
        "time_window_hours": hours,
        "detection_count": detection_count or 0,
        "alert_count": alert_count or 0,
        "avg_motion_percentage": float(avg_motion) if avg_motion else 0.0
    }

