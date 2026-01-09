"""
Detection API endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.db.session import get_db
from app.models.models import Detection

router = APIRouter()


class DetectionResponse(BaseModel):
    id: int
    vehicle_id: int
    timestamp: datetime
    motion_percentage: float
    confidence_score: float

    class Config:
        from_attributes = True


@router.get("/history", response_model=List[DetectionResponse])
async def get_detection_history(
    vehicle_id: int = None,
    hours: int = 24,
    limit: int = 1000,
    db: Session = Depends(get_db)
):
    """Get historical detection data."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    query = db.query(Detection).filter(Detection.timestamp >= cutoff)

    if vehicle_id:
        query = query.filter(Detection.vehicle_id == vehicle_id)

    detections = query.order_by(Detection.timestamp.desc()).limit(limit).all()
    return detections

