"""
Alert API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.db.session import get_db
from app.models.models import Alert, AlertSeverity
from app.services.alert_service import AlertManager

router = APIRouter()
alert_manager = AlertManager()


class AlertResponse(BaseModel):
    id: int
    vehicle_id: int
    severity: str
    message: str
    alert_type: str
    is_acknowledged: bool
    triggered_at: datetime

    class Config:
        from_attributes = True


class AlertAcknowledge(BaseModel):
    notes: Optional[str] = None


@router.get("/", response_model=List[AlertResponse])
async def get_alerts(
    vehicle_id: Optional[int] = None,
    severity: Optional[AlertSeverity] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get alerts with filtering."""
    query = db.query(Alert)

    if vehicle_id:
        query = query.filter(Alert.vehicle_id == vehicle_id)
    if severity:
        query = query.filter(Alert.severity == severity)
    if acknowledged is not None:
        query = query.filter(Alert.is_acknowledged == acknowledged)

    alerts = query.order_by(Alert.triggered_at.desc()).limit(limit).all()
    return alerts


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: int, db: Session = Depends(get_db)):
    """Get specific alert details."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    ack_data: AlertAcknowledge,
    db: Session = Depends(get_db)
):
    """Acknowledge an alert."""
    # In production, get user_id from JWT token
    user_id = 1  # Mock user

    alert = alert_manager.acknowledge_alert(db, alert_id, user_id, ack_data.notes)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert acknowledged", "alert_id": alert_id}

