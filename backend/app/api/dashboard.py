"""
Dashboard API endpoints - KPIs and metrics.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import Dict

from app.db.session import get_db
from app.models.models import Vehicle, Detection, Alert, VehicleStatus
from app.services.alert_service import AlertManager

router = APIRouter()
alert_manager = AlertManager()


@router.get("/metrics")
async def get_dashboard_metrics(
    hours: int = 24,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Get key performance indicators for dashboard.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    # Total vehicles
    total_vehicles = db.query(func.count(Vehicle.id)).scalar()

    # Vehicles by status
    online_vehicles = db.query(func.count(Vehicle.id)).filter(
        Vehicle.status == VehicleStatus.ONLINE
    ).scalar()

    alert_vehicles = db.query(func.count(Vehicle.id)).filter(
        Vehicle.status == VehicleStatus.ALERT
    ).scalar()

    # Active alerts
    active_alerts = db.query(func.count(Alert.id)).filter(
        Alert.is_acknowledged == False
    ).scalar()

    # Detections in time window
    recent_detections = db.query(func.count(Detection.id)).filter(
        Detection.timestamp >= cutoff
    ).scalar()

    # Detection rate (per hour)
    detection_rate = recent_detections / hours if hours > 0 else 0

    # Alert statistics
    alert_stats = alert_manager.get_alert_statistics(db, hours=hours)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "time_window_hours": hours,
        "vehicles": {
            "total": total_vehicles,
            "online": online_vehicles,
            "alert_status": alert_vehicles,
            "offline": total_vehicles - online_vehicles
        },
        "alerts": {
            "active": active_alerts,
            "total_period": alert_stats['total_alerts'],
            "by_severity": alert_stats['by_severity'],
            "avg_response_time": alert_stats['avg_response_time_seconds']
        },
        "detections": {
            "total": recent_detections,
            "rate_per_hour": round(detection_rate, 2)
        }
    }


@router.get("/fleet-status")
async def get_fleet_status(db: Session = Depends(get_db)):
    """Get current status of all vehicles."""
    vehicles = db.query(Vehicle).all()

    vehicle_list = []
    for vehicle in vehicles:
        vehicle_list.append({
            "id": vehicle.id,
            "vehicle_number": vehicle.vehicle_number,
            "status": vehicle.status.value,
            "last_heartbeat": vehicle.last_heartbeat.isoformat() if vehicle.last_heartbeat else None,
            "location": {
                "lat": vehicle.latitude,
                "lon": vehicle.longitude
            } if vehicle.latitude and vehicle.longitude else None
        })

    return {"vehicles": vehicle_list, "total": len(vehicle_list)}

