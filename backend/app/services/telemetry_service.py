"""
Vehicle Telematics Service.
Simulates CAN bus data including speed, RPM, location, brake status, etc.
"""

from typing import Dict, Optional
from datetime import datetime
import random
import math
from dataclasses import dataclass
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class TelemetryData:
    """Vehicle telemetry snapshot."""
    vehicle_id: int
    timestamp: datetime
    speed: float  # km/h
    rpm: int
    engine_temp: float  # Celsius
    fuel_level: float  # percentage
    brake_status: bool
    latitude: float
    longitude: float
    altitude: float
    heading: float  # degrees
    odometer: float  # km


class VehicleTelemetrySimulator:
    """
    Simulates realistic vehicle telemetry data.
    Mimics CAN bus data streams for testing without real hardware.
    """

    def __init__(self, vehicle_id: int, initial_location: tuple = (40.7128, -74.0060)):
        """
        Initialize simulator for a vehicle.

        Args:
            vehicle_id: Vehicle ID
            initial_location: Starting (latitude, longitude) - defaults to NYC
        """
        self.vehicle_id = vehicle_id
        self.speed = 0.0
        self.rpm = 800  # Idle RPM
        self.engine_temp = 90.0  # Normal operating temp
        self.fuel_level = random.uniform(20, 95)
        self.brake_status = False
        self.latitude, self.longitude = initial_location
        self.altitude = random.uniform(0, 100)
        self.heading = random.uniform(0, 360)
        self.odometer = random.uniform(1000, 50000)

        # Movement parameters
        self.target_speed = 0.0
        self.acceleration = 2.0  # km/h per update
        self.is_moving = False

        logger.info(f"Telemetry simulator initialized for vehicle {vehicle_id}")

    def update(self) -> TelemetryData:
        """
        Generate next telemetry reading.
        Simulates realistic vehicle behavior with smooth transitions.

        Returns:
            TelemetryData object with current state
        """
        # Decide if vehicle should move (70% chance if stopped, 90% chance if moving)
        if not self.is_moving and random.random() < 0.7:
            self.is_moving = True
            self.target_speed = random.uniform(20, settings.MAX_VEHICLE_SPEED)
        elif self.is_moving and random.random() < 0.1:
            self.is_moving = False
            self.target_speed = 0.0

        # Update speed gradually towards target
        if self.speed < self.target_speed:
            self.speed = min(self.speed + self.acceleration, self.target_speed)
        elif self.speed > self.target_speed:
            self.speed = max(self.speed - self.acceleration * 1.5, self.target_speed)  # Brake faster

        # Update RPM based on speed
        if self.speed > 0:
            self.rpm = int(800 + (self.speed * 40))  # Rough approximation
        else:
            self.rpm = 800  # Idle

        # Update engine temperature (tends towards 90°C)
        if self.speed > 40:
            self.engine_temp = min(self.engine_temp + 0.5, 105)
        else:
            self.engine_temp = max(self.engine_temp - 0.2, 85)

        # Brake status (true when decelerating)
        self.brake_status = self.speed > self.target_speed

        # Update fuel level (decreases slowly)
        if self.speed > 0:
            self.fuel_level = max(self.fuel_level - 0.001, 0)

        # Update GPS position (simulate movement)
        if self.speed > 0:
            # Convert speed to degrees per update (rough approximation)
            # 1 degree ≈ 111 km, update every second
            distance_km = (self.speed / 3600) * settings.TELEMETRY_UPDATE_INTERVAL
            distance_deg = distance_km / 111

            # Update position based on heading
            self.latitude += distance_deg * math.cos(math.radians(self.heading))
            self.longitude += distance_deg * math.sin(math.radians(self.heading))

            # Occasionally change heading
            if random.random() < 0.1:
                self.heading = (self.heading + random.uniform(-30, 30)) % 360

            # Update odometer
            self.odometer += distance_km

        return TelemetryData(
            vehicle_id=self.vehicle_id,
            timestamp=datetime.utcnow(),
            speed=round(self.speed, 2),
            rpm=self.rpm,
            engine_temp=round(self.engine_temp, 1),
            fuel_level=round(self.fuel_level, 1),
            brake_status=self.brake_status,
            latitude=round(self.latitude, 6),
            longitude=round(self.longitude, 6),
            altitude=round(self.altitude, 1),
            heading=round(self.heading, 1),
            odometer=round(self.odometer, 2)
        )

    def set_speed(self, speed: float):
        """
        Manually set target speed.

        Args:
            speed: Target speed in km/h
        """
        self.target_speed = min(max(speed, 0), settings.MAX_VEHICLE_SPEED)
        self.is_moving = speed > 0

    def trigger_brake(self):
        """Trigger sudden braking."""
        self.target_speed = 0.0
        self.brake_status = True
        self.is_moving = False

    def get_state_summary(self) -> Dict:
        """
        Get current state as dictionary.

        Returns:
            Dictionary with current telemetry
        """
        data = self.update()
        return {
            'vehicle_id': data.vehicle_id,
            'timestamp': data.timestamp.isoformat(),
            'speed': data.speed,
            'rpm': data.rpm,
            'engine_temp': data.engine_temp,
            'fuel_level': data.fuel_level,
            'brake_status': data.brake_status,
            'location': {
                'latitude': data.latitude,
                'longitude': data.longitude,
                'altitude': data.altitude,
                'heading': data.heading
            },
            'odometer': data.odometer
        }


class TelemetryService:
    """
    Service managing telemetry simulators for all vehicles.
    Provides centralized access to vehicle data.
    """

    def __init__(self):
        """Initialize telemetry service."""
        self.simulators: Dict[int, VehicleTelemetrySimulator] = {}
        logger.info("Telemetry service initialized")

    def register_vehicle(
        self,
        vehicle_id: int,
        initial_location: Optional[tuple] = None
    ) -> VehicleTelemetrySimulator:
        """
        Register a vehicle for telemetry simulation.

        Args:
            vehicle_id: Vehicle ID
            initial_location: Optional starting location

        Returns:
            Simulator instance
        """
        if vehicle_id in self.simulators:
            return self.simulators[vehicle_id]

        location = initial_location or self._generate_random_location()
        simulator = VehicleTelemetrySimulator(vehicle_id, location)
        self.simulators[vehicle_id] = simulator

        logger.info(f"Vehicle {vehicle_id} registered for telemetry")
        return simulator

    def _generate_random_location(self) -> tuple:
        """Generate random location in North America."""
        # Random location in continental US
        latitude = random.uniform(25, 49)
        longitude = random.uniform(-125, -65)
        return (latitude, longitude)

    def get_telemetry(self, vehicle_id: int) -> Optional[TelemetryData]:
        """
        Get current telemetry for vehicle.

        Args:
            vehicle_id: Vehicle ID

        Returns:
            TelemetryData or None if vehicle not registered
        """
        simulator = self.simulators.get(vehicle_id)
        if simulator:
            return simulator.update()
        return None

    def get_all_telemetry(self) -> Dict[int, TelemetryData]:
        """
        Get telemetry for all registered vehicles.

        Returns:
            Dictionary mapping vehicle_id to TelemetryData
        """
        return {
            vehicle_id: simulator.update()
            for vehicle_id, simulator in self.simulators.items()
        }

    def unregister_vehicle(self, vehicle_id: int):
        """
        Remove vehicle from telemetry tracking.

        Args:
            vehicle_id: Vehicle ID
        """
        if vehicle_id in self.simulators:
            del self.simulators[vehicle_id]
            logger.info(f"Vehicle {vehicle_id} unregistered from telemetry")


# Global telemetry service instance
telemetry_service = TelemetryService()

