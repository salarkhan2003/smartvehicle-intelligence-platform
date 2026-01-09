-- TNT Surveillance T-SA Fleet Management Database Schema
-- PostgreSQL 15+

CREATE DATABASE tnt_fleet_management;

\c tnt_fleet_management;

-- Vehicles table
CREATE TABLE vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(50) UNIQUE NOT NULL,
    vehicle_number VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'offline',
    last_heartbeat TIMESTAMP,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    make VARCHAR(50),
    model VARCHAR(50),
    year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detections table
CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(50) REFERENCES vehicles(vehicle_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_type VARCHAR(50) NOT NULL,  -- 'person', 'car', 'truck', etc.
    confidence DOUBLE PRECISION NOT NULL,
    distance DOUBLE PRECISION,  -- meters
    threat_level VARCHAR(20),  -- 'none', 'warning', 'high', 'critical'
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    image_snapshot TEXT,  -- Base64 encoded
    gps_latitude DOUBLE PRECISION,
    gps_longitude DOUBLE PRECISION,
    vehicle_speed DOUBLE PRECISION,
    turn_signal VARCHAR(10),
    gear VARCHAR(5)
);

-- Traffic Violations table (NEW for Traffic Enforcement)
CREATE TABLE violations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    violation_type VARCHAR(50) NOT NULL,  -- 'no_helmet','no_seatbelt','over_speeding','wrong_lane','red_light','rash_driving'
    vehicle_type VARCHAR(20),  -- 'motorcycle','car','truck','bus'
    license_plate VARCHAR(20),
    confidence_score DOUBLE PRECISION,  -- 0.0-1.0
    speed_estimated DOUBLE PRECISION,  -- km/h if speed violation
    speed_limit DOUBLE PRECISION,  -- posted limit
    gps_lat DOUBLE PRECISION,
    gps_lon DOUBLE PRECISION,
    location_name VARCHAR(100),
    evidence_image_path VARCHAR(255),
    evidence_metadata JSONB,  -- Additional data: shoulder_angle, trajectory, etc
    oat_alert_sent BOOLEAN DEFAULT FALSE,
    city_traffic_api_synced BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_violation_type (violation_type),
    INDEX idx_timestamp (timestamp),
    INDEX idx_license_plate (license_plate),
    INDEX idx_location (gps_lat, gps_lon)
);

-- Violation statistics (aggregate data)
CREATE TABLE violation_stats (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    location_name VARCHAR(100),
    violation_type VARCHAR(50),
    total_count INTEGER DEFAULT 0,
    avg_confidence DOUBLE PRECISION,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(date, location_name, violation_type)
);

-- Repeat violators tracking
CREATE TABLE repeat_violators (
    id SERIAL PRIMARY KEY,
    license_plate VARCHAR(20) UNIQUE NOT NULL,
    violation_count INTEGER DEFAULT 1,
    last_violation_date TIMESTAMP,
    violation_types JSONB,  -- Array of violation types
    flagged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_plate (license_plate),
    INDEX idx_flagged (flagged)
);

-- Alerts table
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(50) REFERENCES vehicles(vehicle_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    threat_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by INTEGER,
    acknowledged_at TIMESTAMP,
    detection_count INTEGER DEFAULT 1
);

-- Events log table
CREATE TABLE event_log (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(50) REFERENCES vehicles(vehicle_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50),
    event_data JSONB,
    severity VARCHAR(20)
);

-- Driver fatigue detection
CREATE TABLE fatigue_events (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(50) REFERENCES vehicles(vehicle_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fatigue_level INTEGER,  -- 0-100
    eye_closure_duration DOUBLE PRECISION,  -- seconds
    yawn_detected BOOLEAN,
    head_pose_angle DOUBLE PRECISION,
    alert_triggered BOOLEAN,
    image_snapshot TEXT
);

-- Forward collision warnings
CREATE TABLE collision_warnings (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(50) REFERENCES vehicles(vehicle_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttc DOUBLE PRECISION,  -- Time to collision (seconds)
    relative_speed DOUBLE PRECISION,  -- km/h
    object_type VARCHAR(50),
    distance DOUBLE PRECISION,  -- meters
    brake_applied BOOLEAN,
    collision_avoided BOOLEAN
);

-- Indexes for performance
CREATE INDEX idx_detections_vehicle_id ON detections(vehicle_id);
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
CREATE INDEX idx_detections_type ON detections(detection_type);
CREATE INDEX idx_alerts_vehicle_id ON alerts(vehicle_id);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX idx_alerts_acknowledged ON alerts(is_acknowledged);
CREATE INDEX idx_event_log_vehicle_id ON event_log(vehicle_id);
CREATE INDEX idx_event_log_timestamp ON event_log(timestamp);

-- Partitioning for large datasets (optional)
-- Partition detections table by date for better query performance
CREATE TABLE detections_2026_01 PARTITION OF detections
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE detections_2026_02 PARTITION OF detections
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- Sample data for testing
INSERT INTO vehicles (vehicle_id, vehicle_number, status, latitude, longitude, speed) VALUES
('TNT-VEH-001', 'TRUCK-001', 'online', 40.7128, -74.0060, 45.5),
('TNT-VEH-002', 'TRUCK-002', 'online', 40.7580, -73.9855, 60.2),
('TNT-VEH-003', 'TRUCK-003', 'online', 40.7489, -73.9680, 55.0),
('TNT-VEH-004', 'TRUCK-004', 'offline', 40.7614, -73.9776, 0.0),
('TNT-VEH-005', 'TRUCK-005', 'maintenance', 40.7831, -73.9712, 0.0);

-- Views for analytics
CREATE VIEW daily_detection_summary AS
SELECT
    vehicle_id,
    DATE(timestamp) as detection_date,
    COUNT(*) as total_detections,
    SUM(CASE WHEN detection_type = 'person' THEN 1 ELSE 0 END) as pedestrian_detections,
    SUM(CASE WHEN threat_level = 'critical' THEN 1 ELSE 0 END) as critical_threats,
    AVG(distance) as avg_distance
FROM detections
GROUP BY vehicle_id, DATE(timestamp);

CREATE VIEW active_alerts_view AS
SELECT
    a.id,
    a.vehicle_id,
    v.vehicle_number,
    a.timestamp,
    a.threat_level,
    a.message,
    a.is_acknowledged,
    a.detection_count
FROM alerts a
JOIN vehicles v ON a.vehicle_id = v.vehicle_id
WHERE a.is_acknowledged = FALSE
ORDER BY a.timestamp DESC;

-- Functions for data retention
CREATE OR REPLACE FUNCTION cleanup_old_detections(days_to_keep INTEGER)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM detections
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update vehicle last_heartbeat
CREATE OR REPLACE FUNCTION update_vehicle_heartbeat()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE vehicles
    SET last_heartbeat = NEW.timestamp,
        status = 'online'
    WHERE vehicle_id = NEW.vehicle_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER detection_heartbeat_trigger
AFTER INSERT ON detections
FOR EACH ROW
EXECUTE FUNCTION update_vehicle_heartbeat();

-- Grant permissions (adjust username as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO tnt_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO tnt_admin;

