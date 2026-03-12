"""
Central SQLite database for the City Event Causality Engine.
Stores live API data, events, causal discovery results, audit trail, and LLM logs.
"""
import sqlite3
import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "city_engine.db")

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Thread-safe connection (one per thread)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
    -- Live weather data from API
    CREATE TABLE IF NOT EXISTS weather_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        temperature REAL,
        humidity REAL,
        rainfall_mm REAL,
        wind_speed REAL,
        condition TEXT,
        raw_json TEXT,
        source TEXT DEFAULT 'openweathermap',
        created_at TEXT DEFAULT (datetime('now'))
    );

    -- Live traffic data from API
    CREATE TABLE IF NOT EXISTS traffic_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        congestion_level REAL,
        average_speed REAL,
        incident_count INTEGER DEFAULT 0,
        road_segment TEXT,
        raw_json TEXT,
        source TEXT DEFAULT 'tomtom',
        created_at TEXT DEFAULT (datetime('now'))
    );

    -- Ingested events (replaces in-memory event_store)
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE NOT NULL,
        event_type TEXT NOT NULL,
        zone TEXT NOT NULL,
        severity TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        source TEXT,
        metadata TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );

    -- Causal discovery results
    CREATE TABLE IF NOT EXISTS causal_discovery (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        algorithm TEXT NOT NULL,
        zone TEXT,
        edges TEXT NOT NULL,
        scores TEXT,
        p_values TEXT,
        lag_info TEXT,
        parameters TEXT,
        data_points_used INTEGER,
        timestamp TEXT DEFAULT (datetime('now'))
    );

    -- LLM interaction logs
    CREATE TABLE IF NOT EXISTS llm_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone TEXT,
        event_context TEXT,
        prompt TEXT NOT NULL,
        response TEXT NOT NULL,
        model TEXT,
        tokens_used INTEGER,
        timestamp TEXT DEFAULT (datetime('now'))
    );

    -- Unknown cause discoveries
    CREATE TABLE IF NOT EXISTS unknown_causes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone TEXT NOT NULL,
        event_type TEXT NOT NULL,
        discovered_cause TEXT NOT NULL,
        confidence REAL,
        evidence TEXT,
        algorithm TEXT,
        status TEXT DEFAULT 'pending',
        timestamp TEXT DEFAULT (datetime('now'))
    );

    -- Audit trail
    CREATE TABLE IF NOT EXISTS audit_trail (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone TEXT NOT NULL,
        event_type TEXT NOT NULL,
        event_description TEXT,
        event_timestamp TEXT NOT NULL,
        severity TEXT,
        detection_method TEXT,
        action_taken TEXT,
        action_timestamp TEXT,
        resolved INTEGER DEFAULT 0,
        resolution_timestamp TEXT,
        resolution_description TEXT,
        outcome TEXT,
        operator TEXT DEFAULT 'system',
        notes TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );

    -- Time-series data for causal discovery (common attribute space)
    CREATE TABLE IF NOT EXISTS timeseries_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        -- weather
        rainfall REAL DEFAULT 0,
        temperature REAL DEFAULT 0,
        humidity REAL DEFAULT 0,
        wind_speed REAL DEFAULT 0,
        -- infrastructure
        drainage_load REAL DEFAULT 0,
        power_grid_load REAL DEFAULT 0,
        water_supply_pressure REAL DEFAULT 0,
        -- incidents
        flooding_level REAL DEFAULT 0,
        accident_count REAL DEFAULT 0,
        power_outage REAL DEFAULT 0,
        fire_incident REAL DEFAULT 0,
        -- traffic
        traffic_congestion REAL DEFAULT 0,
        emergency_delay REAL DEFAULT 0,
        -- environment
        heatwave_index REAL DEFAULT 0,
        air_quality_index REAL DEFAULT 0,
        -- human activity
        construction_activity REAL DEFAULT 0,
        public_event_crowd REAL DEFAULT 0,
        industrial_discharge REAL DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_weather_zone_ts ON weather_data(zone, timestamp);
    CREATE INDEX IF NOT EXISTS idx_traffic_zone_ts ON traffic_data(zone, timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone, timestamp);
    CREATE INDEX IF NOT EXISTS idx_timeseries_zone ON timeseries_data(zone, timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_zone ON audit_trail(zone, event_timestamp);
    CREATE INDEX IF NOT EXISTS idx_causal_algo ON causal_discovery(algorithm, timestamp);
    """)
    conn.commit()


# ── Generic helpers ──────────────────────────────────────────────────────────

def insert_row(table: str, data: Dict[str, Any]) -> int:
    conn = get_connection()
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    cur = conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        list(data.values()),
    )
    conn.commit()
    return cur.lastrowid


def query_rows(table: str, where: str = "", params: tuple = (), limit: int = 500, order: str = "id DESC") -> List[Dict]:
    conn = get_connection()
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += f" ORDER BY {order} LIMIT {limit}"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def query_raw(sql: str, params: tuple = ()) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ── Weather data ─────────────────────────────────────────────────────────────

def store_weather(zone: str, data: Dict) -> int:
    return insert_row("weather_data", {
        "zone": zone,
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "temperature": data.get("temperature"),
        "humidity": data.get("humidity"),
        "rainfall_mm": data.get("rainfall_mm", 0),
        "wind_speed": data.get("wind_speed"),
        "condition": data.get("condition"),
        "raw_json": json.dumps(data),
        "source": data.get("source", "openweathermap"),
    })


def get_weather(zone: str, hours: int = 24) -> List[Dict]:
    return query_rows(
        "weather_data",
        where="zone = ? AND timestamp >= datetime('now', ?)",
        params=(zone, f"-{hours} hours"),
        order="timestamp ASC",
    )


# ── Traffic data ─────────────────────────────────────────────────────────────

def store_traffic(zone: str, data: Dict) -> int:
    return insert_row("traffic_data", {
        "zone": zone,
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "congestion_level": data.get("congestion_level"),
        "average_speed": data.get("average_speed"),
        "incident_count": data.get("incident_count", 0),
        "road_segment": data.get("road_segment"),
        "raw_json": json.dumps(data),
        "source": data.get("source", "tomtom"),
    })


def get_traffic(zone: str, hours: int = 24) -> List[Dict]:
    return query_rows(
        "traffic_data",
        where="zone = ? AND timestamp >= datetime('now', ?)",
        params=(zone, f"-{hours} hours"),
        order="timestamp ASC",
    )


# ── Events ───────────────────────────────────────────────────────────────────

def store_event(event: Dict) -> int:
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        metadata = json.dumps(metadata)
    return insert_row("events", {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "zone": event["zone"],
        "severity": event["severity"],
        "timestamp": event.get("timestamp", datetime.now().isoformat()),
        "source": event.get("source", "system"),
        "metadata": metadata,
    })


def get_events(zone: Optional[str] = None, limit: int = 100) -> List[Dict]:
    if zone:
        return query_rows("events", where="zone = ?", params=(zone,), limit=limit, order="timestamp DESC")
    return query_rows("events", limit=limit, order="timestamp DESC")


# ── Time-series data ─────────────────────────────────────────────────────────

def store_timeseries(zone: str, data: Dict) -> int:
    from causal_engine.attribute_space import ATTRIBUTE_NAMES
    row = {
        "zone": zone,
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
    }
    for attr in ATTRIBUTE_NAMES:
        row[attr] = data.get(attr, 0)
    return insert_row("timeseries_data", row)


def get_timeseries(zone: str, hours: int = 168) -> List[Dict]:
    """Get time-series for a zone. Default 7 days = 168 hours."""
    return query_rows(
        "timeseries_data",
        where="zone = ? AND timestamp >= datetime('now', ?)",
        params=(zone, f"-{hours} hours"),
        order="timestamp ASC",
        limit=5000,
    )


# ── Causal discovery results ─────────────────────────────────────────────────

def store_causal_result(data: Dict) -> int:
    return insert_row("causal_discovery", {
        "algorithm": data["algorithm"],
        "zone": data.get("zone"),
        "edges": json.dumps(data["edges"]),
        "scores": json.dumps(data.get("scores", {})),
        "p_values": json.dumps(data.get("p_values", {})),
        "lag_info": json.dumps(data.get("lag_info", {})),
        "parameters": json.dumps(data.get("parameters", {})),
        "data_points_used": data.get("data_points_used", 0),
    })


def get_causal_results(algorithm: Optional[str] = None, zone: Optional[str] = None, limit: int = 20) -> List[Dict]:
    conditions = []
    params = []
    if algorithm:
        conditions.append("algorithm = ?")
        params.append(algorithm)
    if zone:
        conditions.append("zone = ?")
        params.append(zone)
    where = " AND ".join(conditions) if conditions else ""
    results = query_rows("causal_discovery", where=where, params=tuple(params), limit=limit)
    for r in results:
        for key in ("edges", "scores", "p_values", "lag_info", "parameters"):
            if r.get(key) and isinstance(r[key], str):
                try:
                    r[key] = json.loads(r[key])
                except (json.JSONDecodeError, TypeError):
                    pass
    return results


# ── LLM logs ─────────────────────────────────────────────────────────────────

def store_llm_log(data: Dict) -> int:
    return insert_row("llm_logs", {
        "zone": data.get("zone"),
        "event_context": json.dumps(data.get("event_context", {})),
        "prompt": data["prompt"],
        "response": data["response"],
        "model": data.get("model", "unknown"),
        "tokens_used": data.get("tokens_used", 0),
    })


def get_llm_logs(zone: Optional[str] = None, limit: int = 50) -> List[Dict]:
    if zone:
        return query_rows("llm_logs", where="zone = ?", params=(zone,), limit=limit)
    return query_rows("llm_logs", limit=limit)


# ── Unknown causes ───────────────────────────────────────────────────────────

def store_unknown_cause(data: Dict) -> int:
    return insert_row("unknown_causes", {
        "zone": data["zone"],
        "event_type": data["event_type"],
        "discovered_cause": data["discovered_cause"],
        "confidence": data.get("confidence", 0.0),
        "evidence": json.dumps(data.get("evidence", {})),
        "algorithm": data.get("algorithm", "anomaly_detection"),
        "status": data.get("status", "pending"),
    })


def get_unknown_causes(zone: Optional[str] = None, limit: int = 50) -> List[Dict]:
    if zone:
        return query_rows("unknown_causes", where="zone = ?", params=(zone,), limit=limit)
    return query_rows("unknown_causes", limit=limit)


# ── Audit trail ──────────────────────────────────────────────────────────────

def store_audit_entry(data: Dict) -> int:
    return insert_row("audit_trail", {
        "zone": data["zone"],
        "event_type": data["event_type"],
        "event_description": data.get("event_description", ""),
        "event_timestamp": data["event_timestamp"],
        "severity": data.get("severity", "medium"),
        "detection_method": data.get("detection_method", "bayesian_network"),
        "action_taken": data.get("action_taken"),
        "action_timestamp": data.get("action_timestamp"),
        "resolved": data.get("resolved", 0),
        "resolution_timestamp": data.get("resolution_timestamp"),
        "resolution_description": data.get("resolution_description"),
        "outcome": data.get("outcome"),
        "operator": data.get("operator", "system"),
        "notes": data.get("notes"),
    })


def update_audit_entry(entry_id: int, updates: Dict):
    conn = get_connection()
    sets = ", ".join(f"{k} = ?" for k in updates.keys())
    conn.execute(f"UPDATE audit_trail SET {sets} WHERE id = ?", list(updates.values()) + [entry_id])
    conn.commit()


def get_audit_trail(zone: Optional[str] = None, limit: int = 100) -> List[Dict]:
    if zone:
        return query_rows("audit_trail", where="zone = ?", params=(zone,), limit=limit, order="event_timestamp DESC")
    return query_rows("audit_trail", limit=limit, order="event_timestamp DESC")


def get_audit_trail_range(zone: str, start: str, end: str) -> List[Dict]:
    return query_rows(
        "audit_trail",
        where="zone = ? AND event_timestamp >= ? AND event_timestamp <= ?",
        params=(zone, start, end),
        order="event_timestamp ASC",
        limit=1000,
    )
