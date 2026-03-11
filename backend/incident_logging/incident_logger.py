import sqlite3
import json
import os
import uuid
from typing import List, Dict, Any
from datetime import datetime

# Store the DB in the data folder to be clean, or just the backend root
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "incidents.db")

def _get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_db():
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                zone TEXT,
                timestamp TEXT,
                event_type TEXT,
                prediction TEXT,
                confidence REAL,
                recommended_action TEXT,
                human_decision TEXT,
                final_action TEXT
            )
            """)
            conn.commit()
    except Exception as e:
        print(f"Error initializing DB: {e}")

# Run once on import
initialize_db()

def log_event(zone: str, event_type: str, timestamp: str = None) -> str:
    """Logs a new incident event and returns the generated incident_id."""
    incident_id = f"INC_{uuid.uuid4().hex[:8].upper()}"
    ts = timestamp or datetime.now().isoformat()
    
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incidents (incident_id, zone, timestamp, event_type)
            VALUES (?, ?, ?, ?)
        """, (incident_id, zone, ts, event_type))
        conn.commit()
        
    return incident_id

def get_latest_incident_for_zone(zone: str) -> str:
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT incident_id FROM incidents 
            WHERE zone = ? ORDER BY timestamp DESC LIMIT 1
        """, (zone,))
        row = cursor.fetchone()
        return row[0] if row else None

def log_prediction(zone: str, predictions: List[Dict[str, Any]]):
    """Updates the latest incident for the zone with prediction data."""
    if not predictions:
        return
        
    incident_id = get_latest_incident_for_zone(zone)
    if not incident_id:
        return
        
    top_pred = max(predictions, key=lambda x: x.get("probability", 0))
    pred_str = json.dumps({top_pred["event"]: {"probability": top_pred["probability"], "confidence": top_pred.get("confidence", 0.0)}})
    
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE incidents 
            SET prediction = ?, confidence = ?
            WHERE incident_id = ?
        """, (pred_str, top_pred.get("confidence", 0.0), incident_id))
        conn.commit()

def log_recommendation(zone: str, recommended_action: str):
    incident_id = get_latest_incident_for_zone(zone)
    if not incident_id:
        return
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE incidents 
            SET recommended_action = ?
            WHERE incident_id = ?
        """, (recommended_action, incident_id))
        conn.commit()

def log_decision(zone: str, human_decision: str, final_action: str = None):
    incident_id = get_latest_incident_for_zone(zone)
    if not incident_id:
        return
    if not final_action:
        final_action = "executed" if human_decision.lower() == "approved" else "discarded"
        
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE incidents 
            SET human_decision = ?, final_action = ?
            WHERE incident_id = ?
        """, (human_decision, final_action, incident_id))
        conn.commit()

def get_incident_history(zone: str = None, start_time: str = None, end_time: str = None) -> List[Dict[str, Any]]:
    query = "SELECT incident_id, zone, timestamp, event_type, prediction, confidence, recommended_action, human_decision, final_action FROM incidents WHERE 1=1"
    params = []
    
    if zone:
        query += " AND zone = ?"
        params.append(zone)
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
        
    query += " ORDER BY timestamp DESC"
    
    records = []
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            prediction_dict = None
            if row[4]:
                try:
                    prediction_dict = json.loads(row[4])
                except Exception:
                    prediction_dict = row[4] # Fallback if text
                    
            records.append({
                "incident_id": row[0],
                "zone": row[1],
                "timestamp": row[2],
                "event_type": row[3],
                "prediction": prediction_dict,
                "confidence": row[5],
                "recommended_action": row[6],
                "human_decision": row[7],
                "final_action": row[8]
            })
            
    return records
