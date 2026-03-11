from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import uuid

from models.event_model import EventModel, CreateEventRequest
from database.memory_store import EventRepository
from ingestion.event_processor import normalize_event, dispatch_event
from utils.logger import SystemLogger
from incident_logging.incident_logger import log_event

router = APIRouter()
logger = SystemLogger(module_name="ingestion")

@router.post("/events", status_code=201)
async def submit_event(request_data: CreateEventRequest):
    # Log reception
    logger.log(f"{request_data.event_type.value} event received from {request_data.source}")
    
    # Validation is handled by Pydantic Model (CreateEventRequest)
    
    # Create the internal EventModel object
    event_id = f"EVT_{uuid.uuid4().hex[:8]}"
    
    event = EventModel(
        event_id=event_id,
        event_type=request_data.event_type,
        zone=request_data.zone,
        severity=request_data.severity,
        timestamp=request_data.timestamp,
        source=request_data.source,
        metadata=request_data.metadata or {}
    )
    
    # Normalize event
    normalized_event = normalize_event(event)
    
    # Store event
    EventRepository.add_event(normalized_event)
    logger.log(f"event {event_id} stored")
    
    # Stage-12 Incident Logging
    log_event(normalized_event.zone, normalized_event.event_type.value, normalized_event.timestamp.isoformat())
    
    # Dispatch event
    dispatch_event(normalized_event)
    
    return {
        "status": "event_received",
        "event_id": event_id
    }

@router.get("/events", response_model=List[EventModel])
async def get_events(
    zone: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1)
):
    if zone:
        events = EventRepository.get_events_by_zone(zone)
    elif event_type:
        events = EventRepository.get_events_by_type(event_type)
    else:
        events = EventRepository.get_all_events(limit)
        
    if limit and (zone or event_type):
       events = events[-limit:]
        
    return events
