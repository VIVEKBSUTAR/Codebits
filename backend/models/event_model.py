from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

class EventType(str, Enum):
    rainfall = "rainfall"
    traffic = "traffic"
    construction = "construction"
    accident = "accident"
    flood = "flood"
    manual_report = "manual_report"

class EventSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class EventModel(BaseModel):
    event_id: str
    event_type: EventType
    zone: str
    severity: EventSeverity
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CreateEventRequest(BaseModel):
    event_type: EventType
    zone: str
    severity: EventSeverity
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None
