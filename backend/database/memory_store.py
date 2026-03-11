from typing import List, Optional
from models.event_model import EventModel
from models.prediction_model import PredictionModel

# In-memory storage structures
event_store: List[EventModel] = []
prediction_store: List[PredictionModel] = []

class EventRepository:
    @staticmethod
    def add_event(event: EventModel) -> None:
        event_store.append(event)
        
    @staticmethod
    def get_all_events(limit: Optional[int] = None) -> List[EventModel]:
        return event_store[-limit:] if limit else event_store
        
    @staticmethod
    def get_events_by_zone(zone: str) -> List[EventModel]:
        return [e for e in event_store if e.zone == zone]
        
    @staticmethod
    def get_events_by_type(event_type: str) -> List[EventModel]:
        return [e for e in event_store if e.event_type.value == event_type]
