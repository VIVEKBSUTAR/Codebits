from models.event_model import EventModel, EventSeverity, EventType
from utils.logger import SystemLogger
from causal_engine.causal_graph import get_causal_graph

logger = SystemLogger(module_name="ingestion")

def normalize_event(event: EventModel) -> EventModel:
    """
    Normalizes a validated event to unified internal structures.
    """
    logger.log("event normalized")
    
    # Normalization logic
    if event.event_type == EventType.rainfall:
        if "rain_intensity" in event.metadata:
            try:
                intensity_str = str(event.metadata["rain_intensity"])
                if intensity_str.endswith("mm"):
                    intensity = float(intensity_str.replace("mm", ""))
                    if intensity > 15:
                        event.severity = EventSeverity.high
            except ValueError:
                pass

    return event

def dispatch_event(event: EventModel):
    """
    Dispatches the stored event to core system modules.
    """
    logger.log(f"Event dispatched to causal engine: {event.event_id}")
    
    # Inject evidence into causal graph
    graph = get_causal_graph(event.zone)
    graph.process_evidence(event.event_type.value, event.severity.value)
    
    logger.log(f"Event dispatched to simulation engine: {event.event_id}")
