from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictionItemModel(BaseModel):
    event: str
    probability: float
    confidence: float

class PredictionModel(BaseModel):
    zone: str
    predictions: List[PredictionItemModel]

class TimelineItemModel(BaseModel):
    predicted_time: str
    event_name: str
    probability: float

class ZoneTimelineResponse(BaseModel):
    zone: str
    escalation_risk_score: str
    predicted_events: List[TimelineItemModel]

class InterventionRequest(BaseModel):
    zone: str
    intervention: str

class InterventionTimelineChanges(BaseModel):
    baseline_timeline: List[TimelineItemModel]
    intervened_timeline: List[TimelineItemModel]

class InterventionResponse(BaseModel):
    zone: str
    intervention: str
    baseline_risk: dict
    after_intervention: dict
    benefit: dict
    timeline_changes: InterventionTimelineChanges

class DeploymentItemModel(BaseModel):
    resource: str
    zone: str
    action: str
    benefit_expected: float

class OptimalDeploymentResponse(BaseModel):
    plan: List[DeploymentItemModel]
    expected_citywide_risk_reduction: float

class CauseItemModel(BaseModel):
    event: str
    contribution: float
    confidence: float

class CauseAnalysisResponse(BaseModel):
    zone: str
    event: str
    causes: List[CauseItemModel]

class IncidentRecordModel(BaseModel):
    incident_id: str
    zone: str
    timestamp: str
    event_type: str
    prediction: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    recommended_action: Optional[str] = None
    human_decision: Optional[str] = None
    final_action: Optional[str] = None

class IncidentHistoryResponse(BaseModel):
    zone: str
    incidents: List[IncidentRecordModel]
