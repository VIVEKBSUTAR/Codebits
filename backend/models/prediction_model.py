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

# Advanced Analytics API Models

class AdvancedInferenceRequest(BaseModel):
    zone: str
    algorithm: str = "auto"  # auto, variational, junction_tree, mcmc
    uncertainty_quantification: bool = True
    continuous_variables: Optional[List[str]] = None
    evidence: Optional[Dict[str, Any]] = None
    query_variables: Optional[List[str]] = None

class UncertaintyBounds(BaseModel):
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    min: float
    max: float
    reliability: float

class AdvancedPredictionItem(BaseModel):
    event: str
    probability: float
    uncertainty_bounds: UncertaintyBounds
    algorithm_used: str
    computation_time: float

class AdvancedInferenceResponse(BaseModel):
    zone: str
    algorithm_selected: str
    performance_stats: Dict[str, Any]
    predictions: List[AdvancedPredictionItem]
    learned_parameters: Optional[Dict[str, Any]] = None

class MultiObjectiveRequest(BaseModel):
    objectives: List[str] = ["risk_minimization", "cost_efficiency"]
    uncertainty_tolerance: float = 0.1
    intervention_budget: Dict[str, int]
    time_horizon: int = 24
    pareto_solutions_limit: int = 10

class ParetoSolution(BaseModel):
    intervention_portfolio: List[str]
    objective_values: Dict[str, float]
    total_cost: float
    risk_reduction: float
    robustness_score: float

class MultiObjectiveResponse(BaseModel):
    pareto_solutions: List[ParetoSolution]
    recommended_solution: ParetoSolution
    optimization_stats: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]

class CausalEffect(BaseModel):
    intervention: str
    target_variable: str
    effect_size: float
    confidence_interval: List[float]
    significance: float

class CounterfactualExplanation(BaseModel):
    original_outcome: str
    counterfactual_outcome: str
    required_changes: Dict[str, str]
    probability_change: float

class CausalityAnalysisResponse(BaseModel):
    zone: str
    causal_effects: List[CausalEffect]
    sensitivity_bounds: Dict[str, Dict[str, float]]
    counterfactual_explanations: List[CounterfactualExplanation]
    backdoor_adjustment_sets: List[List[str]]
    identification_strategy: str
    causal_graph_summary: Dict[str, Any]


# ── Event Timeline Prediction Models ────────────────────────────────────────

class CascadeEventItem(BaseModel):
    event_name: str
    status: str           # "happened" | "predicted"
    escalation_time: str  # "T=0 min", "T+10 min", etc.
    probability: float
    delay_minutes: int

class InterventionOption(BaseModel):
    id: str
    label: str
    target_event: str
    resources_required: int
    resources_unit: str

class EventCascadeTimelineResponse(BaseModel):
    zone: str
    trigger_event: str
    cascade_events: List[CascadeEventItem]
    available_interventions: List[InterventionOption]
    escalation_risk_score: str


# ── Projected Impact Analysis Models ────────────────────────────────────────

class ProjectedImpactRequest(BaseModel):
    zone: str
    target_event: str
    intervention_id: str

class ProjectedImpactResponse(BaseModel):
    zone: str
    target_event: str
    proposed_action: str
    resources_consumed: int
    resources_unit: str
    resources_remaining: Optional[str] = None
    risk_reduction_pct: float
    description: str
    after_intervention_risks: Dict[str, float]
    baseline_risks: Dict[str, float]
