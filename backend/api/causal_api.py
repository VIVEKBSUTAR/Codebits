from fastapi import APIRouter, Query, HTTPException
from causal_engine.causal_graph import get_causal_graph
from models.prediction_model import PredictionModel, ZoneTimelineResponse, InterventionResponse, InterventionRequest, OptimalDeploymentResponse, CauseAnalysisResponse, IncidentHistoryResponse
from simulation.cascade_engine import generate_timeline
from simulation.intervention_engine import simulate_intervention
from optimization.resource_optimizer import generate_optimal_deployment
from analysis.cause_analyzer import compute_causal_contributions
from analysis.prediction_confidence import attach_confidence_to_predictions
from incident_logging.incident_logger import log_prediction, log_recommendation, log_decision, get_incident_history

router = APIRouter()

@router.get("/zone-risk", response_model=PredictionModel)
async def get_zone_risk(zone: str = Query(..., description="City zone to query")):
    graph = get_causal_graph(zone)
    probs = graph.run_inference()
    
    # Attach computed historical confidence
    predictions = attach_confidence_to_predictions(zone, probs)
    
    # Stage-12: Log the prediction output
    # predictions list contains item models, we can format them for the logger
    log_prediction(zone, predictions)
    
    return PredictionModel(
        zone=zone,
        predictions=predictions
    )

@router.get("/cause-analysis", response_model=CauseAnalysisResponse)
async def api_get_cause_analysis(zone: str = Query(...), event: str = Query(...)):
    causes = compute_causal_contributions(zone, event)
    return CauseAnalysisResponse(
        zone=zone,
        event=event,
        causes=causes
    )

@router.get("/zone-timeline", response_model=ZoneTimelineResponse)
async def get_zone_timeline(zone: str = Query(...)):
    timeline = generate_timeline(zone)
    return timeline

@router.post("/simulate-intervention", response_model=InterventionResponse)
async def api_simulate_intervention(req: InterventionRequest):
    try:
        response = simulate_intervention(req.zone, req.intervention)
        
        # Stage-12: Treat a simulation request as a logged decision action for testing
        log_decision(req.zone, "approved", req.intervention)
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/optimal-deployment", response_model=OptimalDeploymentResponse)
async def get_optimal_deployment(resources: str = Query(...)):
    try:
        response = generate_optimal_deployment(resources)
        
        # Stage-12: Log recommended actions for each zone targeted
        for plan_item in response.plan:
            log_recommendation(plan_item.zone, plan_item.action)
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/incident-history", response_model=IncidentHistoryResponse)
async def api_get_incident_history(
    zone: str = Query(..., description="Target zone"),
    start_time: str = Query(None, description="Start ISO timestamp"),
    end_time: str = Query(None, description="End ISO timestamp")
):
    incidents_list = get_incident_history(zone, start_time, end_time)
    
    return IncidentHistoryResponse(
        zone=zone,
        incidents=incidents_list
    )
