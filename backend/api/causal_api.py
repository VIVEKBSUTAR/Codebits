from fastapi import APIRouter, Query, HTTPException, File, UploadFile
from causal_engine.causal_graph import get_causal_graph
from models.prediction_model import (
    PredictionModel, ZoneTimelineResponse, InterventionResponse,
    InterventionRequest, OptimalDeploymentResponse, CauseAnalysisResponse,
    IncidentHistoryResponse, AdvancedInferenceRequest, AdvancedInferenceResponse,
    MultiObjectiveRequest, MultiObjectiveResponse, CausalityAnalysisResponse,
    AdvancedPredictionItem, UncertaintyBounds, ParetoSolution, CausalEffect,
    CounterfactualExplanation,
    EventCascadeTimelineResponse, CascadeEventItem, InterventionOption,
    ProjectedImpactRequest, ProjectedImpactResponse
)
from simulation.cascade_engine import generate_timeline, EDGE_DELAYS, EVENT_NAMES
from simulation.intervention_engine import simulate_intervention
from optimization.resource_optimizer import generate_optimal_deployment
from analysis.cause_analyzer import compute_causal_contributions
from analysis.prediction_confidence import attach_confidence_to_predictions
from incident_logging.incident_logger import log_prediction, log_recommendation, log_decision, get_incident_history
from forecasting.predictive_analytics import get_predictive_forecast
from computer_vision.damage_assessment import analyze_infrastructure_image

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

@router.get("/predictive-forecast")
async def api_get_predictive_forecast(
    zone: str = Query(..., description="Target zone for forecasting"),
    metric: str = Query("overall", description="Risk metric to forecast (flood, traffic, emergency, overall)"),
    hours: int = Query(24, ge=1, le=168, description="Forecast horizon in hours (1-168)")
):
    """
    🚀 Advanced AI-Powered Predictive Analytics

    Features:
    - Facebook Prophet time series forecasting
    - Multi-method anomaly detection (Z-score, IQR, SPC)
    - Probabilistic confidence intervals
    - Real-time trend analysis
    """
    try:
        forecast_result = get_predictive_forecast(zone, metric, hours)
        return forecast_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

@router.post("/analyze-infrastructure-image")
async def api_analyze_infrastructure_image(
    file: UploadFile = File(..., description="Infrastructure image for analysis"),
    zone: str = Query("Unknown", description="Zone where image was taken")
):
    """
    📱 Advanced Computer Vision Infrastructure Damage Assessment

    Features:
    - YOLO v8 object detection for infrastructure elements
    - AI-powered damage classification with severity scoring
    - Automated repair cost estimation
    - Priority-based action recommendations
    - Annotated image output with damage overlays
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        image_data = await file.read()

        # Analyze with computer vision
        analysis_result = analyze_infrastructure_image(image_data, zone)

        # Add upload metadata
        analysis_result["upload_metadata"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(image_data)
        }

        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")


# ── Event Cascade Timeline ──────────────────────────────────────────────────────

# Cascade chain: the DAG path for display purposes
CASCADE_CHAIN = [
    "Rainfall",
    "DrainageCapacity",
    "Flooding",
    "TrafficCongestion",
    "EmergencyDelay",
]

CASCADE_DISPLAY_NAMES = {
    "Rainfall": "RAINFALL HIGH",
    "DrainageCapacity": "DRAINAGE OVERLOAD",
    "Flooding": "ROAD FLOODING",
    "TrafficCongestion": "TRAFFIC CONGESTION",
    "EmergencyDelay": "EMERGENCY DELAY",
}

# Cumulative delay from the trigger event (minutes along the chain)
CASCADE_CUMULATIVE_DELAY = {
    "Rainfall": 0,
    "DrainageCapacity": 10,
    "Flooding": 30,
    "TrafficCongestion": 35,
    "EmergencyDelay": 50,
}

# Available interventions per event in the cascade
CASCADE_INTERVENTIONS = {
    "Flooding": [
        InterventionOption(id="deploy_pump", label="Deploy High-Capacity Pumps", target_event="Flooding", resources_required=1, resources_unit="Pump Unit"),
        InterventionOption(id="open_drainage", label="Open Emergency Drainage Valves", target_event="Flooding", resources_required=1, resources_unit="Valve Unit"),
    ],
    "TrafficCongestion": [
        InterventionOption(id="close_road", label="Close Road & Divert Traffic", target_event="TrafficCongestion", resources_required=1, resources_unit="Traffic Unit"),
        InterventionOption(id="deploy_traffic_police", label="Deploy Traffic Police", target_event="TrafficCongestion", resources_required=2, resources_unit="Officer"),
    ],
    "EmergencyDelay": [
        InterventionOption(id="dispatch_ambulance", label="Pre-position Ambulance", target_event="EmergencyDelay", resources_required=1, resources_unit="Ambulance"),
        InterventionOption(id="clear_corridor", label="Clear Emergency Corridor", target_event="EmergencyDelay", resources_required=1, resources_unit="Corridor Unit"),
    ],
    "DrainageCapacity": [
        InterventionOption(id="deploy_pump", label="Deploy High-Capacity Pumps", target_event="DrainageCapacity", resources_required=1, resources_unit="Pump Unit"),
    ],
    "Rainfall": [
        InterventionOption(id="issue_advisory", label="Issue Public Advisory", target_event="Rainfall", resources_required=1, resources_unit="Unit"),
    ],
}


@router.get("/event-cascade-timeline", response_model=EventCascadeTimelineResponse)
async def get_event_cascade_timeline(zone: str = Query(..., description="Target zone")):
    """
    ⏱️ Event Cascade Timeline Prediction

    Returns the full cascade chain showing which events have happened vs predicted,
    with escalation times and available interventions for each predicted event.
    """
    try:
        graph = get_causal_graph(zone)
        evidence = graph.evidence.copy()
        probs = graph.run_inference()

        # Determine the trigger event (first evidence node, or first in chain)
        trigger_event = "Rainfall"
        for node in CASCADE_CHAIN:
            if node in evidence:
                trigger_event = node
                break

        # Also check DrainageCapacity explicitly
        drain_prob = 0.0
        try:
            drain_result = graph.infer.query(variables=["DrainageCapacity"], evidence=evidence)
            drain_prob = float(drain_result.values[1])  # P(Poor)
        except Exception:
            pass

        all_probs = {
            "Rainfall": 1.0 if "Rainfall" in evidence else 0.1,
            "DrainageCapacity": drain_prob,
            "Flooding": probs.get("Flooding", 0.0),
            "TrafficCongestion": probs.get("TrafficCongestion", 0.0),
            "EmergencyDelay": probs.get("EmergencyDelay", 0.0),
        }

        cascade_events = []
        for node in CASCADE_CHAIN:
            is_happened = node in evidence
            prob = all_probs.get(node, 0.0)
            delay = CASCADE_CUMULATIVE_DELAY.get(node, 0)

            if is_happened:
                status = "happened"
                time_label = f"T=0 min" if node == trigger_event else f"T+{delay} min"
            else:
                status = "predicted"
                time_label = f"T+{delay} min"

            cascade_events.append(CascadeEventItem(
                event_name=CASCADE_DISPLAY_NAMES.get(node, node),
                status=status,
                escalation_time=time_label,
                probability=round(prob, 2),
                delay_minutes=delay,
            ))

        # Gather interventions for predicted events
        available_interventions = []
        for node in CASCADE_CHAIN:
            if node not in evidence and all_probs.get(node, 0) >= 0.2:
                for iv in CASCADE_INTERVENTIONS.get(node, []):
                    available_interventions.append(iv)

        # Escalation risk score
        from simulation.cascade_engine import compute_escalation_risk
        risk_score = compute_escalation_risk(probs)

        return EventCascadeTimelineResponse(
            zone=zone,
            trigger_event=CASCADE_DISPLAY_NAMES.get(trigger_event, trigger_event),
            cascade_events=cascade_events,
            available_interventions=available_interventions,
            escalation_risk_score=risk_score,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cascade timeline error: {str(e)}")


# ── Projected Impact Analysis ───────────────────────────────────────────────────

# Maps intervention IDs to their causal effects and display info
INTERVENTION_REGISTRY = {
    "deploy_pump": {
        "label": "Deploy High-Capacity Pumps",
        "effect": {"DrainageCapacity": "Good"},
        "resources": 1,
        "unit": "Pump Unit",
    },
    "open_drainage": {
        "label": "Open Emergency Drainage Valves",
        "effect": {"DrainageCapacity": "Good"},
        "resources": 1,
        "unit": "Valve Unit",
    },
    "close_road": {
        "label": "Close Road & Divert Traffic",
        "effect": {"TrafficCongestion": "Low"},
        "resources": 1,
        "unit": "Traffic Unit",
    },
    "deploy_traffic_police": {
        "label": "Deploy Traffic Police",
        "effect": {"TrafficCongestion": "Low"},
        "resources": 2,
        "unit": "Officer",
    },
    "dispatch_ambulance": {
        "label": "Pre-position Ambulance",
        "effect": {"EmergencyDelay": "Low"},
        "resources": 1,
        "unit": "Ambulance",
    },
    "clear_corridor": {
        "label": "Clear Emergency Corridor",
        "effect": {"EmergencyDelay": "Low"},
        "resources": 1,
        "unit": "Corridor Unit",
    },
    "issue_advisory": {
        "label": "Issue Public Advisory",
        "effect": {},
        "resources": 1,
        "unit": "Unit",
    },
}


@router.post("/projected-impact", response_model=ProjectedImpactResponse)
async def get_projected_impact(req: ProjectedImpactRequest):
    """
    📊 Projected Impact Analysis

    Simulates the mathematical impact of executing a specific intervention
    on downstream cascade probabilities using do-calculus on the Bayesian network.
    """
    try:
        intervention_id = req.intervention_id
        if intervention_id not in INTERVENTION_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown intervention: {intervention_id}")

        config = INTERVENTION_REGISTRY[intervention_id]
        graph = get_causal_graph(req.zone)
        evidence = graph.evidence.copy()

        # Baseline risks
        baseline_probs = graph.run_inference()
        baseline_risks = {
            "flooding": round(float(baseline_probs.get("Flooding", 0.0)), 3),
            "traffic": round(float(baseline_probs.get("TrafficCongestion", 0.0)), 3),
            "emergency_delay": round(float(baseline_probs.get("EmergencyDelay", 0.0)), 3),
        }

        # Apply intervention evidence (do-calculus: set the target node)
        intervened_evidence = evidence.copy()
        intervened_evidence.update(config["effect"])

        # Compute counterfactual posteriors
        after_risks = {}
        for target, key in [("Flooding", "flooding"), ("TrafficCongestion", "traffic"), ("EmergencyDelay", "emergency_delay")]:
            if target in config["effect"]:
                # Directly intervened node -> forced to safe state
                after_risks[key] = 0.0
            else:
                try:
                    result = graph.infer.query(variables=[target], evidence=intervened_evidence)
                    after_risks[key] = round(float(result.values[1]), 3)
                except Exception:
                    after_risks[key] = baseline_risks[key]

        # Calculate overall risk reduction
        total_baseline = sum(baseline_risks.values())
        total_after = sum(after_risks.values())
        reduction_pct = round(((total_baseline - total_after) / max(total_baseline, 0.01)) * 100, 1)

        return ProjectedImpactResponse(
            zone=req.zone,
            target_event=req.target_event,
            proposed_action=config["label"],
            resources_consumed=config["resources"],
            resources_unit=config["unit"],
            resources_remaining="∞",
            risk_reduction_pct=reduction_pct,
            description=f"Executing this intervention mathematically reduces the probability of downstream cascading failures originating from this node.",
            after_intervention_risks=after_risks,
            baseline_risks=baseline_risks,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impact analysis error: {str(e)}")


# Advanced Analytics APIs

@router.post("/inference/advanced", response_model=AdvancedInferenceResponse)
async def advanced_inference_api(req: AdvancedInferenceRequest):
    """
    🧠 Advanced Bayesian Inference with Algorithm Selection

    Features:
    - Dynamic algorithm selection (Junction Tree, Variational Inference, MCMC)
    - Uncertainty quantification with statistical bounds
    - Online parameter learning with EM algorithm
    - Support for continuous variables
    - Performance optimization and tracking
    """
    try:
        from causal_engine.advanced_inference.inference_controller import InferenceController
        from causal_engine.causal_graph import get_causal_graph
        import time

        start_time = time.time()

        # Get enhanced causal graph with advanced inference capabilities
        graph = get_causal_graph(req.zone, advanced_mode=True)

        # Initialize advanced inference controller
        controller = InferenceController(graph.model)

        # Select and configure inference algorithm
        if req.algorithm == "auto":
            algorithm = controller.select_optimal_algorithm(
                evidence=req.evidence or {},
                query_variables=req.query_variables
            )
        else:
            algorithm = req.algorithm

        # Perform advanced inference
        inference_result = controller.run_advanced_inference(
            algorithm=algorithm,
            evidence=req.evidence or {},
            query_variables=req.query_variables,
            uncertainty_quantification=req.uncertainty_quantification
        )

        computation_time = time.time() - start_time

        # Format response with advanced predictions
        predictions = []
        for var, result in inference_result["marginals"].items():
            uncertainty = result.get("uncertainty_bounds", {})
            predictions.append(AdvancedPredictionItem(
                event=var,
                probability=result["probability"],
                uncertainty_bounds=UncertaintyBounds(**uncertainty) if uncertainty else UncertaintyBounds(
                    mean=result["probability"], std=0.0, ci_lower=result["probability"],
                    ci_upper=result["probability"], min=result["probability"],
                    max=result["probability"], reliability=1.0
                ),
                algorithm_used=algorithm,
                computation_time=computation_time
            ))

        return AdvancedInferenceResponse(
            zone=req.zone,
            algorithm_selected=algorithm,
            performance_stats=inference_result.get("performance_stats", {}),
            predictions=predictions,
            learned_parameters=inference_result.get("learned_parameters")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced inference error: {str(e)}")

@router.post("/interventions/optimize-advanced", response_model=MultiObjectiveResponse)
async def advanced_intervention_optimization(req: MultiObjectiveRequest):
    """
    🎯 Advanced Multi-Objective Intervention Optimization

    Features:
    - NSGA-II multi-objective genetic algorithm
    - Pareto-optimal solution discovery
    - Robust optimization under uncertainty
    - Dynamic objective weighting
    - Comprehensive sensitivity analysis
    """
    try:
        from optimization.multi_objective_optimizer import MultiObjectiveOptimizer
        from causal_engine.causal_graph import get_causal_graph

        # Get sophisticated causal knowledge
        graph = get_causal_graph("citywide", advanced_mode=True)

        # Initialize advanced optimizer
        optimizer = MultiObjectiveOptimizer(
            objectives=req.objectives,
            uncertainty_tolerance=req.uncertainty_tolerance,
            time_horizon=req.time_horizon
        )

        # Perform multi-objective optimization
        optimization_result = optimizer.optimize_intervention_portfolio(
            available_resources=req.intervention_budget,
            pareto_limit=req.pareto_solutions_limit,
            graph=graph
        )

        # Format Pareto solutions
        pareto_solutions = []
        for solution in optimization_result["pareto_front"]:
            pareto_solutions.append(ParetoSolution(
                intervention_portfolio=solution["interventions"],
                objective_values=solution["objectives"],
                total_cost=solution["total_cost"],
                risk_reduction=solution["risk_reduction"],
                robustness_score=solution["robustness_score"]
            ))

        return MultiObjectiveResponse(
            pareto_solutions=pareto_solutions,
            recommended_solution=ParetoSolution(**optimization_result["recommended"]),
            optimization_stats=optimization_result["stats"],
            sensitivity_analysis=optimization_result["sensitivity"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-objective optimization error: {str(e)}")

@router.get("/causality/analyze/{zone}", response_model=CausalityAnalysisResponse)
async def advanced_causality_analysis(zone: str):
    """
    🔬 Advanced Causal Reasoning Analysis

    Features:
    - Pearl's do-calculus for causal identification
    - Backdoor criterion analysis for confounding
    - Counterfactual reasoning and explanations
    - Sensitivity analysis for causal assumptions
    - Front-door criterion when backdoor unavailable
    """
    try:
        from simulation.do_calculus_engine import DoCalculusEngine
        from causal_engine.causal_graph import get_causal_graph

        # Get causal graph with sophisticated structure
        graph = get_causal_graph(zone, advanced_mode=True)

        # Initialize do-calculus engine
        do_engine = DoCalculusEngine(graph.model)

        # Perform comprehensive causal analysis
        causal_analysis = do_engine.comprehensive_causal_analysis(zone)

        # Format causal effects
        causal_effects = []
        for effect in causal_analysis["causal_effects"]:
            causal_effects.append(CausalEffect(
                intervention=effect["intervention"],
                target_variable=effect["target"],
                effect_size=effect["effect_size"],
                confidence_interval=effect["confidence_interval"],
                significance=effect["significance"]
            ))

        # Format counterfactual explanations
        counterfactuals = []
        for cf in causal_analysis["counterfactuals"]:
            counterfactuals.append(CounterfactualExplanation(
                original_outcome=cf["original"],
                counterfactual_outcome=cf["counterfactual"],
                required_changes=cf["changes"],
                probability_change=cf["prob_change"]
            ))

        return CausalityAnalysisResponse(
            zone=zone,
            causal_effects=causal_effects,
            sensitivity_bounds=causal_analysis["sensitivity_bounds"],
            counterfactual_explanations=counterfactuals,
            backdoor_adjustment_sets=causal_analysis["backdoor_sets"],
            identification_strategy=causal_analysis["identification_strategy"],
            causal_graph_summary=causal_analysis["graph_summary"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Causal analysis error: {str(e)}")

@router.get("/learning/adaptation/{zone}")
async def get_learning_adaptation_stats(zone: str):
    """
    📈 Bayesian Learning and Adaptation Statistics

    Shows how the system learns and adapts from data:
    - Parameter learning convergence
    - Evidence accumulation over time
    - Model improvement metrics
    - Uncertainty reduction tracking
    """
    try:
        from causal_engine.causal_graph import get_causal_graph

        graph = get_causal_graph(zone, advanced_mode=True)

        # Get learning statistics from the enhanced graph
        learning_stats = graph.get_learning_statistics()

        return {
            "zone": zone,
            "learning_progress": learning_stats,
            "parameter_convergence": learning_stats.get("convergence_metrics", {}),
            "evidence_count": learning_stats.get("evidence_observations", 0),
            "uncertainty_reduction": learning_stats.get("uncertainty_metrics", {}),
            "model_improvement": learning_stats.get("improvement_score", 0.0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning statistics error: {str(e)}")

@router.post("/uncertainty/analysis")
async def uncertainty_analysis_api(
    zone: str = Query(..., description="Target zone"),
    intervention: str = Query(..., description="Intervention type"),
    monte_carlo_samples: int = Query(1000, description="Number of MC samples")
):
    """
    📊 Advanced Uncertainty Analysis

    Features:
    - Monte Carlo uncertainty propagation
    - Sensitivity analysis for model assumptions
    - Value-at-Risk and Conditional VaR calculations
    - Robustness testing under parameter uncertainty
    """
    try:
        from simulation.probabilistic_interventions import ProbabilisticInterventionEngine
        from causal_engine.causal_graph import get_causal_graph

        # Get current evidence
        graph = get_causal_graph(zone)
        current_evidence = graph.run_inference()

        # Initialize probabilistic intervention engine
        intervention_engine = ProbabilisticInterventionEngine()

        # Perform comprehensive uncertainty analysis
        uncertainty_result = intervention_engine.analyze_uncertainty_bounds(
            intervention_type=intervention,
            current_evidence=current_evidence,
            n_samples=monte_carlo_samples
        )

        # Calculate VaR metrics
        var_analysis = intervention_engine.calculate_risk_metrics(
            intervention, current_evidence, confidence_levels=[0.05, 0.01]
        )

        return {
            "zone": zone,
            "intervention": intervention,
            "uncertainty_bounds": uncertainty_result,
            "risk_metrics": var_analysis,
            "monte_carlo_samples": monte_carlo_samples,
            "robustness_score": var_analysis.get("robustness_score", 0.0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uncertainty analysis error: {str(e)}")
