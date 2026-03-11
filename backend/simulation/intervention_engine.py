from typing import Dict, Any

from causal_engine.causal_graph import get_causal_graph
from simulation.cascade_engine import generate_timeline # Use Stage-4 logic
from models.prediction_model import InterventionResponse, InterventionTimelineChanges
import datetime

# Standardized Interventions targeting specific DAG nodes
INTERVENTION_EFFECTS = {
    "deploy_pump": {"DrainageCapacity": "Good"},
    "close_road": {"TrafficCongestion": "Low"},
    "dispatch_ambulance": {"EmergencyDelay": "Low"}
}

def recalculate_timeline_with_sandbox(zone: str, simulated_evidence: Dict[str, str], counterfactual_probs: Dict[str, float]) -> list:
    """A scoped version of generate_timeline using counterfactual data"""
    from simulation.cascade_engine import EDGE_DELAYS, EVENT_NAMES, TimelineItemModel
    
    # 1. Filter significant risks threshold
    significant_nodes = {k: v for k, v in counterfactual_probs.items() if v >= 0.40}
    
    # If Drainage is part of the counterfactual targets, fetch it specifically (unless intervened upon)
    if "DrainageCapacity" not in simulated_evidence:
        graph = get_causal_graph(zone)
        drain_prob_result = graph.infer.query(variables=["DrainageCapacity"], evidence=simulated_evidence)
        drain_prob = float(drain_prob_result.values[1]) # 'Poor'
        if drain_prob >= 0.40:
             significant_nodes["DrainageCapacity"] = drain_prob
    
    t_zero = datetime.datetime.now()
    projected_times = {}

    # Calculate propagation times (BFS from counterfactual active evidence)
    active_sources = list(simulated_evidence.keys())
    graph_nodes = get_causal_graph(zone).model.nodes()
    
    for target in significant_nodes:
        min_total_delay = float('inf')
        
        for source in active_sources:
            if (source, target) in EDGE_DELAYS:
                min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, target)])
            else:
                for intermediate in graph_nodes:
                    if (source, intermediate) in EDGE_DELAYS and (intermediate, target) in EDGE_DELAYS:
                        min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, intermediate)] + EDGE_DELAYS[(intermediate, target)])
        
        if min_total_delay != float('inf'):
            projected_times[target] = min_total_delay
        else:
            projected_times[target] = 30 # Backstop

    predicted_events = []
    
    for node, prob in significant_nodes.items():
        delay_mins = projected_times.get(node, 0)
        proj_time = t_zero + datetime.timedelta(minutes=delay_mins)
        formatted_time = proj_time.strftime("%H:%M")
        
        predicted_events.append(TimelineItemModel(
            predicted_time=formatted_time,
            event_name=EVENT_NAMES.get(node, node),
            probability=round(float(prob), 2)
        ))
        
    predicted_events.sort(key=lambda x: x.predicted_time)
    return predicted_events


def simulate_intervention(zone: str, intervention_action: str) -> InterventionResponse:
    # 1. Fetch current baseline
    baseline_graph = get_causal_graph(zone)
    current_evidence = baseline_graph.evidence.copy()
    
    baseline_probs = baseline_graph.run_inference()
    baseline_timeline_model = generate_timeline(zone)
    baseline_timeline = baseline_timeline_model.predicted_events

    # 2. Lookup Intervention Effects
    if intervention_action not in INTERVENTION_EFFECTS:
        raise ValueError(f"Unknown Intervention: {intervention_action}")
    effect_evidence = INTERVENTION_EFFECTS[intervention_action]

    # 3. Create Sandbox Evidence
    simulated_evidence = current_evidence.copy()
    simulated_evidence.update(effect_evidence)

    # 4. Run Bayesian Inference (Counterfactual)
    counterfactual_probs = {}
    for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
        if target in effect_evidence and effect_evidence[target] in ['Good', 'Low', 'False']:
             counterfactual_probs[target] = 0.0
             continue
             
        if target not in simulated_evidence:
            result = baseline_graph.infer.query(variables=[target], evidence=simulated_evidence)
            # Probability of 'True' or 'High' state
            val = float(result.values[1])
            counterfactual_probs[target] = val

    # 5. Recompute Timeline
    counterfactual_timeline = recalculate_timeline_with_sandbox(zone, simulated_evidence, counterfactual_probs)

    # 6. Compute Risk Reduction (Benefit)
    benefit_metrics = {}
    baseline_risk_out = {}
    after_intervention_out = {}

    for k in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
        # Rename for API output
        key_name = 'flooding' if k == 'Flooding' else ('traffic' if k == 'TrafficCongestion' else 'emergency_delay')
        
        base_val = round(float(baseline_probs.get(k, 0.0)), 2)
        cf_val = round(float(counterfactual_probs.get(k, 0.0)), 2)
        
        baseline_risk_out[key_name] = base_val
        after_intervention_out[key_name] = cf_val
        benefit_metrics[f"{key_name}_reduction"] = round(base_val - cf_val, 2)

    # 7. Package Response
    return InterventionResponse(
        zone=zone,
        intervention=intervention_action,
        baseline_risk=baseline_risk_out,
        after_intervention=after_intervention_out,
        benefit=benefit_metrics,
        timeline_changes=InterventionTimelineChanges(
            baseline_timeline=baseline_timeline,
            intervened_timeline=counterfactual_timeline
        )
    )
