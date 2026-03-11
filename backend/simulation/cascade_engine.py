from datetime import datetime, timedelta
from typing import List, Dict, Any

from models.prediction_model import TimelineItemModel, ZoneTimelineResponse
from causal_engine.causal_graph import get_causal_graph

# Time delays in minutes
EDGE_DELAYS = {
    ("Rainfall", "DrainageCapacity"): 10,
    ("DrainageCapacity", "Flooding"): 15,
    ("Rainfall", "Flooding"): 20, 
    ("Flooding", "TrafficCongestion"): 15,
    ("ConstructionActivity", "TrafficCongestion"): 5,
    ("Accident", "TrafficCongestion"): 0, 
    ("TrafficCongestion", "EmergencyDelay"): 20
}

# User-friendly event names
EVENT_NAMES = {
    "DrainageCapacity": "DrainageCapacity stress",
    "Flooding": "Flooding",
    "TrafficCongestion": "Traffic congestion",
    "EmergencyDelay": "Emergency delays"
}

def compute_escalation_risk(probs: Dict[str, float]) -> str:
    # A simple weighted configuration
    score = (probs.get("Flooding", 0) * 1.5 + probs.get("TrafficCongestion", 0) * 1.0 + probs.get("EmergencyDelay", 0) * 2.0) / 4.5
    
    if score < 0.3:
        return "LOW"
    elif score < 0.7:
        return "MEDIUM"
    return "HIGH"

def generate_timeline(zone: str) -> ZoneTimelineResponse:
    graph = get_causal_graph(zone)
    
    if not graph.evidence:
        return ZoneTimelineResponse(zone=zone, escalation_risk_score="LOW", predicted_events=[])
        
    # 2. Determine T_zero
    # In a real system, evidence objects store timestamps. We fake it with current time.
    t_zero = datetime.now()
    
    # 3. Run Inference
    posterior_probs = graph.run_inference()
    
    # 4. Filter significant risks threshold
    significant_nodes = {k: v for k, v in posterior_probs.items() if v >= 0.40}
    
    # Check Drainage explicitly as it's an intermediate mapping not normally returned in `run_inference`
    drain_prob = graph.infer.query(variables=["DrainageCapacity"], evidence=graph.evidence).values[1] # 'Poor'
    if drain_prob >= 0.40:
        significant_nodes["DrainageCapacity"] = drain_prob

    projected_times = {}

    # 5. Calculate Propagation Time (BFS from active evidence)
    # Start with active evidence paths
    active_sources = list(graph.evidence.keys())
    
    for target in significant_nodes:
        min_total_delay = float('inf')
        
        # Traverse from every source to target to find shortest delay
        for source in active_sources:
            # Simple check if path exists structurally
            if (source, target) in EDGE_DELAYS:
                min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, target)])
            else:
                # 2-hop checks logic for this DAG
                # e.g Rainfall -> Drainage -> Flooding
                for intermediate in graph.model.nodes():
                    if (source, intermediate) in EDGE_DELAYS and (intermediate, target) in EDGE_DELAYS:
                        min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, intermediate)] + EDGE_DELAYS[(intermediate, target)])
        
        if min_total_delay != float('inf'):
            projected_times[target] = min_total_delay
        else:
             # Default backstop
            projected_times[target] = 30

    # 6. Build Escalation Timeline
    predicted_events = []
    
    for node, prob in significant_nodes.items():
        delay_mins = projected_times.get(node, 0)
        proj_time = t_zero + timedelta(minutes=delay_mins)
        formatted_time = proj_time.strftime("%H:%M")
        
        predicted_events.append(TimelineItemModel(
            predicted_time=formatted_time,
            event_name=EVENT_NAMES.get(node, node),
            probability=round(float(prob), 2)
        ))
        
    # Sort chronologically
    predicted_events.sort(key=lambda x: x.predicted_time)
    
    risk_score = compute_escalation_risk(posterior_probs)

    return ZoneTimelineResponse(
        zone=zone,
        escalation_risk_score=risk_score,
        predicted_events=predicted_events
    )
