import networkx as nx
from causal_engine.causal_graph import get_causal_graph

EVIDENCE_MATRIX = {
    ("Rainfall", "Flooding"): {"support": 342, "total": 432},
    ("DrainageCapacity", "Flooding"): {"support": 150, "total": 200},
    ("Flooding", "TrafficCongestion"): {"support": 280, "total": 350},
    ("ConstructionActivity", "TrafficCongestion"): {"support": 190, "total": 210},
    ("Accident", "TrafficCongestion"): {"support": 110, "total": 200},
    ("TrafficCongestion", "EmergencyDelay"): {"support": 210, "total": 300}
}

def get_edge_confidence(source: str, target: str) -> float:
    """Computes basic confidence for a single edge from historical evidence."""
    data = EVIDENCE_MATRIX.get((source, target))
    if data and data["total"] > 0:
        return data["support"] / data["total"]
    return 0.5  # Fallback

def compute_prediction_confidence(target: str, graph) -> float:
    """
    Computes overall confidence for a target node by finding the highest 
    confidence active path from evidence nodes to the target.
    """
    active_evidence = list(graph.evidence.keys())
    if not active_evidence:
        return 0.5

    best_confidence = 0.0

    for source in active_evidence:
        if source == target:
            # Direct evidence for this node
            return 1.0
            
        try:
            # Find all simple paths in the DAG
            paths = list(nx.all_simple_paths(graph.model, source, target))
            for path in paths:
                path_confidence = 1.0
                # Multiply (product) confidence of each edge along the path
                for i in range(len(path) - 1):
                    edge_conf = get_edge_confidence(path[i], path[i+1])
                    path_confidence *= edge_conf
                
                if path_confidence > best_confidence:
                    best_confidence = path_confidence
        except nx.NetworkXNoPath:
            continue
        except Exception:
            continue

    return round(best_confidence, 2) if best_confidence > 0 else 0.5

def attach_confidence_to_predictions(zone: str, raw_probs: dict) -> list:
    """
    Wraps inference outputs with their computed historical confidence.
    """
    graph = get_causal_graph(zone)
    predicted_events = []
    
    for event, prob in raw_probs.items():
        if event in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
            conf = compute_prediction_confidence(event, graph)
            predicted_events.append({
                "event": event,
                "probability": round(prob, 2),
                "confidence": conf
            })
            
    return predicted_events
