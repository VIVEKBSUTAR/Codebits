from causal_engine.causal_graph import get_causal_graph
from analysis.confidence_engine import get_confidence

def get_probability_of_node(graph, node_name: str) -> float:
    """
    Helper function to get the probability of 'True'/'High'/'Poor'
    state for a node given the current evidence in the graph.
    """
    # If the node is explicitly in the evidence, its probability is 1.0 (or 0.0) depending on state
    if node_name in graph.evidence:
        state = graph.evidence[node_name]
        if state in ['High', 'True', 'Poor']:
            return 1.0
        else:
            return 0.0

    # Otherwise query the inference engine
    try:
        result = graph.infer.query(variables=[node_name], evidence=graph.evidence)
        # We assume the index 1 is the 'Failure/True/High/Poor' state
        return float(result.values[1])
    except Exception:
        # Fallback if inference fails or structure differs
        return 0.0

def compute_causal_contributions(zone: str, target_event: str) -> list:
    """
    Computes and mathematically ranks the normalized contributions of 
    all parent nodes heavily influencing the target_event.
    """
    # 1. Fetch active causal graph for the zone
    graph = get_causal_graph(zone)
    
    # 2. Identify parent nodes of the target_event
    try:
        parents = list(graph.model.get_parents(target_event))
    except ValueError:
        # target_event not in DAG
        return []
        
    if not parents:
        return []

    raw_contributions = []
    total_raw_contribution = 0.0

    # 3. Compute Contribution Scores
    for parent in parents:
        # A. Get P(parent)
        p_parent = get_probability_of_node(graph, parent) 
        
        # B. Get Edge Weight (Default to 1.0 for this stage)
        edge_weight = 1.0 
        
        # C. Get Confidence
        confidence = get_confidence(parent, target_event)
        
        # D. Calculate Raw Contribution
        contribution = p_parent * edge_weight * confidence
        
        raw_contributions.append({
            "event": parent,
            "raw_value": contribution,
            "confidence": confidence
        })
        total_raw_contribution += contribution

    # 4. Normalize Contributions
    final_causes = []
    for item in raw_contributions:
        if total_raw_contribution > 0:
            normalized = item["raw_value"] / total_raw_contribution
        else:
            normalized = 0.0
            
        final_causes.append({
            "event": item["event"],
            "contribution": round(normalized, 2),
            "confidence": round(item["confidence"], 2)
        })

    # 5. Rank Causes (Descending)
    final_causes.sort(key=lambda x: x["contribution"], reverse=True)
    
    return final_causes
