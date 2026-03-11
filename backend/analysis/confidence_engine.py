# The explicit confidence matrix
CONFIDENCE_MATRIX = {
    ("Rainfall", "DrainageCapacity"): 0.78,
    ("Rainfall", "Flooding"): 0.82,
    ("DrainageCapacity", "Flooding"): 0.77,
    ("Flooding", "TrafficCongestion"): 0.76,
    ("ConstructionActivity", "TrafficCongestion"): 0.91,
    ("Accident", "TrafficCongestion"): 0.55,
    ("TrafficCongestion", "EmergencyDelay"): 0.84
}

def get_confidence(source: str, target: str) -> float:
    """
    Returns the confidence score for a given causal edge.
    Default fallback confidence is 0.5 if the edge is not explicitly mapped.
    """
    return CONFIDENCE_MATRIX.get((source, target), 0.5)
