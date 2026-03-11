# Stage-11 Backend System Specification: Prediction Confidence Engine

This document provides the backend design and logic for the Stage-11 Prediction Confidence Engine, which layers historical-evidence-based confidence scores onto existing probabilistic predictions.

---

## 1. Module Architecture

A new module will be created to manage historical evidence data and calculate the statistical reliability of the causal inference outputs.

**Directory Structure:**
```text
backend/
└── analysis/
    ├── __init__.py
    ├── confidence_engine.py      # (Stage 10)
    ├── cause_analyzer.py         # (Stage 10)
    └── prediction_confidence.py  # NEW: Handles evidence matrix and prediction confidence
```

---

## 2. Evidence Matrix Implementation

Historical evidence is mapped per directed edge. Confidence is computed as `support / total`.

**File: `analysis/prediction_confidence.py`**

```python
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
```

---

## 3. Propagation Algorithm 

Predictions rely on the cascading chain of events. Because uncertainty compounds over a sequential logical path, we use the **Product (multiplication) Method** for propagating confidence along causal chains to ensure mathematical stability and appropriately penalize long speculative cascades.

**File: `analysis/prediction_confidence.py`**

### Pseudocode

```python
def compute_prediction_confidence(target: str, graph) -> float:
    """
    Computes overall confidence for a target node by finding the highest 
    confidence active path from evidence nodes to the target.
    """
    import networkx as nx
    
    active_evidence = list(graph.evidence.keys())
    if not active_evidence:
        return 0.5

    best_confidence = 0.0

    # Test all paths from any active evidence to the target
    for source in active_evidence:
        if source == target:
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
                
                # We retain the most reliable path's confidence
                if path_confidence > best_confidence:
                    best_confidence = path_confidence
        except nx.NetworkXNoPath:
            continue

    return round(best_confidence, 2) if best_confidence > 0 else 0.5


def attach_confidence_to_predictions(zone: str, raw_probs: dict) -> list:
    """
    Wraps inference outputs with their computed historical confidence.
    """
    graph = get_causal_graph(zone)
    predicted_events = []
    
    for event, prob in raw_probs.items():
        # Only process standard target risk nodes
        if event in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
            conf = compute_prediction_confidence(event, graph)
            predicted_events.append({
                "event": event,
                "probability": round(prob, 2),
                "confidence": conf
            })
            
    return predicted_events
```

---

## 4. API Integration

We will overwrite the existing `PredictionModel` and update the `GET /zone-risk` logic in `api/causal_api.py`.

**Data Models (`models/prediction_model.py`):**
```python
class PredictionItemModel(BaseModel):
    event: str
    probability: float
    confidence: float

# Overwrite existing PredictionModel
class PredictionModel(BaseModel):
    zone: str
    predictions: List[PredictionItemModel]
```

**API Router (`api/causal_api.py`):**
```python
from analysis.prediction_confidence import attach_confidence_to_predictions

# Overwrite existing /zone-risk
@router.get("/zone-risk", response_model=PredictionModel)
async def get_zone_risk(zone: str = Query(..., description="City zone to query")):
    graph = get_causal_graph(zone)
    probs = graph.run_inference()
    
    # Attach computed historical confidence
    predictions = attach_confidence_to_predictions(zone, probs)
    
    return PredictionModel(
        zone=zone,
        predictions=predictions
    )
```

---

## 5. Complete Testing Suite Specification

**File: `backend/test_stage_11.py`**

### TEST 1 — Edge Confidence
*   **Action**: Call `get_edge_confidence("Rainfall", "Flooding")`.
*   **Assertion**: Must equal `342 / 432_ ≈ 0.79`.

### TEST 2 — Missing Evidence
*   **Action**: Call `get_edge_confidence("Unknown", "Flooding")`.
*   **Assertion**: Must return the default fallback value `0.5`.

### TEST 3 — Cascade Confidence
*   **Action**: Setup Graph Evidence with `Rainfall = High`.
*   **Action**: Call `compute_prediction_confidence("TrafficCongestion", graph)`.
*   **Assertion**: Must equate to `get_edge_confidence("Rainfall", "Flooding") * get_edge_confidence("Flooding", "TrafficCongestion")`.
*   **Assertion**: Ensure `prediction_confidence < get_edge_confidence("Rainfall", "Flooding")` (Product method reduces scores slightly across long chains).

### TEST 4 — API Output
*   **Action**: Perform `GET /zone-risk?zone=Bibwewadi`.
*   **Assertion**: Check `response.status_code == 200`.
*   **Assertion**: Parse JSON. Validate the outer structure contains `zone` and `predictions` list.
*   **Assertion**: Ensure items inside `predictions` explicitly contain `event`, `probability`, and `confidence` float values.

### TEST 5 — Normalization
*   **Action**: Iterate through all `predictions` returned from Test 4.
*   **Assertion**: Validate mathematically that `0.0 <= item["confidence"] <= 1.0` holds true for every object in the array.
