# Stage-10 Backend System Specification: Cause Analysis Engine

This document provides the clear backend design and algorithmic specifications for the Stage-10 Cause Analysis and Confidence Engine.

---

## 1. Module Architecture

The Stage-10 logic will be housed in a new `analysis/` domain directory to cleanly separate observational insights from predictive simulation code.

**Directory Structure:**
```text
backend/
└── analysis/
    ├── __init__.py
    ├── confidence_engine.py
    └── cause_analyzer.py
```

*   `confidence_engine.py`: Encapsulates the static confidence matrix and provides deterministic getter functions.
*   `cause_analyzer.py`: Orchestrates the causal graph, triggers Bayesian inference, and applies the contribution normalization math.

---

## 2. Confidence Matrix Implementation

The confidence matrix stores predefined reliability metrics for known causal edges.

**File: `analysis/confidence_engine.py`**

```python
# The explicit confidence matrix
CONFIDENCE_MATRIX = {
    ("Rainfall", "Flooding"): 0.82,
    ("DrainageCapacity", "Flooding"): 0.77,
    ("Flooding", "TrafficCongestion"): 0.76,
    ("ConstructionActivity", "TrafficCongestion"): 0.91,
    ("Accident", "TrafficCongestion"): 0.55,
    ("TrafficCongestion", "EmergencyDelay"): 0.84
}

def get_confidence(source: str, target: str) -> float:
    """
    Returns the confidence score for a given edge.
    Default fallback confidence is 0.5 if the edge is missing.
    """
    return CONFIDENCE_MATRIX.get((source, target), 0.5)
```

---

## 3. Cause Analysis Algorithm

This algorithm combines computed parent probabilities, theoretical edge weights (derived from CPDs or assumed equal), and the static confidence matrix.

**File: `analysis/cause_analyzer.py`**

### Pseudocode

```python
def compute_causal_contributions(zone: str, target_event: str) -> list:
    # 1. Fetch active causal graph for the zone
    graph = get_causal_graph(zone)
    
    # 2. Identify parent nodes of the target_event
    parents = graph.model.get_parents(target_event)
    if not parents:
        return []

    # 3. Retrieve node probabilities (run inference to get current state)
    posterior_probs = graph.run_inference()
    
    # Also fetch the current probability of 'True'/'High' for each parent
    # Note: If a parent like Rainfall is in evidence, probability is 1.0 (or 0.0)
    # Ensure all parent probabilities are accessible.
    
    raw_contributions = []
    total_raw_contribution = 0.0

    # 4. Compute Contribution Scores
    for parent in parents:
        # A. Get P(parent)
        p_parent = get_probability_of_node(graph, parent) 
        
        # B. Get Edge Weight (Default to 1.0 for this stage if not using explicit partial weights)
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

    # 5. Normalize Contributions
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

    # 6. Rank Causes (Descending)
    final_causes.sort(key=lambda x: x["contribution"], reverse=True)
    
    return final_causes
```

---

## 4. API Integration

We will implement a new explicitly dedicated endpoint `GET /cause-analysis` using Stage-10 logic. Note: In Stage-3, a stub endpoint existed; it must be completely overwritten by this advanced schema.

**Data Models (`models/prediction_model.py`):**
```python
class CauseItemModel(BaseModel):
    event: str
    contribution: float
    confidence: float

class CauseAnalysisResponse(BaseModel):
    zone: str
    event: str
    causes: List[CauseItemModel]
```

**API Router Update (`api/causal_api.py`):**
```python
# OVERWRITE the Stage-3 version of /cause-analysis
@router.get("/cause-analysis", response_model=CauseAnalysisResponse)
async def api_get_cause_analysis(zone: str = Query(...), event: str = Query(...)):
    causes = compute_causal_contributions(zone, event)
    return CauseAnalysisResponse(
        zone=zone,
        event=event,
        causes=causes
    )
```

---

## 5. Complete Testing Suite Specification

**File: `backend/test_stage_10.py`**

### TEST 1 — Cause Identification
*   **Action**: Mock `get_parents("TrafficCongestion")`.
*   **Assertion**: Array must strictly contain exactly `['Flooding', 'ConstructionActivity', 'Accident']`.

### TEST 2 — Contribution Calculation
*   **Action**: Inject mock probabilities: `Flooding = 0.40`, `ConstructionActivity = 0.70`, `Accident = 0.15`.
*   **Action**: Compute causes.
*   **Assertion**: The ranked order returned must be `[ConstructionActivity, Flooding, Accident]`.

### TEST 3 — Confidence Matrix Usage
*   **Action**: Call `get_confidence("ConstructionActivity", "TrafficCongestion")`.
*   **Assertion**: Must equal `0.91`.
*   **Action**: Call `get_confidence("Unknown", "TrafficCongestion")`.
*   **Assertion**: Must equal the fallback `0.5`.

### TEST 4 — Normalization Check
*   **Action**: Aggregate the `contribution` floats from the resulting array of Test 2.
*   **Assertion**: `math.isclose(sum(contributions), 1.0, rel_tol=1e-2)` must be `True`.

### TEST 5 — API Endpoint
*   **Action**: Perform an HTTP `GET /cause-analysis?zone=Bibwewadi&event=TrafficCongestion`.
*   **Assertion**: `response.status_code == 200`.
*   **Assertion**: Validate the JSON keys exist (`zone`, `event`, `causes`). Check that the inner `causes` array dict contains `event`, `contribution`, and `confidence`.
