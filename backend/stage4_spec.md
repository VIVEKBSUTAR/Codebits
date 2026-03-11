# Stage-4 Backend System Specification: Temporal Cascade Prediction Engine

This document outlines the Stage-4 implementation of the Temporal Cascade Prediction Engine. This component introduces time dynamics to the probabilistic outputs of the Causal Inference Engine, forecasting when and how infrastructure failures will escalate.

---

## Temporal Cascade Model

The Temporal Cascade Model overlays a time-to-impact (propagation delay) onto the directed edges of the Bayesian Network. While the causal graph computes *if* an event happens (probability), the cascade engine computes *when* it happens.

### Edge Delay System

Time delays are stored programmatically as an explicit mapping of directed edges `(SourceNode, TargetNode)` to an integer representing minutes.

**Delay Dictionary Definition:**
```python
EDGE_DELAYS = {
    ("Rainfall", "DrainageCapacity"): 10,
    ("DrainageCapacity", "Flooding"): 15,
    ("Rainfall", "Flooding"): 20, # Direct structural impact
    ("Flooding", "TrafficCongestion"): 15,
    ("ConstructionActivity", "TrafficCongestion"): 5,
    ("Accident", "TrafficCongestion"): 0, # Immediate impact
    ("TrafficCongestion", "EmergencyDelay"): 20
}
```

---

## Propagation Algorithm

The propagation algorithm merges the topological ordering of the DAG with the calculated posterior probabilities to generate a chronological escalation path.

### Timeline Generation Logic

**Algorithm Steps:**
1.  **Retrieve State**: Fetch the active `CausalGraphService` for the given zone.
2.  **Validate Evidence**: Determine the timestamp of the earliest active evidence node (e.g., Rainfall reported at 09:00). This serves as `T_zero`.
3.  **Run Inference**: Execute `graph.run_inference()` to obtain posterior probabilities for all downstream nodes.
4.  **Filter Probabilities**: Discard nodes where the calculated risk probability is below a significant threshold (e.g., `< 0.40`).
5.  **Calculate Propagation Time**:
    *   Traverse the DAG using Breadth-First Search (BFS) starting from the active evidence nodes.
    *   For a target node `Y` with an evidence root `X`, the total delay is the sum of delays along the shortest path `X -> ... -> Y`.
    *   `Projected_Time(Y) = T_zero + timedelta(minutes=Total_Delay)`
    *   *Conflict Resolution*: If multiple evidence nodes lead to `Y`, select the path that results in the *earliest* `Projected_Time`.
6.  **Build Escalation Timeline**: Format the results into a chronological array of predicted events.

---

## Data Models

Additional schemas are required to represent timeline items and the final API response.

### `TimelineItemModel`
```text
TimelineItemModel {
    predicted_time: str      // e.g., "09:15"
    event_name: str          // Derived from node name (e.g., "DrainageCapacity stress")
    probability: float       // Exact posterior probability (e.g., 0.60)
}
```

### `ZoneTimelineResponse`
```text
ZoneTimelineResponse {
    zone: str
    escalation_risk_score: str   // "LOW", "MEDIUM", "HIGH"
    predicted_events: List[TimelineItemModel]
}
```

---

## Risk Escalation Scoring

The `escalation_risk_score` summarizes the overall severity of the predicted cascade for quick dashboard consumption.

**Computation Strategy:**
1. Calculate the Root Mean Square (RMS) or a weighted sum of the probabilities of critical terminal nodes (e.g., `Flooding`, `TrafficCongestion`, `EmergencyDelay`).
2. Map the scalar value to categorical thresholds:
    *   Weighted Score `< 0.3` => `LOW`
    *   Weighted Score between `0.3` and `0.7` => `MEDIUM`
    *   Weighted Score `> 0.7` => `HIGH`

---

## Cascade Engine Module Structure

A new module will be created to house the temporal logic.

**File:** `simulation/cascade_engine.py`

**Responsibilities:**
1.  Maintain the `EDGE_DELAYS` configuration.
2.  Provide a `generate_timeline(zone: str) -> ZoneTimelineResponse` function.
3.  Implement the BFS DAG traversal for cumulative time calculation.
4.  Implement the dictionary-based phrase mapping (e.g., `TrafficCongestion` -> "Traffic congestion likely").

### Integration Flow (Stage-4)

1. External APIs POST an event (`api/events_api.py`).
2. Pipeline validates and normalizes the event.
3. Event is stored and instantly dispatched to the Causal Graph.
4. Graph updates its internal CPT states per the new evidence.
5. Client makes a GET request to `/zone-timeline`.
6. API routes request to `simulation/cascade_engine.py`.
7. Engine fetches probabilities from Causal Graph, runs the temporal BFS, and formats the chronological timeline response.

---

## API Interface

A new specific endpoint exposes the temporal engine.

### Retrieve Zone Timeline (`GET /zone-timeline`)

*   **Description**: Returns the forecasted timeline of cascading events given the current active evidence.
*   **Query Parameters**:
    *   `zone` (string, required)
*   **Success Response Example (200 OK)**:
    ```json
    {
      "zone": "Bibwewadi",
      "escalation_risk_score": "HIGH",
      "predicted_events": [
        {
          "predicted_time": "09:10",
          "event_name": "DrainageCapacity stress",
          "probability": 0.60
        },
        {
          "predicted_time": "09:25",
          "event_name": "Flooding",
          "probability": 0.75
        },
        {
          "predicted_time": "09:40",
          "event_name": "Traffic congestion",
          "probability": 0.65
        }
      ]
    }
    ```
