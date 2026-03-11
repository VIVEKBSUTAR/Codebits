# Stage-5 Backend System Specification: Intervention Simulation Engine

This document outlines the Stage-5 implementation for the Counterfactual Intervention Simulation Engine. This module introduces "Decision Intelligence" by allowing the system to simulate explicit actions (Do-Calculus) and recompute the future timeline and risk metrics, proving mathematical benefit before physical deployment.

---

## Intervention Model

In a Bayesian Network, an intervention $do(X = x)$ mathematically equates to removing all incoming edges to node $X$ and forcing its probability distribution to $1.0$ at state $x$. 

For Stage-5, instead of dynamically mutating the core structure of the DAG (which is computationally expensive and complex to reset), we model interventions as **Virtual Evidence Sets** or **CPD Overrides** that artificially inject counterfactual states.

### Standardized Interventions
1.  **`deploy_pump`**: Targets the `DrainageCapacity` node. Forces state to `Good`.
2.  **`close_road`**: Targets the `TrafficCongestion` node. Forces state to `Low`.
3.  **`dispatch_ambulance`**: Acts as a mitigating factor on `EmergencyDelay`. Forces state to `Low`.

---

## Probability Adjustment Structure

When an intervention is submitted, it maps to specific overriding evidence for the inference engine. 

**Mapping Dictionary:**
```python
INTERVENTION_EFFECTS = {
    "deploy_pump": {"DrainageCapacity": "Good"},
    "close_road": {"TrafficCongestion": "Low"},
    "dispatch_ambulance": {"EmergencyDelay": "Low"}
}
```

*Note on Temporal Adjustments: Interventions not only cap the probability of failures but also inherently delay the propagation of downstream nodes. A `deploy_pump` intervention mathematically nullifies the `DrainageCapacity -> Flooding` risk vector.*

---

## Counterfactual Simulation Algorithm

The simulation algorithm computes the delta between the current real-world trajectory and the hypothetical intervened trajectory.

### Algorithm Pseudocode

```python
def simulate_intervention(zone: str, intervention_action: str):
    
    # 1. Fetch current baseline graph and inference object
    baseline_graph = get_causal_graph(zone)
    current_evidence = baseline_graph.evidence.copy()
    
    # Calculate BEFORE metrics
    baseline_probs = baseline_graph.run_inference()
    from simulation.cascade_engine import generate_timeline # Use logic from Stage-4
    baseline_timeline = generate_timeline(zone) # Uses current_evidence

    # 2. Lookup Intervention Effects
    if intervention_action not in INTERVENTION_EFFECTS:
        raise ValueError("Unknown Intervention")
    effect_evidence = INTERVENTION_EFFECTS[intervention_action]

    # 3. Create a Counterfactual Inference Sandbox
    # We do NOT want to permanently alter the zone's active graph
    # So we use a localized copy of the evidence dictionary merged with the intervention
    simulated_evidence = current_evidence.copy()
    simulated_evidence.update(effect_evidence)

    # 4. Run Bayesian Inference (Counterfactual)
    # Perform variable elimination using the merged simulated evidence
    counterfactual_probs = {}
    for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
        # If the target itself is intervened on, risk is technically 0 for the failure state
        if target in effect_evidence and effect_evidence[target] in ['Good', 'Low', 'False']:
             counterfactual_probs[target] = 0.0
             continue
             
        if target not in simulated_evidence:
            result = baseline_graph.infer.query(variables=[target], evidence=simulated_evidence)
            # Extract probability of 'True' or 'High'
            val = result.values[1] 
            counterfactual_probs[target] = val

    # 5. Recompute Timeline (Counterfactual)
    # Run the BFS propagation algorithm from Stage-4 using `simulated_evidence` and `counterfactual_probs`
    # (Excludes nodes whose probabilities dropped below the threshold)
    counterfactual_timeline = recalculate_timeline_with_sandbox(zone, simulated_evidence, counterfactual_probs)

    # 6. Compute Risk Reduction (Benefit)
    benefit_metrics = calculate_risk_reduction(baseline_probs, counterfactual_probs)

    # 7. Return payload
    return prepare_simulation_response(...)
```

---

## Risk Reduction Metrics

The benefit metric quantifies the absolute reduction in risk likelihood for critical city infrastructure.

### Calculation

$Benefit(Node) = Risk_{baseline}(Node) - Risk_{intervention}(Node)$

If `baseline_flood = 0.72` and `simulated_flood = 0.40`, the `flooding_reduction = 0.32`. If the value is negative (an intervention accidentally made things worse), it is represented exactly as computed to allow the future optimizer (Stage-6) to avoid it.

---

## Simulation Module Architecture

**File:** `simulation/intervention_engine.py`

**Responsibilities:**
1.  **Counterfactual Storage**: Maintain the `INTERVENTION_EFFECTS` mapping matrix.
2.  **Sandbox Execution**: Orchestrate the `simulate_intervention` flow, ensuring the actual `zone_graph` state is never mutated during a hypothetical query.
3.  **Cross-pollination**: Import and reuse the BFS topological delay logic from `cascade_engine.py` to recompute the timeline on the fly.
4.  **Delta Math**: Perform the arithmetic for the Risk Reduction Benefit schemas.

---

## API Design

A new endpoint exclusively for testing hypothetical actions.

### Simulate Intervention (`POST /simulate-intervention`)

*   **Description**: Receives an intervention request, executes the counterfactual graph simulation, and returns the comparative risk and timeline deltas.
*   **Request Body**:
    ```json
    {
      "zone": "Bibwewadi",
      "intervention": "deploy_pump"
    }
    ```
*   **Success Response (200 OK)**:
    ```json
    {
      "zone": "Bibwewadi",
      "intervention": "deploy_pump",
      "baseline_risk": {
        "flooding": 0.72,
        "traffic": 0.64,
        "emergency_delay": 0.35
      },
      "after_intervention": {
        "flooding": 0.30,
        "traffic": 0.45,
        "emergency_delay": 0.20
      },
      "benefit": {
        "flooding_reduction": 0.42,
        "traffic_reduction": 0.19,
        "emergency_delay_reduction": 0.15
      },
      "timeline_changes": {
        "baseline_timeline": [
            {"time": "09:15", "event": "DrainageCapacity stress"},
            {"time": "09:30", "event": "Flooding"}
        ],
        "intervened_timeline": [
             {"time": "09:30", "event": "Flooding"} // Event delayed or removed depending on threshold
        ]
      }
    }
    ```

---

## Integration Flow (System Pipeline)

1.  **Stage-2**: City Event triggers `POST /events` -> Ingestion -> Normalization.
2.  **Stage-3**: Causal Graph Engine receives evidence, mathematical CPDs update.
3.  **Stage-4**: Client polls `GET /zone-timeline`. System predicts standard escalation path.
4.  **Stage-5**: City Operator identifies high risk and submits `POST /simulate-intervention`.
5.  **Intervention Engine**: Locates the Graph, clones the evidence vector, applies the intervention effect, and runs the inference/timeline BFS algorithms again in a sandbox.
6.  **Response**: Returns the comparative metrics to the Operator to aid decision making.
