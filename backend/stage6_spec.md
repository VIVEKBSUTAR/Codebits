# Stage-6 Backend System Specification: Resource Optimization Engine

This document outlines the Stage-6 implementation for the Resource Optimization Engine. Operating at the apex of the causal intelligence pyramid, this module evaluates limited physical resources against the predicted chronological risks across the city, generating an optimal deployment strategy to minimize total localized hazard escalation.

---

## Optimization Model

The model determines optimal allocations by evaluating the marginal utility of each resource within each geography. The core metric is $Benefit$.

### Benefit Calculation Definition
Using the Stage-5 Counterfactual Simulation algorithm, the benefit of an action is defined as the summed reduction of risk probabilities it provides to that zone.

**Formula:**
$Benefit(Z, A) = \sum (Risk\_baseline(Z) - Risk\_intervention_{A}(Z))$
Where $Z$ = Zone, $A$ = Action.

If a `deploy_pump` in Bibwewadi reduces Flooding risk by 0.32 and Traffic risk by 0.10, the Net Benefit for the pair (`Bibwewadi`, `deploy_pump`) is **0.42**.

---

## Benefit Matrix Generation

The Optimizer builds an $N \times M$ matrix ($N$ = Number of affected zones, $M$ = Number of distinct interventions) by executing bulk hypothetical queries against the Causal Inference Engine.

### Generation Process
1.  **Identify Active Zones**: Query the `memory_store` for zones that have active events.
2.  **Define Candidates**: List all supported standard interventions (`deploy_pump`, `close_road`, `dispatch_ambulance`).
3.  **Simulate Matrix**: Iteratively run `simulate_intervention(zone, action)` for every $(Z_i, A_j)$ combination.
4.  **Extract Scores**: Sum the `benefit` dictionary properties from the Stage-5 response to calculate a single scalar value for that cell.

**Example Internal Structure:**
```python
matrix = {
    "Bibwewadi": {"deploy_pump": 0.42, "close_road": 0.15, "dispatch_ambulance": 0.05},
    "Katraj": {"deploy_pump": 0.10, "close_road": 0.18, "dispatch_ambulance": 0.25},
    "Shivajinagar": {"deploy_pump": 0.05, "close_road": 0.22, "dispatch_ambulance": 0.18}
}
```

---

## Greedy Optimization Algorithm

Due to processing constraints typical of real-time disaster response engines, a greedy optimization strategy is implemented for allocation.

### Algorithm Pseudocode

```python
def generate_optimal_deployment(available_resources: dict, active_zones: list) -> dict:
    # 1. Generate Benefit Matrix
    flat_benefits = []
    for zone in active_zones:
        for action in ["deploy_pump", "close_road", "dispatch_ambulance"]:
            response = simulate_intervention(zone, action)
            total_benefit = sum(response.benefit.values())
            
            # Action Mapping to Resource Type
            resource_type = get_resource_type_for_action(action) 
            
            if total_benefit > 0:
                flat_benefits.append({
                    "zone": zone,
                    "action": action,
                    "resource": resource_type,
                    "benefit_score": total_benefit
                })

    # 2. Sort by Maximum Utility
    flat_benefits.sort(key=lambda x: x["benefit_score"], reverse=True)

    # 3. Allocate based on Constraints
    deployment_plan = []
    total_expected_reduction = 0.0
    
    # Track inventory limits locally
    inventory = available_resources.copy()

    for item in flat_benefits:
        res = item["resource"]
        
        # Check constraint limits
        if inventory.get(res, 0) > 0:
            # Allocate
            deployment_plan.append({
                "resource": res,
                "zone": item["zone"],
                "action": item["action"],
                "benefit_expected": item["benefit_score"]
            })
            
            # Deduct inventory
            inventory[res] -= 1
            
            # Aggregate total metric
            total_expected_reduction += item["benefit_score"]

    return {
        "plan": deployment_plan,
        "expected_citywide_risk_reduction": round(total_expected_reduction, 2)
    }
```

---

## Resource Constraint Model

Inventory configuration is passed dynamically during the API query. 

**Structure (`resources` dictionary):**
*   `pumps`: Integer mapped to `deploy_pump` action.
*   `ambulances`: Integer mapped to `dispatch_ambulance` action.
*   `traffic_units`: Integer mapped to `close_road` action.

---

## Optimizer Module Architecture

**File:** `optimization/resource_optimizer.py`

**Responsibilities:**
1.  **State Discovery**: Scanning all stored events to determine which zones actually have active probabilistic chains running.
2.  **Matrix Formulation**: Coordinating bulk simulation calls to the `intervention_engine`.
3.  **Allocation Engine**: Executing the greedy sorting and constraint deduction loops.
4.  **Schema Formatting**: Generating the final `OptimalDeploymentResponse` schema.

---

## API Design

### Optimal Deployment API (`GET /optimal-deployment`)

*   **Description**: Evaluates resource parameters against citywide active Causal Graphs to recommend optimal tactical allocations.
*   **Query Parameters**:
    *   `resources` (string format: `pumps:1,ambulances:2,traffic:1`)
*   **Success Response Example (200 OK)**:
    ```json
    {
      "plan": [
        {
           "resource": "pump",
           "zone": "Bibwewadi",
           "action": "deploy_pump",
           "benefit_expected": 0.42
        },
        {
           "resource": "ambulance",
           "zone": "Katraj",
           "action": "dispatch_ambulance",
           "benefit_expected": 0.25
        },
        {
           "resource": "ambulance",
           "zone": "Shivajinagar",
           "action": "dispatch_ambulance",
           "benefit_expected": 0.18
        }
      ],
      "expected_citywide_risk_reduction": 0.85
    }
    ```

---

## Final Integration Flow (System Pipeline)

1.  **Ingestion**: `POST /events` loads disparate telemetry into standard schemas.
2.  **Evidence Matrix**: Incoming events translate directly into hard logical overrides within local Bayesian DAGs.
3.  **Bayesian Causal Inference**: Variable Elimination algorithm dynamically updates unobserved variable expectations `P(Consequence | Evidence)`.
4.  **Temporal Cascade Prediction**: Breadth-First-Search calculates delay matrices to project a chronological `GET /zone-timeline`.
5.  **Intervention Simulation**: Virtual sandbox clones the DAG, overriding states (Do-Calculus) to compute risk reductions (`Benefit`).
6.  **Resource Optimization (Stage-6)**: `GET /optimal-deployment` wraps Stage 5 in a programmatic loop across all affected zones. It builds an extensive hypothetical output matrix, ranks utilities, and limits assignments based on parameterized supply pools, resulting in a mathematical **Decision Plan**.
