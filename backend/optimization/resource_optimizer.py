from typing import Dict, List

from causal_engine.causal_graph import zone_graphs
from simulation.intervention_engine import simulate_intervention
from models.prediction_model import OptimalDeploymentResponse, DeploymentItemModel

# Action to resource mapping
ACTION_TO_RESOURCE = {
    "deploy_pump": "pumps",
    "close_road": "traffic_units",
    "dispatch_ambulance": "ambulances"
}

def parse_resources(resources_str: str) -> Dict[str, int]:
    """
    Parses a string like 'pumps:1,ambulances:2,traffic:1' into a dict.
    Note: mapping 'traffic' to 'traffic_units' if needed, or keeping it as provided.
    For standardizing, we expect the key to match the ACTION_TO_RESOURCE values.
    """
    inventory = {}
    if not resources_str:
        return inventory
        
    parts = resources_str.split(',')
    for part in parts:
        if ':' in part:
            key, val = part.split(':', 1)
            key = key.strip()
            # Standardize names just in case the API user uses shorthand
            if key == "traffic":
                key = "traffic_units"
            if key == "pump":
                key = "pumps"
            if key == "ambulance":
                key = "ambulances"
                
            try:
                inventory[key] = int(val.strip())
            except ValueError:
                pass
    return inventory

def generate_optimal_deployment(resources_str: str) -> OptimalDeploymentResponse:
    available_resources = parse_resources(resources_str)
    
    # 1. Generate Benefit Matrix
    flat_benefits = []
    
    # Identify active zones (zones that have had events processed)
    active_zones = list(zone_graphs.keys())
    
    for zone in active_zones:
        for action, resource_type in ACTION_TO_RESOURCE.items():
            # Only simulate if we actually have that resource type available
            if available_resources.get(resource_type, 0) > 0:
                # Simulate intervention
                response = simulate_intervention(zone, action)
                
                # Sum the benefit reductions
                total_benefit = sum(response.benefit.values())
                
                if total_benefit > 0.0:
                    flat_benefits.append({
                        "zone": zone,
                        "action": action,
                        "resource": resource_type,
                        "benefit_score": total_benefit
                    })

    # 2. Sort by Maximum Utility (Greedy approach)
    flat_benefits.sort(key=lambda x: x["benefit_score"], reverse=True)

    # 3. Allocate based on Constraints
    deployment_plan = []
    total_expected_reduction = 0.0
    
    inventory = available_resources.copy()

    for item in flat_benefits:
        res = item["resource"]
        
        # Check constraint limits
        if inventory.get(res, 0) > 0:
            # Drop the plural 's' for the output UI model if preferred, or keep it.
            display_resource = res[:-1] if res.endswith('s') else res
            if res == "traffic_units":
                display_resource = "traffic_unit"
                
            # Allocate
            deployment_plan.append(DeploymentItemModel(
                resource=display_resource,
                zone=item["zone"],
                action=item["action"],
                benefit_expected=round(item["benefit_score"], 2)
            ))
            
            # Deduct from inventory
            inventory[res] -= 1
            
            # Aggregate total reduction metric
            total_expected_reduction += item["benefit_score"]

    return OptimalDeploymentResponse(
        plan=deployment_plan,
        expected_citywide_risk_reduction=round(total_expected_reduction, 2)
    )
