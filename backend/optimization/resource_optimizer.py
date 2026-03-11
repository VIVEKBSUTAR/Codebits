from typing import Dict, List, Optional, Any
import numpy as np

from causal_engine.causal_graph import zone_graphs
from simulation.intervention_engine import simulate_intervention
from models.prediction_model import OptimalDeploymentResponse, DeploymentItemModel
from optimization.multi_objective_optimizer import (
    NSGA2Optimizer, RobustOptimizer, OptimizationObjective,
    diminishing_returns_utility, threshold_utility
)
from optimization.chance_constrained_optimizer import (
    ChanceConstrainedOptimizer, create_response_time_constraint,
    create_coverage_constraint, create_budget_risk_constraint
)
from utils.logger import SystemLogger

logger = SystemLogger(module_name="advanced_optimization")

# Action to resource mapping (legacy compatibility)
ACTION_TO_RESOURCE = {
    "deploy_pump": "pumps",
    "close_road": "traffic_units",
    "dispatch_ambulance": "ambulances"
}

class AdvancedResourceOptimizer:
    """
    Sophisticated multi-objective resource allocation optimizer combining:
    - NSGA-II genetic algorithm for Pareto-optimal solutions
    - Robust optimization for worst-case scenario protection
    - Chance-constrained programming for probabilistic guarantees
    - Advanced utility functions with diminishing returns
    """

    def __init__(self):
        self.nsga2_optimizer = NSGA2Optimizer(
            population_size=50,  # Reduced for practical performance
            max_generations=100,
            crossover_rate=0.9,
            mutation_rate=0.1
        )
        self.robust_optimizer = RobustOptimizer()
        self.chance_optimizer = ChanceConstrainedOptimizer()

    def optimize_advanced_deployment(self,
                                   resources: Dict[str, int],
                                   optimization_method: str = 'nsga2',
                                   objectives: List[str] = None,
                                   constraints: List[str] = None,
                                   risk_tolerance: float = 0.1,
                                   budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Advanced multi-objective resource optimization

        Args:
            resources: Available resources {resource_type: count}
            optimization_method: 'nsga2', 'robust', 'chance_constrained', or 'legacy'
            objectives: List of objectives to optimize ['risk_reduction', 'cost_efficiency', 'social_equity', 'robustness']
            constraints: List of constraints ['budget', 'coverage', 'response_time']
            risk_tolerance: Risk tolerance for robust optimization (0.0 to 1.0)
            budget_limit: Maximum budget constraint

        Returns:
            Comprehensive optimization results with Pareto-optimal solutions
        """

        logger.log(f"Starting advanced optimization with method: {optimization_method}")

        # Get active zones
        active_zones = list(zone_graphs.keys())
        if not active_zones:
            logger.log("No active zones found, returning empty optimization")
            return self._create_empty_result()

        # Set default objectives if not specified
        if objectives is None:
            objectives = ['risk_reduction', 'cost_efficiency', 'social_equity']

        # Create intervention evaluator function
        def intervention_evaluator(zone: str, intervention: str) -> float:
            try:
                response = simulate_intervention(zone, intervention)
                return sum(response.benefit.values())
            except Exception as e:
                logger.log(f"Error evaluating intervention {intervention} in {zone}: {str(e)}")
                return 0.0

        # Route to appropriate optimizer
        if optimization_method == 'nsga2':
            return self._optimize_with_nsga2(
                active_zones, resources, objectives, intervention_evaluator, budget_limit
            )
        elif optimization_method == 'robust':
            return self._optimize_robust(
                active_zones, resources, intervention_evaluator, risk_tolerance
            )
        elif optimization_method == 'chance_constrained':
            return self._optimize_chance_constrained(
                active_zones, resources, constraints or [], intervention_evaluator, budget_limit
            )
        elif optimization_method == 'legacy':
            return self._optimize_legacy(resources)
        else:
            logger.log(f"Unknown optimization method: {optimization_method}, falling back to NSGA-II")
            return self._optimize_with_nsga2(
                active_zones, resources, objectives, intervention_evaluator, budget_limit
            )

    def _optimize_with_nsga2(self,
                           zones: List[str],
                           resources: Dict[str, int],
                           objectives: List[str],
                           intervention_evaluator,
                           budget_limit: Optional[float]) -> Dict[str, Any]:
        """Optimize using NSGA-II multi-objective algorithm"""

        # Create optimization objectives with sophisticated utility functions
        optimization_objectives = []

        for obj_name in objectives:
            if obj_name == 'risk_reduction':
                optimization_objectives.append(OptimizationObjective(
                    name='risk_reduction',
                    weight=1.0,
                    maximize=True,
                    utility_function=lambda x: diminishing_returns_utility(x, saturation_point=15.0)
                ))
            elif obj_name == 'cost_efficiency':
                optimization_objectives.append(OptimizationObjective(
                    name='cost_efficiency',
                    weight=1.0,
                    maximize=True,
                    utility_function=lambda x: threshold_utility(x, threshold=2.0)
                ))
            elif obj_name == 'social_equity':
                optimization_objectives.append(OptimizationObjective(
                    name='social_equity',
                    weight=1.0,
                    maximize=True
                ))
            elif obj_name == 'robustness':
                optimization_objectives.append(OptimizationObjective(
                    name='robustness',
                    weight=1.0,
                    maximize=True
                ))

        # Create problem definition
        problem_definition = {
            'zones': zones,
            'resources': resources,
            'objectives': optimization_objectives,
            'constraints': [],
            'intervention_evaluator': intervention_evaluator
        }

        # Add budget constraint if specified
        if budget_limit:
            from optimization.multi_objective_optimizer import OptimizationConstraint
            budget_constraint = OptimizationConstraint(
                constraint_type='budget',
                parameters={'budget_limit': budget_limit}
            )
            problem_definition['constraints'] = [budget_constraint]

        # Run NSGA-II optimization
        pareto_solutions = self.nsga2_optimizer.optimize(problem_definition)

        # Convert results to deployment format
        return self._format_nsga2_results(pareto_solutions, zones)

    def _optimize_robust(self,
                       zones: List[str],
                       resources: Dict[str, int],
                       intervention_evaluator,
                       risk_tolerance: float) -> Dict[str, Any]:
        """Optimize using robust optimization for worst-case protection"""

        # Create uncertainty scenarios
        uncertainty_scenarios = []

        # Scenario 1: Low effectiveness (pessimistic)
        low_eff_scenario = {
            'effectiveness_factors': {
                'pumps': 0.5,
                'traffic_units': 0.6,
                'ambulances': 0.4
            }
        }
        uncertainty_scenarios.append(low_eff_scenario)

        # Scenario 2: High cost scenario
        high_cost_scenario = {
            'effectiveness_factors': {
                'pumps': 0.8,
                'traffic_units': 0.7,
                'ambulances': 0.9
            },
            'cost_multiplier': 1.5
        }
        uncertainty_scenarios.append(high_cost_scenario)

        # Scenario 3: Resource availability issues
        resource_constraint_scenario = {
            'effectiveness_factors': {
                'pumps': 0.7,
                'traffic_units': 0.8,
                'ambulances': 0.6
            },
            'availability_factor': 0.8
        }
        uncertainty_scenarios.append(resource_constraint_scenario)

        # Prepare robust optimization problem
        base_problem = {
            'zones': zones,
            'resources': resources,
            'intervention_evaluator': intervention_evaluator
        }

        # Run robust optimization
        robust_solution = self.robust_optimizer.optimize_robust_allocation(
            base_problem, uncertainty_scenarios
        )

        # Convert to deployment format
        return self._format_robust_results(robust_solution, zones)

    def _optimize_chance_constrained(self,
                                   zones: List[str],
                                   resources: Dict[str, int],
                                   constraints: List[str],
                                   intervention_evaluator,
                                   budget_limit: Optional[float]) -> Dict[str, Any]:
        """Optimize using chance-constrained programming"""

        # Create probabilistic constraints
        probabilistic_constraints = []

        if 'response_time' in constraints:
            response_time_constraint = create_response_time_constraint(
                min_reduction=2.0, confidence=0.9
            )
            probabilistic_constraints.append(response_time_constraint)

        if 'coverage' in constraints:
            coverage_constraint = create_coverage_constraint(
                min_coverage=0.8, confidence=0.95
            )
            probabilistic_constraints.append(coverage_constraint)

        if 'budget' in constraints or budget_limit:
            budget_constraint = create_budget_risk_constraint(
                budget_limit=budget_limit or 1000.0, confidence=0.9
            )
            probabilistic_constraints.append(budget_constraint)

        # Define stochastic objective function
        def stochastic_objective(allocation_dict):
            total_expected_benefit = 0.0
            for zone, zone_allocations in allocation_dict.items():
                for resource_type, count in zone_allocations.items():
                    if count > 0:
                        intervention = self._resource_to_intervention(resource_type)
                        if intervention:
                            base_benefit = intervention_evaluator(zone, intervention)
                            expected_benefit = base_benefit * 0.8  # Expected effectiveness
                            total_expected_benefit += expected_benefit * count
            return total_expected_benefit

        # Run chance-constrained optimization
        stochastic_solution = self.chance_optimizer.solve_chance_constrained_problem(
            zones, resources, probabilistic_constraints, stochastic_objective, intervention_evaluator
        )

        # Convert to deployment format
        return self._format_chance_constrained_results(stochastic_solution, zones)

    def _optimize_legacy(self, resources: Dict[str, int]) -> Dict[str, Any]:
        """Legacy optimization method for backward compatibility"""
        logger.log("Using legacy optimization method")

        # Use the original greedy algorithm
        legacy_result = generate_optimal_deployment_legacy(resources)

        return {
            'method': 'legacy',
            'solutions': [{
                'allocation': self._deployment_to_allocation_dict(legacy_result.plan),
                'objectives': {
                    'risk_reduction': legacy_result.expected_citywide_risk_reduction,
                    'total_cost': self._calculate_deployment_cost(legacy_result.plan)
                },
                'deployment_plan': legacy_result.plan,
                'pareto_rank': 0
            }],
            'optimization_summary': {
                'pareto_solutions_count': 1,
                'optimization_method': 'greedy',
                'total_expected_reduction': legacy_result.expected_citywide_risk_reduction
            }
        }

    def _resource_to_intervention(self, resource_type: str) -> Optional[str]:
        """Map resource type to intervention"""
        resource_mapping = {
            'pumps': 'deploy_pump',
            'traffic_units': 'close_road',
            'ambulances': 'dispatch_ambulance'
        }
        return resource_mapping.get(resource_type)

    def _format_nsga2_results(self, pareto_solutions, zones: List[str]) -> Dict[str, Any]:
        """Format NSGA-II results for API response"""

        formatted_solutions = []

        for solution in pareto_solutions:
            # Convert to deployment plan
            deployment_plan = self._allocation_to_deployment_plan(solution.allocations)

            formatted_solution = {
                'allocation': solution.allocations,
                'objectives': solution.objective_values,
                'pareto_rank': solution.pareto_rank,
                'crowding_distance': solution.crowding_distance,
                'robustness_score': solution.robustness_score,
                'risk_metrics': solution.risk_metrics,
                'deployment_plan': deployment_plan,
                'constraints_satisfied': solution.constraints_satisfied
            }

            formatted_solutions.append(formatted_solution)

        # Sort solutions by Pareto rank and crowding distance
        formatted_solutions.sort(key=lambda x: (x['pareto_rank'], -x['crowding_distance']))

        return {
            'method': 'nsga2',
            'solutions': formatted_solutions,
            'optimization_summary': {
                'pareto_solutions_count': len(pareto_solutions),
                'optimization_method': 'NSGA-II Multi-Objective Genetic Algorithm',
                'best_solution_objectives': formatted_solutions[0]['objectives'] if formatted_solutions else {},
                'pareto_front_diversity': self._calculate_front_diversity(formatted_solutions)
            },
            'recommendations': self._generate_optimization_recommendations(formatted_solutions)
        }

    def _format_robust_results(self, robust_solution, zones: List[str]) -> Dict[str, Any]:
        """Format robust optimization results"""

        deployment_plan = self._allocation_to_deployment_plan(robust_solution.allocations)

        return {
            'method': 'robust',
            'solutions': [{
                'allocation': robust_solution.allocations,
                'objectives': robust_solution.objective_values,
                'deployment_plan': deployment_plan,
                'robustness_guarantees': 'Optimized for worst-case scenario performance',
                'risk_tolerance': 'Conservative approach with uncertainty protection'
            }],
            'optimization_summary': {
                'optimization_method': 'Robust Optimization (Minimax)',
                'worst_case_performance': robust_solution.objective_values,
                'robustness_level': 'High'
            }
        }

    def _format_chance_constrained_results(self, stochastic_solution, zones: List[str]) -> Dict[str, Any]:
        """Format chance-constrained optimization results"""

        deployment_plan = self._allocation_to_deployment_plan(stochastic_solution.allocation)

        return {
            'method': 'chance_constrained',
            'solutions': [{
                'allocation': stochastic_solution.allocation,
                'objectives': {'expected_benefit': stochastic_solution.objective_value},
                'deployment_plan': deployment_plan,
                'constraint_satisfaction_probabilities': stochastic_solution.constraint_satisfaction_probabilities,
                'expected_outcomes': stochastic_solution.expected_outcomes,
                'risk_measures': stochastic_solution.risk_measures,
                'robustness_metrics': stochastic_solution.robustness_metrics
            }],
            'optimization_summary': {
                'optimization_method': 'Chance-Constrained Programming',
                'probabilistic_guarantees': stochastic_solution.constraint_satisfaction_probabilities,
                'expected_performance': stochastic_solution.expected_outcomes
            }
        }

    def _allocation_to_deployment_plan(self, allocation: Dict[str, Dict[str, int]]) -> List[DeploymentItemModel]:
        """Convert allocation dictionary to deployment plan"""

        deployment_plan = []

        for zone, zone_allocations in allocation.items():
            for resource_type, count in zone_allocations.items():
                if count > 0:
                    # Map resource to action
                    action = self._resource_to_intervention(resource_type)

                    if action:
                        # Calculate expected benefit
                        try:
                            response = simulate_intervention(zone, action)
                            benefit = sum(response.benefit.values())
                        except:
                            benefit = 1.0  # Default benefit

                        # Create deployment items (one per resource unit)
                        for _ in range(count):
                            deployment_plan.append(DeploymentItemModel(
                                resource=resource_type.rstrip('s'),  # Remove plural
                                zone=zone,
                                action=action,
                                benefit_expected=round(benefit, 2)
                            ))

        return deployment_plan

    def _deployment_to_allocation_dict(self, deployment_plan: List[DeploymentItemModel]) -> Dict[str, Dict[str, int]]:
        """Convert deployment plan to allocation dictionary"""

        allocation = {}

        for item in deployment_plan:
            zone = item.zone
            resource_type = item.resource + 's'  # Add plural

            if zone not in allocation:
                allocation[zone] = {}

            if resource_type not in allocation[zone]:
                allocation[zone][resource_type] = 0

            allocation[zone][resource_type] += 1

        return allocation

    def _calculate_deployment_cost(self, deployment_plan: List[DeploymentItemModel]) -> float:
        """Calculate total cost of deployment plan"""

        cost_per_resource = {
            'pump': 100.0,
            'traffic_unit': 25.0,
            'ambulance': 50.0
        }

        total_cost = 0.0

        for item in deployment_plan:
            unit_cost = cost_per_resource.get(item.resource, 10.0)
            total_cost += unit_cost

        return total_cost

    def _calculate_front_diversity(self, solutions: List[Dict[str, Any]]) -> float:
        """Calculate diversity of Pareto front solutions"""

        if len(solutions) <= 1:
            return 0.0

        # Calculate average crowding distance as diversity metric
        crowding_distances = [sol.get('crowding_distance', 0.0) for sol in solutions]
        return float(np.mean(crowding_distances))

    def _generate_optimization_recommendations(self, solutions: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on optimization results"""

        recommendations = []

        if not solutions:
            return ["No feasible solutions found. Consider relaxing constraints or increasing resource availability."]

        best_solution = solutions[0]
        best_objectives = best_solution.get('objectives', {})

        # Risk reduction recommendations
        risk_reduction = best_objectives.get('risk_reduction', 0.0)
        if risk_reduction > 10.0:
            recommendations.append("Excellent risk reduction potential identified - deployment highly recommended")
        elif risk_reduction > 5.0:
            recommendations.append("Good risk reduction achievable with optimal deployment")
        else:
            recommendations.append("Limited risk reduction potential - consider additional resources")

        # Cost efficiency recommendations
        cost_efficiency = best_objectives.get('cost_efficiency', 0.0)
        if cost_efficiency > 2.0:
            recommendations.append("Highly cost-efficient deployment strategy identified")
        elif cost_efficiency > 1.0:
            recommendations.append("Balanced cost-efficiency achieved")
        else:
            recommendations.append("Consider cost optimization - current deployment may be expensive")

        # Equity recommendations
        social_equity = best_objectives.get('social_equity', 0.0)
        if social_equity > 0.8:
            recommendations.append("Excellent social equity in resource distribution")
        elif social_equity > 0.6:
            recommendations.append("Good equity balance achieved")
        else:
            recommendations.append("Consider more equitable resource distribution across zones")

        # Pareto front analysis
        if len(solutions) > 1:
            recommendations.append(f"Multiple Pareto-optimal solutions available ({len(solutions)} options)")
            recommendations.append("Review trade-offs between objectives to select preferred solution")

        return recommendations

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no zones are active"""
        return {
            'method': 'none',
            'solutions': [],
            'optimization_summary': {
                'pareto_solutions_count': 0,
                'optimization_method': 'No active zones',
                'message': 'No zones with active events found for optimization'
            },
            'recommendations': ['Process events in zones to enable resource optimization']
        }


# Global advanced optimizer instance
_advanced_optimizer = AdvancedResourceOptimizer()

# Legacy compatibility functions

def parse_resources(resources_str: str) -> Dict[str, int]:
    """
    Parses a string like 'pumps:1,ambulances:2,traffic:1' into a dict.
    """
    inventory = {}
    if not resources_str:
        return inventory

    parts = resources_str.split(',')
    for part in parts:
        if ':' in part:
            key, val = part.split(':', 1)
            key = key.strip()
            # Standardize names
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

def generate_optimal_deployment_legacy(resources: Dict[str, int]) -> OptimalDeploymentResponse:
    """Legacy greedy optimization for backward compatibility"""

    # Generate benefit matrix
    flat_benefits = []
    active_zones = list(zone_graphs.keys())

    for zone in active_zones:
        for action, resource_type in ACTION_TO_RESOURCE.items():
            if resources.get(resource_type, 0) > 0:
                # Simulate intervention
                response = simulate_intervention(zone, action)
                total_benefit = sum(response.benefit.values())

                if total_benefit > 0.0:
                    flat_benefits.append({
                        "zone": zone,
                        "action": action,
                        "resource": resource_type,
                        "benefit_score": total_benefit
                    })

    # Sort by maximum utility (greedy approach)
    flat_benefits.sort(key=lambda x: x["benefit_score"], reverse=True)

    # Allocate based on constraints
    deployment_plan = []
    total_expected_reduction = 0.0
    inventory = resources.copy()

    for item in flat_benefits:
        res = item["resource"]

        if inventory.get(res, 0) > 0:
            # Format resource name
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

            inventory[res] -= 1
            total_expected_reduction += item["benefit_score"]

    return OptimalDeploymentResponse(
        plan=deployment_plan,
        expected_citywide_risk_reduction=round(total_expected_reduction, 2)
    )

def generate_optimal_deployment(resources_str: str) -> OptimalDeploymentResponse:
    """
    Main entry point for resource optimization with advanced capabilities

    This function now supports sophisticated multi-objective optimization
    while maintaining backward compatibility with the original API.
    """

    available_resources = parse_resources(resources_str)
    logger.log(f"Starting deployment optimization with resources: {available_resources}")

    # Use advanced optimization by default, fall back to legacy if needed
    try:
        advanced_results = _advanced_optimizer.optimize_advanced_deployment(
            available_resources,
            optimization_method='nsga2',  # Default to sophisticated multi-objective optimization
            objectives=['risk_reduction', 'cost_efficiency', 'social_equity']
        )

        # Extract best solution for backward compatibility
        if advanced_results['solutions']:
            best_solution = advanced_results['solutions'][0]
            deployment_plan = best_solution['deployment_plan']

            # Calculate total expected reduction
            total_reduction = sum(item.benefit_expected for item in deployment_plan)

            logger.log(f"Advanced optimization completed with {len(advanced_results['solutions'])} Pareto-optimal solutions")

            return OptimalDeploymentResponse(
                plan=deployment_plan,
                expected_citywide_risk_reduction=round(total_reduction, 2)
            )
        else:
            logger.log("Advanced optimization found no solutions, using legacy method")
            return generate_optimal_deployment_legacy(available_resources)

    except Exception as e:
        logger.log(f"Advanced optimization failed: {str(e)}, falling back to legacy method")
        return generate_optimal_deployment_legacy(available_resources)

def get_advanced_optimization_results(resources_str: str,
                                    method: str = 'nsga2',
                                    objectives: List[str] = None,
                                    budget_limit: Optional[float] = None) -> Dict[str, Any]:
    """
    Get detailed advanced optimization results for sophisticated analysis

    Args:
        resources_str: Resource string in format 'pumps:2,ambulances:1,traffic_units:3'
        method: Optimization method ('nsga2', 'robust', 'chance_constrained')
        objectives: List of objectives to optimize
        budget_limit: Optional budget constraint

    Returns:
        Detailed optimization results with Pareto-optimal solutions and analysis
    """

    available_resources = parse_resources(resources_str)

    return _advanced_optimizer.optimize_advanced_deployment(
        available_resources,
        optimization_method=method,
        objectives=objectives,
        budget_limit=budget_limit
    )

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
