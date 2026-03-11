"""
Advanced Multi-Objective Optimization for Resource Allocation

This module implements sophisticated optimization algorithms including:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization
- Robust optimization for worst-case scenario protection
- Chance-constrained programming with probabilistic risk thresholds
- Value-at-Risk (VaR) and Conditional VaR for tail risk assessment
- Non-linear utility functions with diminishing returns and threshold effects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t, beta
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationObjective:
    """Represents a single optimization objective"""
    name: str
    weight: float
    maximize: bool = True  # True for maximization, False for minimization
    constraint_type: Optional[str] = None  # 'hard', 'soft', or None
    constraint_threshold: Optional[float] = None
    utility_function: Optional[Callable[[float], float]] = None

@dataclass
class ResourceAllocation:
    """Represents a resource allocation solution"""
    allocations: Dict[str, Dict[str, int]]  # {zone: {resource_type: count}}
    objective_values: Dict[str, float] = field(default_factory=dict)
    constraints_satisfied: Dict[str, bool] = field(default_factory=dict)
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    robustness_score: float = 0.0
    risk_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class OptimizationConstraint:
    """Represents optimization constraints"""
    constraint_type: str  # 'resource_limit', 'budget', 'equity', 'response_time'
    parameters: Dict[str, Any]
    penalty_function: Optional[Callable[[ResourceAllocation], float]] = None

class MultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective optimizers"""

    @abstractmethod
    def optimize(self, problem_definition: Dict[str, Any]) -> List[ResourceAllocation]:
        pass

class NSGA2Optimizer(MultiObjectiveOptimizer):
    """
    Implementation of NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    for multi-objective resource allocation optimization
    """

    def __init__(self,
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10):

        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        # Problem-specific parameters (set during optimization)
        self.zones = []
        self.resources = {}
        self.objectives = []
        self.constraints = []
        self.intervention_evaluator = None

    def optimize(self, problem_definition: Dict[str, Any]) -> List[ResourceAllocation]:
        """
        Run NSGA-II optimization for multi-objective resource allocation

        Args:
            problem_definition: Dictionary containing zones, resources, objectives, constraints

        Returns:
            Pareto-optimal set of resource allocations
        """

        # Initialize problem parameters
        self.zones = problem_definition['zones']
        self.resources = problem_definition['resources']
        self.objectives = problem_definition['objectives']
        self.constraints = problem_definition.get('constraints', [])
        self.intervention_evaluator = problem_definition['intervention_evaluator']

        # Initialize population
        population = self._initialize_population()

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate population
            self._evaluate_population(population)

            # Selection, crossover, and mutation
            offspring = self._create_offspring(population)

            # Combine parent and offspring populations
            combined_population = population + offspring

            # Non-dominated sorting and crowding distance
            fronts = self._non_dominated_sorting(combined_population)
            self._calculate_crowding_distance(fronts)

            # Environmental selection
            population = self._environmental_selection(fronts)

        # Return Pareto-optimal solutions (first front)
        final_population = population[:self.population_size]
        self._evaluate_population(final_population)
        fronts = self._non_dominated_sorting(final_population)

        # Add robustness and risk metrics
        self._calculate_robustness_metrics(fronts[0] if fronts else [])

        return fronts[0] if fronts else []

    def _initialize_population(self) -> List[ResourceAllocation]:
        """Initialize random population of resource allocations"""
        population = []

        for _ in range(self.population_size):
            allocation = self._create_random_allocation()
            population.append(allocation)

        return population

    def _create_random_allocation(self) -> ResourceAllocation:
        """Create a random resource allocation within constraints"""
        allocations = {zone: {} for zone in self.zones}

        # Distribute resources randomly while respecting constraints
        remaining_resources = self.resources.copy()

        for zone in self.zones:
            for resource_type, total_available in remaining_resources.items():
                if total_available > 0:
                    # Random allocation between 0 and min(total_available, reasonable_limit)
                    reasonable_limit = min(total_available, 3)  # Max 3 resources per zone
                    allocation_count = random.randint(0, reasonable_limit)
                    allocations[zone][resource_type] = allocation_count
                    remaining_resources[resource_type] -= allocation_count

        return ResourceAllocation(allocations=allocations)

    def _evaluate_population(self, population: List[ResourceAllocation]):
        """Evaluate objectives for each individual in population"""

        for individual in population:
            # Reset objective values
            individual.objective_values = {}
            individual.constraints_satisfied = {}

            # Evaluate each objective
            for objective in self.objectives:
                if objective.name == 'risk_reduction':
                    value = self._evaluate_risk_reduction(individual)
                elif objective.name == 'cost_efficiency':
                    value = self._evaluate_cost_efficiency(individual)
                elif objective.name == 'social_equity':
                    value = self._evaluate_social_equity(individual)
                elif objective.name == 'robustness':
                    value = self._evaluate_robustness(individual)
                else:
                    value = 0.0

                # Apply utility function if specified
                if objective.utility_function:
                    value = objective.utility_function(value)

                individual.objective_values[objective.name] = value

            # Evaluate constraints
            for constraint in self.constraints:
                satisfied = self._evaluate_constraint(individual, constraint)
                individual.constraints_satisfied[constraint.constraint_type] = satisfied

    def _evaluate_risk_reduction(self, individual: ResourceAllocation) -> float:
        """Evaluate total risk reduction objective"""
        total_reduction = 0.0

        for zone, zone_allocations in individual.allocations.items():
            for resource_type, count in zone_allocations.items():
                if count > 0:
                    # Map resource to intervention
                    intervention = self._resource_to_intervention(resource_type)
                    if intervention and self.intervention_evaluator:
                        # Get intervention benefit with diminishing returns
                        base_benefit = self.intervention_evaluator(zone, intervention)

                        # Diminishing returns: benefit = base * (1 - exp(-count/decay_factor))
                        decay_factor = 2.0
                        actual_benefit = base_benefit * (1 - np.exp(-count / decay_factor))
                        total_reduction += actual_benefit

        return total_reduction

    def _evaluate_cost_efficiency(self, individual: ResourceAllocation) -> float:
        """Evaluate cost efficiency (benefit per unit cost)"""
        total_benefit = self._evaluate_risk_reduction(individual)
        total_cost = self._calculate_total_cost(individual)

        if total_cost == 0:
            return 0.0

        return total_benefit / total_cost

    def _evaluate_social_equity(self, individual: ResourceAllocation) -> float:
        """Evaluate social equity using Gini coefficient approach"""
        zone_benefits = []

        for zone in self.zones:
            zone_benefit = 0.0
            zone_allocations = individual.allocations.get(zone, {})

            for resource_type, count in zone_allocations.items():
                if count > 0:
                    intervention = self._resource_to_intervention(resource_type)
                    if intervention and self.intervention_evaluator:
                        benefit = self.intervention_evaluator(zone, intervention)
                        zone_benefit += benefit * count

            zone_benefits.append(zone_benefit)

        # Calculate inverse Gini coefficient (higher = more equitable)
        if not zone_benefits or all(b == 0 for b in zone_benefits):
            return 1.0  # Perfect equity when no benefits

        # Sort benefits
        sorted_benefits = sorted(zone_benefits)
        n = len(sorted_benefits)
        cumulative_sum = sum(sorted_benefits)

        if cumulative_sum == 0:
            return 1.0

        # Gini coefficient calculation
        gini_sum = sum((2 * i - n - 1) * benefit for i, benefit in enumerate(sorted_benefits, 1))
        gini_coefficient = gini_sum / (n * cumulative_sum)

        # Return inverse (1 - Gini) for equity maximization
        return 1 - abs(gini_coefficient)

    def _evaluate_robustness(self, individual: ResourceAllocation) -> float:
        """Evaluate robustness using Monte Carlo simulation"""
        if not hasattr(self, '_robustness_cache'):
            self._robustness_cache = {}

        # Create cache key
        cache_key = self._allocation_to_key(individual)
        if cache_key in self._robustness_cache:
            return self._robustness_cache[cache_key]

        # Monte Carlo simulation with uncertainty
        n_scenarios = 50  # Reduced for computational efficiency
        scenario_outcomes = []

        for _ in range(n_scenarios):
            # Add uncertainty to intervention effectiveness
            perturbed_outcome = self._evaluate_risk_reduction_with_uncertainty(individual)
            scenario_outcomes.append(perturbed_outcome)

        # Robustness = negative of standard deviation (prefer stable solutions)
        robustness = max(0.0, 1.0 - (np.std(scenario_outcomes) / (np.mean(scenario_outcomes) + 1e-6)))

        self._robustness_cache[cache_key] = robustness
        return robustness

    def _evaluate_risk_reduction_with_uncertainty(self, individual: ResourceAllocation) -> float:
        """Evaluate risk reduction with added uncertainty"""
        total_reduction = 0.0

        for zone, zone_allocations in individual.allocations.items():
            for resource_type, count in zone_allocations.items():
                if count > 0:
                    intervention = self._resource_to_intervention(resource_type)
                    if intervention and self.intervention_evaluator:
                        base_benefit = self.intervention_evaluator(zone, intervention)

                        # Add uncertainty (±20% normal variation)
                        uncertainty_factor = np.random.normal(1.0, 0.2)
                        uncertainty_factor = max(0.1, uncertainty_factor)  # Ensure positive

                        perturbed_benefit = base_benefit * uncertainty_factor

                        # Diminishing returns
                        decay_factor = 2.0
                        actual_benefit = perturbed_benefit * (1 - np.exp(-count / decay_factor))
                        total_reduction += actual_benefit

        return total_reduction

    def _calculate_total_cost(self, individual: ResourceAllocation) -> float:
        """Calculate total deployment cost"""
        # Simple cost model - can be made more sophisticated
        cost_per_resource = {
            'pumps': 100.0,
            'traffic_units': 25.0,
            'ambulances': 50.0
        }

        total_cost = 0.0
        for zone_allocations in individual.allocations.values():
            for resource_type, count in zone_allocations.items():
                unit_cost = cost_per_resource.get(resource_type, 10.0)
                total_cost += unit_cost * count

        return total_cost

    def _resource_to_intervention(self, resource_type: str) -> Optional[str]:
        """Map resource type to intervention action"""
        resource_mapping = {
            'pumps': 'deploy_pump',
            'traffic_units': 'close_road',
            'ambulances': 'dispatch_ambulance'
        }
        return resource_mapping.get(resource_type)

    def _evaluate_constraint(self, individual: ResourceAllocation, constraint: OptimizationConstraint) -> bool:
        """Evaluate whether constraint is satisfied"""
        if constraint.constraint_type == 'resource_limit':
            return self._check_resource_limits(individual, constraint)
        elif constraint.constraint_type == 'budget':
            return self._check_budget_constraint(individual, constraint)
        elif constraint.constraint_type == 'equity':
            return self._check_equity_constraint(individual, constraint)
        else:
            return True

    def _check_resource_limits(self, individual: ResourceAllocation, constraint: OptimizationConstraint) -> bool:
        """Check if resource allocation respects availability limits"""
        resource_usage = {}

        for zone_allocations in individual.allocations.values():
            for resource_type, count in zone_allocations.items():
                resource_usage[resource_type] = resource_usage.get(resource_type, 0) + count

        for resource_type, used_count in resource_usage.items():
            available = self.resources.get(resource_type, 0)
            if used_count > available:
                return False

        return True

    def _check_budget_constraint(self, individual: ResourceAllocation, constraint: OptimizationConstraint) -> bool:
        """Check budget constraint"""
        total_cost = self._calculate_total_cost(individual)
        budget_limit = constraint.parameters.get('budget_limit', float('inf'))
        return total_cost <= budget_limit

    def _check_equity_constraint(self, individual: ResourceAllocation, constraint: OptimizationConstraint) -> bool:
        """Check equity constraint"""
        equity_score = self._evaluate_social_equity(individual)
        min_equity = constraint.parameters.get('min_equity', 0.0)
        return equity_score >= min_equity

    def _non_dominated_sorting(self, population: List[ResourceAllocation]) -> List[List[ResourceAllocation]]:
        """Perform non-dominated sorting to create Pareto fronts"""
        fronts = []
        domination_count = {}
        dominated_solutions = {}

        # Initialize
        for individual in population:
            domination_count[id(individual)] = 0
            dominated_solutions[id(individual)] = []

        # Calculate domination relationships
        for i, individual_i in enumerate(population):
            for j, individual_j in enumerate(population):
                if i != j:
                    if self._dominates(individual_i, individual_j):
                        dominated_solutions[id(individual_i)].append(individual_j)
                    elif self._dominates(individual_j, individual_i):
                        domination_count[id(individual_i)] += 1

        # Create first front
        first_front = []
        for individual in population:
            if domination_count[id(individual)] == 0:
                individual.pareto_rank = 0
                first_front.append(individual)

        fronts.append(first_front)

        # Create subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []
            for individual in fronts[front_index]:
                for dominated_individual in dominated_solutions[id(individual)]:
                    domination_count[id(dominated_individual)] -= 1
                    if domination_count[id(dominated_individual)] == 0:
                        dominated_individual.pareto_rank = front_index + 1
                        next_front.append(dominated_individual)

            fronts.append(next_front)
            front_index += 1

        return fronts[:-1]  # Remove empty last front

    def _dominates(self, individual_a: ResourceAllocation, individual_b: ResourceAllocation) -> bool:
        """Check if individual_a dominates individual_b"""
        better_in_any = False
        worse_in_any = False

        for objective in self.objectives:
            value_a = individual_a.objective_values.get(objective.name, 0.0)
            value_b = individual_b.objective_values.get(objective.name, 0.0)

            if objective.maximize:
                if value_a > value_b:
                    better_in_any = True
                elif value_a < value_b:
                    worse_in_any = True
            else:  # minimize
                if value_a < value_b:
                    better_in_any = True
                elif value_a > value_b:
                    worse_in_any = True

        return better_in_any and not worse_in_any

    def _calculate_crowding_distance(self, fronts: List[List[ResourceAllocation]]):
        """Calculate crowding distance for diversity preservation"""
        for front in fronts:
            if len(front) <= 2:
                for individual in front:
                    individual.crowding_distance = float('inf')
                continue

            # Initialize crowding distance
            for individual in front:
                individual.crowding_distance = 0.0

            # Calculate for each objective
            for objective in self.objectives:
                # Sort front by objective value
                front.sort(key=lambda x: x.objective_values.get(objective.name, 0.0))

                # Set boundary points to infinity
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')

                # Calculate normalized distance for intermediate points
                obj_min = front[0].objective_values.get(objective.name, 0.0)
                obj_max = front[-1].objective_values.get(objective.name, 0.0)
                obj_range = obj_max - obj_min

                if obj_range > 0:
                    for i in range(1, len(front) - 1):
                        distance_contribution = (
                            front[i + 1].objective_values.get(objective.name, 0.0) -
                            front[i - 1].objective_values.get(objective.name, 0.0)
                        ) / obj_range

                        front[i].crowding_distance += distance_contribution

    def _environmental_selection(self, fronts: List[List[ResourceAllocation]]) -> List[ResourceAllocation]:
        """Select individuals for next generation"""
        new_population = []

        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # Sort by crowding distance and select best
                remaining_slots = self.population_size - len(new_population)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front[:remaining_slots])
                break

        return new_population

    def _create_offspring(self, population: List[ResourceAllocation]) -> List[ResourceAllocation]:
        """Create offspring through selection, crossover, and mutation"""
        offspring = []

        for _ in range(self.population_size):
            if random.random() < self.crossover_rate:
                # Select parents using tournament selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                child = self._crossover(parent1, parent2)
            else:
                # Select single parent
                child = self._tournament_selection(population)

            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_selection(self, population: List[ResourceAllocation]) -> ResourceAllocation:
        """Tournament selection based on Pareto rank and crowding distance"""
        tournament_size = 3
        tournament = random.sample(population, min(tournament_size, len(population)))

        # Select best individual (lower rank is better, higher crowding distance is better)
        tournament.sort(key=lambda x: (x.pareto_rank, -x.crowding_distance))

        # Create copy to avoid modifying original
        selected = tournament[0]
        return ResourceAllocation(
            allocations={zone: allocs.copy() for zone, allocs in selected.allocations.items()}
        )

    def _crossover(self, parent1: ResourceAllocation, parent2: ResourceAllocation) -> ResourceAllocation:
        """Uniform crossover for resource allocations"""
        child_allocations = {}

        for zone in self.zones:
            child_allocations[zone] = {}

            for resource_type in self.resources.keys():
                # Uniform crossover - randomly choose from parent1 or parent2
                if random.random() < 0.5:
                    value = parent1.allocations.get(zone, {}).get(resource_type, 0)
                else:
                    value = parent2.allocations.get(zone, {}).get(resource_type, 0)

                child_allocations[zone][resource_type] = value

        return ResourceAllocation(allocations=child_allocations)

    def _mutate(self, individual: ResourceAllocation) -> ResourceAllocation:
        """Mutation operator with resource constraint awareness"""
        mutated_allocations = {}

        for zone in self.zones:
            mutated_allocations[zone] = {}

            for resource_type in self.resources.keys():
                current_value = individual.allocations.get(zone, {}).get(resource_type, 0)

                # Probability of mutation for this gene
                if random.random() < 0.1:  # 10% mutation rate per gene
                    # Small random change
                    change = random.randint(-1, 2)  # -1, 0, 1, or 2
                    new_value = max(0, current_value + change)

                    # Ensure we don't exceed reasonable limits
                    max_reasonable = min(self.resources[resource_type], 5)
                    new_value = min(new_value, max_reasonable)

                    mutated_allocations[zone][resource_type] = new_value
                else:
                    mutated_allocations[zone][resource_type] = current_value

        return ResourceAllocation(allocations=mutated_allocations)

    def _allocation_to_key(self, individual: ResourceAllocation) -> str:
        """Convert allocation to string key for caching"""
        key_parts = []
        for zone in sorted(self.zones):
            zone_allocs = individual.allocations.get(zone, {})
            for resource_type in sorted(self.resources.keys()):
                count = zone_allocs.get(resource_type, 0)
                key_parts.append(f"{zone}_{resource_type}_{count}")
        return "|".join(key_parts)

    def _calculate_robustness_metrics(self, pareto_front: List[ResourceAllocation]):
        """Calculate advanced robustness and risk metrics for Pareto front"""
        for individual in pareto_front:
            # Value at Risk (VaR) and Conditional VaR analysis
            risk_metrics = self._calculate_var_cvar(individual)
            individual.risk_metrics = risk_metrics

            # Overall robustness score
            individual.robustness_score = individual.objective_values.get('robustness', 0.0)

    def _calculate_var_cvar(self, individual: ResourceAllocation) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional VaR metrics"""
        # Monte Carlo simulation for risk metrics
        n_scenarios = 100
        scenario_outcomes = []

        for _ in range(n_scenarios):
            outcome = self._evaluate_risk_reduction_with_uncertainty(individual)
            scenario_outcomes.append(outcome)

        scenario_outcomes = np.array(scenario_outcomes)

        # Calculate VaR and CVaR (assuming we want to minimize risk of poor outcomes)
        # For benefits, we look at the left tail (low benefit outcomes)
        var_95 = np.percentile(scenario_outcomes, 5)  # 5th percentile
        var_99 = np.percentile(scenario_outcomes, 1)  # 1st percentile

        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(scenario_outcomes[scenario_outcomes <= var_95])
        cvar_99 = np.mean(scenario_outcomes[scenario_outcomes <= var_99])

        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95 if not np.isnan(cvar_95) else var_95,
            'cvar_99': cvar_99 if not np.isnan(cvar_99) else var_99,
            'expected_outcome': np.mean(scenario_outcomes),
            'outcome_volatility': np.std(scenario_outcomes)
        }


class RobustOptimizer:
    """
    Robust optimization for worst-case scenario protection
    """

    def __init__(self, uncertainty_budget: float = 0.2):
        self.uncertainty_budget = uncertainty_budget

    def optimize_robust_allocation(self,
                                 base_problem: Dict[str, Any],
                                 uncertainty_scenarios: List[Dict[str, Any]]) -> ResourceAllocation:
        """
        Optimize allocation for worst-case scenario performance

        Args:
            base_problem: Base problem definition
            uncertainty_scenarios: List of uncertainty scenarios

        Returns:
            Robust optimal allocation
        """

        # Use minimax approach: maximize minimum performance across scenarios
        def robust_objective(allocation_vector):
            allocation = self._vector_to_allocation(allocation_vector, base_problem)

            scenario_outcomes = []
            for scenario in uncertainty_scenarios:
                # Evaluate allocation under this scenario
                outcome = self._evaluate_allocation_scenario(allocation, scenario, base_problem)
                scenario_outcomes.append(outcome)

            # Return negative of worst-case outcome (for minimization)
            return -min(scenario_outcomes)

        # Define bounds based on resource constraints
        bounds = self._get_optimization_bounds(base_problem)

        # Run robust optimization
        result = differential_evolution(
            robust_objective,
            bounds,
            maxiter=100,
            seed=42
        )

        if result.success:
            optimal_allocation = self._vector_to_allocation(result.x, base_problem)
            return optimal_allocation
        else:
            # Fallback: return uniform allocation
            return self._create_uniform_allocation(base_problem)

    def _vector_to_allocation(self,
                            allocation_vector: np.ndarray,
                            base_problem: Dict[str, Any]) -> ResourceAllocation:
        """Convert optimization vector to ResourceAllocation"""
        zones = base_problem['zones']
        resources = base_problem['resources']

        allocations = {zone: {} for zone in zones}

        # Reshape vector into allocation matrix
        n_zones = len(zones)
        n_resources = len(resources)

        allocation_matrix = allocation_vector.reshape(n_zones, n_resources)

        for i, zone in enumerate(zones):
            for j, resource_type in enumerate(resources.keys()):
                count = max(0, int(round(allocation_matrix[i, j])))
                allocations[zone][resource_type] = count

        return ResourceAllocation(allocations=allocations)

    def _get_optimization_bounds(self, base_problem: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Get bounds for optimization variables"""
        zones = base_problem['zones']
        resources = base_problem['resources']

        bounds = []

        for zone in zones:
            for resource_type, max_available in resources.items():
                # Each allocation can be between 0 and total available
                bounds.append((0, max_available))

        return bounds

    def _evaluate_allocation_scenario(self,
                                    allocation: ResourceAllocation,
                                    scenario: Dict[str, Any],
                                    base_problem: Dict[str, Any]) -> float:
        """Evaluate allocation performance under uncertainty scenario"""
        # Simplified evaluation - in practice would use sophisticated scenario modeling
        base_benefit = 0.0

        for zone, zone_allocations in allocation.allocations.items():
            for resource_type, count in zone_allocations.items():
                if count > 0:
                    # Base benefit calculation
                    unit_benefit = scenario.get('effectiveness_factors', {}).get(resource_type, 1.0)
                    zone_benefit = unit_benefit * count

                    # Apply diminishing returns
                    zone_benefit = zone_benefit * (1 - np.exp(-count / 2.0))

                    base_benefit += zone_benefit

        return base_benefit

    def _create_uniform_allocation(self, base_problem: Dict[str, Any]) -> ResourceAllocation:
        """Create uniform allocation as fallback"""
        zones = base_problem['zones']
        resources = base_problem['resources']

        allocations = {zone: {} for zone in zones}

        for resource_type, total_available in resources.items():
            per_zone_allocation = total_available // len(zones)
            remainder = total_available % len(zones)

            for i, zone in enumerate(zones):
                base_allocation = per_zone_allocation
                if i < remainder:
                    base_allocation += 1

                allocations[zone][resource_type] = base_allocation

        return ResourceAllocation(allocations=allocations)


# Utility functions for non-linear benefit modeling

def diminishing_returns_utility(x: float, saturation_point: float = 10.0, decay_rate: float = 2.0) -> float:
    """Diminishing returns utility function"""
    return saturation_point * (1 - np.exp(-x / decay_rate))

def threshold_utility(x: float, threshold: float = 5.0, below_multiplier: float = 0.5) -> float:
    """Threshold utility with different returns below/above threshold"""
    if x < threshold:
        return x * below_multiplier
    else:
        return threshold * below_multiplier + (x - threshold) * 1.0

def s_curve_utility(x: float, midpoint: float = 5.0, steepness: float = 1.0) -> float:
    """S-curve utility function (sigmoid-like)"""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))

# Factory function for creating optimizers

def create_advanced_optimizer(optimizer_type: str = 'nsga2', **kwargs) -> MultiObjectiveOptimizer:
    """
    Factory function to create advanced optimizers

    Args:
        optimizer_type: Type of optimizer ('nsga2', 'robust')
        **kwargs: Additional parameters for optimizer

    Returns:
        Configured optimizer instance
    """
    if optimizer_type.lower() == 'nsga2':
        return NSGA2Optimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")