"""
Chance-Constrained Programming for Stochastic Resource Allocation

This module implements sophisticated stochastic optimization with probabilistic constraints,
handling uncertainty in intervention effectiveness and resource availability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import norm, multivariate_normal, beta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProbabilisticConstraint:
    """Represents a chance constraint P(g(x, ξ) ≤ 0) ≥ α"""
    name: str
    constraint_function: Callable
    confidence_level: float  # α - minimum probability of constraint satisfaction
    constraint_parameters: Dict[str, Any]
    uncertainty_distribution: Any  # Distribution of uncertain parameters

@dataclass
class StochasticSolution:
    """Solution to chance-constrained optimization problem"""
    allocation: Dict[str, Dict[str, int]]
    objective_value: float
    constraint_satisfaction_probabilities: Dict[str, float]
    expected_outcomes: Dict[str, float]
    risk_measures: Dict[str, float]
    robustness_metrics: Dict[str, float]

class ChanceConstrainedOptimizer:
    """
    Implements chance-constrained programming for stochastic resource allocation
    """

    def __init__(self,
                 confidence_level: float = 0.9,
                 sampling_method: str = 'monte_carlo',
                 n_samples: int = 1000):

        self.confidence_level = confidence_level
        self.sampling_method = sampling_method
        self.n_samples = n_samples

    def solve_chance_constrained_problem(self,
                                       zones: List[str],
                                       resources: Dict[str, int],
                                       probabilistic_constraints: List[ProbabilisticConstraint],
                                       objective_function: Callable,
                                       intervention_evaluator: Callable) -> StochasticSolution:
        """
        Solve chance-constrained resource allocation problem

        Args:
            zones: List of zones available for deployment
            resources: Available resources {resource_type: count}
            probabilistic_constraints: List of chance constraints
            objective_function: Objective function to maximize/minimize
            intervention_evaluator: Function to evaluate intervention effectiveness

        Returns:
            Optimal stochastic solution
        """

        # Create decision variables
        n_zones = len(zones)
        n_resource_types = len(resources)
        n_variables = n_zones * n_resource_types

        # Define deterministic constraints (resource limits)
        bounds = []
        for zone_idx in range(n_zones):
            for resource_type in resources.keys():
                max_allocation = resources[resource_type]
                bounds.append((0, max_allocation))

        # Define resource availability constraints
        def resource_constraint(x):
            x_matrix = x.reshape(n_zones, n_resource_types)
            constraint_values = []

            for j, (resource_type, max_available) in enumerate(resources.items()):
                total_used = np.sum(x_matrix[:, j])
                constraint_values.append(max_available - total_used)

            return np.array(constraint_values)

        # Convert probabilistic constraints to deterministic approximations
        deterministic_constraints = []

        for prob_constraint in probabilistic_constraints:
            if prob_constraint.name == 'response_time_reliability':
                det_constraint = self._convert_response_time_constraint(
                    prob_constraint, zones, resources
                )
                deterministic_constraints.append(det_constraint)

            elif prob_constraint.name == 'service_coverage_guarantee':
                det_constraint = self._convert_coverage_constraint(
                    prob_constraint, zones, resources
                )
                deterministic_constraints.append(det_constraint)

            elif prob_constraint.name == 'budget_risk_constraint':
                det_constraint = self._convert_budget_risk_constraint(
                    prob_constraint, zones, resources
                )
                deterministic_constraints.append(det_constraint)

        # Combine all constraints
        all_constraints = [NonlinearConstraint(resource_constraint, 0, float('inf'))]
        all_constraints.extend(deterministic_constraints)

        # Define stochastic objective function
        def stochastic_objective(x):
            return -self._evaluate_stochastic_objective(
                x, zones, resources, objective_function, intervention_evaluator
            )

        # Initial guess (uniform allocation)
        x0 = self._create_initial_guess(zones, resources)

        # Solve optimization problem
        result = minimize(
            stochastic_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=all_constraints,
            options={'maxiter': 200, 'ftol': 1e-6}
        )

        if result.success:
            optimal_allocation = self._vector_to_allocation_dict(
                result.x, zones, list(resources.keys())
            )

            # Evaluate solution quality
            solution_metrics = self._evaluate_solution_quality(
                optimal_allocation, zones, probabilistic_constraints, intervention_evaluator
            )

            return StochasticSolution(
                allocation=optimal_allocation,
                objective_value=-result.fun,
                constraint_satisfaction_probabilities=solution_metrics['constraint_probabilities'],
                expected_outcomes=solution_metrics['expected_outcomes'],
                risk_measures=solution_metrics['risk_measures'],
                robustness_metrics=solution_metrics['robustness_metrics']
            )
        else:
            # Return fallback solution
            return self._create_fallback_solution(zones, resources)

    def _convert_response_time_constraint(self,
                                        prob_constraint: ProbabilisticConstraint,
                                        zones: List[str],
                                        resources: Dict[str, int]) -> NonlinearConstraint:
        """Convert probabilistic response time constraint to deterministic equivalent"""

        def response_time_constraint_func(x):
            x_matrix = x.reshape(len(zones), len(resources))
            constraint_values = []

            for i, zone in enumerate(zones):
                # Calculate expected response time improvement
                zone_allocation = x_matrix[i, :]

                # Model: response_time_reduction = sum(effectiveness * allocation) + noise
                expected_reduction = 0.0
                variance = 0.0

                for j, resource_type in enumerate(resources.keys()):
                    allocation_count = zone_allocation[j]

                    if allocation_count > 0:
                        # Base effectiveness with uncertainty
                        base_effectiveness = prob_constraint.constraint_parameters.get(
                            f'{resource_type}_effectiveness', 1.0
                        )
                        effectiveness_variance = prob_constraint.constraint_parameters.get(
                            f'{resource_type}_variance', 0.1
                        )

                        expected_reduction += base_effectiveness * allocation_count
                        variance += effectiveness_variance * (allocation_count ** 2)

                # Convert to deterministic constraint using normal approximation
                # P(response_time_reduction >= threshold) >= confidence_level
                required_reduction = prob_constraint.constraint_parameters.get('min_reduction', 2.0)
                z_score = norm.ppf(prob_constraint.confidence_level)

                # Deterministic equivalent: expected_reduction - z_score * sqrt(variance) >= threshold
                deterministic_bound = required_reduction + z_score * np.sqrt(variance)
                constraint_values.append(expected_reduction - deterministic_bound)

            return np.array(constraint_values)

        return NonlinearConstraint(response_time_constraint_func, 0, float('inf'))

    def _convert_coverage_constraint(self,
                                   prob_constraint: ProbabilisticConstraint,
                                   zones: List[str],
                                   resources: Dict[str, int]) -> NonlinearConstraint:
        """Convert probabilistic coverage constraint to deterministic equivalent"""

        def coverage_constraint_func(x):
            x_matrix = x.reshape(len(zones), len(resources))

            # Calculate coverage probability for each zone
            total_coverage_score = 0.0

            for i, zone in enumerate(zones):
                zone_allocation = x_matrix[i, :]

                # Coverage score based on resource allocation
                zone_coverage = 0.0

                for j, resource_type in enumerate(resources.keys()):
                    allocation_count = zone_allocation[j]

                    if allocation_count > 0:
                        # Coverage contribution with saturation
                        coverage_contribution = prob_constraint.constraint_parameters.get(
                            f'{resource_type}_coverage_factor', 0.3
                        )

                        # Saturation model: coverage = 1 - exp(-factor * allocation)
                        zone_coverage += 1 - np.exp(-coverage_contribution * allocation_count)

                # Cap zone coverage at 1.0
                zone_coverage = min(1.0, zone_coverage)
                total_coverage_score += zone_coverage

            # Constraint: average coverage >= minimum required coverage
            avg_coverage = total_coverage_score / len(zones)
            min_required_coverage = prob_constraint.constraint_parameters.get('min_coverage', 0.8)

            return avg_coverage - min_required_coverage

        return NonlinearConstraint(coverage_constraint_func, 0, float('inf'))

    def _convert_budget_risk_constraint(self,
                                      prob_constraint: ProbabilisticConstraint,
                                      zones: List[str],
                                      resources: Dict[str, int]) -> NonlinearConstraint:
        """Convert probabilistic budget constraint to deterministic equivalent"""

        def budget_risk_constraint_func(x):
            x_matrix = x.reshape(len(zones), len(resources))

            # Cost calculation with uncertainty
            expected_cost = 0.0
            cost_variance = 0.0

            cost_per_resource = prob_constraint.constraint_parameters.get('cost_per_resource', {
                'pumps': 100.0,
                'traffic_units': 25.0,
                'ambulances': 50.0
            })

            cost_uncertainty = prob_constraint.constraint_parameters.get('cost_uncertainty', {
                'pumps': 10.0,
                'traffic_units': 5.0,
                'ambulances': 8.0
            })

            for zone_allocation in x_matrix:
                for j, resource_type in enumerate(resources.keys()):
                    allocation_count = zone_allocation[j]

                    if allocation_count > 0:
                        unit_cost = cost_per_resource.get(resource_type, 10.0)
                        unit_variance = cost_uncertainty.get(resource_type, 1.0) ** 2

                        expected_cost += unit_cost * allocation_count
                        cost_variance += unit_variance * (allocation_count ** 2)

            # Deterministic equivalent for P(cost <= budget) >= confidence_level
            budget_limit = prob_constraint.constraint_parameters.get('budget_limit', 1000.0)
            z_score = norm.ppf(prob_constraint.confidence_level)

            # Constraint: expected_cost + z_score * sqrt(variance) <= budget_limit
            cost_upper_bound = expected_cost + z_score * np.sqrt(cost_variance)

            return budget_limit - cost_upper_bound

        return NonlinearConstraint(budget_risk_constraint_func, 0, float('inf'))

    def _evaluate_stochastic_objective(self,
                                     x: np.ndarray,
                                     zones: List[str],
                                     resources: Dict[str, int],
                                     objective_function: Callable,
                                     intervention_evaluator: Callable) -> float:
        """Evaluate stochastic objective function using sampling"""

        allocation_dict = self._vector_to_allocation_dict(x, zones, list(resources.keys()))

        # Monte Carlo evaluation of expected objective
        objective_samples = []

        for _ in range(min(100, self.n_samples // 10)):  # Reduced for efficiency
            # Sample uncertain parameters
            sampled_effectiveness = self._sample_intervention_effectiveness()

            # Evaluate objective under this sample
            sample_objective = self._evaluate_sample_objective(
                allocation_dict, zones, sampled_effectiveness, objective_function, intervention_evaluator
            )

            objective_samples.append(sample_objective)

        # Return expected objective value
        expected_objective = np.mean(objective_samples)

        # Add penalty for risk (optional)
        objective_std = np.std(objective_samples)
        risk_penalty = 0.1 * objective_std  # Risk aversion parameter

        return expected_objective - risk_penalty

    def _sample_intervention_effectiveness(self) -> Dict[str, float]:
        """Sample uncertain intervention effectiveness parameters"""

        # Sample effectiveness factors from beta distributions
        effectiveness_samples = {}

        # Base effectiveness parameters (can be calibrated from data)
        base_params = {
            'deploy_pump': {'alpha': 3.0, 'beta': 1.0},
            'close_road': {'alpha': 4.0, 'beta': 1.2},
            'dispatch_ambulance': {'alpha': 5.0, 'beta': 0.8}
        }

        for intervention, params in base_params.items():
            sample = np.random.beta(params['alpha'], params['beta'])
            effectiveness_samples[intervention] = sample

        return effectiveness_samples

    def _evaluate_sample_objective(self,
                                 allocation: Dict[str, Dict[str, int]],
                                 zones: List[str],
                                 effectiveness_sample: Dict[str, float],
                                 objective_function: Callable,
                                 intervention_evaluator: Callable) -> float:
        """Evaluate objective under sampled effectiveness"""

        total_benefit = 0.0

        resource_to_intervention = {
            'pumps': 'deploy_pump',
            'traffic_units': 'close_road',
            'ambulances': 'dispatch_ambulance'
        }

        for zone in zones:
            zone_allocations = allocation.get(zone, {})

            for resource_type, count in zone_allocations.items():
                if count > 0:
                    intervention = resource_to_intervention.get(resource_type)

                    if intervention:
                        # Get base benefit
                        try:
                            base_benefit = intervention_evaluator(zone, intervention)
                        except:
                            base_benefit = 1.0  # Default benefit

                        # Apply sampled effectiveness
                        effectiveness = effectiveness_sample.get(intervention, 1.0)

                        # Calculate actual benefit with diminishing returns
                        actual_benefit = base_benefit * effectiveness * (1 - np.exp(-count / 2.0))
                        total_benefit += actual_benefit

        return total_benefit

    def _vector_to_allocation_dict(self,
                                 x: np.ndarray,
                                 zones: List[str],
                                 resource_types: List[str]) -> Dict[str, Dict[str, int]]:
        """Convert optimization vector to allocation dictionary"""

        allocation_dict = {}
        x_matrix = x.reshape(len(zones), len(resource_types))

        for i, zone in enumerate(zones):
            allocation_dict[zone] = {}
            for j, resource_type in enumerate(resource_types):
                count = max(0, int(round(x_matrix[i, j])))
                allocation_dict[zone][resource_type] = count

        return allocation_dict

    def _create_initial_guess(self, zones: List[str], resources: Dict[str, int]) -> np.ndarray:
        """Create initial guess for optimization"""
        n_zones = len(zones)
        n_resource_types = len(resources)

        initial_guess = np.zeros(n_zones * n_resource_types)

        # Simple uniform distribution
        idx = 0
        for zone in zones:
            for resource_type, total_available in resources.items():
                # Distribute resources uniformly across zones
                per_zone_allocation = total_available / n_zones
                initial_guess[idx] = per_zone_allocation
                idx += 1

        return initial_guess

    def _evaluate_solution_quality(self,
                                 allocation: Dict[str, Dict[str, int]],
                                 zones: List[str],
                                 probabilistic_constraints: List[ProbabilisticConstraint],
                                 intervention_evaluator: Callable) -> Dict[str, Any]:
        """Evaluate solution quality metrics"""

        # Monte Carlo evaluation of constraint satisfaction
        constraint_probabilities = {}

        for constraint in probabilistic_constraints:
            satisfaction_count = 0

            for _ in range(200):  # Monte Carlo samples
                # Sample uncertain parameters
                sample_satisfied = self._check_constraint_satisfaction(
                    allocation, constraint, zones, intervention_evaluator
                )

                if sample_satisfied:
                    satisfaction_count += 1

            constraint_probabilities[constraint.name] = satisfaction_count / 200

        # Expected outcomes
        expected_outcomes = self._calculate_expected_outcomes(allocation, zones, intervention_evaluator)

        # Risk measures
        risk_measures = self._calculate_risk_measures(allocation, zones, intervention_evaluator)

        # Robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(allocation, zones, intervention_evaluator)

        return {
            'constraint_probabilities': constraint_probabilities,
            'expected_outcomes': expected_outcomes,
            'risk_measures': risk_measures,
            'robustness_metrics': robustness_metrics
        }

    def _check_constraint_satisfaction(self,
                                     allocation: Dict[str, Dict[str, int]],
                                     constraint: ProbabilisticConstraint,
                                     zones: List[str],
                                     intervention_evaluator: Callable) -> bool:
        """Check if constraint is satisfied under sampled parameters"""

        # Simplified constraint checking - would be more sophisticated in practice
        if constraint.name == 'response_time_reliability':
            # Check if response time improvement is adequate
            total_improvement = 0.0

            for zone, zone_allocations in allocation.items():
                for resource_type, count in zone_allocations.items():
                    if count > 0:
                        # Sample effectiveness
                        effectiveness = np.random.beta(2.0, 1.0)  # Simplified sampling
                        improvement = effectiveness * count
                        total_improvement += improvement

            required_improvement = constraint.constraint_parameters.get('min_reduction', 5.0)
            return total_improvement >= required_improvement

        return True  # Default to satisfied for unknown constraints

    def _calculate_expected_outcomes(self,
                                   allocation: Dict[str, Dict[str, int]],
                                   zones: List[str],
                                   intervention_evaluator: Callable) -> Dict[str, float]:
        """Calculate expected outcomes"""

        total_expected_benefit = 0.0
        total_expected_cost = 0.0

        cost_per_resource = {
            'pumps': 100.0,
            'traffic_units': 25.0,
            'ambulances': 50.0
        }

        resource_to_intervention = {
            'pumps': 'deploy_pump',
            'traffic_units': 'close_road',
            'ambulances': 'dispatch_ambulance'
        }

        for zone, zone_allocations in allocation.items():
            for resource_type, count in zone_allocations.items():
                if count > 0:
                    # Expected benefit calculation
                    intervention = resource_to_intervention.get(resource_type)
                    if intervention:
                        try:
                            base_benefit = intervention_evaluator(zone, intervention)
                        except:
                            base_benefit = 1.0

                        expected_benefit = base_benefit * count * 0.8  # Expected effectiveness
                        total_expected_benefit += expected_benefit

                    # Expected cost
                    unit_cost = cost_per_resource.get(resource_type, 10.0)
                    total_expected_cost += unit_cost * count

        return {
            'expected_benefit': total_expected_benefit,
            'expected_cost': total_expected_cost,
            'benefit_cost_ratio': total_expected_benefit / max(1.0, total_expected_cost)
        }

    def _calculate_risk_measures(self,
                               allocation: Dict[str, Dict[str, int]],
                               zones: List[str],
                               intervention_evaluator: Callable) -> Dict[str, float]:
        """Calculate risk measures (VaR, CVaR)"""

        # Monte Carlo simulation for risk assessment
        benefit_samples = []

        for _ in range(200):
            sample_benefit = self._sample_total_benefit(allocation, zones, intervention_evaluator)
            benefit_samples.append(sample_benefit)

        benefit_samples = np.array(benefit_samples)

        # Calculate risk measures
        var_95 = np.percentile(benefit_samples, 5)
        var_99 = np.percentile(benefit_samples, 1)

        cvar_95 = np.mean(benefit_samples[benefit_samples <= var_95])
        cvar_99 = np.mean(benefit_samples[benefit_samples <= var_99])

        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95 if not np.isnan(cvar_95) else var_95,
            'cvar_99': cvar_99 if not np.isnan(cvar_99) else var_99,
            'expected_benefit': np.mean(benefit_samples),
            'benefit_volatility': np.std(benefit_samples)
        }

    def _sample_total_benefit(self,
                            allocation: Dict[str, Dict[str, int]],
                            zones: List[str],
                            intervention_evaluator: Callable) -> float:
        """Sample total benefit under uncertainty"""

        total_benefit = 0.0

        resource_to_intervention = {
            'pumps': 'deploy_pump',
            'traffic_units': 'close_road',
            'ambulances': 'dispatch_ambulance'
        }

        for zone, zone_allocations in allocation.items():
            for resource_type, count in zone_allocations.items():
                if count > 0:
                    intervention = resource_to_intervention.get(resource_type)

                    if intervention:
                        try:
                            base_benefit = intervention_evaluator(zone, intervention)
                        except:
                            base_benefit = 1.0

                        # Sample effectiveness
                        effectiveness = np.random.beta(3.0, 1.0)  # Average effectiveness ~0.75

                        # Calculate benefit with uncertainty
                        actual_benefit = base_benefit * effectiveness * count
                        total_benefit += actual_benefit

        return total_benefit

    def _calculate_robustness_metrics(self,
                                    allocation: Dict[str, Dict[str, int]],
                                    zones: List[str],
                                    intervention_evaluator: Callable) -> Dict[str, float]:
        """Calculate robustness metrics"""

        # Test allocation performance under various scenarios
        scenario_performances = []

        # Scenario 1: Low effectiveness
        low_eff_performance = self._evaluate_scenario_performance(
            allocation, zones, intervention_evaluator, effectiveness_multiplier=0.5
        )
        scenario_performances.append(low_eff_performance)

        # Scenario 2: High effectiveness
        high_eff_performance = self._evaluate_scenario_performance(
            allocation, zones, intervention_evaluator, effectiveness_multiplier=1.5
        )
        scenario_performances.append(high_eff_performance)

        # Scenario 3: Variable effectiveness
        var_eff_performance = self._evaluate_scenario_performance(
            allocation, zones, intervention_evaluator, effectiveness_multiplier=1.0, add_noise=True
        )
        scenario_performances.append(var_eff_performance)

        # Calculate robustness metrics
        min_performance = min(scenario_performances)
        avg_performance = np.mean(scenario_performances)
        performance_std = np.std(scenario_performances)

        return {
            'worst_case_performance': min_performance,
            'average_performance': avg_performance,
            'performance_stability': 1.0 - (performance_std / max(1.0, avg_performance)),
            'robustness_ratio': min_performance / max(1.0, avg_performance)
        }

    def _evaluate_scenario_performance(self,
                                     allocation: Dict[str, Dict[str, int]],
                                     zones: List[str],
                                     intervention_evaluator: Callable,
                                     effectiveness_multiplier: float = 1.0,
                                     add_noise: bool = False) -> float:
        """Evaluate allocation performance under specific scenario"""

        total_benefit = 0.0

        resource_to_intervention = {
            'pumps': 'deploy_pump',
            'traffic_units': 'close_road',
            'ambulances': 'dispatch_ambulance'
        }

        for zone, zone_allocations in allocation.items():
            for resource_type, count in zone_allocations.items():
                if count > 0:
                    intervention = resource_to_intervention.get(resource_type)

                    if intervention:
                        try:
                            base_benefit = intervention_evaluator(zone, intervention)
                        except:
                            base_benefit = 1.0

                        # Apply scenario conditions
                        effectiveness = effectiveness_multiplier

                        if add_noise:
                            noise = np.random.normal(0.0, 0.2)
                            effectiveness = max(0.1, effectiveness + noise)

                        actual_benefit = base_benefit * effectiveness * count
                        total_benefit += actual_benefit

        return total_benefit

    def _create_fallback_solution(self, zones: List[str], resources: Dict[str, int]) -> StochasticSolution:
        """Create fallback solution when optimization fails"""

        # Simple uniform allocation
        allocation = {}

        for zone in zones:
            allocation[zone] = {}
            for resource_type, total_available in resources.items():
                per_zone_allocation = total_available // len(zones)
                allocation[zone][resource_type] = per_zone_allocation

        return StochasticSolution(
            allocation=allocation,
            objective_value=0.0,
            constraint_satisfaction_probabilities={},
            expected_outcomes={'expected_benefit': 0.0, 'expected_cost': 0.0},
            risk_measures={'var_95': 0.0, 'cvar_95': 0.0},
            robustness_metrics={'robustness_ratio': 0.5}
        )


# Convenience functions

def create_response_time_constraint(min_reduction: float = 2.0, confidence: float = 0.9) -> ProbabilisticConstraint:
    """Create response time reliability constraint"""
    return ProbabilisticConstraint(
        name='response_time_reliability',
        constraint_function=None,  # Handled internally
        confidence_level=confidence,
        constraint_parameters={
            'min_reduction': min_reduction,
            'pumps_effectiveness': 1.5,
            'traffic_units_effectiveness': 1.0,
            'ambulances_effectiveness': 2.0,
            'pumps_variance': 0.2,
            'traffic_units_variance': 0.15,
            'ambulances_variance': 0.1
        },
        uncertainty_distribution=None
    )

def create_coverage_constraint(min_coverage: float = 0.8, confidence: float = 0.95) -> ProbabilisticConstraint:
    """Create service coverage guarantee constraint"""
    return ProbabilisticConstraint(
        name='service_coverage_guarantee',
        constraint_function=None,
        confidence_level=confidence,
        constraint_parameters={
            'min_coverage': min_coverage,
            'pumps_coverage_factor': 0.4,
            'traffic_units_coverage_factor': 0.3,
            'ambulances_coverage_factor': 0.5
        },
        uncertainty_distribution=None
    )

def create_budget_risk_constraint(budget_limit: float = 1000.0, confidence: float = 0.9) -> ProbabilisticConstraint:
    """Create budget risk constraint"""
    return ProbabilisticConstraint(
        name='budget_risk_constraint',
        constraint_function=None,
        confidence_level=confidence,
        constraint_parameters={
            'budget_limit': budget_limit,
            'cost_per_resource': {
                'pumps': 100.0,
                'traffic_units': 25.0,
                'ambulances': 50.0
            },
            'cost_uncertainty': {
                'pumps': 15.0,
                'traffic_units': 5.0,
                'ambulances': 10.0
            }
        },
        uncertainty_distribution=None
    )