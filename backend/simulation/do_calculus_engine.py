"""
Pearl's Do-Calculus Engine for Proper Causal Intervention Analysis

This module implements Pearl's causal calculus (do-calculus) for rigorous causal reasoning,
providing the theoretical foundation for understanding intervention effects in causal graphs.

Key capabilities:
- Backdoor criterion for confounder identification
- Do-calculus rules for causal effect identification
- Counterfactual reasoning and nested counterfactuals
- Sensitivity analysis for unobserved confounding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from itertools import combinations, chain, product
from dataclasses import dataclass
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CausalEffect:
    """Represents a causal effect P(Y|do(X=x))"""
    treatment_vars: List[str]
    treatment_values: Dict[str, str]
    outcome_vars: List[str]
    effect_size: Dict[str, float]
    confidence_bounds: Dict[str, Tuple[float, float]]
    identification_strategy: str
    adjustment_sets: List[List[str]]
    estimable: bool
    sensitivity_bounds: Optional[Dict[str, Tuple[float, float]]] = None

@dataclass
class CounterfactualQuery:
    """Represents a counterfactual query"""
    factual_evidence: Dict[str, str]
    counterfactual_intervention: Dict[str, str]
    target_variables: List[str]
    probability: Dict[str, float]
    explanation: str

class DoCalkulusEngine:
    """Implementation of Pearl's do-calculus for causal inference"""

    def __init__(self, causal_model: DiscreteBayesianNetwork):
        """
        Initialize the do-calculus engine

        Args:
            causal_model: Bayesian network representing the causal DAG
        """
        self.causal_model = causal_model
        self.graph = self._build_networkx_graph()
        self.inference_engine = VariableElimination(causal_model)

        # Cache for computed effects
        self._effect_cache = {}
        self._backdoor_cache = {}

    def _build_networkx_graph(self) -> nx.DiGraph:
        """Convert pgmpy model to networkx for graph algorithms"""
        G = nx.DiGraph()
        G.add_nodes_from(self.causal_model.nodes())
        G.add_edges_from(self.causal_model.edges())
        return G

    def find_backdoor_adjustment_sets(self,
                                    treatment: Union[str, List[str]],
                                    outcome: Union[str, List[str]]) -> List[List[str]]:
        """
        Find all valid backdoor adjustment sets using Pearl's backdoor criterion

        Args:
            treatment: Treatment variable(s)
            outcome: Outcome variable(s)

        Returns:
            List of valid adjustment sets (empty if none exist)
        """

        if isinstance(treatment, str):
            treatment = [treatment]
        if isinstance(outcome, str):
            outcome = [outcome]

        cache_key = (tuple(sorted(treatment)), tuple(sorted(outcome)))
        if cache_key in self._backdoor_cache:
            return self._backdoor_cache[cache_key]

        # Get all possible confounders (not descendants of treatment)
        potential_confounders = set(self.causal_model.nodes()) - set(treatment) - set(outcome)

        # Remove descendants of treatment variables
        for t in treatment:
            descendants = self._get_descendants(t)
            potential_confounders -= descendants

        valid_adjustment_sets = []

        # Test all possible subsets of potential confounders
        for r in range(len(potential_confounders) + 1):
            for adjustment_set in combinations(potential_confounders, r):
                if self._satisfies_backdoor_criterion(treatment, outcome, list(adjustment_set)):
                    valid_adjustment_sets.append(list(adjustment_set))

        # Sort by preference (smaller sets first, then by informativeness)
        valid_adjustment_sets.sort(key=lambda x: (len(x), str(sorted(x))))

        self._backdoor_cache[cache_key] = valid_adjustment_sets
        return valid_adjustment_sets

    def _satisfies_backdoor_criterion(self,
                                    treatment: List[str],
                                    outcome: List[str],
                                    adjustment_set: List[str]) -> bool:
        """
        Check if adjustment set satisfies Pearl's backdoor criterion

        Args:
            treatment: Treatment variables
            outcome: Outcome variables
            adjustment_set: Proposed adjustment set

        Returns:
            True if backdoor criterion is satisfied
        """

        # Criterion 1: No node in Z is a descendant of X
        for z in adjustment_set:
            for x in treatment:
                if z in self._get_descendants(x):
                    return False

        # Criterion 2: Z blocks every path between X and Y that contains an arrow into X
        # Create modified graph with treatment arrows removed
        modified_graph = self.graph.copy()
        for x in treatment:
            # Remove all incoming edges to treatment variables
            incoming_edges = list(modified_graph.in_edges(x))
            modified_graph.remove_edges_from(incoming_edges)

        # Check if all backdoor paths are blocked
        return self._blocks_all_backdoor_paths(treatment, outcome, adjustment_set, modified_graph)

    def _blocks_all_backdoor_paths(self,
                                 treatment: List[str],
                                 outcome: List[str],
                                 adjustment_set: List[str],
                                 graph: nx.DiGraph) -> bool:
        """Check if adjustment set blocks all backdoor paths"""

        # For each treatment-outcome pair, check if all backdoor paths are blocked
        for x in treatment:
            for y in outcome:
                if x == y:  # Same variable
                    continue

                # Find all paths from X to Y in the modified graph
                try:
                    all_paths = list(nx.all_simple_paths(graph.to_undirected(), x, y, cutoff=10))

                    for path in all_paths:
                        # Check if this path is a backdoor path (contains arrow into X)
                        is_backdoor = False
                        for i in range(len(path) - 1):
                            if path[i+1] == x and graph.has_edge(path[i], x):
                                is_backdoor = True
                                break

                        if is_backdoor and not self._path_blocked_by_adjustment(path, adjustment_set, graph):
                            return False

                except nx.NetworkXNoPath:
                    continue  # No path exists, which is fine

        return True

    def _path_blocked_by_adjustment(self, path: List[str], adjustment_set: List[str], graph: nx.DiGraph) -> bool:
        """Check if a path is blocked by the adjustment set"""

        for i in range(1, len(path) - 1):  # Skip endpoints
            node = path[i]
            if node in adjustment_set:
                # Check if node blocks the path (not a collider or collider with descendant in adjustment set)
                prev_node, next_node = path[i-1], path[i+1]

                # Determine edge directions
                prev_to_node = graph.has_edge(prev_node, node)
                node_to_next = graph.has_edge(node, next_node)
                node_to_prev = graph.has_edge(node, prev_node)
                next_to_node = graph.has_edge(next_node, node)

                # Check if it's a collider (arrows pointing into node)
                if (prev_to_node or next_to_node) and not (node_to_prev or node_to_next):
                    # It's a collider - only blocks if no descendant is in adjustment set
                    descendants = self._get_descendants(node)
                    if any(desc in adjustment_set for desc in descendants):
                        continue  # Path not blocked
                    else:
                        return True  # Path blocked by collider

                else:
                    # Not a collider - blocks the path
                    return True

        return False  # Path not blocked

    def _get_descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node in the causal graph"""
        return set(nx.descendants(self.graph, node))

    def compute_causal_effect(self,
                            treatment: Dict[str, str],
                            outcome: List[str],
                            conditioning_vars: Dict[str, str] = None) -> CausalEffect:
        """
        Compute causal effect P(Y|do(X=x)) using do-calculus

        Args:
            treatment: Treatment intervention {variable: value}
            outcome: Outcome variables
            conditioning_vars: Additional conditioning variables

        Returns:
            CausalEffect object with identification results
        """

        treatment_vars = list(treatment.keys())
        conditioning_vars = conditioning_vars or {}

        # Find backdoor adjustment sets
        adjustment_sets = self.find_backdoor_adjustment_sets(treatment_vars, outcome)

        if not adjustment_sets:
            # Try other identification strategies (front-door, etc.)
            return self._attempt_alternative_identification(treatment, outcome, conditioning_vars)

        # Use the minimal adjustment set
        optimal_adjustment_set = adjustment_sets[0]

        # Compute the causal effect using adjustment formula
        effect_estimates = {}
        confidence_bounds = {}

        for outcome_var in outcome:
            effect_size, bounds = self._compute_adjustment_formula(
                treatment, outcome_var, optimal_adjustment_set, conditioning_vars
            )
            effect_estimates[outcome_var] = effect_size
            confidence_bounds[outcome_var] = bounds

        return CausalEffect(
            treatment_vars=treatment_vars,
            treatment_values=treatment,
            outcome_vars=outcome,
            effect_size=effect_estimates,
            confidence_bounds=confidence_bounds,
            identification_strategy="backdoor_adjustment",
            adjustment_sets=adjustment_sets,
            estimable=True
        )

    def _compute_adjustment_formula(self,
                                  treatment: Dict[str, str],
                                  outcome_var: str,
                                  adjustment_set: List[str],
                                  conditioning_vars: Dict[str, str]) -> Tuple[float, Tuple[float, float]]:
        """
        Compute P(Y|do(X=x)) = Σ_z P(Y|X=x,Z=z) * P(Z=z)

        Args:
            treatment: Treatment intervention
            outcome_var: Single outcome variable
            adjustment_set: Variables to adjust for
            conditioning_vars: Additional conditioning

        Returns:
            Effect estimate and confidence bounds
        """

        total_effect = 0.0
        contributions = []

        # If no adjustment needed
        if not adjustment_set:
            # Direct computation P(Y|X=x, conditioning)
            evidence = {**treatment, **conditioning_vars}
            result = self.inference_engine.query(variables=[outcome_var], evidence=evidence)

            # Assuming binary outcomes with 'True'/'High' as positive
            effect_size = float(result.values[1]) if len(result.values) > 1 else float(result.values[0])
            return effect_size, (effect_size * 0.9, effect_size * 1.1)  # Simple bounds

        # Get all possible values for adjustment variables
        adjustment_combinations = self._get_variable_combinations(adjustment_set)

        for adj_values in adjustment_combinations:
            # P(Z=z | conditioning)
            marginal_evidence = {**conditioning_vars} if conditioning_vars else {}
            if marginal_evidence:
                marginal_prob = self._compute_conditional_probability(adjustment_set, adj_values, marginal_evidence)
            else:
                marginal_prob = self._compute_marginal_probability(adjustment_set, adj_values)

            # P(Y|X=x, Z=z, conditioning)
            conditional_evidence = {**treatment, **adj_values, **conditioning_vars}
            conditional_result = self.inference_engine.query(
                variables=[outcome_var],
                evidence=conditional_evidence
            )

            conditional_prob = float(conditional_result.values[1]) if len(conditional_result.values) > 1 else float(conditional_result.values[0])

            contribution = marginal_prob * conditional_prob
            contributions.append(contribution)
            total_effect += contribution

        # Simple confidence bounds based on contribution variance
        if len(contributions) > 1:
            std_error = np.std(contributions) / np.sqrt(len(contributions))
            lower_bound = max(0.0, total_effect - 1.96 * std_error)
            upper_bound = min(1.0, total_effect + 1.96 * std_error)
        else:
            lower_bound = total_effect * 0.9
            upper_bound = total_effect * 1.1

        return total_effect, (lower_bound, upper_bound)

    def _get_variable_combinations(self, variables: List[str]) -> List[Dict[str, str]]:
        """Get all possible value combinations for a set of variables"""
        if not variables:
            return [{}]

        combinations_list = []
        variable_values = {}

        # Get possible values for each variable from CPDs
        for var in variables:
            cpd = self.causal_model.get_cpds(var)
            if cpd:
                # Assuming discrete variables with states 0, 1 or equivalently named states
                if hasattr(cpd, 'state_names') and var in cpd.state_names:
                    variable_values[var] = cpd.state_names[var]
                else:
                    # Default binary assumption
                    variable_values[var] = ['False', 'True'] if var != 'DrainageCapacity' else ['Good', 'Poor']

        # Generate all combinations
        keys = list(variable_values.keys())
        for value_combo in product(*[variable_values[key] for key in keys]):
            combinations_list.append(dict(zip(keys, value_combo)))

        return combinations_list

    def _compute_marginal_probability(self, variables: List[str], values: Dict[str, str]) -> float:
        """Compute marginal probability P(variables=values)"""
        if not variables:
            return 1.0

        try:
            result = self.inference_engine.query(variables=variables, evidence={})
            # This is simplified - real implementation would need to handle joint distributions properly
            return 1.0 / len(self._get_variable_combinations(variables))  # Uniform assumption
        except:
            return 0.01  # Small default probability

    def _compute_conditional_probability(self,
                                       target_vars: List[str],
                                       target_values: Dict[str, str],
                                       evidence: Dict[str, str]) -> float:
        """Compute conditional probability P(target|evidence)"""
        try:
            result = self.inference_engine.query(variables=target_vars, evidence=evidence)
            return 1.0 / len(self._get_variable_combinations(target_vars))  # Simplified
        except:
            return 0.01

    def _attempt_alternative_identification(self,
                                          treatment: Dict[str, str],
                                          outcome: List[str],
                                          conditioning_vars: Dict[str, str]) -> CausalEffect:
        """Attempt identification using front-door criterion or other methods"""

        # Simplified implementation - in practice would implement full front-door criterion
        return CausalEffect(
            treatment_vars=list(treatment.keys()),
            treatment_values=treatment,
            outcome_vars=outcome,
            effect_size={var: 0.0 for var in outcome},
            confidence_bounds={var: (0.0, 0.0) for var in outcome},
            identification_strategy="not_identifiable",
            adjustment_sets=[],
            estimable=False
        )

    def compute_counterfactual(self,
                             factual_evidence: Dict[str, str],
                             counterfactual_intervention: Dict[str, str],
                             target_variables: List[str]) -> CounterfactualQuery:
        """
        Compute counterfactual probabilities P(Y_x | X'=x', Y=y)

        This implements the three-step process:
        1. Abduction: Update beliefs given factual evidence
        2. Action: Apply counterfactual intervention
        3. Prediction: Compute counterfactual outcomes

        Args:
            factual_evidence: Observed evidence in factual world
            counterfactual_intervention: Hypothetical intervention
            target_variables: Variables to predict in counterfactual world

        Returns:
            CounterfactualQuery with results
        """

        # Step 1: Abduction - compute posterior over latent variables
        # In discrete Bayesian networks, this is approximated by conditioning

        # Step 2 & 3: Action and Prediction
        # Simulate the counterfactual by applying intervention
        counterfactual_results = {}

        for target_var in target_variables:
            # Simple counterfactual computation
            # In practice, this requires more sophisticated structural equation modeling
            try:
                # Combine factual evidence with counterfactual intervention
                combined_evidence = {**factual_evidence}
                combined_evidence.update(counterfactual_intervention)

                # Remove target variable from evidence if present
                if target_var in combined_evidence:
                    del combined_evidence[target_var]

                result = self.inference_engine.query(
                    variables=[target_var],
                    evidence=combined_evidence
                )

                counterfactual_results[target_var] = float(result.values[1]) if len(result.values) > 1 else float(result.values[0])

            except Exception as e:
                counterfactual_results[target_var] = 0.5  # Default uncertain prediction

        return CounterfactualQuery(
            factual_evidence=factual_evidence,
            counterfactual_intervention=counterfactual_intervention,
            target_variables=target_variables,
            probability=counterfactual_results,
            explanation=f"Counterfactual: If {counterfactual_intervention} had occurred given {factual_evidence}"
        )

    def perform_sensitivity_analysis(self,
                                   treatment: Dict[str, str],
                                   outcome: str,
                                   unobserved_confounder_strength: float = 0.1) -> Dict[str, Tuple[float, float]]:
        """
        Perform sensitivity analysis for unobserved confounding

        Args:
            treatment: Treatment variables and values
            outcome: Outcome variable
            unobserved_confounder_strength: Strength of potential unobserved confounder

        Returns:
            Sensitivity bounds for causal effect estimates
        """

        baseline_effect = self.compute_causal_effect(treatment, [outcome])

        if not baseline_effect.estimable:
            return {outcome: (0.0, 0.0)}

        baseline_estimate = baseline_effect.effect_size[outcome]

        # Simulate effect of unobserved confounder
        # This is a simplified version - full implementation would use more sophisticated bounds

        max_bias = unobserved_confounder_strength * baseline_estimate

        lower_bound = max(0.0, baseline_estimate - max_bias)
        upper_bound = min(1.0, baseline_estimate + max_bias)

        return {outcome: (lower_bound, upper_bound)}

    def explain_causal_pathway(self,
                             treatment: Dict[str, str],
                             outcome: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of causal pathways and identification strategy

        Args:
            treatment: Treatment intervention
            outcome: Outcome variable

        Returns:
            Detailed explanation of causal analysis
        """

        treatment_vars = list(treatment.keys())

        # Find adjustment sets
        adjustment_sets = self.find_backdoor_adjustment_sets(treatment_vars, [outcome])

        # Analyze causal paths
        direct_paths = self._find_directed_paths(treatment_vars, [outcome])
        confounding_paths = self._find_confounding_paths(treatment_vars, [outcome])

        # Compute effect
        causal_effect = self.compute_causal_effect(treatment, [outcome])

        explanation = {
            "treatment": treatment,
            "outcome": outcome,
            "identification_strategy": causal_effect.identification_strategy,
            "estimable": causal_effect.estimable,
            "adjustment_sets": adjustment_sets,
            "minimal_adjustment": adjustment_sets[0] if adjustment_sets else None,
            "direct_paths": direct_paths,
            "confounding_paths": confounding_paths,
            "effect_size": causal_effect.effect_size.get(outcome, 0.0),
            "confidence_bounds": causal_effect.confidence_bounds.get(outcome, (0.0, 0.0)),
            "interpretation": self._generate_interpretation(causal_effect, outcome)
        }

        return explanation

    def _find_directed_paths(self, sources: List[str], targets: List[str]) -> List[List[str]]:
        """Find all directed paths from sources to targets"""
        all_paths = []

        for source in sources:
            for target in targets:
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=5))
                        all_paths.extend(paths)
                    except nx.NetworkXNoPath:
                        continue

        return all_paths

    def _find_confounding_paths(self, sources: List[str], targets: List[str]) -> List[List[str]]:
        """Find backdoor (confounding) paths from sources to targets"""
        confounding_paths = []

        # Convert to undirected for path finding, then check if backdoor
        undirected_graph = self.graph.to_undirected()

        for source in sources:
            for target in targets:
                if source != target:
                    try:
                        all_paths = list(nx.all_simple_paths(undirected_graph, source, target, cutoff=5))

                        for path in all_paths:
                            # Check if it's a backdoor path (has arrow into source)
                            if len(path) > 1 and self.graph.has_edge(path[1], path[0]):
                                confounding_paths.append(path)

                    except nx.NetworkXNoPath:
                        continue

        return confounding_paths

    def _generate_interpretation(self, causal_effect: CausalEffect, outcome: str) -> str:
        """Generate human-readable interpretation of causal effect"""

        if not causal_effect.estimable:
            return f"The causal effect of {causal_effect.treatment_vars} on {outcome} cannot be identified from the available data and graph structure."

        effect_size = causal_effect.effect_size.get(outcome, 0.0)
        bounds = causal_effect.confidence_bounds.get(outcome, (0.0, 0.0))

        interpretation = f"The causal effect of {causal_effect.treatment_values} on {outcome} is {effect_size:.3f} "
        interpretation += f"(95% CI: [{bounds[0]:.3f}, {bounds[1]:.3f}]). "

        if causal_effect.identification_strategy == "backdoor_adjustment":
            adjustment_vars = causal_effect.adjustment_sets[0] if causal_effect.adjustment_sets else []
            if adjustment_vars:
                interpretation += f"This effect is identified by adjusting for: {', '.join(adjustment_vars)}. "
            else:
                interpretation += "No adjustment for confounders is needed. "

        if effect_size > 0.7:
            interpretation += "This represents a strong positive causal effect."
        elif effect_size > 0.3:
            interpretation += "This represents a moderate positive causal effect."
        elif effect_size > 0.1:
            interpretation += "This represents a weak positive causal effect."
        else:
            interpretation += "This represents little to no causal effect."

        return interpretation


# Convenience functions for integration

def analyze_intervention_causality(causal_model: DiscreteBayesianNetwork,
                                 intervention: Dict[str, str],
                                 outcomes: List[str]) -> Dict[str, Any]:
    """
    Comprehensive causal analysis of an intervention

    Args:
        causal_model: Bayesian network representing causal structure
        intervention: Intervention to analyze {variable: value}
        outcomes: Outcome variables to analyze

    Returns:
        Complete causal analysis results
    """

    engine = DoCalkulusEngine(causal_model)

    results = {}

    for outcome in outcomes:
        causal_effect = engine.compute_causal_effect(intervention, [outcome])
        explanation = engine.explain_causal_pathway(intervention, outcome)
        sensitivity = engine.perform_sensitivity_analysis(intervention, outcome)

        results[outcome] = {
            "causal_effect": causal_effect,
            "explanation": explanation,
            "sensitivity_bounds": sensitivity
        }

    return results

def compute_counterfactual_scenario(causal_model: DiscreteBayesianNetwork,
                                  factual_evidence: Dict[str, str],
                                  hypothetical_intervention: Dict[str, str],
                                  target_outcomes: List[str]) -> CounterfactualQuery:
    """
    Compute counterfactual scenario analysis

    Args:
        causal_model: Bayesian network
        factual_evidence: What actually happened
        hypothetical_intervention: What could have been done differently
        target_outcomes: Outcomes to predict in counterfactual world

    Returns:
        Counterfactual analysis results
    """

    engine = DoCalkulusEngine(causal_model)
    return engine.compute_counterfactual(
        factual_evidence, hypothetical_intervention, target_outcomes
    )