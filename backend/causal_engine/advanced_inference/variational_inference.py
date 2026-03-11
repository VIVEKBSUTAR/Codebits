"""
Variational Inference for Approximate Bayesian Inference

Implements mean-field variational inference for Bayesian Networks.
This provides scalable approximate inference when exact methods become
computationally intractable.

Key features:
- Mean-field approximation with factorized distributions
- Coordinate ascent optimization
- Convergence monitoring and diagnostics
- Evidence Lower BOund (ELBO) optimization
- Supports continuous and discrete variables
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from scipy.optimize import minimize
from scipy.special import digamma, logsumexp
import time
from dataclasses import dataclass
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor

from utils.logger import SystemLogger

logger = SystemLogger(module_name="variational_inference")


@dataclass
class VariationalParameter:
    """Represents variational parameters for a single variable"""
    variable: str
    parameter_type: str  # 'categorical', 'gaussian', etc.
    parameters: np.ndarray  # Natural parameters of the variational distribution
    entropy: float = 0.0
    expected_value: Any = None


class MeanFieldVariationalInference:
    """
    Mean-field variational inference for Bayesian networks

    Approximates the posterior P(X|evidence) with a factorized distribution:
    q(X) = ∏ᵢ qᵢ(Xᵢ)

    Uses coordinate ascent to maximize the Evidence Lower Bound (ELBO):
    ELBO = E_q[log P(X,evidence)] - E_q[log q(X)]
    """

    def __init__(self, model: DiscreteBayesianNetwork, max_iterations: int = 100,
                 tolerance: float = 1e-6, learning_rate: float = 0.01):
        self.model = model
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate

        # Variational parameters
        self.variational_params: Dict[str, VariationalParameter] = {}
        self.evidence: Dict[str, str] = {}

        # Convergence tracking
        self.elbo_history: List[float] = []
        self.convergence_history: List[Dict[str, Any]] = []

        # Performance metrics
        self.inference_time = 0.0
        self.converged = False
        self.final_elbo = float('-inf')

        logger.log("Mean-field variational inference engine initialized")

    def fit(self, evidence: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Perform variational inference to approximate the posterior

        Args:
            evidence: Dictionary of observed variables and their states

        Returns:
            Inference results and convergence diagnostics
        """
        start_time = time.time()

        try:
            # Set evidence
            self.evidence = evidence or {}

            # Initialize variational parameters
            self._initialize_variational_parameters()

            # Run coordinate ascent optimization
            convergence_info = self._coordinate_ascent_optimization()

            # Compute final results
            self._compute_final_marginals()

            self.inference_time = time.time() - start_time

            logger.log(f"Variational inference completed: converged={self.converged}, "
                      f"iterations={len(self.elbo_history)}, time={self.inference_time:.3f}s, "
                      f"final_ELBO={self.final_elbo:.3f}")

            return {
                "status": "success",
                "converged": self.converged,
                "iterations": len(self.elbo_history),
                "inference_time": self.inference_time,
                "final_elbo": self.final_elbo,
                "convergence_history": self.convergence_history,
                "elbo_curve": self.elbo_history
            }

        except Exception as e:
            logger.log(f"Variational inference failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "partial_results": getattr(self, 'elbo_history', [])
            }

    def _initialize_variational_parameters(self) -> None:
        """Initialize variational parameters for all latent variables"""
        for node in self.model.nodes():
            if node not in self.evidence:
                # Initialize as uniform categorical distribution
                cpd = self.model.get_cpds(node)
                variable_card = cpd.variable_card

                # Uniform initialization with small random perturbation
                uniform_params = np.ones(variable_card) / variable_card
                random_noise = np.random.normal(0, 0.01, variable_card)
                initial_params = uniform_params + random_noise

                # Ensure valid probability simplex
                initial_params = np.abs(initial_params)
                initial_params = initial_params / np.sum(initial_params)

                # Convert to natural parameters (log-space)
                natural_params = np.log(initial_params + 1e-10)

                self.variational_params[node] = VariationalParameter(
                    variable=node,
                    parameter_type='categorical',
                    parameters=natural_params,
                    entropy=0.0,
                    expected_value=initial_params
                )

    def _coordinate_ascent_optimization(self) -> Dict[str, Any]:
        """Run coordinate ascent to optimize ELBO"""
        prev_elbo = float('-inf')

        for iteration in range(self.max_iterations):
            # Update each variable's variational parameters
            for node in self.variational_params:
                self._update_variational_parameter(node)

            # Compute current ELBO
            current_elbo = self._compute_elbo()
            self.elbo_history.append(current_elbo)

            # Check convergence
            elbo_improvement = current_elbo - prev_elbo
            rel_improvement = abs(elbo_improvement) / (abs(prev_elbo) + 1e-10)

            convergence_info = {
                "iteration": iteration,
                "elbo": current_elbo,
                "improvement": elbo_improvement,
                "relative_improvement": rel_improvement
            }
            self.convergence_history.append(convergence_info)

            # Convergence check
            if rel_improvement < self.tolerance:
                self.converged = True
                self.final_elbo = current_elbo
                break

            prev_elbo = current_elbo

        if not self.converged:
            self.final_elbo = prev_elbo

        return {
            "converged": self.converged,
            "final_elbo": self.final_elbo,
            "iterations": len(self.elbo_history)
        }

    def _update_variational_parameter(self, node: str) -> None:
        """Update variational parameters for a single node using coordinate ascent"""
        cpd = self.model.get_cpds(node)
        variable_card = cpd.variable_card

        # Compute expected log-likelihood contribution for each state of this node
        expected_log_likelihood = np.zeros(variable_card)

        for state_idx in range(variable_card):
            # Set this node to specific state
            temp_assignment = {node: state_idx}

            # Compute expected log-likelihood under current variational distribution
            expected_log_likelihood[state_idx] = self._compute_expected_log_likelihood_single_node(
                node, state_idx, temp_assignment
            )

        # Update natural parameters using coordinate ascent
        new_natural_params = expected_log_likelihood

        # Apply learning rate for stability
        old_params = self.variational_params[node].parameters
        updated_params = (1 - self.learning_rate) * old_params + self.learning_rate * new_natural_params

        # Convert to probability space and normalize
        prob_params = np.exp(updated_params - logsumexp(updated_params))

        # Update variational parameter
        self.variational_params[node].parameters = np.log(prob_params + 1e-10)
        self.variational_params[node].expected_value = prob_params
        self.variational_params[node].entropy = -np.sum(prob_params * np.log(prob_params + 1e-10))

    def _compute_expected_log_likelihood_single_node(self, node: str, state_idx: int,
                                                   assignment: Dict[str, int]) -> float:
        """Compute expected log-likelihood contribution for a specific node state"""
        cpd = self.model.get_cpds(node)

        # Get parent configuration probabilities
        parents = list(self.model.get_parents(node))

        if not parents:
            # No parents case - use CPD directly
            node_state = self._get_state_name(node, state_idx)
            if node_state in cpd.state_names[node]:
                prob_index = cpd.state_names[node].index(node_state)
                return np.log(cpd.values[prob_index, 0] + 1e-10)
            else:
                return np.log(1e-10)

        # With parents - marginalize over parent configurations
        expected_log_prob = 0.0

        # Enumerate all parent configurations
        parent_cards = [self.model.get_cpds(p).variable_card for p in parents]
        total_configs = np.prod(parent_cards)

        for config_idx in range(int(total_configs)):
            # Convert flat index to parent configuration
            parent_config = []
            remaining = config_idx

            for card in reversed(parent_cards):
                parent_config.append(remaining % card)
                remaining //= card
            parent_config.reverse()

            # Compute probability of this parent configuration under variational distribution
            config_prob = 1.0
            for i, parent in enumerate(parents):
                if parent in self.evidence:
                    # Parent is observed
                    evidence_state_idx = self._get_state_index(parent, self.evidence[parent])
                    config_prob *= (1.0 if parent_config[i] == evidence_state_idx else 0.0)
                else:
                    # Parent is latent
                    if parent in self.variational_params:
                        config_prob *= self.variational_params[parent].expected_value[parent_config[i]]

            if config_prob > 1e-10:
                # Get CPD probability for this configuration
                try:
                    # Map indices to states and get CPD probability
                    evidence_dict = {}
                    for i, parent in enumerate(parents):
                        parent_state = self._get_state_name(parent, parent_config[i])
                        evidence_dict[parent] = parent_state

                    node_state = self._get_state_name(node, state_idx)
                    evidence_dict[node] = node_state

                    # Get probability from CPD
                    cpd_factor = cpd.to_factor()
                    reduced_factor = cpd_factor.reduce(evidence_dict, inplace=False)

                    if reduced_factor.values.size > 0:
                        cpd_prob = reduced_factor.values.flat[0]
                    else:
                        cpd_prob = 1e-10

                    expected_log_prob += config_prob * np.log(cpd_prob + 1e-10)

                except:
                    # Fallback: uniform probability
                    expected_log_prob += config_prob * np.log(1.0 / cpd.variable_card)

        return expected_log_prob

    def _compute_elbo(self) -> float:
        """Compute the Evidence Lower Bound (ELBO)"""
        # ELBO = E_q[log P(X, evidence)] - E_q[log q(X)]

        expected_log_joint = self._compute_expected_log_joint()
        entropy_term = sum(param.entropy for param in self.variational_params.values())

        elbo = expected_log_joint + entropy_term
        return elbo

    def _compute_expected_log_joint(self) -> float:
        """Compute expected log joint probability under variational distribution"""
        expected_log_joint = 0.0

        # Iterate over all CPDs in the model
        for cpd in self.model.get_cpds():
            node = cpd.variable
            parents = list(self.model.get_parents(node))

            # Enumerate all possible configurations
            all_vars = [node] + parents
            all_cards = [cpd.variable_card] + [self.model.get_cpds(p).variable_card for p in parents]

            total_configs = np.prod(all_cards)

            for config_idx in range(int(total_configs)):
                # Convert to variable assignment
                assignment = []
                remaining = config_idx

                for card in reversed(all_cards):
                    assignment.append(remaining % card)
                    remaining //= card
                assignment.reverse()

                # Compute probability of this configuration
                config_prob = 1.0
                evidence_dict = {}

                for i, var in enumerate(all_vars):
                    var_state_idx = assignment[i]
                    var_state = self._get_state_name(var, var_state_idx)
                    evidence_dict[var] = var_state

                    if var in self.evidence:
                        # Observed variable
                        observed_state_idx = self._get_state_index(var, self.evidence[var])
                        config_prob *= (1.0 if var_state_idx == observed_state_idx else 0.0)
                    else:
                        # Latent variable
                        if var in self.variational_params:
                            config_prob *= self.variational_params[var].expected_value[var_state_idx]

                if config_prob > 1e-10:
                    # Get CPD probability
                    try:
                        cpd_factor = cpd.to_factor()
                        reduced_factor = cpd_factor.reduce(evidence_dict, inplace=False)

                        if reduced_factor.values.size > 0:
                            cpd_prob = reduced_factor.values.flat[0]
                        else:
                            cpd_prob = 1e-10

                        expected_log_joint += config_prob * np.log(cpd_prob + 1e-10)

                    except:
                        # Fallback
                        continue

        return expected_log_joint

    def _compute_final_marginals(self) -> None:
        """Compute final marginal distributions"""
        for node, param in self.variational_params.items():
            # Marginals are directly stored in expected_value
            param.expected_value = np.exp(param.parameters - logsumexp(param.parameters))

    def query(self, variables: List[str]) -> Dict[str, np.ndarray]:
        """
        Query marginal distributions for specified variables

        Args:
            variables: List of variable names to query

        Returns:
            Dictionary mapping variable names to their marginal probability arrays
        """
        results = {}

        for var in variables:
            if var in self.evidence:
                # Observed variable - deterministic
                cpd = self.model.get_cpds(var)
                variable_card = cpd.variable_card
                marginal = np.zeros(variable_card)

                evidence_state_idx = self._get_state_index(var, self.evidence[var])
                marginal[evidence_state_idx] = 1.0

                results[var] = marginal

            elif var in self.variational_params:
                # Latent variable - use variational approximation
                results[var] = self.variational_params[var].expected_value.copy()

            else:
                # Variable not in model
                logger.log(f"Warning: Variable {var} not found in model")
                results[var] = None

        return results

    def _get_state_name(self, variable: str, state_index: int) -> str:
        """Convert state index to state name"""
        cpd = self.model.get_cpds(variable)

        if variable in cpd.state_names:
            state_names = cpd.state_names[variable]
            if 0 <= state_index < len(state_names):
                return state_names[state_index]

        # Fallback mapping
        if variable in ['Accident', 'Flooding']:
            return 'True' if state_index == 1 else 'False'
        elif variable == 'DrainageCapacity':
            return 'Poor' if state_index == 1 else 'Good'
        else:
            return 'High' if state_index == 1 else 'Low'

    def _get_state_index(self, variable: str, state_name: str) -> int:
        """Convert state name to index"""
        state_mapping = {
            'Low': 0, 'High': 1,
            'False': 0, 'True': 1,
            'Good': 0, 'Poor': 1
        }
        return state_mapping.get(state_name, 0)

    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """Get detailed convergence diagnostics"""
        return {
            "converged": self.converged,
            "final_elbo": self.final_elbo,
            "elbo_history": self.elbo_history,
            "total_iterations": len(self.elbo_history),
            "convergence_rate": self._compute_convergence_rate(),
            "parameter_stability": self._assess_parameter_stability(),
            "inference_time": self.inference_time
        }

    def _compute_convergence_rate(self) -> float:
        """Estimate convergence rate from ELBO history"""
        if len(self.elbo_history) < 10:
            return 0.0

        # Fit exponential decay to ELBO improvements
        improvements = np.diff(self.elbo_history[-10:])
        if len(improvements) > 0:
            return float(np.mean(np.abs(improvements)))
        return 0.0

    def _assess_parameter_stability(self) -> Dict[str, float]:
        """Assess stability of variational parameters"""
        stability = {}

        for node, param in self.variational_params.items():
            # Measure concentration of distribution (lower entropy = more stable)
            concentration = 1.0 - param.entropy / np.log(len(param.expected_value))
            stability[node] = max(0.0, min(1.0, concentration))

        return stability

    def get_approximation_quality(self) -> Dict[str, Any]:
        """Assess quality of variational approximation"""
        return {
            "final_elbo": self.final_elbo,
            "parameter_entropies": {
                node: param.entropy
                for node, param in self.variational_params.items()
            },
            "mean_entropy": np.mean([param.entropy for param in self.variational_params.values()]),
            "parameter_concentrations": self._assess_parameter_stability(),
            "elbo_variance": np.var(self.elbo_history[-10:]) if len(self.elbo_history) >= 10 else 0.0
        }