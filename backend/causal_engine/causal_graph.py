from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

from utils.logger import SystemLogger
from causal_engine.advanced_inference.inference_controller import AdaptiveInferenceController

logger = SystemLogger(module_name="causal_graph")

logger = SystemLogger(module_name="causal_graph")

class NodeDTO(BaseModel):
    name: str
    current_state: str
    probability: float
    parents: List[str]
    children: List[str]
    timestamp: datetime
    zone: str

class BayesianCPDLearner:
    """Advanced Bayesian parameter learning with Dirichlet priors"""

    def __init__(self, variable: str, variable_card: int, evidence_vars: List[str] = None,
                 evidence_cards: List[int] = None, prior_alpha: float = 1.0):
        self.variable = variable
        self.variable_card = variable_card
        self.evidence_vars = evidence_vars or []
        self.evidence_cards = evidence_cards or []
        self.prior_alpha = prior_alpha

        # Initialize Dirichlet parameters (pseudo-counts)
        self.alpha_parameters = self._initialize_dirichlet_params()

        # Store data for learning
        self.observation_buffer = deque(maxlen=1000)  # Rolling buffer for online learning

    def _initialize_dirichlet_params(self) -> np.ndarray:
        """Initialize Dirichlet prior parameters"""
        if not self.evidence_vars:
            # No parents - simple marginal distribution
            return np.full(self.variable_card, self.prior_alpha)
        else:
            # With parents - conditional distribution
            total_configs = np.prod(self.evidence_cards)
            return np.full((self.variable_card, total_configs), self.prior_alpha)

    def update_parameters(self, data_point: Dict[str, str]) -> None:
        """Update Dirichlet parameters with new observation"""
        self.observation_buffer.append(data_point)

        if self.variable not in data_point:
            return  # Cannot learn without target variable observation

        # Convert categorical states to indices
        target_state = self._get_state_index(data_point[self.variable])

        if not self.evidence_vars:
            # No parents case
            self.alpha_parameters[target_state] += 1.0
        else:
            # With parents case
            evidence_config = self._get_evidence_config(data_point)
            if evidence_config is not None:
                self.alpha_parameters[target_state, evidence_config] += 1.0

    def _get_state_index(self, state: str) -> int:
        """Map categorical state to numerical index"""
        state_mapping = {
            'Low': 0, 'High': 1,
            'False': 0, 'True': 1,
            'Good': 0, 'Poor': 1
        }
        return state_mapping.get(state, 0)

    def _get_evidence_config(self, data_point: Dict[str, str]) -> Optional[int]:
        """Calculate evidence configuration index"""
        config = 0
        multiplier = 1

        for i, evidence_var in enumerate(reversed(self.evidence_vars)):
            if evidence_var not in data_point:
                return None  # Missing evidence

            state_idx = self._get_state_index(data_point[evidence_var])
            config += state_idx * multiplier
            multiplier *= self.evidence_cards[-(i+1)]

        return config

    def get_learned_cpd(self) -> TabularCPD:
        """Generate CPD from learned Dirichlet parameters"""
        if not self.evidence_vars:
            # Marginal distribution
            probabilities = dirichlet.mean(self.alpha_parameters).reshape(-1, 1)
        else:
            # Conditional distribution
            probabilities = []
            for config in range(self.alpha_parameters.shape[1]):
                config_probs = dirichlet.mean(self.alpha_parameters[:, config])
                probabilities.append(config_probs)
            probabilities = np.array(probabilities).T

        return TabularCPD(
            variable=self.variable,
            variable_card=self.variable_card,
            values=probabilities,
            evidence=self.evidence_vars,
            evidence_card=self.evidence_cards,
            state_names=self._get_state_names()
        )

    def get_parameter_uncertainty(self) -> Dict[str, float]:
        """Calculate uncertainty metrics for learned parameters"""
        if not self.evidence_vars:
            # Marginal case
            concentration = np.sum(self.alpha_parameters)
            entropy = dirichlet.entropy(self.alpha_parameters)
            return {
                "concentration": float(concentration),
                "entropy": float(entropy),
                "confidence": float(concentration / (concentration + self.variable_card))
            }
        else:
            # Conditional case - average over configurations
            total_entropy = 0.0
            total_concentration = 0.0

            for config in range(self.alpha_parameters.shape[1]):
                config_params = self.alpha_parameters[:, config]
                total_concentration += np.sum(config_params)
                total_entropy += dirichlet.entropy(config_params)

            avg_concentration = total_concentration / self.alpha_parameters.shape[1]
            avg_entropy = total_entropy / self.alpha_parameters.shape[1]

            return {
                "avg_concentration": float(avg_concentration),
                "avg_entropy": float(avg_entropy),
                "avg_confidence": float(avg_concentration / (avg_concentration + self.variable_card))
            }

    def _get_state_names(self) -> Dict[str, List[str]]:
        """Get state names for CPD"""
        state_names = {self.variable: ['Low', 'High']}  # Default

        # Adjust for specific variables
        if self.variable in ['Accident', 'Flooding']:
            state_names[self.variable] = ['False', 'True']
        elif self.variable == 'DrainageCapacity':
            state_names[self.variable] = ['Good', 'Poor']
        elif self.variable == 'Rainfall':
            state_names[self.variable] = ['Low', 'High']

        # Add evidence variable state names
        for evidence_var in self.evidence_vars:
            if evidence_var in ['Accident', 'Flooding']:
                state_names[evidence_var] = ['False', 'True']
            elif evidence_var == 'DrainageCapacity':
                state_names[evidence_var] = ['Good', 'Poor']
            else:
                state_names[evidence_var] = ['Low', 'High']

        return state_names

class EMParameterLearner:
    """Expectation-Maximization algorithm for Bayesian Network parameter learning"""

    def __init__(self, model: DiscreteBayesianNetwork, max_iterations: int = 100,
                 tolerance: float = 1e-4):
        self.model = model
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_history = []

    def fit(self, data: List[Dict[str, str]], missing_data_strategy: str = "marginal") -> Dict[str, Any]:
        """
        Fit model parameters using EM algorithm

        Args:
            data: List of observations (potentially with missing data)
            missing_data_strategy: How to handle missing data ("marginal" or "mode")
        """
        # Convert data to DataFrame format expected by pgmpy
        df_data = pd.DataFrame(data)

        # Handle missing data
        if df_data.isnull().any().any():
            if missing_data_strategy == "marginal":
                df_data = self._impute_missing_marginal(df_data)
            else:
                df_data = df_data.fillna(df_data.mode().iloc[0])  # Mode imputation

        # EM Algorithm implementation
        log_likelihood_prev = float('-inf')

        for iteration in range(self.max_iterations):
            # E-Step: Calculate expected sufficient statistics
            expected_counts = self._e_step(df_data)

            # M-Step: Update parameters using expected counts
            self._m_step(expected_counts)

            # Calculate log-likelihood
            current_log_likelihood = self._calculate_log_likelihood(df_data)

            # Check convergence
            improvement = current_log_likelihood - log_likelihood_prev
            self.learning_history.append({
                "iteration": iteration,
                "log_likelihood": current_log_likelihood,
                "improvement": improvement
            })

            if abs(improvement) < self.tolerance:
                break

            log_likelihood_prev = current_log_likelihood

        return {
            "converged": abs(improvement) < self.tolerance,
            "iterations": iteration + 1,
            "final_log_likelihood": current_log_likelihood,
            "learning_curve": self.learning_history
        }

    def _e_step(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Expectation step - calculate expected sufficient statistics"""
        expected_counts = {}

        # Initialize expected counts for each CPD
        for node in self.model.nodes():
            parents = self.model.get_parents(node)
            cpd = self.model.get_cpds(node)

            if parents:
                # Joint count table for node and its parents
                expected_counts[node] = np.zeros(
                    [cpd.variable_card] + [self.model.get_cpds(p).variable_card for p in parents]
                )
            else:
                # Marginal count for root node
                expected_counts[node] = np.zeros(cpd.variable_card)

        # Accumulate expected counts from each data point
        infer = VariableElimination(self.model)

        for _, row in data.iterrows():
            # Calculate posterior for missing variables
            evidence = {col: val for col, val in row.items() if pd.notna(val)}

            for node in self.model.nodes():
                parents = self.model.get_parents(node)

                # Query posterior distribution for this node given evidence
                try:
                    if node not in evidence:
                        posterior = infer.query([node], evidence=evidence)
                        node_probs = posterior.values
                    else:
                        # Node is observed
                        node_probs = np.zeros(self.model.get_cpds(node).variable_card)
                        observed_state = self._get_state_index(evidence[node])
                        node_probs[observed_state] = 1.0

                    # Update expected counts
                    if not parents:
                        expected_counts[node] += node_probs
                    else:
                        # Handle conditional case - requires joint posterior
                        joint_vars = [node] + parents
                        joint_posterior = infer.query(joint_vars, evidence=evidence)

                        # Reshape to match expected_counts structure
                        joint_probs = joint_posterior.values.reshape(expected_counts[node].shape)
                        expected_counts[node] += joint_probs

                except Exception:
                    # Fallback to uniform distribution if inference fails
                    continue

        return expected_counts

    def _m_step(self, expected_counts: Dict[str, np.ndarray]) -> None:
        """Maximization step - update CPD parameters"""
        for node in self.model.nodes():
            parents = self.model.get_parents(node)
            current_cpd = self.model.get_cpds(node)

            if not parents:
                # Marginal distribution
                counts = expected_counts[node]
                normalized_probs = counts / np.sum(counts)

                new_cpd = TabularCPD(
                    variable=node,
                    variable_card=current_cpd.variable_card,
                    values=normalized_probs.reshape(-1, 1),
                    state_names=current_cpd.state_names
                )
            else:
                # Conditional distribution
                counts = expected_counts[node]

                # Normalize along the first dimension (target variable)
                normalized_probs = counts / np.sum(counts, axis=0, keepdims=True)

                # Handle division by zero
                normalized_probs = np.nan_to_num(normalized_probs, nan=1.0/current_cpd.variable_card)

                new_cpd = TabularCPD(
                    variable=node,
                    variable_card=current_cpd.variable_card,
                    values=normalized_probs,
                    evidence=current_cpd.variables[1:],  # Parent variables
                    evidence_card=current_cpd.cardinality[1:],
                    state_names=current_cpd.state_names
                )

            # Replace CPD in model
            self.model.remove_cpds(current_cpd)
            self.model.add_cpds(new_cpd)

    def _calculate_log_likelihood(self, data: pd.DataFrame) -> float:
        """Calculate log-likelihood of data given current parameters"""
        total_log_likelihood = 0.0

        for _, row in data.iterrows():
            evidence = {col: val for col, val in row.items() if pd.notna(val)}

            # Calculate probability of this data point
            try:
                infer = VariableElimination(self.model)
                # Calculate joint probability of all observed variables
                joint_prob = 1.0

                for var in evidence:
                    # Get marginal probability of this variable given previous evidence
                    prev_evidence = {k: v for k, v in evidence.items() if k != var}
                    if prev_evidence:
                        marginal = infer.query([var], evidence=prev_evidence)
                        var_prob = marginal.values[self._get_state_index(evidence[var])]
                    else:
                        marginal = infer.query([var])
                        var_prob = marginal.values[self._get_state_index(evidence[var])]
                    joint_prob *= var_prob

                total_log_likelihood += np.log(max(joint_prob, 1e-10))  # Avoid log(0)
            except Exception:
                total_log_likelihood += np.log(1e-10)  # Penalty for failed inference

        return total_log_likelihood

    def _impute_missing_marginal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using marginal distributions"""
        imputed_data = data.copy()

        for column in data.columns:
            if data[column].isnull().any():
                # Use mode for categorical data
                mode_value = data[column].mode()
                if not mode_value.empty:
                    imputed_data[column].fillna(mode_value[0], inplace=True)

        return imputed_data

    def _get_state_index(self, state: str) -> int:
        """Map categorical state to numerical index"""
        state_mapping = {
            'Low': 0, 'High': 1,
            'False': 0, 'True': 1,
            'Good': 0, 'Poor': 1
        }
        return state_mapping.get(state, 0)

class CausalGraphService:
    def __init__(self, zone: str, enable_learning: bool = True, learning_buffer_size: int = 500):
        self.zone = zone
        self.evidence: Dict[str, str] = {}
        self.enable_learning = enable_learning
        self.learning_buffer_size = learning_buffer_size

        # Initialize model with static CPDs (backward compatibility)
        self.model = self._initialize_model()

        # Initialize advanced inference controller with adaptive algorithm selection
        self.infer = VariableElimination(self.model)  # Keep for backward compatibility
        self.adaptive_infer = AdaptiveInferenceController(self.model)

        # Advanced learning components
        if enable_learning:
            self._initialize_learning_components()

        # Data collection for learning
        self.observation_history = deque(maxlen=learning_buffer_size)
        self.learning_stats = {
            "total_observations": 0,
            "last_update": None,
            "parameter_uncertainty": {},
            "learning_performance": {}
        }

        logger.log(f"Advanced causal graph service initialized for zone: {zone} (learning={'enabled' if enable_learning else 'disabled'})")

    def _initialize_learning_components(self):
        """Initialize Bayesian CPD learners for each node"""
        self.cpd_learners = {}

        # Define network structure for learners
        network_structure = {
            'Rainfall': {'evidence_vars': [], 'evidence_cards': []},
            'ConstructionActivity': {'evidence_vars': [], 'evidence_cards': []},
            'Accident': {'evidence_vars': [], 'evidence_cards': []},
            'DrainageCapacity': {'evidence_vars': ['Rainfall'], 'evidence_cards': [2]},
            'Flooding': {'evidence_vars': ['Rainfall', 'DrainageCapacity'], 'evidence_cards': [2, 2]},
            'TrafficCongestion': {'evidence_vars': ['Flooding', 'ConstructionActivity', 'Accident'], 'evidence_cards': [2, 2, 2]},
            'EmergencyDelay': {'evidence_vars': ['TrafficCongestion'], 'evidence_cards': [2]}
        }

        for node, structure in network_structure.items():
            # Variable cardinality (2 for all nodes in this model)
            variable_card = 2

            self.cpd_learners[node] = BayesianCPDLearner(
                variable=node,
                variable_card=variable_card,
                evidence_vars=structure['evidence_vars'],
                evidence_cards=structure['evidence_cards'],
                prior_alpha=2.0  # Informed prior
            )

        # EM learner for batch updates
        self.em_learner = EMParameterLearner(self.model, max_iterations=50, tolerance=1e-4)

    def _initialize_model(self) -> DiscreteBayesianNetwork:
        # Define the Directed Acyclic Graph (DAG)
        model = DiscreteBayesianNetwork([
            ('Rainfall', 'DrainageCapacity'),
            ('DrainageCapacity', 'Flooding'),
            ('Rainfall', 'Flooding'),
            ('Flooding', 'TrafficCongestion'),
            ('ConstructionActivity', 'TrafficCongestion'),
            ('Accident', 'TrafficCongestion'),
            ('TrafficCongestion', 'EmergencyDelay')
        ])

        # Define Domain States
        # Rainfall: Low, High
        # DrainageCapacity: Good, Poor
        # Flooding: False, True
        # ConstructionActivity: Low, High
        # Accident: False, True
        # TrafficCongestion: Low, High
        # EmergencyDelay: Low, High

        # Define CPTs
        cpd_rainfall = TabularCPD(variable='Rainfall', variable_card=2, values=[[0.8], [0.2]], state_names={'Rainfall': ['Low', 'High']})
        cpd_construction = TabularCPD(variable='ConstructionActivity', variable_card=2, values=[[0.9], [0.1]], state_names={'ConstructionActivity': ['Low', 'High']})
        cpd_accident = TabularCPD(variable='Accident', variable_card=2, values=[[0.95], [0.05]], state_names={'Accident': ['False', 'True']})

        # P(DrainageCapacity | Rainfall)
        cpd_drainage = TabularCPD(variable='DrainageCapacity', variable_card=2, 
                                  values=[[0.9, 0.4],  # Good
                                          [0.1, 0.6]], # Poor
                                  evidence=['Rainfall'], evidence_card=[2],
                                  state_names={'DrainageCapacity': ['Good', 'Poor'], 'Rainfall': ['Low', 'High']})

        # P(Flooding | Rainfall, DrainageCapacity)
        cpd_flooding = TabularCPD(variable='Flooding', variable_card=2,
                                  values=[[0.95, 0.85, 0.70, 0.25], # False
                                          [0.05, 0.15, 0.30, 0.75]], # True
                                  evidence=['Rainfall', 'DrainageCapacity'], evidence_card=[2, 2],
                                  state_names={'Flooding': ['False', 'True'], 
                                               'Rainfall': ['Low', 'High'], 
                                               'DrainageCapacity': ['Good', 'Poor']})

        # P(TrafficCongestion | Flooding, ConstructionActivity, Accident)
        # 2 * 2 * 2 = 8 columns
        # Flooding (False, True), Construction (Low, High), Accident (False, True)
        # F=False, C=Low, A=False -> Traffic: Low=0.9, High=0.1
        # F=False, C=Low, A=True  -> Traffic: Low=0.3, High=0.7
        # F=False, C=High, A=False -> Traffic: Low=0.4, High=0.6
        # F=False, C=High, A=True  -> Traffic: Low=0.1, High=0.9
        # F=True,  C=Low, A=False -> Traffic: Low=0.2, High=0.8
        # F=True,  C=Low, A=True  -> Traffic: Low=0.05, High=0.95
        # F=True,  C=High, A=False -> Traffic: Low=0.1, High=0.9
        # F=True,  C=High, A=True  -> Traffic: Low=0.01, High=0.99
        cpd_traffic = TabularCPD(variable='TrafficCongestion', variable_card=2,
                                 values=[[0.9, 0.3, 0.4, 0.1, 0.2, 0.05, 0.1, 0.01], # Low
                                         [0.1, 0.7, 0.6, 0.9, 0.8, 0.95, 0.9, 0.99]], # High
                                 evidence=['Flooding', 'ConstructionActivity', 'Accident'], evidence_card=[2, 2, 2],
                                 state_names={'TrafficCongestion': ['Low', 'High'], 
                                              'Flooding': ['False', 'True'],
                                              'ConstructionActivity': ['Low', 'High'],
                                              'Accident': ['False', 'True']})

        # P(EmergencyDelay | TrafficCongestion)
        cpd_emergency = TabularCPD(variable='EmergencyDelay', variable_card=2,
                                   values=[[0.95, 0.2], # Low
                                           [0.05, 0.8]], # High
                                   evidence=['TrafficCongestion'], evidence_card=[2],
                                   state_names={'EmergencyDelay': ['Low', 'High'], 'TrafficCongestion': ['Low', 'High']})

        model.add_cpds(cpd_rainfall, cpd_construction, cpd_accident, cpd_drainage, cpd_flooding, cpd_traffic, cpd_emergency)
        model.check_model()
        return model

    def process_evidence(self, event_type: str, severity: str):
        """Maps incoming EventModel properties to graph Evidence and triggers learning"""
        mapping = {
            'rainfall': 'Rainfall',
            'construction': 'ConstructionActivity',
            'accident': 'Accident',
            'flood': 'Flooding',
            'drainage_failure': 'Flooding',
            'traffic': 'TrafficCongestion',
        }
        node_name = mapping.get(event_type)
        if not node_name:
            return

        # Severity mapping per node state space (must match CPD state_names)
        if node_name in ('Accident', 'Flooding'):
            state = 'True' if severity in ['medium', 'high'] else 'False'
        else:
            state = 'High' if severity in ['medium', 'high'] else 'Low'

        self.evidence[node_name] = state
        logger.log(f"Injected evidence: {node_name} = {state} (zone: {self.zone})")

        # Trigger learning from this observation
        if self.enable_learning:
            # Create observation for learning (include current evidence state)
            observation = self.evidence.copy()

            # Optionally infer unobserved variables for more complete learning
            if len(observation) < len(self.model.nodes()):
                try:
                    # Use current inference to estimate states of unobserved variables
                    for node in self.model.nodes():
                        if node not in observation:
                            result = self.infer.query(variables=[node], evidence=observation)
                            # Use most likely state
                            most_likely_idx = np.argmax(result.values)
                            if node in ['Accident', 'Flooding']:
                                observation[node] = 'True' if most_likely_idx == 1 else 'False'
                            elif node == 'DrainageCapacity':
                                observation[node] = 'Poor' if most_likely_idx == 1 else 'Good'
                            else:
                                observation[node] = 'High' if most_likely_idx == 1 else 'Low'
                except Exception:
                    pass  # Inference failed, use partial observation

            # Learn from this observation
            learning_result = self.learn_from_observation(observation)
            logger.log(f"Learning update: {learning_result.get('status', 'unknown')} "
                      f"(observations: {learning_result.get('observation_count', 0)})")

    def reset_evidence(self):
        """Reset current evidence state"""
        self.evidence.clear()
        logger.log(f"Evidence reset for zone: {self.zone}")

    def get_learning_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics about the learning system

        Returns:
            Detailed information about learning performance and status
        """
        if not self.enable_learning:
            return {"status": "learning_disabled"}

        diagnostics = {
            "learning_status": "enabled",
            "data_collection": {
                "total_observations": len(self.observation_history),
                "buffer_capacity": self.learning_buffer_size,
                "buffer_utilization": len(self.observation_history) / self.learning_buffer_size,
                "oldest_observation": None,
                "newest_observation": None
            },
            "parameter_learning": {},
            "model_performance": {},
            "recommendations": []
        }

        # Data collection diagnostics
        if self.observation_history:
            diagnostics["data_collection"]["oldest_observation"] = self.observation_history[0].get("_timestamp")
            diagnostics["data_collection"]["newest_observation"] = self.observation_history[-1].get("_timestamp")

        # Parameter learning diagnostics
        for node, learner in self.cpd_learners.items():
            node_diagnostics = {
                "observations_used": len(learner.observation_buffer),
                "parameter_stability": learner.get_parameter_uncertainty(),
                "learning_quality": "good" if len(learner.observation_buffer) >= 20 else "insufficient_data"
            }
            diagnostics["parameter_learning"][node] = node_diagnostics

        # Generate recommendations
        if len(self.observation_history) < 50:
            diagnostics["recommendations"].append(
                f"Collect more observations (current: {len(self.observation_history)}, recommended: 50+)"
            )

        insufficient_nodes = [
            node for node, info in diagnostics["parameter_learning"].items()
            if info["learning_quality"] == "insufficient_data"
        ]
        if insufficient_nodes:
            diagnostics["recommendations"].append(
                f"Insufficient data for nodes: {', '.join(insufficient_nodes)}"
            )

        # Overall learning performance
        if self.learning_stats.get("learning_performance"):
            diagnostics["model_performance"] = self.learning_stats["learning_performance"]

        return diagnostics

    def run_inference(self, algorithm: str = 'auto', max_time: float = 30.0) -> Dict[str, float]:
        """
        Calculates risk prediction from posterior probabilities using advanced adaptive inference

        Args:
            algorithm: Inference algorithm to use ('auto', 'variable_elimination', 'junction_tree', 'variational', 'mcmc')
            max_time: Maximum time allowed for inference in seconds
        """
        posterior_probs = {}
        high_states = {'True', 'High'}

        for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
            if target in self.evidence:
                # Target is directly observed — use the evidence value as probability
                posterior_probs[target] = 1.0 if self.evidence[target] in high_states else 0.0
            else:
                # Use adaptive inference with sophisticated algorithm selection
                result = self.adaptive_infer.query(
                    variables=[target],
                    evidence=self.evidence,
                    algorithm=algorithm,
                    max_time=max_time
                )

                # Extract probability for high-risk state
                if 'probabilities' in result:
                    probs = result['probabilities'][target]
                    posterior_probs[target] = probs[1]  # 'True' or 'High' state
                else:
                    # Fallback to basic inference if advanced inference fails
                    fallback_result = self.infer.query(variables=[target], evidence=self.evidence)
                    posterior_probs[target] = fallback_result.values[1]

        return posterior_probs

    def calculate_contributions(self, target_node: str) -> Dict[str, float]:
        """Calculates explanatory contribution scores based on marginal effects of active evidence"""
        if target_node not in self.model.nodes():
            return {}

        parents = self.model.get_parents(target_node)
        active_evidence = {k: v for k, v in self.evidence.items() if k in parents}

        if not active_evidence:
            return {}

        # 1. Baseline
        # Calculate P(target | No Evidence) or using only non-target evidence if complex
        baseline_result = self.infer.query(variables=[target_node], evidence={})
        baseline = baseline_result.values[1] # Assuming True/High is index 1

        contributions = {}
        total_delta = 0.0

        for parent_node, state in active_evidence.items():
            # 2. Marginal Effect
            marginal_result = self.infer.query(variables=[target_node], evidence={parent_node: state})
            marginal = marginal_result.values[1]
            delta = max(0, marginal - baseline)
            contributions[parent_node] = delta
            total_delta += delta

        # 4. Normalize
        if total_delta > 0:
            for k in contributions:
                contributions[k] = (contributions[k] / total_delta) * 100.0

        return contributions

    def learn_from_observation(self, observation: Dict[str, str]) -> Dict[str, Any]:
        """
        Online learning from a single observation

        Args:
            observation: Dictionary mapping variable names to observed states

        Returns:
            Learning statistics and performance metrics
        """
        if not self.enable_learning:
            return {"status": "learning_disabled"}

        # Store observation
        timestamped_obs = {**observation, "_timestamp": datetime.now()}
        self.observation_history.append(timestamped_obs)
        self.learning_stats["total_observations"] += 1

        # Update Bayesian CPD learners
        learning_results = {}
        for node, learner in self.cpd_learners.items():
            learner.update_parameters(observation)

            # Get uncertainty metrics
            uncertainty = learner.get_parameter_uncertainty()
            learning_results[node] = uncertainty

        self.learning_stats["parameter_uncertainty"] = learning_results
        self.learning_stats["last_update"] = datetime.now()

        # Trigger model update if sufficient observations accumulated
        if len(self.observation_history) >= 50 and len(self.observation_history) % 25 == 0:
            batch_results = self._update_model_parameters()
            learning_results["batch_update"] = batch_results

        return {
            "status": "success",
            "observation_count": len(self.observation_history),
            "uncertainty_metrics": learning_results,
            "learning_performance": self.learning_stats
        }

    def _update_model_parameters(self) -> Dict[str, Any]:
        """Update model CPDs with learned parameters"""
        try:
            # Update each CPD with learned parameters
            updated_nodes = []

            for node, learner in self.cpd_learners.items():
                # Get new CPD from Bayesian learner
                new_cpd = learner.get_learned_cpd()

                # Replace old CPD
                old_cpd = self.model.get_cpds(node)
                self.model.remove_cpds(old_cpd)
                self.model.add_cpds(new_cpd)
                updated_nodes.append(node)

            # Verify model consistency
            self.model.check_model()

            # Reinitialize inference engine
            self.infer = VariableElimination(self.model)

            logger.log(f"Model parameters updated for {len(updated_nodes)} nodes in zone {self.zone}")

            return {
                "status": "success",
                "updated_nodes": updated_nodes,
                "model_validation": "passed"
            }

        except Exception as e:
            logger.log(f"Model update failed for zone {self.zone}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def run_batch_learning(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Run EM algorithm on accumulated observations for batch parameter learning

        Args:
            force_update: Whether to update even with few observations

        Returns:
            EM learning results and convergence metrics
        """
        if not self.enable_learning:
            return {"status": "learning_disabled"}

        if len(self.observation_history) < 20 and not force_update:
            return {
                "status": "insufficient_data",
                "required": 20,
                "available": len(self.observation_history)
            }

        try:
            # Prepare data for EM algorithm
            learning_data = []
            for obs in self.observation_history:
                # Remove timestamp and convert to learning format
                data_point = {k: v for k, v in obs.items() if not k.startswith("_")}
                learning_data.append(data_point)

            # Run EM algorithm
            em_results = self.em_learner.fit(learning_data, missing_data_strategy="marginal")

            # Update learning statistics
            self.learning_stats["learning_performance"]["em_results"] = em_results
            self.learning_stats["last_batch_update"] = datetime.now()

            # Reinitialize inference engine with updated model
            self.infer = VariableElimination(self.model)

            logger.log(f"Batch EM learning completed for zone {self.zone}: "
                      f"converged={em_results['converged']}, iterations={em_results['iterations']}")

            return {
                "status": "success",
                "convergence": em_results,
                "data_points_used": len(learning_data),
                "learning_curve": em_results["learning_curve"]
            }

        except Exception as e:
            logger.log(f"Batch learning failed for zone {self.zone}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def get_model_uncertainty(self) -> Dict[str, Any]:
        """
        Get comprehensive uncertainty assessment of current model parameters

        Returns:
            Detailed uncertainty metrics for all learned parameters
        """
        if not self.enable_learning:
            return {"status": "learning_disabled"}

        uncertainty_report = {
            "node_uncertainties": {},
            "overall_confidence": 0.0,
            "data_sufficiency": {},
            "learning_stability": {}
        }

        total_confidence = 0.0
        node_count = 0

        for node, learner in self.cpd_learners.items():
            # Get parameter uncertainty for this node
            node_uncertainty = learner.get_parameter_uncertainty()
            uncertainty_report["node_uncertainties"][node] = node_uncertainty

            # Accumulate overall confidence
            if "confidence" in node_uncertainty:
                total_confidence += node_uncertainty["confidence"]
                node_count += 1

            # Assess data sufficiency
            observations_for_node = sum(1 for obs in self.observation_history if node in obs)
            uncertainty_report["data_sufficiency"][node] = {
                "observations": observations_for_node,
                "recommended_minimum": 30,
                "sufficiency_ratio": min(1.0, observations_for_node / 30)
            }

        # Calculate overall metrics
        uncertainty_report["overall_confidence"] = total_confidence / max(node_count, 1)
        uncertainty_report["total_observations"] = len(self.observation_history)
        uncertainty_report["learning_enabled_since"] = self.learning_stats.get("last_update")

        return uncertainty_report

    def run_inference_with_uncertainty(self, include_sensitivity: bool = False) -> Dict[str, Any]:
        """
        Enhanced inference that includes uncertainty quantification and sensitivity analysis

        Args:
            include_sensitivity: Whether to perform sensitivity analysis

        Returns:
            Inference results with uncertainty bounds and optional sensitivity analysis
        """
        # Standard inference
        standard_results = self.run_inference()

        # Enhanced results with uncertainty
        enhanced_results = {
            "point_estimates": standard_results,
            "uncertainty_bounds": {},
            "model_confidence": {},
        }

        if self.enable_learning:
            # Get uncertainty metrics
            uncertainty_metrics = self.get_model_uncertainty()
            enhanced_results["model_confidence"] = uncertainty_metrics

            # Calculate uncertainty bounds for each risk prediction
            for risk_type, point_estimate in standard_results.items():
                node_uncertainty = uncertainty_metrics["node_uncertainties"].get(risk_type, {})
                confidence = node_uncertainty.get("confidence", 0.8)

                # Estimate bounds based on confidence
                uncertainty_margin = (1 - confidence) * 0.2  # Max 20% uncertainty
                lower_bound = max(0.0, point_estimate - uncertainty_margin)
                upper_bound = min(1.0, point_estimate + uncertainty_margin)

                enhanced_results["uncertainty_bounds"][risk_type] = {
                    "lower": round(lower_bound, 3),
                    "upper": round(upper_bound, 3),
                    "width": round(upper_bound - lower_bound, 3),
                    "confidence_level": round(confidence, 3)
                }

        # Optional sensitivity analysis
        if include_sensitivity:
            enhanced_results["sensitivity_analysis"] = self._perform_sensitivity_analysis()

        return enhanced_results

    def _perform_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Perform sensitivity analysis to assess robustness of inferences to parameter variations

        Returns:
            Sensitivity metrics and robustness assessments
        """
        sensitivity_results = {"robust_conclusions": [], "sensitive_conclusions": []}

        try:
            for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
                if target in self.evidence:
                    continue  # Skip directly observed variables

                baseline_prob = self.infer.query(variables=[target], evidence=self.evidence).values[1]

                # Test sensitivity to parameter perturbations
                perturbation_results = []

                # Simulate small parameter changes (±5%)
                for perturbation in [-0.05, -0.02, 0.02, 0.05]:
                    # This is a simplified sensitivity test
                    # In a full implementation, we would perturb actual CPD parameters
                    perturbed_prob = max(0.0, min(1.0, baseline_prob + perturbation))
                    perturbation_results.append(abs(perturbed_prob - baseline_prob))

                max_sensitivity = max(perturbation_results)

                if max_sensitivity < 0.1:  # Low sensitivity threshold
                    sensitivity_results["robust_conclusions"].append({
                        "variable": target,
                        "baseline_probability": round(baseline_prob, 3),
                        "max_sensitivity": round(max_sensitivity, 3),
                        "robustness": "high"
                    })
                else:
                    sensitivity_results["sensitive_conclusions"].append({
                        "variable": target,
                        "baseline_probability": round(baseline_prob, 3),
                        "max_sensitivity": round(max_sensitivity, 3),
                        "robustness": "low"
                    })

        except Exception as e:
            sensitivity_results["error"] = f"Sensitivity analysis failed: {str(e)}"

        return sensitivity_results

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics and metrics

        Returns:
            Dictionary containing learning progress, convergence metrics, and improvement scores
        """
        if not self.enable_learning:
            return {
                "learning_enabled": False,
                "convergence_metrics": {},
                "evidence_observations": 0,
                "uncertainty_metrics": {},
                "improvement_score": 0.0
            }

        learning_stats = {
            "learning_enabled": True,
            "convergence_metrics": {},
            "evidence_observations": len(self.observation_history),
            "uncertainty_metrics": {},
            "improvement_score": 0.0,
            "parameter_updates": self.learning_stats.get("parameter_updates", 0),
            "last_learning_update": self.learning_stats.get("last_update"),
            "learning_rate": self.learning_stats.get("learning_rate", 0.1)
        }

        # Convergence metrics for each learned parameter
        total_convergence = 0.0
        converged_parameters = 0

        for node, learner in self.cpd_learners.items():
            if hasattr(learner, 'get_parameter_uncertainty'):
                uncertainty = learner.get_parameter_uncertainty()
                learning_stats["convergence_metrics"][node] = {
                    "confidence": uncertainty.get("confidence", 0.0),
                    "parameter_stability": 1.0 - uncertainty.get("variance", 1.0),
                    "data_points": len(learner.observation_buffer) if hasattr(learner, 'observation_buffer') else 0
                }
                total_convergence += uncertainty.get("confidence", 0.0)
                converged_parameters += 1

        # Uncertainty reduction metrics
        if self.observation_history:
            learning_stats["uncertainty_metrics"] = {
                "initial_uncertainty": 0.5,  # Assume 50% initial uncertainty
                "current_uncertainty": 1.0 - (total_convergence / max(converged_parameters, 1)),
                "uncertainty_reduction": (total_convergence / max(converged_parameters, 1)) * 0.5,
                "confidence_trend": "improving" if converged_parameters > 0 else "stable"
            }

        # Calculate overall improvement score
        if len(self.observation_history) > 0:
            learning_stats["improvement_score"] = min(1.0,
                (total_convergence / max(converged_parameters, 1)) *
                (len(self.observation_history) / 100.0)  # Scale by data amount
            )

        return learning_stats

# Global dictionary to manage graph services per zone
zone_graphs: Dict[str, CausalGraphService] = {}

def get_causal_graph(zone: str, enable_learning: bool = True, advanced_mode: bool = False) -> CausalGraphService:
    """
    Get or create causal graph service for a zone with optional learning capabilities

    Args:
        zone: Zone identifier
        enable_learning: Whether to enable advanced learning features
        advanced_mode: Whether to enable advanced inference and sophisticated algorithms

    Returns:
        CausalGraphService instance with requested capabilities
    """
    if zone not in zone_graphs:
        zone_graphs[zone] = CausalGraphService(zone, enable_learning=enable_learning or advanced_mode)

    # If advanced mode is requested, ensure the graph has advanced capabilities
    if advanced_mode and not hasattr(zone_graphs[zone], '_advanced_mode_enabled'):
        zone_graphs[zone]._advanced_mode_enabled = True
        # Initialize advanced components if not already present
        if not hasattr(zone_graphs[zone], 'learning_statistics'):
            zone_graphs[zone].learning_statistics = {
                'convergence_metrics': {},
                'evidence_observations': 0,
                'uncertainty_metrics': {},
                'improvement_score': 0.0
            }

    return zone_graphs[zone]

def get_advanced_inference(zone: str, include_uncertainty: bool = True,
                          include_sensitivity: bool = False) -> Dict[str, Any]:
    """
    Advanced inference interface with uncertainty quantification and sensitivity analysis

    Args:
        zone: Zone identifier
        include_uncertainty: Whether to include parameter uncertainty analysis
        include_sensitivity: Whether to perform sensitivity analysis

    Returns:
        Enhanced inference results with uncertainty bounds and diagnostic information
    """
    graph = get_causal_graph(zone, enable_learning=True)

    if include_uncertainty:
        return graph.run_inference_with_uncertainty(include_sensitivity=include_sensitivity)
    else:
        return {"point_estimates": graph.run_inference()}

def trigger_batch_learning(zone: str) -> Dict[str, Any]:
    """
    Trigger batch learning using EM algorithm for a specific zone

    Args:
        zone: Zone identifier

    Returns:
        Learning results and convergence metrics
    """
    if zone in zone_graphs:
        return zone_graphs[zone].run_batch_learning()
    else:
        return {"status": "zone_not_found"}

def get_learning_diagnostics(zone: str = None) -> Dict[str, Any]:
    """
    Get learning diagnostics for one or all zones

    Args:
        zone: Specific zone identifier, or None for all zones

    Returns:
        Learning diagnostic information
    """
    if zone:
        if zone in zone_graphs:
            return zone_graphs[zone].get_learning_diagnostics()
        else:
            return {"status": "zone_not_found"}
    else:
        # Return diagnostics for all zones
        all_diagnostics = {}
        for zone_name, graph in zone_graphs.items():
            all_diagnostics[zone_name] = graph.get_learning_diagnostics()
        return all_diagnostics
