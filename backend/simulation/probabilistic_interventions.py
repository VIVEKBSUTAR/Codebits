"""
Advanced Probabilistic Intervention Modeling with Beta-distributed Success Rates

This module replaces deterministic intervention effects with sophisticated probabilistic modeling
featuring context-dependent effectiveness, intervention synergies, and temporal dynamics.
"""

import numpy as np
from scipy.stats import beta, multivariate_normal, gamma
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InterventionConfig:
    """Configuration for a specific intervention type"""
    name: str
    target_nodes: List[str]

    # Beta distribution parameters for success probability
    base_alpha: float = 2.0  # Success count prior
    base_beta: float = 1.0   # Failure count prior

    # Context dependency factors
    context_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Effectiveness bounds
    min_effectiveness: float = 0.1
    max_effectiveness: float = 0.95

    # Temporal dynamics
    effectiveness_decay_rate: float = 0.1  # Per hour decay
    max_duration_hours: float = 24.0

    # Cost and constraints
    deployment_cost: float = 1.0
    resource_requirements: Dict[str, int] = field(default_factory=dict)

    # Interaction effects
    synergy_partners: List[str] = field(default_factory=list)
    conflict_partners: List[str] = field(default_factory=list)

class ProbabilisticInterventionEngine:
    """Advanced intervention modeling with probabilistic effects and sophisticated interactions"""

    def __init__(self):
        self.intervention_configs = self._initialize_intervention_configs()
        self.deployment_history = []  # Track past deployments
        self.learning_data = pd.DataFrame()  # For effectiveness learning

    def _initialize_intervention_configs(self) -> Dict[str, InterventionConfig]:
        """Initialize sophisticated intervention configurations"""

        configs = {}

        # Advanced Pump Deployment
        configs["deploy_pump"] = InterventionConfig(
            name="Advanced Drainage Pump",
            target_nodes=["DrainageCapacity", "Flooding"],
            base_alpha=3.0,  # Generally effective
            base_beta=1.2,   # Some failure cases
            context_modifiers={
                "Rainfall": {"High": 0.7, "Medium": 0.9, "Low": 1.1},  # Less effective in heavy rain
                "ConstructionActivity": {"High": 0.8, "Low": 1.0},     # Construction interferes
                "season": {"wet_season": 0.8, "dry_season": 1.2}       # Seasonal effectiveness
            },
            min_effectiveness=0.15,
            max_effectiveness=0.90,
            effectiveness_decay_rate=0.05,  # Slower decay for infrastructure
            max_duration_hours=48.0,
            deployment_cost=100.0,
            resource_requirements={"pump_units": 1, "maintenance_crew": 2},
            synergy_partners=["traffic_rerouting"],  # Works well with traffic management
            conflict_partners=["construction_halt"]   # Can't deploy during construction
        )

        # Intelligent Traffic Management
        configs["close_road"] = InterventionConfig(
            name="Dynamic Traffic Rerouting",
            target_nodes=["TrafficCongestion"],
            base_alpha=4.0,  # Usually very effective
            base_beta=1.0,
            context_modifiers={
                "Accident": {"True": 0.6, "False": 1.0},              # Less effective during accidents
                "ConstructionActivity": {"High": 0.5, "Low": 1.0},    # Construction complicates routing
                "time_of_day": {"rush_hour": 0.7, "off_peak": 1.2},   # Time-dependent effectiveness
                "weather": {"poor_visibility": 0.8, "clear": 1.0}     # Weather impacts compliance
            },
            min_effectiveness=0.2,
            max_effectiveness=0.85,
            effectiveness_decay_rate=0.15,  # Faster decay as people find alternatives
            max_duration_hours=12.0,
            deployment_cost=25.0,
            resource_requirements={"traffic_officers": 2, "digital_signs": 4},
            synergy_partners=["deploy_pump"],      # Prevents flood-related traffic issues
            conflict_partners=["emergency_routing"] # Can't conflict with emergency access
        )

        # Emergency Response Optimization
        configs["dispatch_ambulance"] = InterventionConfig(
            name="Enhanced Emergency Response",
            target_nodes=["EmergencyDelay"],
            base_alpha=5.0,  # Generally very reliable
            base_beta=0.8,
            context_modifiers={
                "TrafficCongestion": {"High": 0.6, "Medium": 0.8, "Low": 1.0},  # Traffic severely impacts response
                "Flooding": {"True": 0.4, "False": 1.0},                         # Flooding makes access difficult
                "Accident": {"True": 1.3, "False": 1.0},                        # More effective when needed for accidents
                "time_of_day": {"night": 1.1, "day": 0.9}                       # Night has less traffic but worse visibility
            },
            min_effectiveness=0.3,
            max_effectiveness=0.95,
            effectiveness_decay_rate=0.3,   # High decay - emergency response is immediate
            max_duration_hours=4.0,
            deployment_cost=50.0,
            resource_requirements={"ambulance_units": 1, "medical_crew": 3},
            synergy_partners=["close_road"],        # Traffic management helps emergency access
            conflict_partners=[]                    # Emergency services have priority
        )

        # Advanced Flood Barriers
        configs["deploy_barriers"] = InterventionConfig(
            name="Temporary Flood Barriers",
            target_nodes=["Flooding", "DrainageCapacity"],
            base_alpha=2.5,
            base_beta=1.5,  # More variable effectiveness
            context_modifiers={
                "Rainfall": {"High": 1.2, "Medium": 1.0, "Low": 0.8},  # More effective in actual floods
                "terrain": {"flat": 1.1, "sloped": 0.9, "urban": 0.8}  # Terrain dependency
            },
            min_effectiveness=0.1,
            max_effectiveness=0.8,
            effectiveness_decay_rate=0.08,
            max_duration_hours=72.0,
            deployment_cost=150.0,
            resource_requirements={"barrier_units": 10, "deployment_crew": 5},
            synergy_partners=["deploy_pump"],
            conflict_partners=["pedestrian_access"]
        )

        return configs

    def calculate_intervention_effectiveness(self,
                                           intervention_type: str,
                                           current_evidence: Dict[str, str],
                                           environmental_context: Dict[str, Any] = None,
                                           active_interventions: List[str] = None) -> Dict[str, float]:
        """
        Calculate probabilistic effectiveness for each target node

        Args:
            intervention_type: Type of intervention ("deploy_pump", etc.)
            current_evidence: Current state of the causal graph
            environmental_context: Additional contextual factors
            active_interventions: Currently active interventions for synergy/conflict analysis

        Returns:
            Dict mapping target nodes to effectiveness probabilities
        """

        if intervention_type not in self.intervention_configs:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

        config = self.intervention_configs[intervention_type]
        environmental_context = environmental_context or {}
        active_interventions = active_interventions or []

        effectiveness = {}

        for target_node in config.target_nodes:
            # Start with base Beta distribution parameters
            alpha = config.base_alpha
            beta_param = config.base_beta

            # Apply context modifiers
            context_multiplier = 1.0

            # Evidence-based modifiers
            for evidence_var, evidence_value in current_evidence.items():
                if evidence_var in config.context_modifiers:
                    modifier = config.context_modifiers[evidence_var].get(evidence_value, 1.0)
                    context_multiplier *= modifier

            # Environmental context modifiers
            for context_var, context_value in environmental_context.items():
                if context_var in config.context_modifiers:
                    modifier = config.context_modifiers[context_var].get(str(context_value), 1.0)
                    context_multiplier *= modifier

            # Synergy and conflict effects
            synergy_bonus = 0.0
            conflict_penalty = 0.0

            for active_intervention in active_interventions:
                if active_intervention in config.synergy_partners:
                    synergy_bonus += 0.15  # 15% effectiveness bonus per synergy
                if active_intervention in config.conflict_partners:
                    conflict_penalty += 0.25  # 25% effectiveness penalty per conflict

            # Adjust alpha/beta parameters based on context and interactions
            adjusted_alpha = alpha * (context_multiplier + synergy_bonus - conflict_penalty)
            adjusted_beta = beta_param / max(0.1, context_multiplier + synergy_bonus - conflict_penalty)

            # Ensure parameters stay positive
            adjusted_alpha = max(0.1, adjusted_alpha)
            adjusted_beta = max(0.1, adjusted_beta)

            # Sample from Beta distribution
            effectiveness_sample = np.random.beta(adjusted_alpha, adjusted_beta)

            # Apply bounds
            effectiveness_sample = np.clip(effectiveness_sample,
                                         config.min_effectiveness,
                                         config.max_effectiveness)

            effectiveness[target_node] = effectiveness_sample

        return effectiveness

    def calculate_intervention_synergies(self,
                                       intervention_portfolio: List[str],
                                       current_evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate complex synergy effects between multiple interventions

        Args:
            intervention_portfolio: List of interventions to deploy simultaneously
            current_evidence: Current causal graph state

        Returns:
            Combined effectiveness for target nodes considering all interactions
        """

        if len(intervention_portfolio) <= 1:
            # No synergies for single interventions
            if intervention_portfolio:
                return self.calculate_intervention_effectiveness(intervention_portfolio[0], current_evidence)
            return {}

        # Calculate individual effectiveness for each intervention
        individual_effects = {}
        for intervention in intervention_portfolio:
            effects = self.calculate_intervention_effectiveness(
                intervention,
                current_evidence,
                active_interventions=[i for i in intervention_portfolio if i != intervention]
            )
            individual_effects[intervention] = effects

        # Combine effects using sophisticated aggregation
        combined_effects = {}
        all_target_nodes = set()
        for effects in individual_effects.values():
            all_target_nodes.update(effects.keys())

        for target_node in all_target_nodes:
            node_effects = []
            for intervention, effects in individual_effects.items():
                if target_node in effects:
                    node_effects.append(effects[target_node])

            if len(node_effects) == 1:
                combined_effects[target_node] = node_effects[0]
            elif len(node_effects) > 1:
                # Sophisticated combination: not just multiplication or addition
                # Use a logistic combination that accounts for diminishing returns
                combined_prob = 1.0 - np.prod([1.0 - eff for eff in node_effects])

                # Add interaction bonus for synergistic interventions
                synergy_configs = [self.intervention_configs[intervention] for intervention in intervention_portfolio]
                synergy_count = 0

                for i, config1 in enumerate(synergy_configs):
                    for j, config2 in enumerate(synergy_configs[i+1:], i+1):
                        if intervention_portfolio[j] in config1.synergy_partners:
                            synergy_count += 1

                if synergy_count > 0:
                    # Synergy bonus: up to 20% improvement with diminishing returns
                    synergy_bonus = 0.2 * (1 - np.exp(-synergy_count / 2))
                    combined_prob = min(0.95, combined_prob * (1 + synergy_bonus))

                combined_effects[target_node] = combined_prob

        return combined_effects

    def simulate_temporal_dynamics(self,
                                 intervention_type: str,
                                 deployment_time: datetime,
                                 query_time: datetime) -> float:
        """
        Calculate time-dependent effectiveness decay

        Args:
            intervention_type: Type of intervention
            deployment_time: When intervention was deployed
            query_time: Current time for effectiveness calculation

        Returns:
            Effectiveness multiplier (0.0 to 1.0) based on time decay
        """

        if intervention_type not in self.intervention_configs:
            return 0.0

        config = self.intervention_configs[intervention_type]

        # Calculate hours since deployment
        hours_elapsed = (query_time - deployment_time).total_seconds() / 3600

        # Check if intervention has expired
        if hours_elapsed > config.max_duration_hours:
            return 0.0

        # Apply exponential decay
        decay_multiplier = np.exp(-config.effectiveness_decay_rate * hours_elapsed)

        return decay_multiplier

    def estimate_intervention_costs(self,
                                  intervention_portfolio: List[str],
                                  deployment_duration_hours: float = 24.0) -> Dict[str, Any]:
        """
        Estimate comprehensive costs for intervention portfolio

        Args:
            intervention_portfolio: List of interventions to analyze
            deployment_duration_hours: Expected deployment duration

        Returns:
            Detailed cost breakdown
        """

        total_cost = 0.0
        resource_requirements = {}
        individual_costs = {}

        for intervention in intervention_portfolio:
            if intervention not in self.intervention_configs:
                continue

            config = self.intervention_configs[intervention]

            # Base deployment cost
            base_cost = config.deployment_cost

            # Duration-based cost scaling
            duration_multiplier = 1.0 + (deployment_duration_hours / 24.0) * 0.2  # 20% cost increase per day
            intervention_cost = base_cost * duration_multiplier

            individual_costs[intervention] = intervention_cost
            total_cost += intervention_cost

            # Aggregate resource requirements
            for resource, amount in config.resource_requirements.items():
                if resource in resource_requirements:
                    resource_requirements[resource] += amount
                else:
                    resource_requirements[resource] = amount

        # Calculate resource conflict penalties
        conflict_penalty = 0.0
        resource_conflicts = {}

        # Simplified resource conflict detection
        for resource, total_needed in resource_requirements.items():
            if resource == "maintenance_crew" and total_needed > 5:
                conflict_penalty += (total_needed - 5) * 10.0  # Overtime costs
                resource_conflicts[resource] = f"Requires {total_needed}, optimal capacity is 5"

        return {
            "total_cost": total_cost + conflict_penalty,
            "individual_costs": individual_costs,
            "resource_requirements": resource_requirements,
            "resource_conflicts": resource_conflicts,
            "conflict_penalty": conflict_penalty,
            "duration_multiplier": duration_multiplier
        }

    def analyze_uncertainty_bounds(self,
                                 intervention_type: str,
                                 current_evidence: Dict[str, str],
                                 n_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Monte Carlo analysis of intervention effectiveness uncertainty

        Args:
            intervention_type: Type of intervention to analyze
            current_evidence: Current causal graph state
            n_samples: Number of Monte Carlo samples

        Returns:
            Statistical bounds for effectiveness estimates
        """

        effectiveness_samples = {}

        # Collect samples
        for _ in range(n_samples):
            sample_effectiveness = self.calculate_intervention_effectiveness(
                intervention_type, current_evidence
            )

            for target_node, effectiveness in sample_effectiveness.items():
                if target_node not in effectiveness_samples:
                    effectiveness_samples[target_node] = []
                effectiveness_samples[target_node].append(effectiveness)

        # Calculate statistical bounds
        uncertainty_bounds = {}

        for target_node, samples in effectiveness_samples.items():
            samples_array = np.array(samples)

            uncertainty_bounds[target_node] = {
                "mean": float(np.mean(samples_array)),
                "std": float(np.std(samples_array)),
                "ci_lower": float(np.percentile(samples_array, 5)),   # 90% CI
                "ci_upper": float(np.percentile(samples_array, 95)),
                "min": float(np.min(samples_array)),
                "max": float(np.max(samples_array)),
                "reliability": float(1.0 - np.std(samples_array))  # Simple reliability metric
            }

        return uncertainty_bounds

    def calculate_risk_metrics(self,
                             intervention_type: str,
                             current_evidence: Dict[str, str],
                             confidence_levels: List[float] = None) -> Dict[str, Any]:
        """
        Calculate Value-at-Risk (VaR) and Conditional VaR metrics for intervention effectiveness

        Args:
            intervention_type: Type of intervention to analyze
            current_evidence: Current causal graph state
            confidence_levels: List of confidence levels for VaR calculation (e.g., [0.05, 0.01])

        Returns:
            Risk metrics including VaR, CVaR, and robustness scores
        """
        confidence_levels = confidence_levels or [0.05, 0.01]
        n_samples = 10000  # Large sample for accurate risk estimation

        # Collect effectiveness samples for risk analysis
        effectiveness_samples = {}
        for _ in range(n_samples):
            sample_effectiveness = self.calculate_intervention_effectiveness(
                intervention_type, current_evidence
            )
            for target_node, effectiveness in sample_effectiveness.items():
                if target_node not in effectiveness_samples:
                    effectiveness_samples[target_node] = []
                effectiveness_samples[target_node].append(effectiveness)

        # Calculate risk metrics
        risk_metrics = {
            "var_analysis": {},
            "cvar_analysis": {},
            "robustness_score": 0.0,
            "worst_case_scenario": {},
            "expected_shortfall": {}
        }

        total_robustness = 0.0
        node_count = 0

        for target_node, samples in effectiveness_samples.items():
            samples_array = np.array(samples)
            samples_array = 1.0 - samples_array  # Convert to "risk" (failure probability)

            node_metrics = {
                "var": {},
                "cvar": {},
                "expected_loss": float(np.mean(samples_array)),
                "max_loss": float(np.max(samples_array)),
                "loss_volatility": float(np.std(samples_array))
            }

            # Calculate VaR at different confidence levels
            for confidence_level in confidence_levels:
                var_value = float(np.percentile(samples_array, confidence_level * 100))
                node_metrics["var"][f"{confidence_level:.2f}"] = var_value

                # Calculate Conditional VaR (Expected Shortfall)
                tail_losses = samples_array[samples_array >= var_value]
                cvar_value = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_value
                node_metrics["cvar"][f"{confidence_level:.2f}"] = cvar_value

            risk_metrics["var_analysis"][target_node] = node_metrics["var"]
            risk_metrics["cvar_analysis"][target_node] = node_metrics["cvar"]
            risk_metrics["worst_case_scenario"][target_node] = node_metrics["max_loss"]
            risk_metrics["expected_shortfall"][target_node] = node_metrics["expected_loss"]

            # Calculate robustness score (1 - coefficient of variation)
            cv = node_metrics["loss_volatility"] / max(node_metrics["expected_loss"], 1e-6)
            node_robustness = max(0.0, 1.0 - cv)
            total_robustness += node_robustness
            node_count += 1

        # Overall robustness score
        risk_metrics["robustness_score"] = total_robustness / max(node_count, 1)

        # Add interpretation
        risk_metrics["interpretation"] = {
            "robustness_level": "high" if risk_metrics["robustness_score"] > 0.7
                               else "medium" if risk_metrics["robustness_score"] > 0.4
                               else "low",
            "recommendation": self._generate_risk_recommendation(risk_metrics)
        }

        return risk_metrics

    def _generate_risk_recommendation(self, risk_metrics: Dict[str, Any]) -> str:
        """Generate human-readable risk recommendation based on metrics"""
        robustness = risk_metrics["robustness_score"]

        if robustness > 0.8:
            return "Intervention shows high reliability with low risk of failure. Recommended for deployment."
        elif robustness > 0.6:
            return "Intervention shows good reliability with moderate risk. Consider backup measures."
        elif robustness > 0.4:
            return "Intervention has significant uncertainty. Recommend additional risk mitigation."
        else:
            return "High risk intervention with substantial uncertainty. Consider alternative approaches."

# Convenience functions for integration

def get_intervention_effectiveness(intervention_type: str,
                                 current_evidence: Dict[str, str],
                                 environmental_context: Dict[str, Any] = None) -> Dict[str, float]:
    """Convenience function to get probabilistic intervention effectiveness"""
    engine = ProbabilisticInterventionEngine()
    return engine.calculate_intervention_effectiveness(
        intervention_type, current_evidence, environmental_context
    )

def analyze_intervention_portfolio(intervention_list: List[str],
                                 current_evidence: Dict[str, str]) -> Dict[str, Any]:
    """Convenience function for comprehensive intervention portfolio analysis"""
    engine = ProbabilisticInterventionEngine()

    return {
        "synergy_effects": engine.calculate_intervention_synergies(intervention_list, current_evidence),
        "cost_analysis": engine.estimate_intervention_costs(intervention_list),
        "uncertainty_bounds": {
            intervention: engine.analyze_uncertainty_bounds(intervention, current_evidence, n_samples=500)
            for intervention in intervention_list
        }
    }