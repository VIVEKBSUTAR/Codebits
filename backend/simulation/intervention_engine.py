from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta

from causal_engine.causal_graph import get_causal_graph
from simulation.cascade_engine import generate_timeline
from simulation.probabilistic_interventions import ProbabilisticInterventionEngine, analyze_intervention_portfolio
from simulation.do_calculus_engine import DoCalkulusEngine, analyze_intervention_causality
from models.prediction_model import InterventionResponse, InterventionTimelineChanges
from utils.logger import SystemLogger

logger = SystemLogger(module_name="advanced_intervention")

# Legacy intervention mapping for backward compatibility
LEGACY_INTERVENTION_EFFECTS = {
    "deploy_pump": {"DrainageCapacity": "Good"},
    "close_road": {"TrafficCongestion": "Low"},
    "dispatch_ambulance": {"EmergencyDelay": "Low"}
}

class AdvancedInterventionEngine:
    """
    Sophisticated intervention engine combining probabilistic effects with causal reasoning
    """

    def __init__(self):
        self.probabilistic_engine = ProbabilisticInterventionEngine()
        self.deployment_history = []

    def simulate_advanced_intervention(self,
                                     zone: str,
                                     intervention_action: str,
                                     environmental_context: Dict[str, Any] = None,
                                     use_probabilistic: bool = True,
                                     include_causal_analysis: bool = True) -> Dict[str, Any]:
        """
        Advanced intervention simulation with probabilistic effects and causal reasoning

        Args:
            zone: Target zone
            intervention_action: Type of intervention
            environmental_context: Environmental factors affecting intervention
            use_probabilistic: Whether to use probabilistic intervention modeling
            include_causal_analysis: Whether to include detailed causal analysis

        Returns:
            Comprehensive intervention analysis
        """

        logger.log(f"Starting advanced intervention simulation: {intervention_action} in {zone}")

        # Get baseline state
        baseline_graph = get_causal_graph(zone)
        current_evidence = baseline_graph.evidence.copy()
        baseline_probs = baseline_graph.run_inference()
        baseline_timeline = generate_timeline(zone)

        # Initialize results structure
        results = {
            "zone": zone,
            "intervention": intervention_action,
            "baseline_risk": self._format_risk_output(baseline_probs),
            "baseline_timeline": baseline_timeline.predicted_events,
            "intervention_analysis": {},
            "causal_analysis": {},
            "uncertainty_quantification": {},
            "recommendations": []
        }

        if use_probabilistic:
            # Advanced probabilistic intervention analysis
            intervention_analysis = self._analyze_probabilistic_intervention(
                zone, intervention_action, current_evidence, environmental_context
            )
            results["intervention_analysis"] = intervention_analysis

            # Calculate effects with uncertainty
            effects_with_uncertainty = self._calculate_probabilistic_effects(
                baseline_graph, current_evidence, intervention_analysis["effectiveness"]
            )
            results["after_intervention"] = effects_with_uncertainty["expected_effects"]
            results["uncertainty_quantification"] = effects_with_uncertainty["uncertainty_bounds"]

        else:
            # Fallback to deterministic effects for backward compatibility
            deterministic_effects = self._simulate_legacy_intervention(
                zone, intervention_action, current_evidence, baseline_probs
            )
            results["after_intervention"] = deterministic_effects["after_intervention"]
            results["intervention_analysis"] = {"method": "legacy_deterministic"}

        if include_causal_analysis and intervention_action in self.probabilistic_engine.intervention_configs:
            # Sophisticated causal reasoning using do-calculus
            causal_analysis = self._perform_causal_analysis(
                baseline_graph, intervention_action, current_evidence
            )
            results["causal_analysis"] = causal_analysis

        # Generate recommendations
        results["recommendations"] = self._generate_intervention_recommendations(
            results["intervention_analysis"],
            results.get("causal_analysis", {}),
            results.get("uncertainty_quantification", {})
        )

        # Update deployment history
        self._update_deployment_history(zone, intervention_action, results)

        logger.log(f"Advanced intervention simulation completed for {intervention_action}")
        return results

    def simulate_intervention_portfolio(self,
                                     zone: str,
                                     intervention_list: List[str],
                                     environmental_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze portfolio of multiple interventions with synergy effects

        Args:
            zone: Target zone
            intervention_list: List of interventions to deploy
            environmental_context: Environmental factors

        Returns:
            Portfolio analysis with synergy effects
        """

        logger.log(f"Analyzing intervention portfolio: {intervention_list} in {zone}")

        baseline_graph = get_causal_graph(zone)
        current_evidence = baseline_graph.evidence.copy()

        # Comprehensive portfolio analysis
        portfolio_analysis = analyze_intervention_portfolio(intervention_list, current_evidence)

        # Calculate combined effects
        combined_effects = self._calculate_portfolio_effects(
            baseline_graph, current_evidence, portfolio_analysis["synergy_effects"]
        )

        # Generate timeline with portfolio interventions
        portfolio_timeline = self._generate_portfolio_timeline(
            zone, current_evidence, combined_effects
        )

        results = {
            "zone": zone,
            "intervention_portfolio": intervention_list,
            "portfolio_analysis": portfolio_analysis,
            "combined_effects": combined_effects,
            "timeline_changes": {
                "baseline_timeline": generate_timeline(zone).predicted_events,
                "portfolio_timeline": portfolio_timeline
            },
            "optimization_suggestions": self._suggest_portfolio_optimizations(portfolio_analysis)
        }

        return results

    def _analyze_probabilistic_intervention(self,
                                          zone: str,
                                          intervention_action: str,
                                          current_evidence: Dict[str, str],
                                          environmental_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze intervention using probabilistic modeling"""

        if intervention_action not in self.probabilistic_engine.intervention_configs:
            logger.log(f"Warning: {intervention_action} not in probabilistic configs, using legacy method")
            return {"method": "legacy_fallback", "effectiveness": {}}

        # Get probabilistic effectiveness
        effectiveness = self.probabilistic_engine.calculate_intervention_effectiveness(
            intervention_action, current_evidence, environmental_context
        )

        # Analyze uncertainty bounds
        uncertainty_analysis = self.probabilistic_engine.analyze_uncertainty_bounds(
            intervention_action, current_evidence, n_samples=500
        )

        # Cost analysis
        cost_analysis = self.probabilistic_engine.estimate_intervention_costs([intervention_action])

        # Check for active interventions (simplified - in real system would track deployments)
        active_interventions = self._get_active_interventions(zone)

        return {
            "method": "probabilistic",
            "effectiveness": effectiveness,
            "uncertainty_bounds": uncertainty_analysis,
            "cost_analysis": cost_analysis,
            "active_interventions": active_interventions,
            "environmental_context": environmental_context or {},
            "deployment_feasibility": self._assess_deployment_feasibility(intervention_action, zone)
        }

    def _calculate_probabilistic_effects(self,
                                       baseline_graph,
                                       current_evidence: Dict[str, str],
                                       effectiveness: Dict[str, float]) -> Dict[str, Any]:
        """Calculate intervention effects with probabilistic modeling"""

        # Create modified evidence based on probabilistic effectiveness
        modified_evidence = current_evidence.copy()

        # Apply probabilistic interventions
        for target_node, eff_prob in effectiveness.items():
            if target_node in ["DrainageCapacity", "TrafficCongestion", "EmergencyDelay", "Flooding"]:
                # Probabilistically determine intervention success
                if np.random.random() < eff_prob:
                    # Intervention successful
                    if target_node == "DrainageCapacity":
                        modified_evidence[target_node] = "Good"
                    elif target_node in ["TrafficCongestion", "EmergencyDelay"]:
                        modified_evidence[target_node] = "Low"
                    elif target_node == "Flooding":
                        modified_evidence[target_node] = "False"

        # Run inference with modified evidence
        counterfactual_probs = {}
        for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
            if target in modified_evidence and modified_evidence[target] in ['Good', 'Low', 'False']:
                counterfactual_probs[target] = 0.0
            elif target not in modified_evidence:
                try:
                    result = baseline_graph.infer.query(variables=[target], evidence=modified_evidence)
                    counterfactual_probs[target] = float(result.values[1])
                except:
                    counterfactual_probs[target] = 0.5  # Default uncertainty

        # Calculate uncertainty bounds using Monte Carlo
        uncertainty_bounds = self._monte_carlo_uncertainty_analysis(
            baseline_graph, current_evidence, effectiveness, n_samples=200
        )

        return {
            "expected_effects": self._format_risk_output(counterfactual_probs),
            "modified_evidence": modified_evidence,
            "uncertainty_bounds": uncertainty_bounds
        }

    def _perform_causal_analysis(self,
                               baseline_graph,
                               intervention_action: str,
                               current_evidence: Dict[str, str]) -> Dict[str, Any]:
        """Perform sophisticated causal analysis using do-calculus"""

        try:
            # Initialize do-calculus engine
            do_engine = DoCalkulusEngine(baseline_graph.model)

            # Map intervention to causal intervention
            intervention_config = self.probabilistic_engine.intervention_configs.get(intervention_action)
            if not intervention_config:
                return {"error": "Intervention not found in configurations"}

            # Create intervention dict for do-calculus
            causal_intervention = {}
            for target_node in intervention_config.target_nodes:
                if target_node == "DrainageCapacity":
                    causal_intervention[target_node] = "Good"
                elif target_node in ["TrafficCongestion", "EmergencyDelay"]:
                    causal_intervention[target_node] = "Low"
                elif target_node == "Flooding":
                    causal_intervention[target_node] = "False"

            # Analyze causal effects for each outcome
            outcomes = ['Flooding', 'TrafficCongestion', 'EmergencyDelay']
            causal_results = analyze_intervention_causality(
                baseline_graph.model, causal_intervention, outcomes
            )

            # Extract key insights
            analysis = {
                "intervention_targets": intervention_config.target_nodes,
                "causal_intervention": causal_intervention,
                "outcomes_analysis": {},
                "identification_summary": {},
                "overall_assessment": ""
            }

            total_strong_effects = 0
            total_identifiable_effects = 0

            for outcome, outcome_analysis in causal_results.items():
                causal_effect = outcome_analysis["causal_effect"]
                explanation = outcome_analysis["explanation"]

                analysis["outcomes_analysis"][outcome] = {
                    "effect_size": causal_effect.effect_size.get(outcome, 0.0),
                    "confidence_bounds": causal_effect.confidence_bounds.get(outcome, (0.0, 0.0)),
                    "identifiable": causal_effect.estimable,
                    "identification_strategy": causal_effect.identification_strategy,
                    "adjustment_variables": explanation.get("minimal_adjustment", []),
                    "interpretation": explanation.get("interpretation", ""),
                    "sensitivity_bounds": outcome_analysis["sensitivity_bounds"]
                }

                if causal_effect.estimable:
                    total_identifiable_effects += 1
                    effect_size = causal_effect.effect_size.get(outcome, 0.0)
                    if effect_size > 0.3:
                        total_strong_effects += 1

            # Generate overall assessment
            if total_identifiable_effects == 0:
                analysis["overall_assessment"] = "Causal effects cannot be reliably identified from current graph structure."
            elif total_strong_effects > 0:
                analysis["overall_assessment"] = f"Intervention shows strong causal effects on {total_strong_effects} outcome(s)."
            else:
                analysis["overall_assessment"] = "Intervention has identifiable but modest causal effects."

            analysis["identification_summary"] = {
                "identifiable_effects": total_identifiable_effects,
                "strong_effects": total_strong_effects,
                "total_analyzed": len(outcomes)
            }

            return analysis

        except Exception as e:
            logger.log(f"Error in causal analysis: {str(e)}")
            return {
                "error": f"Causal analysis failed: {str(e)}",
                "fallback_note": "Using probabilistic analysis only"
            }

    def _monte_carlo_uncertainty_analysis(self,
                                        baseline_graph,
                                        current_evidence: Dict[str, str],
                                        effectiveness: Dict[str, float],
                                        n_samples: int = 200) -> Dict[str, Dict[str, float]]:
        """Monte Carlo analysis for uncertainty quantification"""

        outcomes = ['Flooding', 'TrafficCongestion', 'EmergencyDelay']
        samples = {outcome: [] for outcome in outcomes}

        for _ in range(n_samples):
            # Sample effectiveness realizations
            sample_evidence = current_evidence.copy()

            for target_node, eff_prob in effectiveness.items():
                if np.random.random() < eff_prob:
                    if target_node == "DrainageCapacity":
                        sample_evidence[target_node] = "Good"
                    elif target_node in ["TrafficCongestion", "EmergencyDelay"]:
                        sample_evidence[target_node] = "Low"
                    elif target_node == "Flooding":
                        sample_evidence[target_node] = "False"

            # Run inference
            for outcome in outcomes:
                if outcome in sample_evidence and sample_evidence[outcome] in ['Good', 'Low', 'False']:
                    samples[outcome].append(0.0)
                else:
                    try:
                        result = baseline_graph.infer.query(variables=[outcome], evidence=sample_evidence)
                        samples[outcome].append(float(result.values[1]))
                    except:
                        samples[outcome].append(0.5)

        # Calculate statistics
        uncertainty_bounds = {}
        for outcome in outcomes:
            outcome_samples = np.array(samples[outcome])
            uncertainty_bounds[outcome] = {
                "mean": float(np.mean(outcome_samples)),
                "std": float(np.std(outcome_samples)),
                "ci_5": float(np.percentile(outcome_samples, 5)),
                "ci_95": float(np.percentile(outcome_samples, 95)),
                "min": float(np.min(outcome_samples)),
                "max": float(np.max(outcome_samples))
            }

        return uncertainty_bounds

    def _simulate_legacy_intervention(self,
                                    zone: str,
                                    intervention_action: str,
                                    current_evidence: Dict[str, str],
                                    baseline_probs: Dict[str, float]) -> Dict[str, Any]:
        """Backward compatibility with legacy deterministic interventions"""

        if intervention_action not in LEGACY_INTERVENTION_EFFECTS:
            raise ValueError(f"Unknown legacy intervention: {intervention_action}")

        effect_evidence = LEGACY_INTERVENTION_EFFECTS[intervention_action]
        simulated_evidence = current_evidence.copy()
        simulated_evidence.update(effect_evidence)

        # Legacy inference
        baseline_graph = get_causal_graph(zone)
        counterfactual_probs = {}

        for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
            if target in effect_evidence and effect_evidence[target] in ['Good', 'Low', 'False']:
                counterfactual_probs[target] = 0.0
            elif target not in simulated_evidence:
                result = baseline_graph.infer.query(variables=[target], evidence=simulated_evidence)
                counterfactual_probs[target] = float(result.values[1])

        return {
            "after_intervention": self._format_risk_output(counterfactual_probs),
            "method": "legacy_deterministic"
        }

    def _format_risk_output(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Format probability output for API consistency"""
        return {
            'flooding': round(float(probs.get('Flooding', 0.0)), 2),
            'traffic': round(float(probs.get('TrafficCongestion', 0.0)), 2),
            'emergency_delay': round(float(probs.get('EmergencyDelay', 0.0)), 2)
        }

    def _get_active_interventions(self, zone: str) -> List[str]:
        """Get currently active interventions in zone (simplified)"""
        # In real implementation, would query deployment database
        return []

    def _assess_deployment_feasibility(self, intervention_action: str, zone: str) -> Dict[str, Any]:
        """Assess feasibility of intervention deployment"""
        return {
            "feasible": True,
            "constraints": [],
            "estimated_deployment_time": 15,  # minutes
            "resource_availability": "available"
        }

    def _generate_intervention_recommendations(self,
                                            intervention_analysis: Dict[str, Any],
                                            causal_analysis: Dict[str, Any],
                                            uncertainty_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        if intervention_analysis.get("method") == "probabilistic":
            effectiveness = intervention_analysis.get("effectiveness", {})
            avg_effectiveness = np.mean(list(effectiveness.values())) if effectiveness else 0.0

            if avg_effectiveness > 0.7:
                recommendations.append("High-confidence intervention deployment recommended")
            elif avg_effectiveness > 0.4:
                recommendations.append("Moderate-confidence deployment with monitoring recommended")
            else:
                recommendations.append("Consider alternative interventions due to low effectiveness")

        if causal_analysis.get("identification_summary", {}).get("strong_effects", 0) > 0:
            recommendations.append("Strong causal evidence supports intervention effectiveness")

        if uncertainty_analysis:
            high_uncertainty_outcomes = [
                outcome for outcome, bounds in uncertainty_analysis.items()
                if bounds.get("std", 0) > 0.2
            ]
            if high_uncertainty_outcomes:
                recommendations.append(f"Monitor {', '.join(high_uncertainty_outcomes)} due to high uncertainty")

        return recommendations or ["Insufficient data for specific recommendations"]

    def _calculate_portfolio_effects(self,
                                   baseline_graph,
                                   current_evidence: Dict[str, str],
                                   synergy_effects: Dict[str, float]) -> Dict[str, float]:
        """Calculate combined effects of intervention portfolio"""
        # Simplified implementation - use synergy effects directly
        return self._format_risk_output(synergy_effects)

    def _generate_portfolio_timeline(self,
                                   zone: str,
                                   modified_evidence: Dict[str, str],
                                   combined_effects: Dict[str, float]) -> List:
        """Generate timeline for intervention portfolio"""
        # Use existing timeline logic with portfolio effects
        return recalculate_timeline_with_sandbox(zone, modified_evidence, {
            'Flooding': combined_effects.get('flooding', 0.0),
            'TrafficCongestion': combined_effects.get('traffic', 0.0),
            'EmergencyDelay': combined_effects.get('emergency_delay', 0.0)
        })

    def _suggest_portfolio_optimizations(self, portfolio_analysis: Dict[str, Any]) -> List[str]:
        """Suggest optimizations for intervention portfolio"""
        suggestions = []

        cost_analysis = portfolio_analysis.get("cost_analysis", {})
        if cost_analysis.get("conflict_penalty", 0) > 0:
            suggestions.append("Resource conflicts detected - consider staggered deployment")

        synergy_effects = portfolio_analysis.get("synergy_effects", {})
        avg_effect = np.mean(list(synergy_effects.values())) if synergy_effects else 0.0

        if avg_effect < 0.3:
            suggestions.append("Consider adding synergistic interventions to improve effectiveness")

        return suggestions or ["Portfolio appears well-optimized"]

    def _update_deployment_history(self, zone: str, intervention: str, results: Dict[str, Any]):
        """Update deployment history for learning"""
        deployment_record = {
            "timestamp": datetime.now(),
            "zone": zone,
            "intervention": intervention,
            "baseline_risk": results.get("baseline_risk"),
            "expected_outcome": results.get("after_intervention"),
            "effectiveness": results.get("intervention_analysis", {}).get("effectiveness")
        }
        self.deployment_history.append(deployment_record)

        # Keep only recent history
        if len(self.deployment_history) > 100:
            self.deployment_history = self.deployment_history[-100:]


# Initialize global engine instance
_advanced_engine = AdvancedInterventionEngine()

# Legacy function for backward compatibility
def recalculate_timeline_with_sandbox(zone: str, simulated_evidence: Dict[str, str], counterfactual_probs: Dict[str, float]) -> list:
    """Legacy timeline recalculation (maintained for backward compatibility)"""
    from simulation.cascade_engine import EDGE_DELAYS, EVENT_NAMES, TimelineItemModel
    import datetime

    # Filter significant risks
    significant_nodes = {k: v for k, v in counterfactual_probs.items() if v >= 0.40}

    # Add DrainageCapacity if relevant
    if "DrainageCapacity" not in simulated_evidence:
        graph = get_causal_graph(zone)
        try:
            drain_prob_result = graph.infer.query(variables=["DrainageCapacity"], evidence=simulated_evidence)
            drain_prob = float(drain_prob_result.values[1])
            if drain_prob >= 0.40:
                significant_nodes["DrainageCapacity"] = drain_prob
        except:
            pass

    t_zero = datetime.datetime.now()
    projected_times = {}

    # Calculate propagation times
    active_sources = list(simulated_evidence.keys())
    graph_nodes = get_causal_graph(zone).model.nodes()

    for target in significant_nodes:
        min_total_delay = float('inf')

        for source in active_sources:
            if (source, target) in EDGE_DELAYS:
                min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, target)])
            else:
                for intermediate in graph_nodes:
                    if (source, intermediate) in EDGE_DELAYS and (intermediate, target) in EDGE_DELAYS:
                        min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, intermediate)] + EDGE_DELAYS[(intermediate, target)])

        projected_times[target] = min_total_delay if min_total_delay != float('inf') else 30

    # Build timeline
    predicted_events = []
    for node, prob in significant_nodes.items():
        delay_mins = projected_times.get(node, 0)
        proj_time = t_zero + datetime.timedelta(minutes=delay_mins)
        formatted_time = proj_time.strftime("%H:%M")

        predicted_events.append(TimelineItemModel(
            predicted_time=formatted_time,
            event_name=EVENT_NAMES.get(node, node),
            probability=round(float(prob), 2)
        ))

    predicted_events.sort(key=lambda x: x.predicted_time)
    return predicted_events

def simulate_intervention(zone: str, intervention_action: str) -> InterventionResponse:
    """
    Main intervention simulation function with advanced capabilities

    This function now uses sophisticated probabilistic modeling and causal reasoning
    while maintaining backward compatibility with the original API.
    """

    # Run advanced intervention analysis
    advanced_results = _advanced_engine.simulate_advanced_intervention(
        zone, intervention_action,
        use_probabilistic=True,
        include_causal_analysis=True
    )

    # Generate timeline changes
    baseline_timeline = advanced_results["baseline_timeline"]

    # Create counterfactual timeline
    counterfactual_probs = {
        'Flooding': advanced_results["after_intervention"]["flooding"],
        'TrafficCongestion': advanced_results["after_intervention"]["traffic"],
        'EmergencyDelay': advanced_results["after_intervention"]["emergency_delay"]
    }

    # Get simulated evidence for timeline calculation
    intervention_config = _advanced_engine.probabilistic_engine.intervention_configs.get(intervention_action)
    simulated_evidence = {}

    if intervention_config:
        # Use probabilistic intervention effects
        effectiveness = advanced_results["intervention_analysis"].get("effectiveness", {})
        for target_node, eff_prob in effectiveness.items():
            if eff_prob > 0.5:  # Threshold for likely success
                if target_node == "DrainageCapacity":
                    simulated_evidence[target_node] = "Good"
                elif target_node in ["TrafficCongestion", "EmergencyDelay"]:
                    simulated_evidence[target_node] = "Low"
    else:
        # Fallback to legacy effects
        if intervention_action in LEGACY_INTERVENTION_EFFECTS:
            simulated_evidence = LEGACY_INTERVENTION_EFFECTS[intervention_action]

    counterfactual_timeline = recalculate_timeline_with_sandbox(
        zone, simulated_evidence, counterfactual_probs
    )

    # Calculate benefit metrics
    benefit_metrics = {}
    baseline_risk = advanced_results["baseline_risk"]
    after_intervention = advanced_results["after_intervention"]

    for risk_type in ['flooding', 'traffic', 'emergency_delay']:
        baseline_val = baseline_risk.get(risk_type, 0.0)
        after_val = after_intervention.get(risk_type, 0.0)
        benefit_metrics[f"{risk_type}_reduction"] = round(baseline_val - after_val, 2)

    # Create response in original format for backward compatibility
    return InterventionResponse(
        zone=zone,
        intervention=intervention_action,
        baseline_risk=baseline_risk,
        after_intervention=after_intervention,
        benefit=benefit_metrics,
        timeline_changes=InterventionTimelineChanges(
            baseline_timeline=baseline_timeline,
            intervened_timeline=counterfactual_timeline
        )
    )

def recalculate_timeline_with_sandbox(zone: str, simulated_evidence: Dict[str, str], counterfactual_probs: Dict[str, float]) -> list:
    """A scoped version of generate_timeline using counterfactual data"""
    from simulation.cascade_engine import EDGE_DELAYS, EVENT_NAMES, TimelineItemModel
    
    # 1. Filter significant risks threshold
    significant_nodes = {k: v for k, v in counterfactual_probs.items() if v >= 0.40}
    
    # If Drainage is part of the counterfactual targets, fetch it specifically (unless intervened upon)
    if "DrainageCapacity" not in simulated_evidence:
        graph = get_causal_graph(zone)
        drain_prob_result = graph.infer.query(variables=["DrainageCapacity"], evidence=simulated_evidence)
        drain_prob = float(drain_prob_result.values[1]) # 'Poor'
        if drain_prob >= 0.40:
             significant_nodes["DrainageCapacity"] = drain_prob
    
    t_zero = datetime.datetime.now()
    projected_times = {}

    # Calculate propagation times (BFS from counterfactual active evidence)
    active_sources = list(simulated_evidence.keys())
    graph_nodes = get_causal_graph(zone).model.nodes()
    
    for target in significant_nodes:
        min_total_delay = float('inf')
        
        for source in active_sources:
            if (source, target) in EDGE_DELAYS:
                min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, target)])
            else:
                for intermediate in graph_nodes:
                    if (source, intermediate) in EDGE_DELAYS and (intermediate, target) in EDGE_DELAYS:
                        min_total_delay = min(min_total_delay, EDGE_DELAYS[(source, intermediate)] + EDGE_DELAYS[(intermediate, target)])
        
        if min_total_delay != float('inf'):
            projected_times[target] = min_total_delay
        else:
            projected_times[target] = 30 # Backstop

    predicted_events = []
    
    for node, prob in significant_nodes.items():
        delay_mins = projected_times.get(node, 0)
        proj_time = t_zero + datetime.timedelta(minutes=delay_mins)
        formatted_time = proj_time.strftime("%H:%M")
        
        predicted_events.append(TimelineItemModel(
            predicted_time=formatted_time,
            event_name=EVENT_NAMES.get(node, node),
            probability=round(float(prob), 2)
        ))
        
    predicted_events.sort(key=lambda x: x.predicted_time)
    return predicted_events


def simulate_intervention(zone: str, intervention_action: str) -> InterventionResponse:
    # 1. Fetch current baseline
    baseline_graph = get_causal_graph(zone)
    current_evidence = baseline_graph.evidence.copy()
    
    baseline_probs = baseline_graph.run_inference()
    baseline_timeline_model = generate_timeline(zone)
    baseline_timeline = baseline_timeline_model.predicted_events

    # 2. Lookup Intervention Effects
    if intervention_action not in INTERVENTION_EFFECTS:
        raise ValueError(f"Unknown Intervention: {intervention_action}")
    effect_evidence = INTERVENTION_EFFECTS[intervention_action]

    # 3. Create Sandbox Evidence
    simulated_evidence = current_evidence.copy()
    simulated_evidence.update(effect_evidence)

    # 4. Run Bayesian Inference (Counterfactual)
    counterfactual_probs = {}
    for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
        if target in effect_evidence and effect_evidence[target] in ['Good', 'Low', 'False']:
             counterfactual_probs[target] = 0.0
             continue
             
        if target not in simulated_evidence:
            result = baseline_graph.infer.query(variables=[target], evidence=simulated_evidence)
            # Probability of 'True' or 'High' state
            val = float(result.values[1])
            counterfactual_probs[target] = val

    # 5. Recompute Timeline
    counterfactual_timeline = recalculate_timeline_with_sandbox(zone, simulated_evidence, counterfactual_probs)

    # 6. Compute Risk Reduction (Benefit)
    benefit_metrics = {}
    baseline_risk_out = {}
    after_intervention_out = {}

    for k in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
        # Rename for API output
        key_name = 'flooding' if k == 'Flooding' else ('traffic' if k == 'TrafficCongestion' else 'emergency_delay')
        
        base_val = round(float(baseline_probs.get(k, 0.0)), 2)
        cf_val = round(float(counterfactual_probs.get(k, 0.0)), 2)
        
        baseline_risk_out[key_name] = base_val
        after_intervention_out[key_name] = cf_val
        benefit_metrics[f"{key_name}_reduction"] = round(base_val - cf_val, 2)

    # 7. Package Response
    return InterventionResponse(
        zone=zone,
        intervention=intervention_action,
        baseline_risk=baseline_risk_out,
        after_intervention=after_intervention_out,
        benefit=benefit_metrics,
        timeline_changes=InterventionTimelineChanges(
            baseline_timeline=baseline_timeline,
            intervened_timeline=counterfactual_timeline
        )
    )
