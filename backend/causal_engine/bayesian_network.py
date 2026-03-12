"""
Implements probability propagation and evidence injection for the Bayesian Network.

Provides:
- BayesianNetworkEngine: core inference, evidence handling, and CPD management
- CPDManager: create, update, validate, and learn Conditional Probability Distributions
- NetworkStructure: DAG definition, topological utilities, d-separation checks
- Helper functions: default network factory, state mappings, validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.independencies import Independencies

from utils.logger import SystemLogger

logger = SystemLogger(module_name="bayesian_network")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_INDEX: Dict[str, int] = {
    "Low": 0, "High": 1,
    "False": 0, "True": 1,
    "Good": 0, "Poor": 1,
}

NODE_STATE_NAMES: Dict[str, List[str]] = {
    "Rainfall": ["Low", "High"],
    "ConstructionActivity": ["Low", "High"],
    "Accident": ["False", "True"],
    "DrainageCapacity": ["Good", "Poor"],
    "Flooding": ["False", "True"],
    "TrafficCongestion": ["Low", "High"],
    "EmergencyDelay": ["Low", "High"],
}

DEFAULT_EDGES: List[Tuple[str, str]] = [
    ("Rainfall", "DrainageCapacity"),
    ("DrainageCapacity", "Flooding"),
    ("Rainfall", "Flooding"),
    ("Flooding", "TrafficCongestion"),
    ("ConstructionActivity", "TrafficCongestion"),
    ("Accident", "TrafficCongestion"),
    ("TrafficCongestion", "EmergencyDelay"),
]

EVENT_TO_NODE: Dict[str, str] = {
    "rainfall": "Rainfall",
    "construction": "ConstructionActivity",
    "accident": "Accident",
    "flood": "Flooding",
    "drainage_failure": "Flooding",
    "traffic": "TrafficCongestion",
}

# Nodes whose "high‑risk" state is True/False rather than Low/High
BOOLEAN_NODES: Set[str] = {"Accident", "Flooding"}


# ---------------------------------------------------------------------------
# NetworkStructure – DAG definition and topological helpers
# ---------------------------------------------------------------------------

class NetworkStructure:
    """Manages the directed acyclic graph topology."""

    def __init__(self, edges: Optional[List[Tuple[str, str]]] = None):
        self.edges = list(edges or DEFAULT_EDGES)
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # parent → children
        self._parents: Dict[str, List[str]] = defaultdict(list)
        self._nodes: Set[str] = set()
        self._build(self.edges)

    # ---- construction ----

    def _build(self, edges: List[Tuple[str, str]]) -> None:
        for parent, child in edges:
            self._adjacency[parent].append(child)
            self._parents[child].append(parent)
            self._nodes.update([parent, child])

    @property
    def nodes(self) -> List[str]:
        return sorted(self._nodes)

    def parents(self, node: str) -> List[str]:
        return list(self._parents.get(node, []))

    def children(self, node: str) -> List[str]:
        return list(self._adjacency.get(node, []))

    def root_nodes(self) -> List[str]:
        return [n for n in self.nodes if not self._parents.get(n)]

    def leaf_nodes(self) -> List[str]:
        return [n for n in self.nodes if not self._adjacency.get(n)]

    # ---- topological sort ----

    def topological_order(self) -> List[str]:
        visited: Set[str] = set()
        order: List[str] = []

        def _dfs(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            for child in self._adjacency.get(n, []):
                _dfs(child)
            order.append(n)

        for node in self.nodes:
            _dfs(node)
        order.reverse()
        return order

    # ---- d-separation ----

    def ancestors(self, node: str) -> Set[str]:
        result: Set[str] = set()
        stack = list(self._parents.get(node, []))
        while stack:
            n = stack.pop()
            if n not in result:
                result.add(n)
                stack.extend(self._parents.get(n, []))
        return result

    def d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """
        Check d-separation between *x* and *y* given evidence set *z*
        using the Bayes-Ball algorithm.
        """
        if x == y:
            return False

        z = set(z)
        # Determine ancestors of Z for explaining-away
        z_ancestors: Set[str] = set()
        for node in z:
            z_ancestors |= self.ancestors(node)
            z_ancestors.add(node)

        # BFS / Bayes-Ball reachability
        reachable: Set[Tuple[str, str]] = set()  # (node, direction)
        queue: deque = deque()

        # Start from x going both directions
        queue.append((x, "up"))
        queue.append((x, "down"))

        while queue:
            current, direction = queue.popleft()
            if (current, direction) in reachable:
                continue
            reachable.add((current, direction))

            if current == y:
                return False  # y is reachable → not d-separated

            if direction == "up" and current not in z:
                # Pass through non-evidence node going up: continue to parents & children
                for parent in self._parents.get(current, []):
                    queue.append((parent, "up"))
                for child in self._adjacency.get(current, []):
                    queue.append((child, "down"))

            elif direction == "down":
                if current not in z:
                    # Non-evidence: pass message to children
                    for child in self._adjacency.get(current, []):
                        queue.append((child, "down"))
                if current in z or current in z_ancestors:
                    # Evidence / ancestor-of-evidence: explaining away
                    for parent in self._parents.get(current, []):
                        queue.append((parent, "up"))

        return True  # y not reachable → d-separated

    def markov_blanket(self, node: str) -> Set[str]:
        """Parents + children + co-parents of node."""
        blanket: Set[str] = set()
        blanket.update(self._parents.get(node, []))
        for child in self._adjacency.get(node, []):
            blanket.add(child)
            blanket.update(self._parents.get(child, []))
        blanket.discard(node)
        return blanket

    def summary(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "root_nodes": self.root_nodes(),
            "leaf_nodes": self.leaf_nodes(),
            "topological_order": self.topological_order(),
        }


# ---------------------------------------------------------------------------
# CPDManager – create / validate / perturb CPDs
# ---------------------------------------------------------------------------

class CPDManager:
    """Factory and manager for Conditional Probability Distributions."""

    def __init__(self, structure: Optional[NetworkStructure] = None):
        self.structure = structure or NetworkStructure()

    @staticmethod
    def state_names_for(node: str) -> Dict[str, List[str]]:
        """Return state-name mapping suitable for a TabularCPD."""
        return {node: NODE_STATE_NAMES.get(node, ["Low", "High"])}

    @staticmethod
    def full_state_names(node: str, parents: List[str]) -> Dict[str, List[str]]:
        names: Dict[str, List[str]] = {}
        names[node] = NODE_STATE_NAMES.get(node, ["Low", "High"])
        for p in parents:
            names[p] = NODE_STATE_NAMES.get(p, ["Low", "High"])
        return names

    # ---- default CPDs ----

    def create_default_cpds(self) -> List[TabularCPD]:
        """Return the set of hardcoded CPDs for the 7-node city network."""
        cpds: List[TabularCPD] = []

        # Root nodes
        cpds.append(TabularCPD(
            variable="Rainfall", variable_card=2,
            values=[[0.8], [0.2]],
            state_names={"Rainfall": ["Low", "High"]},
        ))
        cpds.append(TabularCPD(
            variable="ConstructionActivity", variable_card=2,
            values=[[0.9], [0.1]],
            state_names={"ConstructionActivity": ["Low", "High"]},
        ))
        cpds.append(TabularCPD(
            variable="Accident", variable_card=2,
            values=[[0.95], [0.05]],
            state_names={"Accident": ["False", "True"]},
        ))

        # P(DrainageCapacity | Rainfall)
        cpds.append(TabularCPD(
            variable="DrainageCapacity", variable_card=2,
            values=[[0.9, 0.4], [0.1, 0.6]],
            evidence=["Rainfall"], evidence_card=[2],
            state_names={"DrainageCapacity": ["Good", "Poor"], "Rainfall": ["Low", "High"]},
        ))

        # P(Flooding | Rainfall, DrainageCapacity)
        cpds.append(TabularCPD(
            variable="Flooding", variable_card=2,
            values=[[0.95, 0.85, 0.70, 0.25],
                    [0.05, 0.15, 0.30, 0.75]],
            evidence=["Rainfall", "DrainageCapacity"], evidence_card=[2, 2],
            state_names={
                "Flooding": ["False", "True"],
                "Rainfall": ["Low", "High"],
                "DrainageCapacity": ["Good", "Poor"],
            },
        ))

        # P(TrafficCongestion | Flooding, ConstructionActivity, Accident)
        cpds.append(TabularCPD(
            variable="TrafficCongestion", variable_card=2,
            values=[
                [0.9, 0.3, 0.4, 0.1, 0.2, 0.05, 0.1, 0.01],
                [0.1, 0.7, 0.6, 0.9, 0.8, 0.95, 0.9, 0.99],
            ],
            evidence=["Flooding", "ConstructionActivity", "Accident"],
            evidence_card=[2, 2, 2],
            state_names={
                "TrafficCongestion": ["Low", "High"],
                "Flooding": ["False", "True"],
                "ConstructionActivity": ["Low", "High"],
                "Accident": ["False", "True"],
            },
        ))

        # P(EmergencyDelay | TrafficCongestion)
        cpds.append(TabularCPD(
            variable="EmergencyDelay", variable_card=2,
            values=[[0.95, 0.2], [0.05, 0.8]],
            evidence=["TrafficCongestion"], evidence_card=[2],
            state_names={
                "EmergencyDelay": ["Low", "High"],
                "TrafficCongestion": ["Low", "High"],
            },
        ))

        return cpds

    # ---- CPD utilities ----

    @staticmethod
    def validate_cpd(cpd: TabularCPD, tolerance: float = 1e-6) -> Dict[str, Any]:
        """Check that a CPD's columns each sum to 1."""
        values = cpd.get_values()
        col_sums = values.sum(axis=0)
        valid = bool(np.allclose(col_sums, 1.0, atol=tolerance))
        return {
            "variable": cpd.variable,
            "valid": valid,
            "column_sums": col_sums.tolist(),
            "max_deviation": float(np.max(np.abs(col_sums - 1.0))),
        }

    @staticmethod
    def perturb_cpd(cpd: TabularCPD, noise_scale: float = 0.05,
                    rng: Optional[np.random.Generator] = None) -> TabularCPD:
        """
        Return a new CPD whose probabilities are slightly perturbed
        (useful for sensitivity analysis).
        """
        rng = rng or np.random.default_rng()
        values = cpd.get_values().copy().astype(float)
        noise = rng.normal(0, noise_scale, size=values.shape)
        values = np.clip(values + noise, 1e-6, None)
        values /= values.sum(axis=0, keepdims=True)  # re-normalise

        evidence = list(cpd.variables[1:]) if len(cpd.variables) > 1 else None
        evidence_card = list(cpd.cardinality[1:]) if evidence else None

        return TabularCPD(
            variable=cpd.variable,
            variable_card=int(cpd.variable_card),
            values=values.tolist(),
            evidence=evidence,
            evidence_card=[int(c) for c in evidence_card] if evidence_card else None,
            state_names=cpd.state_names,
        )

    @staticmethod
    def cpd_to_dict(cpd: TabularCPD) -> Dict[str, Any]:
        """Serialise a CPD to a plain dictionary."""
        return {
            "variable": cpd.variable,
            "variable_card": int(cpd.variable_card),
            "values": cpd.get_values().tolist(),
            "evidence": list(cpd.variables[1:]) if len(cpd.variables) > 1 else [],
            "evidence_card": [int(c) for c in cpd.cardinality[1:]] if len(cpd.variables) > 1 else [],
            "state_names": {k: list(v) for k, v in cpd.state_names.items()},
        }


# ---------------------------------------------------------------------------
# BayesianNetworkEngine – core propagation / evidence / query
# ---------------------------------------------------------------------------

class BayesianNetworkEngine:
    """
    High-level Bayesian Network engine wrapping pgmpy.

    Responsibilities:
    - Build and validate the model
    - Inject / retract / manage evidence
    - Run exact inference (Variable Elimination or Belief Propagation)
    - Marginal, MAP, conditional queries
    - Sensitivity analysis via CPD perturbation
    - Full network diagnostic reporting
    """

    def __init__(
        self,
        edges: Optional[List[Tuple[str, str]]] = None,
        cpds: Optional[List[TabularCPD]] = None,
    ):
        self.structure = NetworkStructure(edges)
        self.cpd_manager = CPDManager(self.structure)

        # Build pgmpy model
        self.model = DiscreteBayesianNetwork(self.structure.edges)
        for cpd in (cpds or self.cpd_manager.create_default_cpds()):
            self.model.add_cpds(cpd)
        self.model.check_model()

        # Inference engines (lazy-initialised after evidence changes)
        self._ve: Optional[VariableElimination] = None
        self._bp: Optional[BeliefPropagation] = None

        # Evidence state
        self.evidence: Dict[str, str] = {}
        self._evidence_history: List[Dict[str, Any]] = []

        logger.log(
            f"BayesianNetworkEngine initialised: "
            f"{len(self.structure.nodes)} nodes, {len(self.structure.edges)} edges"
        )

    # ---- model access helpers ----

    @property
    def nodes(self) -> List[str]:
        return self.structure.nodes

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return self.structure.edges

    def get_cpd(self, node: str) -> TabularCPD:
        return self.model.get_cpds(node)

    def get_all_cpds(self) -> List[TabularCPD]:
        return list(self.model.get_cpds())

    # ---- inference engine management ----

    def _get_ve(self) -> VariableElimination:
        if self._ve is None:
            self._ve = VariableElimination(self.model)
        return self._ve

    def _get_bp(self) -> BeliefPropagation:
        if self._bp is None:
            self._bp = BeliefPropagation(self.model)
        return self._bp

    def _invalidate_inference_cache(self) -> None:
        """Call after any CPD or structure change."""
        self._ve = None
        self._bp = None

    # ---- evidence management ----

    def set_evidence(self, node: str, state: str) -> None:
        """Inject a single piece of evidence."""
        if node not in self.structure._nodes:
            logger.log(f"Ignoring evidence for unknown node: {node}")
            return
        valid_states = NODE_STATE_NAMES.get(node, [])
        if valid_states and state not in valid_states:
            logger.log(f"Invalid state '{state}' for node {node}, expected {valid_states}")
            return
        self.evidence[node] = state
        self._evidence_history.append({
            "action": "set", "node": node, "state": state,
            "timestamp": datetime.now().isoformat(),
        })

    def set_evidence_bulk(self, evidence: Dict[str, str]) -> None:
        for node, state in evidence.items():
            self.set_evidence(node, state)

    def retract_evidence(self, node: str) -> None:
        """Remove evidence for a single node."""
        if node in self.evidence:
            del self.evidence[node]
            self._evidence_history.append({
                "action": "retract", "node": node,
                "timestamp": datetime.now().isoformat(),
            })

    def reset_evidence(self) -> None:
        self.evidence.clear()
        self._evidence_history.append({
            "action": "reset",
            "timestamp": datetime.now().isoformat(),
        })

    def process_event(self, event_type: str, severity: str) -> Optional[str]:
        """
        Map an incoming event (e.g. 'rainfall', 'high') to Bayesian evidence.

        Returns the node name that was updated, or None if the event is unmapped.
        """
        node_name = EVENT_TO_NODE.get(event_type)
        if node_name is None:
            return None

        if node_name in BOOLEAN_NODES:
            state = "True" if severity in ("medium", "high") else "False"
        else:
            state = "High" if severity in ("medium", "high") else "Low"

        self.set_evidence(node_name, state)
        return node_name

    # ---- queries ----

    def query(
        self,
        variables: List[str],
        evidence: Optional[Dict[str, str]] = None,
        method: str = "ve",
    ) -> Dict[str, np.ndarray]:
        """
        Compute posterior marginals for *variables* given *evidence*.

        Args:
            variables: target query variables
            evidence: optional evidence override (defaults to self.evidence)
            method: 've' for Variable Elimination, 'bp' for Belief Propagation

        Returns:
            {variable_name: probability_array}  (index 0 = Low/False/Good, 1 = High/True/Poor)
        """
        ev = dict(evidence) if evidence is not None else dict(self.evidence)
        # Remove query variables from evidence to avoid pgmpy errors
        ev = {k: v for k, v in ev.items() if k not in variables}

        results: Dict[str, np.ndarray] = {}
        infer = self._get_bp() if method == "bp" else self._get_ve()

        for var in variables:
            try:
                factor = infer.query([var], evidence=ev)
                results[var] = factor.values.copy()
            except Exception as e:
                logger.log(f"Query failed for {var}: {e}")
                # Uniform fallback
                card = int(self.model.get_cpds(var).variable_card)
                results[var] = np.full(card, 1.0 / card)

        return results

    def query_risk(
        self,
        evidence: Optional[Dict[str, str]] = None,
        method: str = "ve",
    ) -> Dict[str, float]:
        """
        Return P(High/True) for the three risk targets:
        Flooding, TrafficCongestion, EmergencyDelay.
        """
        ev = evidence if evidence is not None else self.evidence
        risk_targets = ["Flooding", "TrafficCongestion", "EmergencyDelay"]
        posteriors: Dict[str, float] = {}

        for target in risk_targets:
            if target in ev:
                posteriors[target] = 1.0 if ev[target] in ("High", "True") else 0.0
            else:
                marginals = self.query([target], evidence=ev, method=method)
                posteriors[target] = float(marginals[target][1])

        return posteriors

    def map_query(
        self,
        variables: List[str],
        evidence: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Return the most-probable state for each variable (MAP query)."""
        marginals = self.query(variables, evidence)
        result: Dict[str, str] = {}
        for var, probs in marginals.items():
            idx = int(np.argmax(probs))
            states = NODE_STATE_NAMES.get(var, [str(i) for i in range(len(probs))])
            result[var] = states[idx]
        return result

    def prior_marginals(self) -> Dict[str, np.ndarray]:
        """Compute marginals with no evidence (prior distribution)."""
        return self.query(self.nodes, evidence={})

    # ---- probability propagation ----

    def propagate(self, evidence: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Full forward propagation: return named‑state probabilities for every node.

        Returns:
            { node: { state_name: probability, ... }, ... }
        """
        marginals = self.query(self.nodes, evidence=evidence)
        named: Dict[str, Dict[str, float]] = {}
        for node, probs in marginals.items():
            states = NODE_STATE_NAMES.get(node, [str(i) for i in range(len(probs))])
            named[node] = {s: round(float(p), 6) for s, p in zip(states, probs)}
        return named

    # ---- contribution analysis ----

    def calculate_contributions(
        self, target_node: str, evidence: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Explanatory contribution scores: how much each active parent
        shifts the target's high-risk probability relative to the prior.
        """
        if target_node not in self.structure._nodes:
            return {}
        ev = evidence if evidence is not None else self.evidence
        parents = self.structure.parents(target_node)
        active = {k: v for k, v in ev.items() if k in parents}
        if not active:
            return {}

        baseline = float(self.query([target_node], evidence={})[target_node][1])
        deltas: Dict[str, float] = {}
        total = 0.0
        for parent, state in active.items():
            marginal = float(self.query([target_node], evidence={parent: state})[target_node][1])
            delta = max(0.0, marginal - baseline)
            deltas[parent] = delta
            total += delta

        if total > 0:
            return {k: round(v / total * 100.0, 2) for k, v in deltas.items()}
        return deltas

    # ---- sensitivity analysis ----

    def sensitivity_analysis(
        self,
        target: str,
        perturbation_scale: float = 0.05,
        n_trials: int = 20,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Monte-Carlo sensitivity analysis: perturb each CPD independently
        and measure the effect on ``target``'s posterior.
        """
        rng = rng or np.random.default_rng(42)
        ev = dict(self.evidence)

        # Baseline
        baseline = float(self.query([target], evidence=ev)[target][1])

        sensitivities: Dict[str, Dict[str, float]] = {}
        for cpd in self.get_all_cpds():
            node = cpd.variable
            deltas = []
            for _ in range(n_trials):
                perturbed = self.cpd_manager.perturb_cpd(cpd, perturbation_scale, rng)
                # Temporarily swap CPD
                self.model.remove_cpds(cpd)
                self.model.add_cpds(perturbed)
                self._invalidate_inference_cache()
                try:
                    p = float(self.query([target], evidence=ev)[target][1])
                    deltas.append(abs(p - baseline))
                except Exception:
                    deltas.append(0.0)
                finally:
                    self.model.remove_cpds(perturbed)
                    self.model.add_cpds(cpd)
                    self._invalidate_inference_cache()

            sensitivities[node] = {
                "mean_delta": round(float(np.mean(deltas)), 6),
                "max_delta": round(float(np.max(deltas)), 6),
                "std_delta": round(float(np.std(deltas)), 6),
            }

        return {
            "target": target,
            "baseline": round(baseline, 6),
            "evidence": ev,
            "perturbation_scale": perturbation_scale,
            "n_trials": n_trials,
            "sensitivities": sensitivities,
        }

    # ---- CPD update ----

    def update_cpd(self, new_cpd: TabularCPD) -> bool:
        """Replace an existing CPD and revalidate the model."""
        node = new_cpd.variable
        try:
            old = self.model.get_cpds(node)
            self.model.remove_cpds(old)
            self.model.add_cpds(new_cpd)
            self.model.check_model()
            self._invalidate_inference_cache()
            logger.log(f"CPD updated for node: {node}")
            return True
        except Exception as e:
            logger.log(f"CPD update failed for {node}: {e}")
            # Rollback: re-add old CPD
            try:
                self.model.remove_cpds(new_cpd)
            except Exception:
                pass
            self.model.add_cpds(old)
            self._invalidate_inference_cache()
            return False

    def update_cpds_from_data(
        self, data: pd.DataFrame, method: str = "mle",
    ) -> Dict[str, Any]:
        """
        Re-estimate all CPDs from a DataFrame of observations.

        Args:
            data: columns = node names, rows = observations (state strings)
            method: 'mle' for Maximum Likelihood, 'bayes' for Bayesian estimation
        """
        from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

        try:
            if method == "bayes":
                self.model.fit(data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
            else:
                self.model.fit(data, estimator=MaximumLikelihoodEstimator)

            self.model.check_model()
            self._invalidate_inference_cache()
            logger.log(f"CPDs re-estimated from {len(data)} observations (method={method})")
            return {"status": "success", "observations": len(data), "method": method}
        except Exception as e:
            logger.log(f"CPD estimation failed: {e}")
            return {"status": "failed", "error": str(e)}

    # ---- d-separation queries ----

    def is_d_separated(self, x: str, y: str, z: Optional[Set[str]] = None) -> bool:
        """Check d-separation using internal structure (no pgmpy dependency)."""
        return self.structure.d_separated(x, y, z or set())

    def get_markov_blanket(self, node: str) -> Set[str]:
        return self.structure.markov_blanket(node)

    # ---- diagnostics ----

    def validate(self) -> Dict[str, Any]:
        """Run full model validation."""
        cpd_reports = [self.cpd_manager.validate_cpd(cpd) for cpd in self.get_all_cpds()]
        all_valid = all(r["valid"] for r in cpd_reports)
        try:
            pgmpy_valid = self.model.check_model()
        except Exception as e:
            pgmpy_valid = False
            logger.log(f"Model validation error: {e}")

        return {
            "valid": all_valid and pgmpy_valid,
            "pgmpy_check": pgmpy_valid,
            "cpd_reports": cpd_reports,
            "structure": self.structure.summary(),
        }

    def diagnostics(self) -> Dict[str, Any]:
        """Comprehensive engine diagnostic report."""
        priors = self.prior_marginals()
        prior_summary = {}
        for node, arr in priors.items():
            states = NODE_STATE_NAMES.get(node, [str(i) for i in range(len(arr))])
            prior_summary[node] = {s: round(float(p), 4) for s, p in zip(states, arr)}

        return {
            "structure": self.structure.summary(),
            "current_evidence": dict(self.evidence),
            "evidence_history_length": len(self._evidence_history),
            "prior_marginals": prior_summary,
            "validation": self.validate(),
            "cpds": [self.cpd_manager.cpd_to_dict(c) for c in self.get_all_cpds()],
        }

    def __repr__(self) -> str:
        return (
            f"BayesianNetworkEngine("
            f"nodes={len(self.nodes)}, edges={len(self.edges)}, "
            f"evidence={len(self.evidence)})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def create_default_network() -> BayesianNetworkEngine:
    """Create a BayesianNetworkEngine with the default city DAG and CPDs."""
    return BayesianNetworkEngine()


def get_default_structure() -> NetworkStructure:
    """Return the default 7-node city network structure."""
    return NetworkStructure()


def get_default_cpds() -> List[TabularCPD]:
    """Return the hardcoded CPDs for the default network."""
    return CPDManager().create_default_cpds()


def state_index(state: str) -> int:
    """Map a categorical state string to its integer index."""
    return STATE_INDEX.get(state, 0)


def map_event_to_evidence(event_type: str, severity: str) -> Optional[Tuple[str, str]]:
    """
    Convert an (event_type, severity) pair to a (node_name, state) tuple.

    Returns None if the event_type is not mapped to any node.
    """
    node = EVENT_TO_NODE.get(event_type)
    if node is None:
        return None
    if node in BOOLEAN_NODES:
        state = "True" if severity in ("medium", "high") else "False"
    else:
        state = "High" if severity in ("medium", "high") else "Low"
    return node, state
