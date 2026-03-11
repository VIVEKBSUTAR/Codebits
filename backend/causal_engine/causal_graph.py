from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

from utils.logger import SystemLogger

logger = SystemLogger(module_name="causal_graph")

class NodeDTO(BaseModel):
    name: str
    current_state: str
    probability: float
    parents: List[str]
    children: List[str]
    timestamp: datetime
    zone: str

class CausalGraphService:
    def __init__(self, zone: str):
        self.zone = zone
        self.evidence: Dict[str, str] = {}
        self.model = self._initialize_model()
        self.infer = VariableElimination(self.model)
        logger.log(f"Causal graph service initialized for zone: {zone}")

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
        """Maps incoming EventModel properties to graph Evidence"""
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

    def run_inference(self) -> Dict[str, float]:
        """Calculates risk prediction from posterior probabilities"""
        posterior_probs = {}
        high_states = {'True', 'High'}

        for target in ['Flooding', 'TrafficCongestion', 'EmergencyDelay']:
            if target in self.evidence:
                # Target is directly observed — use the evidence value as probability
                posterior_probs[target] = 1.0 if self.evidence[target] in high_states else 0.0
            else:
                result = self.infer.query(variables=[target], evidence=self.evidence)
                posterior_probs[target] = result.values[1]  # 'True' or 'High' state

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

# Global dictionary to manage graph services per zone
zone_graphs: Dict[str, CausalGraphService] = {}

def get_causal_graph(zone: str) -> CausalGraphService:
    if zone not in zone_graphs:
        zone_graphs[zone] = CausalGraphService(zone)
    return zone_graphs[zone]
