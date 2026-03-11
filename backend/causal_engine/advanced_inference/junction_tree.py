"""
Junction Tree Algorithm for Exact Bayesian Inference

Implements the Junction Tree algorithm (also known as the Clique Tree algorithm)
for efficient exact inference in Bayesian Networks. This algorithm is more efficient
than Variable Elimination for multiple queries on the same network.

Key advantages:
- Precomputes message passing structure
- O(1) query time after initial compilation
- Handles multiple simultaneous queries efficiently
- Maintains exact inference guarantees
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, deque
import itertools
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from dataclasses import dataclass
import time

from utils.logger import SystemLogger

logger = SystemLogger(module_name="junction_tree")


@dataclass
class JunctionTreeNode:
    """Represents a node (clique) in the Junction Tree"""
    clique: Set[str]
    potential: DiscreteFactor
    neighbors: List['JunctionTreeNode']
    messages: Dict['JunctionTreeNode', DiscreteFactor]
    node_id: int


class JunctionTreeInference:
    """
    Junction Tree algorithm implementation for exact Bayesian inference

    The algorithm works in several phases:
    1. Moralize the DAG (add edges between parents of common children)
    2. Triangulate the moral graph (eliminate cycles of length > 3)
    3. Form cliques and build junction tree
    4. Initialize potentials and calibrate the tree
    5. Answer queries in O(1) time
    """

    def __init__(self, model: DiscreteBayesianNetwork):
        self.model = model
        self.nodes = set(model.nodes())
        self.edges = set(model.edges())

        # Junction tree components
        self.junction_tree: List[JunctionTreeNode] = []
        self.clique_graph: Dict[int, JunctionTreeNode] = {}
        self.variable_to_cliques: Dict[str, List[JunctionTreeNode]] = defaultdict(list)
        self.separator_sets: Dict[Tuple[int, int], Set[str]] = {}

        # Compilation results
        self.is_compiled = False
        self.compilation_time = 0.0
        self.tree_width = 0

        # Evidence and calibration state
        self.evidence: Dict[str, str] = {}
        self.is_calibrated = False

        logger.log("Junction Tree inference engine initialized")

    def compile_junction_tree(self) -> Dict[str, Any]:
        """
        Compile the junction tree from the Bayesian network

        Returns:
            Compilation statistics and performance metrics
        """
        start_time = time.time()

        try:
            # Step 1: Moralize the graph
            moral_graph = self._moralize_graph()

            # Step 2: Triangulate the moral graph
            triangulated_graph, elimination_order = self._triangulate_graph(moral_graph)

            # Step 3: Find maximal cliques
            cliques = self._find_maximal_cliques(triangulated_graph)

            # Step 4: Build junction tree
            self._build_junction_tree(cliques)

            # Step 5: Initialize clique potentials
            self._initialize_potentials()

            # Step 6: Initial calibration
            self._calibrate_tree()

            self.compilation_time = time.time() - start_time
            self.is_compiled = True
            self.is_calibrated = True

            # Calculate tree width (largest clique size - 1)
            self.tree_width = max(len(clique.clique) for clique in self.junction_tree) - 1

            logger.log(f"Junction tree compiled: {len(self.junction_tree)} cliques, "
                      f"tree width {self.tree_width}, compilation time {self.compilation_time:.3f}s")

            return {
                "status": "success",
                "compilation_time": self.compilation_time,
                "num_cliques": len(self.junction_tree),
                "tree_width": self.tree_width,
                "elimination_order": elimination_order
            }

        except Exception as e:
            logger.log(f"Junction tree compilation failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def _moralize_graph(self) -> Dict[str, Set[str]]:
        """Convert DAG to undirected moral graph"""
        moral_graph = defaultdict(set)

        # Add all original edges as undirected
        for parent, child in self.edges:
            moral_graph[parent].add(child)
            moral_graph[child].add(parent)

        # Add moral edges: connect all parents of each node
        for node in self.nodes:
            parents = list(self.model.get_parents(node))

            # Connect all pairs of parents
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moral_graph[parents[i]].add(parents[j])
                    moral_graph[parents[j]].add(parents[i])

        return dict(moral_graph)

    def _triangulate_graph(self, moral_graph: Dict[str, Set[str]]) -> Tuple[Dict[str, Set[str]], List[str]]:
        """Triangulate the moral graph using minimum fill-in heuristic"""
        triangulated = {node: neighbors.copy() for node, neighbors in moral_graph.items()}
        elimination_order = []
        remaining_nodes = set(self.nodes)

        while remaining_nodes:
            # Choose node with minimum fill-in (greedy heuristic)
            best_node = None
            min_fill_in = float('inf')

            for node in remaining_nodes:
                neighbors = triangulated[node] & remaining_nodes

                # Calculate fill-in: edges needed to make neighbors form a clique
                fill_in = 0
                neighbor_list = list(neighbors)
                for i in range(len(neighbor_list)):
                    for j in range(i + 1, len(neighbor_list)):
                        if neighbor_list[j] not in triangulated[neighbor_list[i]]:
                            fill_in += 1

                if fill_in < min_fill_in:
                    min_fill_in = fill_in
                    best_node = node

            # Eliminate the chosen node
            elimination_order.append(best_node)
            neighbors = triangulated[best_node] & remaining_nodes

            # Add fill-in edges to make neighbors form a clique
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    triangulated[neighbor_list[i]].add(neighbor_list[j])
                    triangulated[neighbor_list[j]].add(neighbor_list[i])

            # Remove the eliminated node
            remaining_nodes.remove(best_node)
            for neighbor in triangulated[best_node]:
                if neighbor in triangulated:
                    triangulated[neighbor].discard(best_node)

        return triangulated, elimination_order

    def _find_maximal_cliques(self, triangulated_graph: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find all maximal cliques in the triangulated graph"""
        cliques = []

        # Use the elimination order approach: each eliminated variable forms a clique
        # with its neighbors at elimination time
        temp_graph = {node: neighbors.copy() for node, neighbors in triangulated_graph.items()}

        # Reverse the elimination to reconstruct cliques
        remaining_nodes = set(self.nodes)

        while remaining_nodes:
            # Find a simplicial node (node whose neighbors form a clique)
            simplicial_node = None

            for node in remaining_nodes:
                neighbors = temp_graph[node] & remaining_nodes

                # Check if neighbors form a clique
                is_simplicial = True
                neighbor_list = list(neighbors)
                for i in range(len(neighbor_list)):
                    for j in range(i + 1, len(neighbor_list)):
                        if neighbor_list[j] not in temp_graph[neighbor_list[i]]:
                            is_simplicial = False
                            break
                    if not is_simplicial:
                        break

                if is_simplicial:
                    simplicial_node = node
                    break

            if simplicial_node is None:
                # If no simplicial node found, pick any node
                simplicial_node = next(iter(remaining_nodes))

            # Form clique with this node and its neighbors
            neighbors = temp_graph[simplicial_node] & remaining_nodes
            clique = {simplicial_node} | neighbors
            cliques.append(clique)

            # Remove the node
            remaining_nodes.remove(simplicial_node)
            for neighbor in temp_graph[simplicial_node]:
                if neighbor in temp_graph:
                    temp_graph[neighbor].discard(simplicial_node)

        return cliques

    def _build_junction_tree(self, cliques: List[Set[str]]) -> None:
        """Build the junction tree from maximal cliques"""
        # Create junction tree nodes
        self.junction_tree = []
        self.clique_graph = {}

        for i, clique in enumerate(cliques):
            jt_node = JunctionTreeNode(
                clique=clique,
                potential=None,  # Will be initialized later
                neighbors=[],
                messages={},
                node_id=i
            )
            self.junction_tree.append(jt_node)
            self.clique_graph[i] = jt_node

            # Map variables to cliques
            for var in clique:
                self.variable_to_cliques[var].append(jt_node)

        # Build minimum spanning tree using maximum cardinality search
        if len(self.junction_tree) > 1:
            self._build_mst_junction_tree()

    def _build_mst_junction_tree(self) -> None:
        """Build junction tree using minimum spanning tree approach"""
        # Calculate weights between cliques (size of intersection)
        edge_weights = []

        for i in range(len(self.junction_tree)):
            for j in range(i + 1, len(self.junction_tree)):
                clique_i = self.junction_tree[i].clique
                clique_j = self.junction_tree[j].clique
                intersection = clique_i & clique_j

                if intersection:  # Only connect if they share variables
                    weight = len(intersection)
                    edge_weights.append((weight, i, j, intersection))

        # Sort by weight (descending) and build MST
        edge_weights.sort(reverse=True)

        # Kruskal's algorithm for MST
        parent = list(range(len(self.junction_tree)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        for weight, i, j, separator in edge_weights:
            if union(i, j):
                # Add edge to junction tree
                self.junction_tree[i].neighbors.append(self.junction_tree[j])
                self.junction_tree[j].neighbors.append(self.junction_tree[i])
                self.separator_sets[(i, j)] = separator
                self.separator_sets[(j, i)] = separator

    def _initialize_potentials(self) -> None:
        """Initialize clique potentials with CPDs from the Bayesian network"""
        # Initialize all potentials to uniform
        for jt_node in self.junction_tree:
            # Create uniform factor over clique variables
            variables = list(jt_node.clique)
            cardinalities = [self.model.get_cpds(var).variable_card for var in variables]

            uniform_values = np.ones(cardinalities)
            jt_node.potential = DiscreteFactor(
                variables=variables,
                cardinality=cardinalities,
                values=uniform_values
            )

        # Multiply CPDs into appropriate cliques
        for cpd in self.model.get_cpds():
            var = cpd.variable
            cpd_variables = set(cpd.variables)

            # Find a clique that contains all CPD variables
            target_clique = None
            for jt_node in self.junction_tree:
                if cpd_variables.issubset(jt_node.clique):
                    target_clique = jt_node
                    break

            if target_clique:
                # Convert CPD to DiscreteFactor and multiply
                cpd_factor = cpd.to_factor()
                target_clique.potential = target_clique.potential.product(cpd_factor, inplace=False)

    def _calibrate_tree(self) -> None:
        """Calibrate the junction tree using message passing"""
        if len(self.junction_tree) <= 1:
            return

        # Choose root (arbitrary)
        root = self.junction_tree[0]

        # Two-pass message passing
        # Pass 1: Collect evidence (leaves to root)
        self._collect_evidence(root, None)

        # Pass 2: Distribute evidence (root to leaves)
        self._distribute_evidence(root, None)

        self.is_calibrated = True

    def _collect_evidence(self, node: JunctionTreeNode, parent: Optional[JunctionTreeNode]) -> None:
        """Collect evidence phase of calibration"""
        # Recursively collect from children first
        for neighbor in node.neighbors:
            if neighbor != parent:
                self._collect_evidence(neighbor, node)

        # Send message to parent
        if parent is not None:
            message = self._compute_message(node, parent)
            node.messages[parent] = message
            parent.potential = parent.potential.product(message, inplace=False)

    def _distribute_evidence(self, node: JunctionTreeNode, parent: Optional[JunctionTreeNode]) -> None:
        """Distribute evidence phase of calibration"""
        # Send messages to children
        for neighbor in node.neighbors:
            if neighbor != parent:
                message = self._compute_message(node, neighbor)
                node.messages[neighbor] = message
                neighbor.potential = neighbor.potential.product(message, inplace=False)

                # Recursively distribute to subtree
                self._distribute_evidence(neighbor, node)

    def _compute_message(self, sender: JunctionTreeNode, receiver: JunctionTreeNode) -> DiscreteFactor:
        """Compute message from sender to receiver"""
        # Get separator set
        separator = self.separator_sets.get((sender.node_id, receiver.node_id), set())

        # Start with sender's potential
        message_potential = sender.potential.copy()

        # Multiply by incoming messages (except from receiver)
        for neighbor, incoming_message in sender.messages.items():
            if neighbor != receiver:
                message_potential = message_potential.product(incoming_message, inplace=False)

        # Marginalize to separator variables
        if separator:
            vars_to_marginalize = set(message_potential.variables) - separator
            if vars_to_marginalize:
                message_potential = message_potential.marginalize(
                    list(vars_to_marginalize), inplace=False
                )

        return message_potential

    def query(self, variables: List[str], evidence: Dict[str, str] = None) -> DiscreteFactor:
        """
        Perform inference query on the calibrated junction tree

        Args:
            variables: Variables to query
            evidence: Evidence dictionary

        Returns:
            Factor containing query results
        """
        if not self.is_compiled:
            raise ValueError("Junction tree must be compiled before querying")

        # Update evidence if provided
        if evidence != self.evidence:
            self._set_evidence(evidence or {})

        # Find clique containing all query variables
        query_vars = set(variables)
        target_clique = None

        for jt_node in self.junction_tree:
            if query_vars.issubset(jt_node.clique):
                target_clique = jt_node
                break

        if target_clique is None:
            raise ValueError(f"No clique contains all query variables: {variables}")

        # Extract marginal from clique potential
        clique_potential = target_clique.potential.copy()

        # Marginalize out non-query variables
        vars_to_marginalize = set(clique_potential.variables) - query_vars
        if vars_to_marginalize:
            result = clique_potential.marginalize(list(vars_to_marginalize), inplace=False)
        else:
            result = clique_potential

        # Normalize to get probabilities
        result.normalize()

        return result

    def _set_evidence(self, evidence: Dict[str, str]) -> None:
        """Set evidence and recalibrate if needed"""
        if evidence == self.evidence:
            return

        self.evidence = evidence.copy()

        # Apply evidence to clique potentials
        for jt_node in self.junction_tree:
            # Find evidence variables in this clique
            clique_evidence = {}
            for var in jt_node.clique:
                if var in evidence:
                    clique_evidence[var] = evidence[var]

            if clique_evidence:
                # Reduce potential based on evidence
                jt_node.potential = jt_node.potential.reduce(
                    clique_evidence, inplace=False
                )

        # Recalibrate tree
        self._calibrate_tree()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the junction tree"""
        return {
            "is_compiled": self.is_compiled,
            "compilation_time": self.compilation_time,
            "num_cliques": len(self.junction_tree),
            "tree_width": self.tree_width,
            "max_clique_size": max(len(clique.clique) for clique in self.junction_tree) if self.junction_tree else 0,
            "total_potential_size": sum(
                np.prod(node.potential.cardinality) if node.potential else 0
                for node in self.junction_tree
            ),
            "is_calibrated": self.is_calibrated
        }