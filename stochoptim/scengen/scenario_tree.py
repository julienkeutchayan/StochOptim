# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import unicode_literals

from typing import Dict, Union, Optional, List
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from stochoptim.scengen.tree_structure import Node
from stochoptim.scengen.tree_structure import get_data_path
from stochoptim.scengen.scenario_process import ScenarioProcess
from stochoptim.scengen.figure_of_demerit import FigureOfDemerit
from stochoptim.scengen.variability_process import VariabilityProcess


class ScenarioTree(Node):

    def __init__(self, node: Node):
        """
        Arguments:
        ----------
        node: Node
            Root of the scenario tree. 
            If it is not already a root (i.e., if it has a parent), it is made one by cutting its parent. 
        """
        node.make_it_root()
        super().__init__(*node.children, **node.data)
        
    # --- Properties ---   
    def are_scenarios_consistent(self):
        """Return True if all the paths have the same number of variables."""
        array_map_stage_to_rvar_nb = np.array([self._map_stage_to_rvar_nb(leaf) for leaf in self.leaves])
        return (array_map_stage_to_rvar_nb == array_map_stage_to_rvar_nb[0]).all()
    
    @property
    def map_stage_to_rvar_names(self) -> Dict[int, List[str]]:
        return {stage: list(var_dict.keys()) for stage, var_dict in self.map_stage_to_rvar_nb.items()}
    
    @property
    def map_stage_to_rvar_nb(self) -> Dict[int, Dict[str, int]]:
        bottom_most_leaf = self.node_at_address((0,) * (self.depth - 1))
        return self._map_stage_to_rvar_nb(bottom_most_leaf)
    
    def _map_stage_to_rvar_nb(self, leaf: Node) -> Dict[int, Dict[str, int]]:
        """Return the variables names and numbers at each level along a certain path leading to a leaf."""
        return {node.level: {var_name: len(variables) for var_name, variables in node.data['scenario'].items()}
                                 for node in leaf.branch if isinstance(node.data.get('scenario'), dict)}
        
    @staticmethod
    def get_scenario_path(node: Node):
        return get_data_path(node, 'scenario')
        
    def get_figure_of_demerit(self, demerit: Optional[FigureOfDemerit]):
        """Figure of demerit of the scenario tree, as given by the exact formula.
        
        Arugment:
        ---------
        demerit: FigureOfDemerit
            The figure of demerit used to compute the demerit of the children at each node.
            
        Returns:
        --------
        float > 0
        """
        self._is_filled()
        return demerit(self, subtree=True, path=True)
    
    # --- Scenarios filling ---
    def fill(self, 
             scenario_process: ScenarioProcess, 
             optimized: Optional[str] = None, 
             variability_process: Optional[VariabilityProcess] = None, 
             demerit: Optional[FigureOfDemerit] = None):
        """Fills the tree structure with scenarios given by the scenario process.
        
        Arguments:
        ----------           
        optimized: {'forward', 'backward', None}
            The way the scenario tree is filled with scenarios and weights.
            If 'forward', the assignment of scenarios to nodes is optimized from the root to 
            the leaves, and it is guided by the scenario values (under data key 'scenario').
            If 'backward', the assignment of scenarios to nodes is optimized from the leaves 
            to the root, and it is guided by the epsilon values (under data key 'eps').
            If None, the assignment is not optimized.
        
        scenario_process: ScenarioProcess
            The scenario process used to generate the scenarios.

        variability_process: VariabilityProcess
            The variability process used to guide the assignement of scenarios to nodes.

        demerit: FigureOfDemerit
            The figure of demerit used to compute the demerit of the children at each node.
        """        
        self._fill_epsilon(scenario_process) # assign weights ('W') and optionaly epsilon sample ('eps') at each node
        if optimized is None:
            self._fill_scenario(scenario_process)
        elif optimized == 'forward':
            assert variability_process.has_lookback(), \
                "The variability process must have a `lookback` method for the forward-optimized scenario tree."
            self._optimized_assignment_forward(scenario_process, variability_process, demerit) 
        elif optimized == 'backward':
            assert variability_process.has_looknow(), \
                "The variability process must have a `looknow` method for the backward-optimized scenario tree."
            self._optimized_assignment_backward(variability_process, demerit) 
            self._fill_scenario(scenario_process)
        else:
            raise ValueError(f"Wrong 'optimized' keyword: must be None, 'forward', or 'backward', not {optimized}.")

    def _fill_epsilon(self, scenario_process):
        """Fills the scenario tree with points 'eps' and weights 'w'"""
        self.data["w"] = 1 # root
        for node in self.nodes:
            if not node.is_leaf:
                weights, epsilons = scenario_process.get_children_sample(node)
                random_indices = np.random.permutation(range(len(node.children)))
                for i, child in zip(random_indices, node.children):
                    child.data["w"] = weights[i]
                    if epsilons is not None:
                        child.data["eps"] = epsilons[i]

    def _fill_scenario(self, scenario_process):
        """Fills the scenario tree with points 'scenario' and weights 'W' given the assignment of 'eps' 
        and 'w' at each node."""
        for node in self.nodes:
            node.data["W"] = node.parent.data["W"] * node.data["w"] if not node.is_root else 1
            root_scenario = scenario_process.get_node_scenario(node, path=False)
            if root_scenario is not None:
                node.data["scenario"] = root_scenario
            
    def _optimized_assignment_forward(self, scenario_process, variability_process, demerit):  
        for node in self.forward_nodes:
            if node.is_root:
                node.data["W"] = 1
                root_scenario = scenario_process.get_node_scenario(node, path=False)
                if root_scenario is not None:
                    node.data["scenario"] = root_scenario
                
            if not node.is_leaf and not node.is_parent_of_leaf: 
                # sort child (in place) by decreasing demerit
                node.children.sort(key=lambda c: -demerit(c, subtree=False)) 
                # sort child (not in place) by increasing variability
                for child in node.children:
                    child.data["scenario"] = scenario_process.get_node_scenario(child, path=False)
                child_by_variability = sorted(node.children, 
                                              key=lambda c: variability_process.node_lookback(c) * c.data["w"])
                weights = [c.data["w"] for c in child_by_variability]
                epsilons = [c.data.get("eps") for c in child_by_variability]
                scenarios = [c.data['scenario'] for c in child_by_variability]
                
                for i, child in enumerate(node.children):
                    child.data["w"] = weights[i]
                    child.data["W"] = node.data["W"] * child.data["w"]
                    if epsilons[i] is not None:
                        child.data["eps"] = epsilons[i]
                    child.data["scenario"] = scenarios[i]
                    
            if node.is_parent_of_leaf:
                for child in node.children:
                    child.data["scenario"] = scenario_process.get_node_scenario(child, path=False)
                    child.data["W"] = node.data["W"] * child.data["w"]
                    
    def _optimized_assignment_backward(self, variability_process, demerit):
        for node in self.backward_nodes:                
            if not node.is_leaf and not node.is_parent_of_leaf: 
                # sort child (in place) by decreasing demerit
                node.children.sort(key=lambda c: -demerit(c, subtree=True, path=False)) 
                # sort child (not in place) by increasing variability
                child_by_variability = sorted(node.children, 
                                              key=lambda c: variability_process.node_looknow(c) * c.data["w"])
                weights = [c.data["w"] for c in child_by_variability]
                epsilons = [c.data.get("eps") for c in child_by_variability]
                
                for i, child in enumerate(node.children):
                    child.data["w"] = weights[i]
                    if epsilons[i] is not None:
                        child.data["eps"] = epsilons[i]
    
    def _is_filled(self):
        """Check whether the tree is filled with scenarios."""
        assert self.has_key('scenario', excluding_levels=[0]), \
            "Fill the tree structure before computing the scenario-tree figure of demerit."
     
    # --- Forward generation ---                 
    @classmethod
    def forward_generation(cls, 
                           n_stages: int,
                           n_scenarios: int, 
                           scenario_process: ScenarioProcess, 
                           variability_process: VariabilityProcess, 
                           alpha: float):
        """Generates a scenario tree by the forward bound-minimizing heuristic.
        
        Arguments:
        ----------
        n_stages: int >= 1
        
        n_scenarios: int >= 1
                
        scenario_process: ScenarioProcess
            The scenario process used to generate the scenarios.

        variability_process: VariabilityProcess
            The variability process used to guide the assignement of scenarios to nodes.
            
        alpha: float > 0
            Convergence rate of the discretization method (typically from 0.5 to 2).
            
        Returns:
        --------
        ScenarioTree
        """
        assert variability_process.has_average(), \
            "The variability process must have an `average_fct` method for the forward-generation algorithm."
        avg_variability = [variability_process.average(stage) for stage in range(n_stages-1)]
        width_vector = cls.optimal_width_vector(n_scenarios, alpha, avg_variability)
        return cls.forward_generation_from_given_width(width_vector, scenario_process, variability_process, alpha)
         
    @classmethod
    def forward_generation_from_given_width(cls, 
                                            width_vector: List[int], 
                                            scenario_process: ScenarioProcess, 
                                            variability_process: VariabilityProcess, 
                                            alpha: float):
        """Generates a scenario tree by the forward bound-minimizing heuristic.
        
        Arguments:
        ----------
        width_vector: list of int >= 1
        
        scenario_process: ScenarioProcess
            The scenario process used to generate the scenarios.

        variability_process: VariabilityProcess
            The variability process used to guide the assignement of scenarios to nodes.
            
        alpha: float > 0
            Convergence rate of the discretization method (typically from 0.5 to 2).
                
        Returns:
        --------
        ScenarioTree
        """
        last_stage = len(width_vector)
        tree = cls.from_data_dict({(): {"M": width_vector[0], "W": 1, "w": 1}})       
        root_scenario = scenario_process.get_node_scenario(tree, path=False)
        if root_scenario is not None:
            tree.data["scenario"] = root_scenario             
        tree.data["g"] = variability_process.node_lookback(tree)
        
        # difference between the actual width of the leaves and the target one
        node_gap = lambda tree: sum(leaf.data["M"] for leaf in tree.leaves) - width_vector[tree.depth-1]
        
        for stage in range(1, last_stage):
            # 1. Extend and fill the structure
            tree._extend_tree_by_one_stage(scenario_process)
            for leaf in tree.leaves:
                leaf.data["g"] = variability_process.node_lookback(leaf)
            # 2. Compute the optimal number of child nodes
            if width_vector[stage] == width_vector[stage-1]: 
                for leaf in tree.leaves:
                    leaf.data["M"] = 1
            else:
                normalization = sum((leaf.data["W"] * leaf.data["g"])**(1/(alpha+1)) for leaf in tree.leaves)
                for leaf in tree.leaves:
                    leaf.data["m"] = (width_vector[stage] / normalization) \
                                        * (leaf.data["W"] * leaf.data["g"])**(1/(alpha+1))
                    leaf.data["M"] = int(max(1, round(leaf.data["m"])))

            # 3. Correct the rounding off of the number of child nodes (if necessary) so that the actual width 
            # equal the target
            while node_gap(tree) > 0:
                leaf = min([leaf for leaf in tree.leaves if leaf.data["M"] >= 2], 
                               key = lambda leaf: abs(leaf.data["m"] - (leaf.data["M"] - 1)))
                leaf.data["M"] = leaf.data["M"] - 1 # remove one child
                
            while node_gap(tree) < 0:
                leaf = min(tree.leaves, 
                           key = lambda leaf: abs(leaf.data["m"] - (leaf.data["M"] + 1)))
                leaf.data["M"] = leaf.data["M"] + 1 # add one child
                
        # extend and fill the last stage
        tree._extend_tree_by_one_stage(scenario_process)
        # delete temporary data
        tree.delete_data(["M", "m", "g"])
        assert tree.width == list(width_vector),  ("Missmatch between the actual tree width and the target one "
                                            f"actual width: {tree.width}, target width: {list(width_vector)}")
        return tree

    @staticmethod
    def optimal_width_vector(n_scenarios: int, 
                             alpha: float, 
                             gamma: List[float]):
        bush = [1 if gamma[m] == 0 else None for m in range(len(gamma))] # branching factor = 1 if no variability
        inactive_set = [m for m in range(len(gamma)) if gamma[m] != 0] # set of inactive constraints
        found = False
        while not found:
            found = True
            denominator = np.prod([gamma[i]**(1/alpha) for i in inactive_set])**(1/len(inactive_set))
            for m in inactive_set:
                bush[m] = n_scenarios**(1/len(inactive_set)) * gamma[m]**(1/alpha) / denominator
                found = found and (bush[m] > 1)
            if not found:
                # find the index m* such that gamma[m] is the smallest
                min_index = min(inactive_set, key = lambda m: gamma[m]) 
                # remove m* from the set of inactive constraints
                inactive_set.remove(min_index)
                bush[min_index] = 1
        width_vector = np.round(np.cumprod(bush)).astype('int')
        return list(width_vector)
    
    def _extend_tree_by_one_stage(self, scenario_process):
        for leaf in list(self.leaves):
            leaf.add(*[Node() for i in range(leaf.data["M"])])
            weights, epsilons = scenario_process.get_children_sample(leaf)
            for i, child in enumerate(leaf.children):
                child.data["w"] = weights[i]
                if epsilons is not None:
                    child.data["eps"] = epsilons[i]
                child.data["W"] = leaf.data["W"] * child.data["w"]
                child.data["scenario"] = scenario_process.get_node_scenario(child, path=False)
                
    # --- Alternative constructors ---
    @staticmethod
    def _set_equal_weights(tree_structure):
        tree_structure.data["W"] = 1
        for node in tree_structure.nodes:
            if not node.is_root:
                node.data["W"] = node.parent.data["W"] / len(node.parent.children)
        return tree_structure
    
    @classmethod
    def from_topology(cls, topology, equal_weights=False):
        scen_tree = cls(Node.from_topology(topology))
        if equal_weights:
            scen_tree = ScenarioTree._set_equal_weights(scen_tree)
        return scen_tree
    
    @classmethod
    def from_recurrence(cls, last_stage, init, recurrence, equal_weights=False):
        scen_tree = cls(Node.from_recurrence(last_stage, init, recurrence))
        if equal_weights:
            scen_tree = ScenarioTree._set_equal_weights(scen_tree)
        return scen_tree
    
    @classmethod
    def from_bushiness(cls, bushiness, equal_weights=False):
        scen_tree = cls(Node.from_bushiness(bushiness))
        if equal_weights:
            scen_tree = ScenarioTree._set_equal_weights(scen_tree)
        return scen_tree
    
    @classmethod
    def from_data_dict(cls, data_dict, equal_weights=False):
        scen_tree = cls(Node.from_data_dict(data_dict))
        if equal_weights:
            scen_tree = ScenarioTree._set_equal_weights(scen_tree)
        return scen_tree

    @classmethod
    def twostage_from_scenarios(cls, 
                                scenarios, 
                                n_rvar: Dict[str, int],
                                weights=None):
        """
        Constructor of a two-stage scenario tree directly from the set of scenarios and the weights.
    
        Arguments:
        ----------
        scenarios: 2d-array-like
            Array of shape (number_of_scenarios, dimension_of_a_scenario).
    
        n_rvar: Dict[str, int] (optional)
            Dictionary mapping each random variable's name to the number of such variables.
            If None, one variable with name "" is created.
            
        weights: 1d-array (optional)
            Array of shape (number_of_scenarios,). If None, equal-weights are considered.
        """
        if sparse.issparse(scenarios):
            if scenarios.dtype == np.float16: # float16 not directly supported for sparse matrix
                scenarios = scenarios.astype(np.float32).toarray().astype(np.float16)
            else:
                scenarios = scenarios.toarray()
        if isinstance(scenarios, list):
            scenarios = np.array(scenarios)
        assert len(scenarios.shape) == 2, \
            f"The scenarios must be given as a 2d-array, not a {len(scenarios.shape)}d-array."
        n_scenarios, dim_scenario = scenarios.shape
        if weights is None:
            weights = np.ones(n_scenarios) / n_scenarios
            
        if n_rvar is None:
            n_rvar = {"": dim_scenario}
        else:
            assert sum(n_rvar.values()) == dim_scenario, ("Mismatch between the number of random variables "
            f"in `n_rvar` ({sum(n_rvar.values())}) and the number of features in the scenarios ({dim_scenario})")
            
        data_dict = {(): {'W': 1}}
        for i in range(n_scenarios):
            split_points = np.cumsum(np.array(list(n_rvar.values())))
            split_scenario = np.split(scenarios[i], split_points)
            data_dict[(i,)] =  {'scenario': {var_name: split_scenario[j] 
                                                for j, var_name in enumerate(n_rvar.keys())}, 
                                'W': weights[i]}
    
        return cls.from_data_dict(data_dict)
    
    @classmethod
    def combtree_from_scenarios(cls, 
                                scenarios: np.ndarray,
                                map_stage_to_rvar_nb: Dict[int, Dict[str, int]],
                                weights: Optional[np.ndarray] = None):
        """
        Constructor of a multi-stage scenario tree with comb structure from a set of scenarios and their weights.
        A comb structure has all its scenarios linked at the root only.
        
        Arguments:
        ----------
        scenarios: 2d-array of shape (number_of_scenarios, number_of_features)
            The features should be ordered by stage and within a stage by the order in the list of variable names.
            
        map_stage_to_rvar_nb: Dict[int, Dict[str, int]]
            Map stage (int) to a map between the variable names (str) and the variables numbers (int) at that stage.
        
        weights: 1d-array (optional)
            Array of shape (number_of_scenarios,). If not provided, equal-weights are considered.
        """
        if sparse.issparse(scenarios):
            if scenarios.dtype == np.float16: # float16 not directly supported for sparse matrix
                scenarios = scenarios.astype(np.float32).toarray().astype(np.float16)
            else:
                scenarios = scenarios.toarray()
        last_stage = max(map_stage_to_rvar_nb.keys())
        n_var_at_each_stage = [sum(map_stage_to_rvar_nb.get(t, {'': 0}).values()) for t in range(1, last_stage+1)]
        
        # check whether each stage has at least one random variable
        assert (np.array(n_var_at_each_stage) >= 1).all(), f"One stage has no random variable: {map_stage_to_rvar_nb}"
        # check whether the number of variables matches between the input scenarios and map_stage_to_rvar_nb
        assert sum(n_var_at_each_stage) == scenarios.shape[1], \
        (f"Mismatch between the number of random variables expected from `map_stage_to_rvar_nb` "
         f"({sum(n_var_at_each_stage)}) and the number of features in `scenarios` ({scenarios.shape[1]}).")
        
        if weights is None:
            n_scenarios = scenarios.shape[0]
            weights = np.ones(n_scenarios) / n_scenarios

        data_dict = {(): {'W': 1}}
        for i, path_scenario in enumerate(scenarios):
            # decompose into a list of scenario at each stage
            split_points = np.cumsum([sum(map_stage_to_rvar_nb[t].values()) for t in map_stage_to_rvar_nb.keys()])[:-1]
            split_path_scenario = np.split(path_scenario, split_points)
            
            for t, stage_scenario in enumerate(split_path_scenario, 1):
                # decompose into a list of scenario for each var_name
                split_points = np.cumsum(np.array(list(map_stage_to_rvar_nb[t].values())))
                split_stage_scenario = np.split(stage_scenario, split_points)
                # append to data dict
                address = (i,) + tuple(0 for _ in range(t-1))
                data_dict[address] =  {'scenario': {var_name: split_stage_scenario[j] 
                                                        for j, var_name in enumerate(map_stage_to_rvar_nb[t].keys())}, 
                                        'W': weights[i]}
    
        return cls.from_data_dict(data_dict)

    # --- Operations on scenarios ---   
    def merge(self, **kwargs):
        """Merge sibling nodes if they have identical scenarios. Adjust the weights 
        accordingly. (Note that the nodes are merged regardless of whether they have identical
        subtrees or not.)
        
        kwargs: 
        ------
        All kwargs of np.isclose
        """
        for node in self.nodes:
            if node.is_leaf:
                continue
            to_be_removed = []
            for k, child1 in enumerate(node.children):
                for child2 in node.children[k+1:]:
                    # test if same scenario at child node
                    is_equal = True
                    for var_name in self.map_stage_to_rvar_names[node.level + 1]:
                        if not np.isclose(child1.data['scenario'][var_name], 
                                          child2.data['scenario'][var_name], 
                                          **kwargs).all():
                            is_equal = False
                            break
                    if is_equal:
                        weight_coef = (child2.data["W"] + child1.data["W"]) / child2.data["W"]
                        for n in child2.nodes:
                            n.data["W"] *= weight_coef
                        to_be_removed.append(child1)
                        break
            for child in to_be_removed:
                child.remove()
                    
    def average(self, 
                map_stage_to_rvar_names: Optional[Dict[int, List[str]]] = None,
                across_tree: bool = True):
        """Replace some scenarios by their average value in place.        
        
        Arguments:
        ---------
        across_tree: bool (default: True)
            If True, averages are computed across all nodes at a given stage. Otherwise, it
            is compute across all children.
            
        map_stage_to_rvar_names: Dict[int, List[str]] or None (default: None)
            The stages (int) and variables names (List[str]) for which the scenarios are averaged.
            If None, all stages and all variables are averaged.
        """
        if across_tree:
            self._average_across_tree(map_stage_to_rvar_names)
        else:
            self._average_across_children(map_stage_to_rvar_names)

    def _average_across_tree(self, map_stage_to_rvar_names: Optional[Dict[int, List[str]]] = None):
        """Replace some scenarios by their average value in place.
        
        Argument:
        ---------
        map_stage_to_rvar_names: Dict[int, List[str]] or None (default: None)
            The stages (int) and variables names (List[str]) for which the scenarios are averaged.
            If None, all stages and all variables are averaged.
        """
        if map_stage_to_rvar_names is None:
            map_stage_to_rvar_names = self.map_stage_to_rvar_names
        for stage in map_stage_to_rvar_names.keys():
            for var_name in map_stage_to_rvar_names[stage]:
                avg_scen = np.mean(self.to_numpy({stage: [var_name]}), axis=0)
                for node in self.nodes_at_level(stage):
                    node.data['scenario'][var_name] = avg_scen

    def _average_across_children(self, map_stage_to_rvar_names: Optional[Dict[int, List[str]]] = None):
        """Replace some scenarios by their average value in place.
        
        Argument:
        ---------
        map_stage_to_rvar_names: Dict[int, List[str]] or None (default: None)
            The stages (int) and variables names (List[str]) for which the scenarios are averaged.
            If None, all stages and all variables are averaged.
        """
        if map_stage_to_rvar_names is None:
            map_stage_to_rvar_names = self.map_stage_to_rvar_names
        for stage in map_stage_to_rvar_names.keys():
            for node in self.nodes_at_level(stage - 1):
                for var_name in map_stage_to_rvar_names[stage]:                
                    avg_scen = np.mean([child.data['scenario'][var_name] for child in node.children], axis=0)
                    for child in node.children:
                        child.data['scenario'][var_name] = avg_scen
            
    def to_numpy(self, map_stage_to_rvar_names: Optional[Dict[int, List[str]]] = None) -> np.ndarray:
        """ Return the scenarios as a numpy array. 
        
        Scenarios on the same path but at different stages, or corresponding to different random variables, 
        are concatenated along axis = 1 by stage and within a stage by the order of the variable names in input list. 
        Scenarios on different paths are placed along axis = 0.
        
        Argument:
        ---------
        map_stage_to_rvar_names: Dict[int, List[str]] or None (default: None)
            The stages (int) and variables names (List[str]) for which the scenarios are put in an array.
            If None, it returns the scenarios at all stages and for all variable names.
            
        Returns:
        --------
        2d-array of shape (n_leaves, N) where n_leaves is the number of leaves in the scenario tree and
            N is the number of random variables.
        """
        if map_stage_to_rvar_names is None:
            map_stage_to_rvar_names = self.map_stage_to_rvar_names
        return self.get_subtree_as_numpy(self, map_stage_to_rvar_names)
        
    def get_path_as_numpy(self, node, map_stage_to_rvar_names: Optional[Dict[int, List[str]]] = None) -> np.ndarray:
        """Return the scenarios as a numpy 1d-array along the path leading to a certain node. 
        Scenarios are concatenated by stage and within a stage by the order of variables in the input list."""
        if map_stage_to_rvar_names is None:
            map_stage_to_rvar_names = self.map_stage_to_rvar_names
        return np.concatenate([m.data['scenario'][var_name] for m in node.branch if not m.is_root
                                                               if m.level in map_stage_to_rvar_names.keys()
                                                                   for var_name in map_stage_to_rvar_names[m.level]])
    
    def get_subtree_as_numpy(self, node, map_stage_to_rvar_names: Optional[Dict[int, List[str]]] = None) -> np.ndarray:
        """Return the scenarios in a subtree as a numpy array (excluding the scenarios at the subtree root).
        Scenarios on the same path but at different stages, or corresponding to different random variables, are 
        concatenated along axis = 1 by stage and within a stage by the order of the variable names in input list. 
        Scenarios on different paths are placed along axis = 0.
        
        Argument:
        ---------
        map_stage_to_rvar_names: Dict[int, List[str]] or None (default: None)
            The stages (int) and variables names (List[str]) for which the scenarios are put in an array.
            If None, it returns the scenarios at all stages and for all variable names.
            
        Returns:
        --------
        2d-array of shape (n_leaves, N) where n_leaves is the number of leaves in the subtree rooted at `node` and 
            N is the number of random variables.
        """
        if map_stage_to_rvar_names is None:
            map_stage_to_rvar_names = self.map_stage_to_rvar_names
        return np.array([self.get_path_as_numpy(n, map_stage_to_rvar_names) for n in node.leaves if n != node])
    
    # --- Plots ---
    def plot_scenarios(self, 
             var_name: Optional[Union[str, Dict[int, str]]] = None, 
             scenario_precision: int = 2,
             format_weights: str = '.3f',
             **kwargs):
        if var_name is None:
            print_on_nodes = None
        elif isinstance(var_name, str):
            def print_on_nodes(node):
                if node.data.get('scenario') is None:
                    return ""
                elif node.data['scenario'].get(var_name) is None:
                    return ""
                else:
                    return np.array_str(node.data.get('scenario').get(var_name), 
                                        precision=scenario_precision)
        else:
            def print_on_nodes(node):
                if node.data.get('scenario') is None:
                    return ""
                elif var_name.get(node.level) is None:
                    return ""
                elif node.data['scenario'].get(var_name[node.level]) is None:
                    return ""
                else:
                    return np.array_str(node.data['scenario'][var_name[node.level]], 
                                        precision=scenario_precision)
            
        def print_on_edges(node):
            if node.data.get('W') is not None:
                return f"{node.data.get('W'):{format_weights}}"
            else:
                return ""

        return super().plot(print_on_nodes=print_on_nodes,
                             print_on_edges=print_on_edges,
                             **kwargs)
    
    def plot_trajectories(self, 
                       var_name,
                       component=0,
                       figsize=(10,5),
                       color=None):                
        fig, ax = plt.subplots(figsize=figsize)
        i = 0
        color_fct = lambda i: f"C{i%10}" if color is None else lambda i: color
        for node in self.nodes:
            # plot dots at all nodes but leaves
            if not node.is_leaf:
                if node.data.get("scenario", {}).get(var_name) is None:
                    continue
                plt.scatter(node.level, node.data["scenario"][var_name][component], marker="", c=color_fct(i))
                #plot links between the dots
                for m in node.children:
                    if m.data.get("scenario", {}).get(var_name) is None:
                        continue
                    x = [node.level, m.level]
                    y = [node.data["scenario"][var_name][component], m.data["scenario"][var_name][component]]
                    ax.plot(x, y, c=color_fct(i))
                    # plot dots at leaves
                    if node.is_parent_of_leaf:
                        ax.scatter(m.level, m.data["scenario"][var_name][component], marker="", c=color_fct(i))
                i += 1
        return ax
        
    def plot_hist(self, 
                  stage,
                  var_name,
                  component=0,
                  bins=10,
                  figsize=(5,5),
                  return_mean_std=False,
                  xlim=None,
                  ax=None):
        """Plot the weighted histogram of the scenario-tree values at stage."""
        assert stage <= self.depth-1, f"Stage {stage} is higher than maximum scenario-tree stage {self.depth-1}."
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        hist_data = [node.data["scenario"][var_name][component] for node in self.nodes_at_level(stage)]
        hist_weight = [node.data["W"] for node in self.nodes_at_level(stage)]
        ax.hist(hist_data, bins=bins, density=True, weights=hist_weight)
        # empirical mean and std
        mean = np.sum([w*x for (x, w) in zip(hist_data, hist_weight)])
        std = np.sqrt(np.sum([w*(x**2) for (x, w) in zip(hist_data, hist_weight)]) - mean**2)
        ax.set_title(f"mean: {mean:.3f} ; std: {std:.3f}, \n"
                     f"min: {min(hist_data):.3f} ; max: {max(hist_data):.3f}")
        if xlim is not None:
            ax.set_xlim(*xlim)
            
        return ax
        
    # --- Scenario interpolation --- 
    def nearest_nodes(self, n_nearest, scenario_path, across_tree=True, norm_ord=2, metric=None):
        """Finds the nodes closest to a reference scenario path.
        
        Arguments:
        ----------
        n_nearest: integer >= 1
            The number of nearest neighboors to find.
            
        scenario_path: dictionary mapping stage to scenario
            The stages are integer starting at 0 up to some stage, and the scenario are given as 1darray
            
        across_tree: boolean
            If True, closest nodes are found by comparing the whole scenario path to that of all 
            the nodes at the stage of interest. If False, the comparison is done stage by stage with the scenario
            at the child nodes and moving one stage forward in the tree until the stage of interest is reached and 
            the closest node is found.
        
        norm_ord: float > 0 (default 2)
            The order of the norm considered for the distance (only if `metric` is None).
            
        metric: function mapping a tuple of two 1darray to a positive number
            The metric used to compute the distance (if not provided, then the standard norm is used)
            
        Returns:
        iterator: iterator on the nearest nodes ranked by increasing distance.
        """
        stage = max(scenario_path.keys())
        if stage == 0:
            return [self]
        assert n_nearest <= self.width[stage-1], (f"The number of nearest neighbors ({n_nearest}) is larger than "
                                                    f"the number of nodes at stage {stage}: {self.width[stage-1]}")
        if metric is None:
            metric = lambda x, y: np.linalg.norm(x - y, ord=norm_ord)
            
        if across_tree:
            return self._nn_across_tree(n_nearest, scenario_path, metric)
        else:
            return self._nn_across_children(n_nearest, scenario_path, metric)
     
    def _from_scenario_path_to_numpy(self, scenario_path):
        return np.concatenate([scenario_path[stage][var_name]
                                   for stage in scenario_path.keys()
                                       if scenario_path[stage] is not None
                                           for var_name in scenario_path[stage].keys()])
    
    def _from_scenario_path_to_map_stage_to_rvar_nb(self, scenario_path):
        return {stage: {var_name: len(scenario_path[stage][var_name]) for var_name in scenario_path[stage].keys()}
                            for stage in scenario_path.keys() if scenario_path[stage] is not None}
        
    def _nn_across_tree(self, n_nearest, scenario_path, metric=None):
        """Nearest neighbors across tree"""
        ref_scenario = self._from_scenario_path_to_numpy(scenario_path) # numpy array
        map_stage_to_rvar_nb = self._from_scenario_path_to_map_stage_to_rvar_nb(scenario_path)
        map_stage_to_rvar_names = {stage: list(map_stage_to_rvar_nb[stage].keys()) 
                                            for stage in map_stage_to_rvar_nb.keys()}
        distances = {}
        stage = max(scenario_path.keys())
        for node in self.nodes_at_level(stage):
            node_scenario = self.get_path_as_numpy(node, map_stage_to_rvar_names) # numpy array
            distances[node] = metric(ref_scenario, node_scenario)
        if n_nearest == 1:
            return [min(distances.keys(), key=lambda node: distances[node])]
        else:
            return sorted(distances.keys(), key=lambda node: distances[node])[:n_nearest]
    
    def _nn_across_children(self, n_nearest, scenario_path, metric=None):
        """Nearest neighbors across children"""
        nearest_nodes = []
        scen_tree = self.copy()
        for _ in range(n_nearest):
            node = scen_tree
            while node.level + 1 in scenario_path.keys():
                distances = {child: metric(child.data["scenario"], scenario_path[child.level]) 
                                        for child in node.children}
                node = min(distances, key=lambda child: distances[child])
            nearest_nodes.append(node)
            node.remove_branch()
        return nearest_nodes
                       
    # --- Copy, save, load ---
    def copy(self, deep_copy=False):
        return self.__class__(Node.copy(self, deep_copy))
           
    @classmethod
    def from_file(cls, path, extension):
        return cls(Node.from_file(path, extension))

        
def average(scenario_tree: ScenarioTree, 
            map_stage_to_rvar_names: Optional[Dict[int, List[str]]] = None) -> ScenarioTree:
    """Return a new scenario tree with some scenarios replaced by their average value.
    Note: this function deep copies the input scenario tree so as to not replace the scenarios 'in place' unlike 
    the method `average` of ScenarioTree.
    
    Argument:
    ---------
    map_stage_to_rvar_names: Dict[int, List[str]] or None (default: None)
        The stages (int) and variables names (List[str]) for which the scenarios are averaged.
        
    Returns:
    --------
    instance of ScenarioTree: the scenario tree with some scenarios averaged
    """
    scen_tree = copy.deepcopy(scenario_tree)
    scen_tree.average(map_stage_to_rvar_names)
    return scen_tree  

def decompose(scenario_tree: ScenarioTree) -> List[ScenarioTree]:
    """ Return a list of scenario trees each with a single scenario up to the leaf`"""
    return [combtree_from_scenarios(scenario_tree.get_path_as_numpy(leaf)[np.newaxis], 
                                    scenario_tree.map_stage_to_rvar_nb) for leaf in scenario_tree.leaves] 

def collapse_twostage(tree: ScenarioTree) -> ScenarioTree:
    """Return the scenario tree built from `self` by merging identical scenarios. 
    Note: Works only for two-stage."""
    assert tree.depth == 2, "Scenario tree should be two-stage"
    unique_scenarios, inverse_indices = np.unique(tree.to_numpy(), axis=0, return_inverse=True)
    weights = np.array([child.data["W"] for child in tree.children])
    new_weights = [np.sum(weights[inverse_indices == index]) for index in range(len(unique_scenarios))]
    return twostage_from_scenarios(unique_scenarios, tree.map_stage_to_rvar_nb[1], new_weights)
                
def product(tree1: ScenarioTree, tree2: ScenarioTree):
    """Return the product of two scenario trees (works for two-stage only).
    The product scenario tree represents the joint uncertainty of the two distributions defined 
    by the input scenario trees. Note that only data keys "W" and "scenario" are built in 
    the output tree (all other keys in in the input trees will not be copied)."""
    assert tree1.depth == 2 and tree2.depth == 2, "Scenario trees should be two-stage"
    new_tree = from_bushiness([tree1.width[-1] * tree2.width[-1]])
    new_tree.data["W"] = 1
    for k, child in enumerate(new_tree.children):
        k2, k1 = k // tree1.width[-1], k % tree1.width[-1]
        data1 = tree1.node_at_address((k1,)).data
        data2 = tree2.node_at_address((k2,)).data
        child.data["W"] = data1["W"] * data2["W"]
        child.data["scenario"] = {**data2["scenario"], **data1["scenario"]}
    return new_tree

def _concatenate(tree1, tree2, deep_copy=False):
    final_tree = tree1.copy(deep_copy)
    for leaf in list(final_tree.leaves):
        leaf.add(*tree2.copy(deep_copy).children)
        for node in leaf.nodes:
            if node.address != leaf.address:
                node.data["W"] *= leaf.data["W"]
    return final_tree
            
def concatenate(trees: List[ScenarioTree], deep_copy: Optional[bool] = False):
    assert isinstance(trees, (list, tuple)) and len(trees) >= 2, \
        f"There must be at least 2 scenario trees."
    final_tree  = trees[0].copy(deep_copy)
    for tree in trees[1:]:
        final_tree = _concatenate(final_tree, tree.copy(deep_copy))
    return final_tree
    
from_file = ScenarioTree.from_file
from_topology = ScenarioTree.from_topology
from_recurrence = ScenarioTree.from_recurrence
from_bushiness = ScenarioTree.from_bushiness
from_data_dict = ScenarioTree.from_data_dict
twostage_from_scenarios = ScenarioTree.twostage_from_scenarios
combtree_from_scenarios = ScenarioTree.combtree_from_scenarios
optimal_width_vector = ScenarioTree.optimal_width_vector

                
get_scenario_path = ScenarioTree.get_scenario_path

