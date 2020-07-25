# -*- coding: utf-8 -*-

from typing import Dict, Optional, Callable
import numpy as np

from stochoptim.scengen.tree_structure import get_data_path
from stochoptim.scengen.tree_structure import Node

ScenarioType = Dict[str, np.ndarray]
ScenarioPathType = Dict[int, ScenarioType]

class VariabilityProcess:
    """The process that maps scenario paths to variability values"""

    def __init__(self, 
                 lookback_fct: Optional[Callable[[int, ScenarioPathType], float]] = None, 
                 looknow_fct: Optional[Callable[[int, np.ndarray], float]] = None, 
                 average_fct: Optional[Callable[[int], float]] = None, 
                 name: Optional[str] = None,
                 checker: bool = True):
        """
        Arguments:
        ----------
        lookback_fct: Callable[[int, Dict[int, Dict[str, 1d-array]]], float] or None (default: None)
            See the signature of the `lookback` method.
            
        looknow_fct: Callable[[int, 1d-array], float] or None (default: None)
            See the signature of the `looknow` method.
            
        average_fct: Callable[[int], float] or None (default: None)
            See the signature of the `average` method.
            
        name: str or None (default: None)
            Name of the variability process
        """
        self._lookback_fct = lookback_fct
        self._looknow_fct = looknow_fct
        self._average_fct = average_fct
        self._name = name
        self._checker = checker
        
    def has_lookback(self) -> bool:
        return self._lookback_fct is not None

    def has_looknow(self) -> bool:
        return self._looknow_fct is not None
    
    def has_average(self) -> bool:
        return self._average_fct is not None
    
    def lookback(self, stage: int, scenario_path: ScenarioPathType) -> float:
        r""" Path-dependent variability function. 
        
        Mathematically it is the implementation of
            - V_0   (variability at stage 0)
            - V_{stage}(\xi_1, ..., \xi_{stage})    (variability at stage >= 1).
            
        Arguments:
        ----------           
        stage: int >= 0
            Current stage of the variability function to be computed (starting at stage 0).
            
        scenario_path: Dict[int, Dict[str, 1d-array]]
            History of the process up to and including the current stage.
                
        Returns:
        --------
        float >= 0: Value of variability.
        """
        varability = self._lookback_fct(stage, scenario_path)
        if self._checker:
            assert isinstance(varability, (int, float)) or varability.shape == (), \
            f"The `lookback_fct` function must return an int or float, not {type(varability)}"
        return varability
    
    def looknow(self, stage: int, epsilon: np.ndarray) -> float:
        r""" Path-independent variability function. 
        
        Mathematically it is the implementation of
            - V_0   (variability at stage 0)
            - V_{stage}(\epsilon_{stage})    (variability at stage >= 1).
            
        Arguments:
        ----------           
        stage: int >= 0
            Current stage of the variability function to be computed (starting at stage 0).
            
        epsilon: 1d-array
            Discretization point of \epsilon_{stage}, of shape (dim_epsilon,).
                
        Returns:
        --------
        float >= 0: Value of variability.
        """
        varability = self._looknow_fct(stage, epsilon)
        if self._checker:
            assert isinstance(varability, (int, float)) or varability.shape == (), \
            f"The `looknow_fct` function must return an int or float, not {type(varability)}"
        return varability
            
    def average(self, stage: int) -> float:
        r""" Average variability, defined as the expectation of the path-dependent variability under the scenario
        process. 
        
        Mathematically: 
            V_0 (variability at stage 0)
            E[V_{stage}(\xi_1, ..., \xi_{stage})] for stage = 1, ..., last_stage-1
            
        Argument:
        ---------
        stage: int >= 0
        
        Returns:
        --------
        float >= 0: Average variability
        """
        varability = self._average_fct(stage)
        if self._checker:
            assert isinstance(varability, (int, float)) or varability.shape == (), \
            f"The `average_fct` function must return an int or float, not {type(varability)}"
        return varability
 
    def node_lookback(self, node: Node) -> float:
        """Same as `lookback` but callable on a node"""
      #  if node.is_leaf:
      #      return 0
      #  else:
        return self.lookback(node.level, get_data_path(node, 'scenario'))
    
    def node_looknow(self, node: Node) -> float:
        """Same as `looknow` but callable on a node"""
     #   if node.is_leaf:
     #       return 0
     #   else:
        return self.looknow(node.level, node.data.get('eps'))
      
    @classmethod
    def from_scenario_tree(cls, scenario_tree, across_tree: bool) -> 'VariabilityProcess':
        assert scenario_tree.has_key('v'), "Some nodes do not have the data key 'v'."
        return cls(lookback_fct=lambda stage, scenario_path: inferred_lookback(stage, scenario_path, 
                                                                               scenario_tree, across_tree),
                    average_fct=lambda stage: inferred_average(stage, scenario_tree))       
    

from_scenario_tree = VariabilityProcess.from_scenario_tree


def inferred_lookback(stage, scenario_path, scenario_tree, across_tree: bool):
    if stage == 0:
        return np.std([child.data['v'] for child in scenario_tree.children]) 
    elif stage == scenario_tree.depth - 1:
        return 0
    else:
        nearest_node = list(scenario_tree.nearest_nodes(1, scenario_path, across_tree))[0]
        return np.std([child.data['v'] for child in nearest_node.children])


def inferred_average(stage, scenario_tree):
    return sum(node.data["W"] * np.std([child.data['v'] for child in node.children])
                   for node in scenario_tree.nodes_at_level(stage))
            
            
            
            