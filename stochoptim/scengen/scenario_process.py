# -*- coding: utf-8 -*-

from typing import Callable, Tuple, Optional, Dict, Union

import numpy as np
from scipy.stats import norm
 
from stochoptim.scengen.tree_structure import Node

ScenarioType = Dict[str, np.ndarray]
ScenarioPathType = Dict[int, ScenarioType]

class ScenarioProcess:
    """The process that fills a tree structure with scenarios.
    
    Arguments:
    ----------
    scenario_fct: Callable[[int, 1d-array, Dict[int, Dict[str, 1d-array]]], Dict[str, 1d-array]]
        Function that takes as argument the stage, the discretization point (called epsilon by convention), 
        and the scenario path up to (but not including) the stage, and returns the corresponding scenario.
        (See the signature of the `get_scenario` method.)
        
    epsilons_fct: Callable[[int, int], 2d-array] or None (default: None)
        Function that takes as argument the stage and the number of sample points, and returns a sample of 
        discretization points.
        (See the signature of the `get_epsilon_sample` method.)
        
    weights_fct: Callable[[int, int], 1d-array] or None (default: None)
        Function that takes as argument the stage and the number of sample points, and returns a sample of 
        discretization weights (probabilities if they sum to one).
        (See the signature of the `get_epsilon_sample` method.)
        
    name: str or None (default: None)
        Name of the scenario process
        
    checker: bool (default: False)
        If True, the type and size of the scenarios and the discretization points and weights are checked.
        
    stochastic_problem: instance of a subclass of StochasticProblemBasis or None (default: None)
        If provided (and if `checker` is True) the scenarios are checked to be in accordance with what is specified
        in the problem in terms of the names of the random variables and their dimensions.
    """
    
    def __init__(self, 
                 scenario_fct: Callable[[int, np.ndarray, ScenarioPathType], ScenarioType], 
                 epsilons_fct: Optional[Callable[[int, int], np.ndarray]] = None, 
                 weights_fct: Optional[Callable[[int, int], np.ndarray]] = None,
                 name: Optional[str] = None,
                 checker: bool = True,
                 stochastic_problem: Optional['StochasticProblemBasis'] = None):  
        """

        """
        self._scenario_fct = scenario_fct
        self._epsilons_fct = epsilons_fct
        self._weights_fct = weights_fct
        self._name = name
        self._checker = checker
        self._map_rvar_name_to_nb = stochastic_problem.map_rvar_name_to_nb if stochastic_problem is not None else None
    
    def get_epsilon_sample(self, 
                           n_samples: int, 
                           stage: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:   
        """Return the discretization points and weights of the stagewise independent random vector epsilon.
        
        Arguments:
        ----------
        n_samples: int >= 1
            Number of points (and weights) to be generated.
            
        stage: int >= 0
            Stage of the random variable epsilon to be discretized.
            
        Returns:
        --------
        Tuple (1d-array, 2d-array) where:
            
        1d-array: the discretization weights given as an array of shape (n_samples,).
                
        2d-array or None: the discretization points given as an array of shape (n_samples, dim_epsilon), for an 
        arbitrary value of dim_epsilon.
        """
        weights = self._weights_fct(n_samples, stage) if self._weights_fct is not None else np.ones(n_samples) / n_samples
        epsilons = self._epsilons_fct(n_samples, stage) if self._epsilons_fct is not None else None
        if self._checker:
            assert isinstance(weights, np.ndarray), \
                f"The discretization weights must be given as a 1d numpy array, not as type {type(weights)}"
            assert len(weights.shape) == 1, \
                f"The numpy array of weights must be a 1-dimensional, not of dimension {len(weights.shape)}"
            if epsilons is not None:
                assert isinstance(epsilons, np.ndarray), \
                    f"The discretization points (epsilons) must be given as a 2d numpy array, not as type {type(epsilons)}"
                assert len(epsilons.shape) == 2, \
                    f"The numpy array of discretization points (epsilons) must be 2-dimensional, not {len(epsilons.shape)}-dimensional"
                assert weights.shape[0] == epsilons.shape[0], ("Mismatch between the number of discretization weights "
                f"({weights.shape[0]}) and points ({epsilons.shape[0]}).")
        return weights, epsilons
    
    def get_scenario(self, 
                     stage: int, 
                     epsilon: Optional[np.ndarray], 
                     scenario_path: Optional[ScenarioPathType]) -> ScenarioType:
        r"""Return the scenario value given as a function of the epsilon sample point (the current stage contribution) 
        and the scenario path (the contribution of the history up to the current stage). 
        
        Mathematically it is the implementation of the recursion: 
            \xi_{0} = initial value,
            \xi_{stage} = g_t(\epsilon_{stage}, \xi_{..stage-1}), for stage=1,...
        
        Arguments:
        ----------           
        stage: int >= 0
            Current stage of the random parameters to be discretized.
            
        epsilon: 1d-array or None
            Discretization point of epsilon_{stage}, of shape (dim_epsilon,).
            These discretization points are the ones returned by the `get_epsilon_sample` method.
            
        scenario_path: Dict[int, Dict[str, 1d-array]] or None
            History of the process up to (but not including) the current stage.
            
        Returns:
        --------           
        Dict[str, 1d-array]: the discretization point of \xi_{stage}, of shape (dimension_of_\xi_{stage},)
        """
        scenario = self._scenario_fct(stage, epsilon, scenario_path)
        if self._checker and stage >= 1: # we don't check the scenario at stage 0 (at the root)
            assert isinstance(scenario, dict), \
                ("A scenario must be a dictionary mapping the variable names (str) to the variable arrays (1d-array)\n"
                 f"Here scenario was of type {type(scenario)}: {scenario}")
            if self._map_rvar_name_to_nb is not None:
                assert set(scenario.keys()) == set(self._map_rvar_name_to_nb[stage].keys()), \
                    (f"Mismatch between the variable names given in the problem at stage {stage}: "
                     f"{set(self._map_rvar_name_to_nb[stage].keys())}, "
                     f"and those found in the scenario process: {set(scenario.keys())}")
            for var_name, var_array in scenario.items():
                assert isinstance(var_array, np.ndarray), \
                    f"The variables must be given as a 1d numpy array, not as type {type(var_array)}"
                assert len(var_array.shape) == 1, \
                    f"The numpy array must be a 1-dimensional, not of dimension {len(var_array.shape)}"
                if self._map_rvar_name_to_nb is not None:
                    assert var_array.shape[0] == self._map_rvar_name_to_nb[stage][var_name], \
                    (f"Mismatch between the number of components of variable '{var_name}' given in the problem at "
                     f"stage {stage} ({self._map_rvar_name_to_nb[stage][var_name]}) "
                     f"and the number found in the scenario process ({var_array.shape[0]})")
        return scenario
    
    def get_children_sample(self, 
                            node: Node) -> Tuple[np.ndarray, np.ndarray]:
        """Same as .get_epsilon_sample() but callable on a node."""
        assert not node.is_leaf, "Cannot get the children epsilon sample of a leaf"
        return self.get_epsilon_sample(len(node.children), node.level + 1)
        
    def get_node_scenario(self, 
                          node: Node, 
                          path: bool) -> Union[ScenarioType, ScenarioPathType]:
        """Returns the scenario at a node or the scenario path leading to that node.
        
        Arguments:
        ----------
        node: Node
        
        path: bool
            If True, then the scenario path from the root to 'node' is returned.
            Otherwise only the scenario at 'node' is returned.
            
        Returns:
        -------
        Dict[str, 1d-array] (if path is False)
        Dict[int, Dict[str, 1d-array]] (if path is True)
            The scenario at the node or the scenario path leading to the node.
        """
        if path:
            return {stage: self.get_node_scenario(n, path=False) 
                    for stage, n in enumerate(node.branch)}
        else:
            return self.get_scenario(node.level, 
                                     node.data.get("eps"), 
                                     Node.get_data_path(node.parent, 'scenario') 
                                     if not node.is_root else None)


def mc_normal_sample(n_samples, mu=0, sigma=1):
    return np.ones(n_samples) / n_samples, np.random.normal(loc=mu, scale=sigma, size=(n_samples, 1))


def qmc_normal_sample(n_samples, mu=0, sigma=1, u=0.5):
    return np.ones(n_samples) / n_samples, \
            mu + sigma * norm.ppf((np.linspace(0, 1-1/n_samples, n_samples) + u) % 1).reshape(-1, 1)    
        


        
