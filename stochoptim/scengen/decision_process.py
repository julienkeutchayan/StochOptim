# -*- coding: utf-8 -*-

from typing import Dict, Union, Tuple, Callable, List, Optional

import numpy as np

from stochoptim.scengen.scenario_tree import ScenarioTree
from stochoptim.scengen.tree_structure import Node

# shortcuts for important types 
Num = Union[int, float]
SubscriptType = Union[int, str, Tuple[Union[int, str], ...]] # variable subscript
FixedDecisionType = Optional[Num]
ScenarioPathType = Dict[int, Dict[str, np.ndarray]]
CallableDecisionType = Callable[[ScenarioPathType], FixedDecisionType]
DecisionType = Union[FixedDecisionType, CallableDecisionType]


class DecisionProcess:
    """The process that maps scenario paths to decisions"""
    
    def __init__(self, 
                 map_dvar_to_index: Dict[int, Dict[str, Dict[SubscriptType, int]]], 
                 input_decision_dict: Optional[Dict[int, Dict[str, List[DecisionType]]]] = None,
                 checker=False):
        """
        Arguments:
        ----------
        map_dvar_to_index: Dict[int, Dict[str, Dict[SubscriptType, int]]]
            The mapping from the stages to the decision variables that defines the stochastic problem.
            The mapping is: stage (int) -> variable name (str) -> variable subscript (SubscriptType) 
                                                                -> index in the array (int)
            
        input_decision_dict: Dict[int, Dict[str, List[DecisionType]]] or None (default: None)
            The decisions that are fixed beforehand. (More fixed decisions can be added later on via the 
            `update_decision_array` and `update_decision_value` methods)
            
        checker: bool (default: True)
            If True, check that the decision variables have the appropriate size (as specified
            in the stochastic problem) when being updated.
        """
        self._map_dvar_to_index = map_dvar_to_index
        self._map_stage_to_dvar_nb: Dict[int, Dict[str, int]] = {}
        self._decision_dict: Dict[int, Dict[str, List[DecisionType]]] = {}
        self._checker = checker
        self._initialize_decision_dict()
        
        if input_decision_dict is not None:
            self.update_decision_dict(input_decision_dict)                
                    
    # --- Properties ---
    @property
    def n_stages(self):
        return len(self._map_stage_to_dvar_nb.keys())
    
    @property
    def get_dict(self):
        return self._decision_dict
    
    @property
    def map_stage_to_dvar_nb(self):
        return self._map_stage_to_dvar_nb
    
    @property
    def map_dvar_to_index(self):
        return self._map_dvar_to_index
    
    # --- Initialization ---
    def _initialize_decision_dict(self):
        for stage in self._map_dvar_to_index.keys():
            self._decision_dict[stage] = {}
            self._map_stage_to_dvar_nb[stage] = {}
            for var_name, map_subscript_to_index in self._map_dvar_to_index[stage].items():
                self._map_stage_to_dvar_nb[stage][var_name] = len(map_subscript_to_index.values())
                self._decision_dict[stage][var_name] = np.array([None] * self._map_stage_to_dvar_nb[stage][var_name])
           
    # --- Update decisions ---         
    def update_decision_dict(self, 
                             new_decision_dict: Dict[int, Dict[str, List[DecisionType]]]):
        for stage in self._map_dvar_to_index.keys():
            if new_decision_dict.get(stage) is None:
                continue
            for var_name in self._map_dvar_to_index[stage].keys():
                array = new_decision_dict[stage].get(var_name)
                if array is not None:
                    self.update_decision_array(array, stage, var_name)   
                        
    def update_decision_array(self, 
                              new_decision_array: List[DecisionType], 
                              stage: int, 
                              var_name: str):
        """Update a whole array of decisions"""
        new_decision_array = np.array(new_decision_array)
        assert len(new_decision_array.shape) == 1, "The new decision array should be 1-dimensional"
        if self._checker:
            assert new_decision_array.shape[0] == self._map_stage_to_dvar_nb[stage][var_name], \
                (f"The new decision array should have the same size as the previous one, namely "
                 f"{(self._map_stage_to_dvar_nb[stage][var_name],)}, not {new_decision_array.shape}.")
        self._check_var_name(stage, var_name)
        self._decision_dict[stage][var_name] = new_decision_array
        
    def update_decision_value(self, 
                                new_decision: DecisionType, 
                                stage: int, 
                                var_name: str, 
                                var_subscript: SubscriptType):
        """Update a single decision"""
        assert new_decision is None or isinstance(new_decision, (int, float)) or callable(new_decision), \
            f"The new decision should an int, float, None or a callable, not {type(new_decision)}"
        self._check_var_name(stage, var_name)
        index = self._map_dvar_to_index[stage][var_name][var_subscript]
        self._decision_dict[stage][var_name][index] = new_decision
            
    def _check_var_name(self, stage, var_name):
        assert var_name in self._map_dvar_to_index[stage].keys(), (f"Variables {var_name} does not exist at stage "
            f"{stage}, should be one of {list(self._map_dvar_to_index[stage].keys())}")
        
    # --- Get decisons ---               
    def _callable_mask(self, stage, var_name):
        """Returns a mask on the variables with True if it is callable"""
        assert isinstance(self._decision_dict[stage][var_name], np.ndarray), \
            f"Decisions must be of type numpy.ndarray, not {type(self._decision_dict[stage][var_name])}"
        # if type is not object then all decisions are int/float
        if self._decision_dict[stage][var_name].dtype != np.dtype('O'): 
            return np.zeros((self._map_stage_to_dvar_nb[stage][var_name]), dtype='bool')
         # if all decisions are None then none are callable
        if (self._decision_dict[stage][var_name] == None).all():
            return np.zeros((self._map_stage_to_dvar_nb[stage][var_name]), dtype='bool')
        # if some decision are not None, then we check one by one if they are callable
        else: 
            return np.vectorize(callable)(self._decision_dict[stage][var_name])
    
    def get_decision_value(self, 
                            stage: int, 
                            var_name: str, 
                            var_subscript: SubscriptType,
                            scenario_path: Optional[ScenarioPathType] = None) -> FixedDecisionType:
        """Return a single decision"""
        index = self._map_dvar_to_index[stage][var_name][var_subscript]
        if callable(self._decision_dict[stage][var_name][index]):
            return self._decision_dict[stage][var_name][index](scenario_path)
        else:
            return self._decision_dict[stage][var_name][index]
        
    def get_decision_array(self,
                            stage: int,
                            var_name: str,
                            scenario_path: Optional[ScenarioPathType] = None) -> List[FixedDecisionType]:
        """Return a whole array of decisions"""
        callable_mask = self._callable_mask(stage, var_name)
        if not callable_mask.any(): # if no variable is callable
            return self._decision_dict[stage][var_name]
        else:
            decision_array = self._decision_dict[stage][var_name]
            for index in callable_mask.nonzero()[0]:
                decision_array[index] = self._decision_dict[stage][var_name][index](scenario_path)
            return decision_array
        
    def get_decision_dict(self, 
                           stage: int, 
                           scenario_path: Optional[ScenarioPathType] = None) -> Dict[str, List[FixedDecisionType]]:
        """Return a whole dictionary of decisions at one stage"""
        return {var_name: self.get_decision_array(stage, var_name, scenario_path) 
                             for var_name in self._decision_dict[stage].keys()}
                            
    def __call__(self, 
                 node: Node, 
                 var_name: Optional[str] = None, 
                 var_subscript: Optional[SubscriptType] = None) -> Union[FixedDecisionType, 
                                                                         List[FixedDecisionType], 
                                                                         Dict[str, List[FixedDecisionType]]]:
        if var_name is None and var_subscript is None:
            return self.get_decision_dict(node.level, ScenarioTree.get_scenario_path(node))
        
        if var_name is not None and var_subscript is None:
            return self.get_decision_array(node.level, var_name, ScenarioTree.get_scenario_path(node))
        
        if var_name is not None and var_subscript is not None:
            return self.get_decision_value(node.level, var_name, var_subscript, ScenarioTree.get_scenario_path(node))
        
    # --- Representation ---
    def __repr__(self):
        str_ = f"Decision process: ({self.n_stages} stages)\n"
        # dvar
        dvar_nb_list = [sum(self._map_stage_to_dvar_nb[t].values()) for t in range(self.n_stages)]
        str_ += f"  - decision variables: {dvar_nb_list}\n"
        for stage in range(self.n_stages):
            str_ += f"    - stage {stage}: \n"
            for var_name, dict_ in self._map_dvar_to_index[stage].items():
                n_fixed = (self._decision_dict[stage][var_name] != None).sum()
                n_callable = self._callable_mask(stage, var_name).sum()
                if n_fixed > 0 or n_callable > 0:
                    str_ += f"      - {var_name}: {len(dict_.values())} (fixed: {n_fixed}, callable: {n_callable})\n"
                else:
                    str_ += f"      - {var_name}: {len(dict_.values())}\n"
        return str_

    @classmethod
    def from_problem(cls, 
                     stochastic_problem, 
                     input_decision_dict: Optional[Dict[int, Dict[str, List[DecisionType]]]] = None):
        assert hasattr(stochastic_problem, 'map_dvar_to_index'), ("The stochastic problem "
        "must have an attribute `map_dvar_to_index`")
        return cls(stochastic_problem.map_dvar_to_index, input_decision_dict)        

from_problem = DecisionProcess.from_problem