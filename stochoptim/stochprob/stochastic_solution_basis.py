# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Optional, Callable

from stochoptim.scengen.decision_process import DecisionProcess
from stochoptim.scengen.scenario_tree import ScenarioTree

class StochasticSolutionBasis:
        
    def __init__(self, 
                 stochastic_problem,
                 scenario_tree):
        
        self._stochastic_problem = stochastic_problem
        self._scenario_tree = scenario_tree
            
    # --- Properties ---
    @property
    def stochastic_problem(self):
        """The stochastic problem"""
        return self._stochastic_problem
        
    @property
    def scenario_tree(self):
        """The scenario tree"""
        return self._scenario_tree
    
    @property
    def x0(self):
        """The stage-0 decisions"""
        return self._scenario_tree.data.get("decision")
        
    @property
    def status(self):
        return self._scenario_tree.data.get('status')
    
    @property
    def objective_bound(self):
        return self._scenario_tree.data.get('bound', np.nan)

    @property
    def objective_value(self):
        return self._scenario_tree.data.get('v', np.nan)

    @property
    def solve_time(self):
        return self._scenario_tree.data.get('time', np.nan)
    
    @property    
    def objective_value_at_leaves(self):
        return np.array([leaf.data.get('v', np.nan) for leaf in self._scenario_tree.leaves])
    
    @property
    def gap(self):
        return abs(self.objective_bound - self.objective_value)
    
    @property
    def gap_percent(self):
        return 100 * self.gap / abs(10**(-10) + self.objective_value)
    
    @property
    def n_scenarios(self):
        return sum(1 for _ in self._scenario_tree.leaves)
    
    def warmstart(self, excluding_var_names: Optional[Dict[int, List[str]]] = None):
        """Build a warmstart dictionary from the scenario-tree solution.
        This dictionary can be given as argument of StochasticProblemBasis.solve() under the 
        keyword 'warmstart'.
        
        Returns:
        --------
        dictionary mapping variable names (str) to values (int or float).
        """
        if excluding_var_names is None:
            excluding_var_names = dict()
        warmstart_dict = {}
        for node in self._scenario_tree.nodes:
            for var_name, map_subcript_to_index in self._stochastic_problem.map_dvar_to_index[node.level].items():
                if var_name in excluding_var_names.get(node.level, []):
                    continue
                for subscript, index in map_subcript_to_index.items():
                    if isinstance(subscript, tuple):
                        subscript_str = '_'.join([str(s) for s in subscript])
                    else:
                        subscript_str = str(subscript)
                    key = f'{var_name}_{node.address}_{subscript_str}'
                    var_type = self._stochastic_problem.map_stage_to_var_type[node.level][var_name][0]
                    if var_type in ['B', 'I']:
                        warmstart_dict[key] = int(np.round(node.data['decision'][var_name][index]))
                    else:
                        warmstart_dict[key] = float(node.data['decision'][var_name][index])
        return warmstart_dict
    
    # --- To display solution ---
    def set_path_info(self, scen_index: Optional[int] = None):
        if scen_index is None: # root
            self._stochastic_problem.set_path_info(self._scenario_tree)
        else:    
            self._check_scenario_index_validity(scen_index)
            leaf = next(leaf for k, leaf in enumerate(self._scenario_tree.leaves) if k == scen_index)
            self._stochastic_problem.set_path_info(leaf)

    def decision_path(self, scen_index: int) -> Dict[int, Dict[str, np.ndarray]]:
        """Return a path of decisions from the root to a leaf"""
        self._check_scenario_index_validity(scen_index)
        self.set_path_info(scen_index)
        return self._stochastic_problem._decision_path
       # leaf = next(leaf for k, leaf in enumerate(self._scenario_tree.leaves) if k == scen_index)
       # return self._scenario_tree.get_data_path(leaf, 'decision')
  
    def decision_process_fixed(self, 
                               var_names: Dict[int, List[str]],
                               scen_index: Optional[int] = None) -> DecisionProcess:
        """Return a decision process made of a selection of decision variables at each stage. 
        (The decision process is said to be fixed because it does not depend on the scenario path.)
        
        var_names:
            Dictionary mapping a stage to the list of variable names to be included in the decision process
            at that stage.
        """
        assert not (scen_index is None and list(var_names.keys()) != [0]), \
            "Only variables at stage 0 can be considered if `scen_index` is None."
        # generate empty decision process
        decision_process = DecisionProcess(self._stochastic_problem._map_dvar_to_index)
        # get decision path on the required scen_index
        decision_path = self.decision_path(scen_index) if scen_index is not None else {0: self.x0}
        # add one by one the required variables to the process 
        for stage in var_names.keys():
            for var_name in var_names[stage]:
                decision_process.update_decision_array(decision_path[stage][var_name], stage, var_name)
        return decision_process
    
    def generate_empty_decisions(self, null_array: Optional[Callable[[int], np.ndarray]] = None):
        """Initialize an empty numpy array of appropriate length for each and every decision."""
        if null_array is None:
            null_array = lambda n: np.empty(n)
        for stage, map_name_to_nb in self.stochastic_problem.map_dvar_name_to_nb.items():
            for node in self.scenario_tree.nodes_at_level(stage):
                node.data['decision'] = {}
                for name, nb in map_name_to_nb.items():
                    node.data['decision'][name] = null_array(nb)   
            
    # --- Sanity ---
    def _check_scenario_index_validity(self, scen_index):
        assert scen_index in range(self.n_scenarios), \
                (f"Scenario index should be between 0 and {self.n_scenarios-1}, not {scen_index}.")
                
    def _check_x0_validity(self, x0):
        assert len(x0) == sum(self.stochastic_problem.map_dvar_name_to_nb[0].values()), \
            (f"Sanity check failed: mistmatch between the dimension of x0 ({len(x0)}) "
                "and the theoretical number of stage-0 decisions of self.stochastic_problem "
                f"({sum(self.stochastic_problem.map_dvar_name_to_nb[0].values())}")
        
    def _check_sanity(self):
        self._check_x0_validity(self.x0)
       
    # --- Representations ---
    def __repr__(self):
        return (f"obj={self.objective_value:,.3f}, "
                f"gap={self.gap_percent:.3f}%, "
                f"time={self.solve_time:.1f} sec")
        
    # ---- Load, save, copy ---                   
    @classmethod
    def from_file(cls, 
                  problem_class_or_module,
                  path_tree, 
                  path_problem, 
                  extension_tree='pickle', 
                  extension_prob='txt'):
        """ Load network design solution from two files: one for the problem and one for the scenario tree.
        Arguments:
        ----------
        problem_class_or_module: class or module containing the stochastic problem.
            This must have a method/function `from_file`.
            
        path_tree: str
        
        extension_tree: {'pickle', 'txt'}
        
        path_problem: str
        
        extension_prob: {'pickle', 'txt'}
        Returns:
        --------
        instance of StochasticSolutionBasis or a subclass
        """            
        assert hasattr(problem_class_or_module, 'from_file'), \
            "The problem class or module given as input `problem_class_or_module` must have a method `from_file()`."
        problem = problem_class_or_module.from_file(path_problem, extension_prob)
        scenario_tree = ScenarioTree.from_file(path_tree, extension_tree)
        return cls(problem, scenario_tree)
    
    def to_file(self, 
                path_tree,
                extension_tree, 
                path_problem,
                extension_prob, 
                with_keys=None, 
                without_keys=None):
        """ Save the scenario tree (containing the optimal decisions) and the stochastic problem in two
        separated files.
        
        Arguments:
        ----------
        path_tree: string
            Path to the file containing the scenario tree information.
            
        path_prob: string
            Path to the file containing the stochastic problem information.
            
        extension_tree: {'txt', 'pickle'}
            Format in which the scenario tree is solved.
            
        extension_prob: {'txt', 'pickle'}
            Format in which the stochastic problem is solved.
        
        with_keys: list of strings or None
            If not None, it is the data keys to be saved in the scenario tree. 
            If None, then all keys in the list will be saved except those in 'without_keys'.
            
        without_keys: list of strings or None
            This argument is used only if 'with_keys' is None. 
            In that case it contains the data keys not to be saved in the scenario tree.
        """
        self._scenario_tree.to_file(path_tree, extension_tree, with_keys, without_keys)
        self._stochastic_problem.to_file(path_problem, extension_prob)