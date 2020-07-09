# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy

from typing import Union, List, Dict, Tuple, Optional, Callable, Iterator, Any
import numpy as np
from docplex.mp.constr import LinearConstraint
from docplex.mp.linear import Var

from stochoptim.scengen.approx_prob import ApproximateProblem, _timeit
from stochoptim.scengen.scenario_tree import ScenarioTree, decompose
from stochoptim.scengen.tree_structure import Node
from stochoptim.scengen.decision_process import DecisionProcess
from stochoptim.stochprob.stochastic_solution_basis import StochasticSolutionBasis
from stochoptim.util import TimeIt

Num = Union[int, float]
Subscript = Union[int, str, Tuple[Union[int, str], ...]]

class StochasticProblemBasis(ABC):
    
    def __init__(self, 
                 name: str,
                 n_stages: int,
                 objective_sense: str,
                 is_obj_random: bool, 
                 is_mip: bool,
                 solution_class: Optional[StochasticSolutionBasis] = None,
                 initialize_indices: bool = True):
        
        self._name = name
        self._n_stages = n_stages
        self._objective_sense = objective_sense
        self._is_obj_random = is_obj_random
        self._is_mip = is_mip
        
        # class of the stochastic solution
        if solution_class is not None:
            self._solution_class = solution_class
        else:
            self._solution_class = StochasticSolutionBasis
            
        # variables indices
        # map: stage -> var_name -> var_subscript -> index
        self._map_dvar_to_index = {}
        self._map_rvar_to_index = {}
        # map: (stage, index) -> (var_name, var_subscript)
        self._map_index_to_dvar = {}
        self._map_index_to_rvar = {} 
        # map: stage -> var_name -> (var_type, lb, ub)
        self._map_stage_to_var_type = {}
            
        # precomputation
        # map: stage -> list of var_names
        self._name_precomputed_decisions = {}
        self._name_precomputed_parameters = {}        
        
        # path information 
        self._node = None
        self._decision_path = None
        self._scenario_path = None
        self._node_address = None
        self._memory_path = None
        self._sum = None
        self._dot = None
                    
        if initialize_indices:
            self._initialize_indices()
        self._initialize_precomputation()
        
    # --- Property ---   
    @property
    def name(self):
        return self._name
    
    @property
    def n_stages(self):
        return self._n_stages
    
    @property
    def objective_sense(self):
        return self._objective_sense
    
    @property
    def is_obj_random(self):
        return self._is_obj_random
    
    @property
    def is_mip(self):
        return self._is_mip
    
    @property
    def map_stage_to_var_type(self) -> Dict[int, Dict[str, Tuple[str, Optional[Num], Optional[Num]]]]:
        return self._map_stage_to_var_type
    
    @property
    def map_dvar_to_index(self) -> Dict[int, Dict[str, Dict[Subscript, int]]]:
        return self._map_dvar_to_index
    
    @property
    def map_dvar_name_to_nb(self) -> Dict[int, Dict[str, int]]:
        return {stage: {var_name: len(variables) for var_name, variables in dvar.items()} 
                    for stage, dvar in self._map_dvar_to_index.items()}
    
    @property
    def map_stage_to_dvar_names(self) -> Dict[int, List[str]]:
        return {stage: list(dvar.keys()) for stage, dvar in self._map_dvar_to_index.items()}
    
    @property
    def map_rvar_to_index(self) -> Dict[int, Dict[str, Dict[Subscript, int]]]:
        return self._map_rvar_to_index
    
    @property
    def map_rvar_name_to_nb(self) -> Dict[int, Dict[str, int]]:
        return {stage: {var_name: len(variables) for var_name, variables in rvar.items()} 
                    for stage, rvar in self._map_rvar_to_index.items()}
    
    @property
    def map_stage_to_rvar_names(self) -> Dict[int, List[str]]:
        return {stage: list(rvar.keys()) for stage, rvar in self._map_rvar_to_index.items()}
    
    @property       
    def map_index_to_dvar(self) -> Dict[Tuple[int, str, int], Subscript]:
        return self._map_index_to_dvar
    
    @property       
    def map_index_to_rvar(self) -> Dict[Tuple[int, str, int], Subscript]:
        return self._map_index_to_rvar
    
    @property
    def name_precomputed_decisions(self) -> Dict[int, List[str]]:
        return self._name_precomputed_decisions
    
    @property
    def name_precomputed_parameters(self) -> Dict[int, List[str]]:
        return self._name_precomputed_parameters
    
    # --- Abstract methods ---
    @abstractmethod
    def decision_variables_definition(self, stage):
        """For each decision variables of type {y_{t, i} \in A : i \in I} where I is an indexing set 
        (a list of tuples), y_{t,i} is of type A (with A = binary (B), or integer (I), or continuous (C)), 
        and lb <= y_{t,i} <= ub, this function generates a 5-tuple of the form: ('y', I, lb, ub, A) 
        under the conditon stage == t. (lb and ub can be two iterables provided they match same size of I)"""
        pass
    
    @abstractmethod
    def random_variables_definition(self, stage):
        """For each random variable of type {xi_{stage, i} : i \in I} where I is an indexing set (a list of tuples), 
        this method generates a 2-tuple of the form ('xi', I)"""
        pass
    
    @abstractmethod
    def objective(self):
        pass
    
    @abstractmethod
    def deterministic_linear_constraints(self, stage):
        pass
    
    @abstractmethod
    def random_linear_constraints(self, stage):
        pass
    
    @abstractmethod
    def precompute_decision_variables(self, stage: int) -> Iterator[Tuple[str, Callable[[], Any]]]:
        pass
    
    @abstractmethod
    def precompute_parameters(self, stage: int) -> Iterator[Tuple[str, Callable[[], Any]]]:
        pass
    
    @abstractmethod
    def sanity_check(self, stage):
        pass
    
    def deterministic_indicator_constraints(self, stage):
        pass
    
    def random_indicator_constraints(self, stage):
        pass
    
    def deterministic_equivalence_constraints(self, stage):
        pass
    
    def random_equivalence_constraints(self, stage):
        pass
    
    def deterministic_if_then_constraints(self, stage):
        pass
    
    def random_if_then_constraints(self, stage):
        pass
    
    # --- To build approximate problem ---      
    def sum(self, args):
        if isinstance(args, list):
            args = np.array(args)
        if args.dtype == np.dtype('O'):
            return self._sum(args)
        else:
            return np.sum(args)
    
    def dot(self, var, coef):
        """
        var: list or 1d-array
        
        coef: int, float, list or 1d-array
        """
        if isinstance(var, list):
            var = np.array(var)
        # check that coef is a number of an array of the same size as var
        assert np.array(coef).shape == () or np.array(coef).shape[0] == var.shape[0], \
                f"The length of the arrays of variables ({var.shape[0]}) and coefficients ({len(coef)}) do not match."
        if var.dtype == np.dtype('O'): # instance of Var, LinearExpr
            coef = coef.tolist() if isinstance(coef, np.ndarray) else coef # docplex .dot is faster on a list
            return self._dot(var, coef)
        else:
            # if coef is a number we repeat it in an array otherwise np.dot behaves as np.multiply
            if np.array(coef).shape == (): 
                coef = np.tile(coef, var.shape[0])
            return np.dot(var, coef)

    def set_path_info(self, node, model=None):
        """Set the decision path and scenario path (and others) that will be used in building the approximate 
        problem."""
        self._node = node
        self._decision_path = Node.get_data_path(node, 'decision', default={})
        self._scenario_path = ScenarioTree.get_scenario_path(node)
        self._node_address = node.address
        self._memory_path = Node.get_data_path(node, 'memory', default={})
        self._sum = np.sum if model is None else model.sum
        self._dot = np.dot if model is None else model.dot
        self._constraints_generators = \
            {('deterministic', 'linear'): self.deterministic_linear_constraints(node.level),
             ('deterministic', 'indicator'): self.deterministic_indicator_constraints(node.level),
             ('random', 'linear'): self.random_linear_constraints(node.level),
             ('random', 'indicator'): self.random_indicator_constraints(node.level)}
            
    def generate_node_constraints(self, node, ct_type, model=None, remove_constraints=None):
        """
        ct_type: tuple (a, b) where a is 'deterministic' or 'random' and b is 'linear' or 'indicator'.
        
        remove_constraints: list of str or None (default: None)
            Contains the name of the constraints to be removed from the problem.
            If None, no constraint is removed.
        """
        self.set_path_info(node, model)   
        if self._node.data.get(f'{ct_type[0]}-ct') is None:
            self._node.data[f'{ct_type[0]}-ct'] = [] # will contain additionnal linear cts
            
        if not hasattr(self._constraints_generators[ct_type], '__next__'): # if not an iterator
            return iter(()) # empty iterator
        
        for ct_generator in self._constraints_generators[ct_type]:
            if remove_constraints is not None and ct_generator.__name__ in remove_constraints:
                continue
            for ct_tuple in ct_generator:
                if ct_type[1] == 'linear':
                    new_ct_tuple = self._inspect_linear_constraint(ct_tuple)
                elif ct_type[1] == 'indicator':
                    new_ct_tuple = self._inspect_indicator_constraint(ct_tuple, ct_type)
                elif ct_type[1] == 'equivalence':
                    raise NotImplementedError()
                elif ct_type[1] == 'if-then':
                    raise NotImplementedError()
                else:
                    raise ValueError("Wrong type of constraints, must be 'linear', 'indicator', 'equivalence' or "
                                     f"'if-then', not {ct_type[1]}")
                if new_ct_tuple is not None:
                    yield new_ct_tuple
                
    def objective_path(self, leaf, model=None):
        self.set_path_info(leaf, model)
        return self.objective()

    # --- check constraints validity ---
    def _inspect_linear_constraint(self, ct_tuple):
        # check the tuple input
        if not isinstance(ct_tuple, tuple):
            ct_tuple = (ct_tuple, '') # append empty name to it
        else:
            assert len(ct_tuple) == 2, \
            f"Each linear constraint should be a 2-tuple (constraint, name), instead we received type {type(ct_tuple)}."
        # check specifically the constraint    
        ct, name = ct_tuple
        if self.is_constraint_relevant(ct, name):
            return ct_tuple
        else:
            assert self.is_constraint_satisfied(ct, name), \
                    f"Constraint '{name}' is not satified at node address {self._node_address}"
       
    def _inspect_indicator_constraint(self, ct_tuple, ct_type):
        # check the tuple input
        if len(ct_tuple) == 3:
            ct_tuple = ct_tuple + ('',)
        elif len(ct_tuple) != 4:
            raise ValueError("Each indicator constraint should be a 3- or 4-tuple (indicator var, constraint, "
                             f"activation value, optional: name), instead we received {ct_tuple}")
        # check specifically the constraint and indicator variable
        var, ct, true_value, name = ct_tuple
        if isinstance(var, Var): # if indicator variable is not fixed beforehand
            if self.is_constraint_relevant(ct, name):
                assert var.is_binary(), \
                    f"The activation variable '{var}' in the indicator constraint '{name}' must be binary."
                return ct_tuple
            else:
                if not self.is_constraint_satisfied(ct, name):
                    self._node.data[f'{ct_type[0]}-ct'] += [(var == int(not true_value), name)]
        else: # if indicator variable is fixed beforehand
            if var == true_value:
                if self.is_constraint_relevant(ct, name):
                    self._node.data[f'{ct_type[0]}-ct'] += [(ct, name)]
                else:
                    assert self.is_constraint_satisfied(ct, name), (f"Indicator constraint {name} is not satisfied "
                                                                    f"at node address {self._node_address}")
            
    def is_constraint_relevant(self, ct, name):
        """Return True if the constraint includes a least one decision variable.
        Note: a constraint with no decision variables (i.e., with both side constant) may arise when some of the 
        decisions are fixed beforehand."""
        if isinstance(ct, LinearConstraint): # if the constraint is a docplex linear expression
            if ct.rhs.is_constant() and ct.lhs.is_constant(): # if left-hand side and right-hand side are both constant
                return False
            else:
                return True
        elif isinstance(ct, (bool, np.bool_, np.bool)): # if the constraint is a boolean
            return False
        else:
            raise TypeError(f"Unknown type for constraint '{name}' at node address {self._node_address}: {type(ct)}")
                    
    def is_constraint_satisfied(self, ct, name):
        """Return True if a constraint with both sides constant evaluates as True.
        Note: a constraint with both side constant may arise when some of the decisions are fixed beforehand."""
        if isinstance(ct, LinearConstraint):
            return eval(ct.to_string())
        elif isinstance(ct, (bool, np.bool_, np.bool)):
            return ct
        else:
            raise TypeError(f"Unknown type for constraint '{name}' at node address {self._node_address}: {type(ct)}")
            
    # --- Define and access variables and indices ---
    def get_dvar(self, 
                 stage: int, 
                 var_name: str, 
                 var_subscripts: Optional[Subscript] = None) -> Union[Num, np.ndarray]:
        """ Access a decision variable. 
        This parent method is the one that should be used in defining a stochastic problem in a child class of 
        StochasticProblemBasis.
        
        Note: the method `.set_path_info` must be called before calling this method.
        
        Arguments:
        ----------
        stage: int >= 0
        
        var_name: str
        
        var_subscripts: str, int, or tuple of int/str
            The appropriate type is the one used to define the variables in the .decision_variables_definition().
            
        Returns:
        --------
        int/float or 1d-array: the decision variable(s).
        """
        if var_subscripts is None:
            return self._decision_path[stage][var_name]
        else:
            index = self.get_dvar_index(stage, var_name, var_subscripts)
            try:
                return self._decision_path[stage][var_name][index]
            except IndexError:
                raise IndexError(f"Variable subscript {var_subscripts} (index = {index}) is out of bounds for '{var_name}' "
                f"at stage {stage}, which has {self._decision_path[stage][var_name].shape[0]} elements: "
                f"{self._decision_path[stage][var_name]}")
                
    def get_rvar(self, 
                 stage: int, 
                 var_name: str, 
                 var_subscripts: Optional[Subscript] = None) -> Union[Num, np.ndarray]:
        """ Access a random variable. 
        This parent method is the one that should be used in defining a stochastic problem in a child class of 
        StochasticProblemBasis.
        
        Note: the method `.set_path_info` must be called before calling this method.
        
        Arguments:
        ----------
        stage: int >= 1
        
        var_name: str
        
        var_subscripts: str, int, or tuple of int/str
            The appropriate type is the one used to define the variables in the .random_variables_definition().
        
        Returns:
        --------
        int/float or 1d-array: the random variable(s).
        """
        assert var_name in self.map_stage_to_rvar_names[stage], \
            (f"Random variable '{var_name}' is not in the list of existing "
             f"variables at stage {stage}: {self.map_stage_to_rvar_names[stage]}.")
        assert self._scenario_path is not None, \
            "The method `set_path_info(node)` should be called in order to access a random variable at a node."
        if var_subscripts is None:
            return self._scenario_path[stage][var_name]
        else:
            index = self.get_rvar_index(stage, var_name, var_subscripts)
            assert index <= self._scenario_path[stage][var_name].shape[0] - 1, \
                (f"Variable subscript {var_subscripts} (index = {index}) is out of bounds for '{var_name}' "
                f"at stage {stage}, which has {self._scenario_path[stage][var_name].shape[0]} elements: "
                f"{self._scenario_path[stage][var_name]}")
            return self._scenario_path[stage][var_name][index]
                
    def get_memory(self, stage, var_name):
        """Return the value stored in the scenario tree data memory"""
        return self._memory_path[stage].get(var_name)
    
    def get_dvar_index(self, stage, var_name, var_subscripts):
        """Return the index of a decision variable."""
        return self._map_dvar_to_index[stage][var_name][var_subscripts]
        
    def get_rvar_index(self, stage, var_name, var_subscripts):
        """Return the index of a random variable."""
        return self._map_rvar_to_index[stage][var_name][var_subscripts]
        
    def get_dvar_from_index(self, stage, var_name, index):
        return self._map_index_to_dvar[stage, var_name, index]
    
    def get_rvar_from_index(self, stage, var_name, index):
        return self._map_index_to_rvar[stage, var_name, index]

    def _initialize_indices(self):
        """Set the mapping from the decision variables and random parameters to their corresponding indices, 
        and vice versa"""        
        for stage in range(self._n_stages):                        
            # decision variables
            self._map_dvar_to_index[stage] = {}
            self._map_stage_to_var_type[stage] = {}
            for var_name, indexing_set, lb, ub, var_type in self.decision_variables_definition(stage):
                if len(indexing_set) == 0:
                    continue
                # get variable type and lb, ub
                self._map_stage_to_var_type[stage][var_name] = (var_type, lb, ub)
                # get indices
                self._map_dvar_to_index[stage][var_name] = {}
                for index, var_subscripts in enumerate(indexing_set):
                    self._map_dvar_to_index[stage][var_name][var_subscripts] = index
                    self._map_index_to_dvar[stage, var_name, index] = var_subscripts
                    
            # random variables
            self._map_rvar_to_index[stage] = {}
            for var_name, indexing_set in self.random_variables_definition(stage):
                if len(indexing_set) == 0:
                    continue
                # get indices
                self._map_rvar_to_index[stage][var_name] = {}
                for index, var_subscripts in enumerate(indexing_set):
                    self._map_rvar_to_index[stage][var_name][var_subscripts] = index
                    self._map_index_to_rvar[stage, var_name, index] = var_subscripts
               
            var_names_in_common = set(self._map_dvar_to_index[stage].keys())\
                                    .intersection(set(self._map_rvar_to_index[stage].keys()))
            assert var_names_in_common == set(), ("Decision variables and random variables have the same name: "
                                                    f"{var_names_in_common} at stage {stage}.")
             
    # --- Solve methods ---
    def solve(self,
              *scenario_trees: ScenarioTree,
              decision_process: Optional[DecisionProcess] = None,
              decomposition: bool = True,
              precompute_parameters: bool = True,
              precompute_decisions: bool = True,
              relaxation: bool = False,
              keep_integer: Optional[Dict[int, List[str]]] = None,
              remove_constraints: Optional[List[str]] = None,
              warmstart: Optional[Dict[str, Union[float, int]]] = None,
              find_only_feasibility: bool = False,
              with_variable_name: bool = False,
              verbose: int = 3,
              mip_filename: Optional[str] = None,
              solve_problem: bool = True,
              tree_filename: Optional[str] = None,
              tree_extension: str = 'txt',
              with_keys: List[str] = None,
              without_keys: List[str] = None,
              **kwargs) -> Union[StochasticSolutionBasis, List[StochasticSolutionBasis]]:
        """ Main method to solve the stochastic problem on one or several scenario trees.
        
        Arguments:
        ----------
        scenario_trees: instances of ScenarioTree
            The scenario trees for which the stochastic problem will be solved.
            
        decision_process: instance of DecisionProcess
            The decisions variables that are fixed to some value beforehand.
            If None, no decisions are fixed.
            
        decomposition: bool (default: True)
        
        precompute_parameters: bool (default: True)
        
        precompute_decisions: bool (default: True)
        
        relaxation: bool (default: False): 
            If True, the optimizer solves the continuous relaxation of the problem.
         
        keep_integer: Dict[int, List[str]] or None (default: None)
            All the variable names that must stay integer even if the problem is relaxed.
            
        remove_constraints: List[str] or None (default: None)
            Contains the name of the constraints to be removed from the problem.
            If None, no constraint is removed.
            
        warmstart: Dict[str, Union[float, int]] or None (default: None)        
           
        find_only_feasibility: bool (default: False)
            If True, the solver will only look for a feasible solution and not an optimal one.
            
        with_variable_name: bool (default: False)
            If True, the model will refer to the variables by their own names as given in the problem's definition.
            
        verbose: {0, 1, 2, 3} (default: 3)
                
        mip_filename: str or None (default: None)
            If a str is given, it is the path where the problem will be saved.
            The path should contain the extension '.lp' or '.sav'.
            
        solve_problem: bool (default: True)
            If True, solves the stochastic problem. 
            If False, only builds the problem.
            
        kwargs:
        -------
        check_sanity: bool (default: True)
            If True, execute the sanity check specified in the stochastic problem.
        
        check_fixed_constraints: bool (default: False)
            If True, check that the constraints with decisions fixed beforehand in the decision process are satisfied. 
            Note that this may be time consuming.
            
        check_topology: bool (default: True)
            If True, it is checked whether all scenario trees have equal topology (which is a necessary condition
            for them to be solved together).
                 
        fill_scenario_tree: bool (default: True)
            If False, only the solve details at the root and subroots will be available (solve details are 
            the objective value, the best bound, the solve status and the solving time).
                                   
        clear_between_trees: bool (default: True)
            If True, the solution of one scenario tree is not used as warmstart for the next ones.
                        
        timelimit: int >= 1 or None (default: None)
            Time limit for the optimization of one stochastic problem. If None, no limit is set.

        logfile: str or None (default: "")
            This string is appended to the log filename. This file is located in a folder 'logfiles' in the 
            current directory (the folder is created if it doesn't already exist). If None, no log is available. 
            
        Returns:
        --------
        An instance or a list of instances of the class `self._solution_class`.
        `self._solution_class` is either the class StochasticSolutionBasis or a subclass
        """ 
        # check scenario trees
        self._check_scenario_trees(scenario_trees)
        
        # decision process
        if decision_process is None:
            decision_process = DecisionProcess(self._map_dvar_to_index)
        else:
            assert self.map_dvar_to_index == decision_process.map_dvar_to_index, \
             "The decision variables in the decision process do not match those of the stochastic problem."
             
        # check file names
        mip_filename = self._check_filenames(mip_filename, 'mip_filename', len(scenario_trees))
        tree_filename = self._check_filenames(tree_filename, 'tree_filename', len(scenario_trees))
            
        if verbose >= 1:
            print(f"Number of scenario trees: {len(scenario_trees)} (bushiness: {scenario_trees[0].bushiness}) \n")

        # build and solve approximate problem  (trees are copied to store memory data at the nodes)
        approximate_problem = ApproximateProblem(scenario_trees[0].copy(), 
                                                 stochastic_problem=copy.copy(self),
                                                 decision_process=decision_process, 
                                                 decomposition=decomposition, 
                                                 precompute_parameters=precompute_parameters,
                                                 precompute_decisions=precompute_decisions,
                                                 relaxation=relaxation, 
                                                 keep_integer=keep_integer,
                                                 remove_constraints=remove_constraints,
                                                 warmstart=warmstart,
                                                 find_only_feasibility=find_only_feasibility,
                                                 verbose=verbose,
                                                 **kwargs)
        
        solutions = [None] * len(scenario_trees)
        approximate_problem.build_deterministic_part()
        # solve iteratively over the trees
        for index_tree in range(len(scenario_trees)):
            with _timeit(approximate_problem, 
                         f"\rSolve scenario tree #{index_tree+1}... " if verbose <= 2 \
                         else f"\nSolve scenario tree #{index_tree+1}... \n", 1):
                # build, solve, and fill        
                approximate_problem.build_random_part()
                if mip_filename:
                    approximate_problem.write_docplex_models(mip_filename[index_tree])
                if solve_problem:
                    approximate_problem.solve_docplex_models()
                    approximate_problem.fill_decisions_in_tree(tree_filename[index_tree], tree_extension,
                                                               with_keys, without_keys)
                    # append solution
                    solutions[index_tree] = self._solution_class(copy.copy(self), 
                                                                 approximate_problem.scenario_tree)
                # set next scenario tree
                if index_tree < len(scenario_trees)-1:
                    approximate_problem.scenario_tree = scenario_trees[index_tree+1].copy()
                    approximate_problem.scenario_tree.append_data_dict(approximate_problem.dvar_dict)
                    approximate_problem.remove_random_constraints()

        return solutions[0] if len(scenario_trees) == 1 else solutions
            
    def compute_wait_and_see(self, 
                             scenario_tree: ScenarioTree, 
                             **kwargs) -> List[StochasticSolutionBasis]:
        """ 
        Solve the Wait-and-See (WS) problems. Each WS problem is formulated by taking a single scenario in 
        the scenario tree. 
        
        Parameters:
        -----------
        scenario_tree: instance of ScenarioTree
            The scenario tree representing the stochastic problem uncertainty.
            
        kwargs: 
        -------
        see `solve` method
        """
        return self.solve(*decompose(scenario_tree), **kwargs)
        
    def compute_opportunity_cost(self, 
                                 scenario_tree, 
                                 return_matrix=False,
                                 relaxation_step1=False,
                                 relaxation_step2=False,
                                 timelimit_step1=None,
                                 timelimit_step2=None,
                                 fill_scenario_tree_step2=True,
                                 **kwargs) -> List[StochasticSolutionBasis]:            
        r"""
        Compute the square-matrix of opportunity cost. The computation consists in two steps:
            (1) solving the two-stage problem for each scenario independently;
            (2) solving the 2nd-stage problem for each scenario and each 1st-stage solution computed at step (1).
        
        Arguments:
        ----------
        scenario_tree: instance of ScenarioTree
        
        relaxation_step1: bool (optional)
            If True, the optimizer solves the continuous relaxation at step (1).
        
        relaxation_step2: bool (optional)
            If True, the optimizer solves the continuous relaxation at step (2).
            
        timelimit_step1: int >= 1 or None (default: None)
            Time limit for the optimization of one stochastic problem at step (1). If None, no limit is set.
            
        timelimit_step2: int >= 1 or None (default: None)
            Time limit for the optimization of one stochastic problemat step (2). If None, no limit is set.
            
        kwargs:
        -------
        all kwargs of `solve` method, except `decision_process`, `timelimit`, `verbose`, `relaxation`.
        
        Returns:
        --------
        List of instances of StochasticSolutionBasis of length `n_leaves` where n_leaves is the number of leaves 
        in the scenario tree. 
            The i-th element is the solution corresponding to the 2nd-stage evaluation of the 1st-stage decisions 
            of the i-th wait-and-see problem. Formally, it is: sum_{j=1,...,n_leaves} F(x_i^*, \xi_j) where x_i^* 
            is the 1st-stage solution of the i-th WS problem and \xi_j is the j-th scenario of the scenario tree.
        """
        # solve wait and see problems
        with TimeIt(f"Solve Wait-and-See Problems... ", 
                    lambda t: f"Finished {t:.3f} sec\n"):
            ws_solutions = self.compute_wait_and_see(scenario_tree, 
                                                     decision_process=None,
                                                     relaxation=relaxation_step1,
                                                     timelimit=timelimit_step1,
                                                     fill_scenario_tree=True,
                                                     **kwargs)
        # evaluate wait and see problems
        #matrix = np.zeros((len(ws_solutions), len(ws_solutions)))
        eval_ws_solution = []
        for k, ws_sol in enumerate(ws_solutions):
            with TimeIt(f"Evaluate Wait-and-See Solution #{k}... ", 
                        lambda t: f"Finished {t:.3f} sec\n"):
                eval_ws_solution.append(
                        self.solve(scenario_tree,
                                   decision_process=DecisionProcess(self._map_dvar_to_index, {0: ws_sol.x0}),
                                   relaxation=relaxation_step2,
                                   timelimit=timelimit_step2,
                                   fill_scenario_tree=fill_scenario_tree_step2,
                                   **kwargs))
        if return_matrix:
            return np.array([sol.objective_value_at_leaves for sol in eval_ws_solution])
        else:
            return ws_solutions, eval_ws_solution

    def compute_value_proxy(self, 
                            scenario_tree_true,
                            scenario_tree_proxy,
                            **kwargs):
        """ 
        Compute the Value of the Proxy Distribution (VPD).
        
        Note: it is defined as:
            VPD = v(EPD) - v(SP) (for mininization) or v(SP) - v(EPD) (for maximization)
        where:
            v(SP) is the optimal-value of the Two-Stage Problem (SP)
            v(EPD) is the optimal-value of the Expectation of Proxy Distribution Problem (EPD)
        The EPD problem evaluates the value of the Proxy Distribution (PD) solution in the true stochastic problem:
            v(EPD) = v(SP s.t. x0 = x0(PD))
        where x0(PD) is the stage-0 optimal solution obtained using the proxy scenarios.
           
        Arguments:
        ----------
        scenario_tree_true: instance of ScenarioTree
            The scenario tree of the true stochastic problem (SP).
            
        scenario_tree_proxy: instance of ScenarioTree
            The scenario tree of the proxy stochastic problem (PD).
                        
        kwargs: 
        -------
        all kwargs of `solve` method, except `decision_process`.
        
        Returns:
        --------
        dictionary: mapping:
            'PD': instance of StochasticSolutionBasis (the PD solution instance)
            'EPD': instance of StochasticSolutionBasis (the EPD solution instance)
            'SP':  instance of StochasticSolutionBasis (the SP solution instance)
            'VPD': float (the Value of Proxy Distribution)
            'VPD%' float (the Value of Proxy Distribution in percentage of v(2SP))
        """
        # Problem with proxy scenarios
        print("*** Solve Proxy Distribution Problem (PD) ***")
        proxy_dist_sol = self.solve(scenario_tree_proxy,
                                    decision_process=None,
                                    **kwargs)
         # True two-stage problem with proxy solution
        print("\n*** Solve Expectation of Proxy Distribution Problem (EPD) ***")
        exp_proxy_dist_sol = self.solve(scenario_tree_true, 
                                         decision_process=DecisionProcess(self._map_dvar_to_index, 
                                                                          {0: proxy_dist_sol.x0}),
                                         **kwargs)
        # True two-stage problem
        print("\n*** Solve True Stochastic Problem (SP) ***")
        true_stoch_sol = self.solve(scenario_tree_true, 
                                   decision_process=None,
                                   **kwargs)
        # value of proxy distribution
        if self._objective_sense == 'max':
            vpd = true_stoch_sol.objective_value - exp_proxy_dist_sol.objective_value
        else:
            vpd = exp_proxy_dist_sol.objective_value - true_stoch_sol.objective_value
        dict_vpd = {'PD': proxy_dist_sol,
                    'EPD': exp_proxy_dist_sol,
                    '2SP': true_stoch_sol,
                    'VPD': vpd,
                    'VPD(%)': 100 * vpd / (10**-10 + abs(true_stoch_sol.objective_value))}
        return dict_vpd
    
    # --- Precomputation ---
    def _initialize_precomputation(self):
        """Get the name of the precomputed parameters and variables. This sets the dictionary attributes 
        `_name_precomputed_decisions` and `_name_precomputed_parameters` at every stage."""
        for stage in range(self._n_stages):
            # decisions
            self._name_precomputed_decisions[stage] = set()
            if self.precompute_decision_variables(stage) is not None:
                for name, _ in self.precompute_decision_variables(stage):
                    self._name_precomputed_decisions[stage].add(name)
            # parameters
            self._name_precomputed_parameters[stage] = set()
            if self.precompute_parameters(stage) is not None:
                for name, _ in self.precompute_parameters(stage):
                    self._name_precomputed_parameters[stage].add(name)
            
    def precompute_decision_variables_(self, node, model=None):
        self.set_path_info(node, model)
        if self._name_precomputed_decisions[node.level] != set():
            for name, value_fct in self.precompute_decision_variables(node.level):
                self._memory_path[node.level][name] = value_fct()
        
    def precompute_parameters_(self, node):
        self.set_path_info(node)
        if self._name_precomputed_parameters[node.level] != set():
            for name, value_fct in self.precompute_parameters(node.level):
                self._memory_path[node.level][name] = value_fct()
                                    
    # --- Sanity check ---        
    def check_sanity(self, scenario_tree, node_addresses=None):
        """ Check the validity of a scenario tree to solve the stochastic problem.
        
        Arguments:
        ----------
        scenario_tree: instance of ScenarioTree
            The scenario tree from which the information (scenario and decision) is taken.
            
        node_addresses: list of tuples (optional) (default: None)
            Each tuple is the address of a node for which sanity is checked.
            If None, sanity is checked at all nodes.
        """
        for node in scenario_tree.nodes:
            if node_addresses is not None and node.address not in node_addresses:
                continue
            self.set_path_info(node)
            self.sanity_check(node.level)
            
    def _check_scenario_trees(self, scenario_trees):
        """ Check the suitability between the problem and the scenario tree.
        
        Specifically, raise an Error if one of the following is True:
            (1) The scenario trees do not all have the same topology.
            (2) The number of random parameters as defined in the problem is larger than the one available 
                in the scenario tree.
        """
        assert len(set([tree.topology for tree in scenario_trees])) == 1, \
            f"Scenario trees should have the same topology to be solve together."
        for tree_index, scenario_tree in enumerate(scenario_trees, 1):
            for stage in range(1, self._n_stages):
                for var_name in self.map_rvar_name_to_nb[stage].keys():
                    n_var1 = self.map_rvar_name_to_nb[stage][var_name]
                    n_var2 = scenario_tree.map_stage_to_rvar_nb.get(stage, dict()).get(var_name, 0)
                    assert n_var1 <= n_var2, (f"Random parameter '{var_name}' has {n_var1} variables in the problem, "
                                              f"but only {n_var2} in the scenario tree #{tree_index}.")
                
    def _check_filenames(self, filename, which_filename, n_scenario_trees):
        if filename is None:
            return [None] * n_scenario_trees
        if isinstance(filename, str):
            assert n_scenario_trees == 1, \
                f"{which_filename} must be a list of strings if there are more than one scenario tree."
            return [filename]
        elif isinstance(filename, list):
            assert len(filename) == n_scenario_trees, (f"{which_filename} must contain as many strings as there are "
                  f"scenario trees, i.e., {n_scenario_trees}, not {len(filename)}.")
            return filename
        else:
            raise TypeError("{which_filename} should be a string or a list of strings.")
                
    # --- Representation ---
    def __repr__(self):
        dict_var_type = {'B': 'b', 'I': 'i', 'C': 'c'}
        str_ = f"{self._name}: ({self._n_stages} stages)\n"
        # dvar
        dvar_nb_list = [sum(self.map_dvar_name_to_nb[t].values()) for t in range(self._n_stages)]
        str_ += f"  - decision variables: {dvar_nb_list}\n"
        for stage in range(self._n_stages):
            str_ += f"    - stage {stage}: \n"
            for var, dict_ in self._map_dvar_to_index[stage].items():
                var_type = dict_var_type[self._map_stage_to_var_type[stage][var][0]]
                str_ += f"      - {var}: {len(dict_.values())} ({var_type})\n"
        # rvar
        rvar_nb_list = [sum(self.map_rvar_name_to_nb[t].values()) for t in range(self._n_stages)]
        str_ += f"  - random variables: {rvar_nb_list}\n"
        for stage in range(1, self._n_stages):
            str_ += f"    - stage {stage}: \n"
            for var, dict_ in self._map_rvar_to_index[stage].items():
                str_ += f"      - {var}: {len(dict_.values())}\n"
        return str_
    
    # ---  Copy, save and load ---
    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new
    
    def to_file(self, path, extension):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
   

def compare_scenario_trees(optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]], 
                           evaluate_fct: Callable[[ScenarioTree, DecisionProcess], StochasticSolutionBasis], 
                           scenario_tree_opt1: ScenarioTree,
                           scenario_tree_opt2: ScenarioTree,
                           scenario_tree_eval: ScenarioTree) -> Dict[str, Union[float, StochasticSolutionBasis]]:
    """
    Optimize the same problem using two scenario trees and evaluate the two solutions in a third testing scenario tree.
       
    Arguments:
    ----------
    optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]]
        The optimizer that takes a scenario tree and returns the solution and its corresponding decision process.
        
    evaluate_fct: Callable[[ScenarioTree, DecisionProcess], StochasticSolutionBasis]
        The evaluator that takes a scenario tree and a decision process and returns the solution.
        
    scenario_tree_opt1: ScenarioTree
        The scenario tree of the 1st uncertainty representation.
        
    scenario_tree_opt2: ScenarioTree
         The scenario tree of the 2nd uncertainty representation.
           
    scenario_tree_eval: ScenarioTree
        The scenario tree of the true uncertainty representation used to evaluate and compare the solutions.
               
    Returns:
    --------
    dict with keys and values:
         'SOL#1': instance of StochasticSolutionBasis (the solution with the 1st uncertainty)
         'SOL#2': instance of StochasticSolutionBasis (the solution with the 2nd uncertainty) 
         'E[SOL#1]':  instance of StochasticSolutionBasis (the evaluation of the solution of the 1st uncertainty)
         'E[SOL#2]':  instance of StochasticSolutionBasis (the evaluation of the solution of the 2nd uncertainty)
         'delta[#2-#1]': float 
         'delta[(#2-#1)/#1](%)': float 
    """
    # Problem scenario tree #1
    print("*** Solve Optimization Problem #1 (SOL#1) ***")
    sol1, dp1 = optimize_fct(scenario_tree_opt1)
    
    # Problem scenario tree #2
    print("\n*** Solve Optimization Problem #1 (SOL#2) ***")
    sol2, dp2 = optimize_fct(scenario_tree_opt2)
    
     # Evaluate solution of problem scenario tree #1
    print("\n*** Solve Expectation of Solution #1 (E[SOL#1]) ***")
    exp_sol1 = evaluate_fct(scenario_tree_eval, dp1)
    
     # Evaluate solution of problem scenario tree #2
    print("\n*** Solve Expectation of Solution #2 (E[SOL#2]) ***")
    exp_sol2 = evaluate_fct(scenario_tree_eval, dp2)
    
    # value of proxy distribution
    delta = exp_sol2.objective_value - exp_sol1.objective_value
    return {'SOL#1': sol1,
            'SOL#2': sol2,
            'E[SOL#1]': exp_sol1,
            'E[SOL#2]': exp_sol2,
            'delta[#2-#1]': delta,
            'delta[(#2-#1)/#1](%)': 100 * delta / (10**-10 + abs(exp_sol1.objective_value))}
    
    
def compute_value_right_dist(optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]], 
                            evaluate_fct: Callable[[ScenarioTree, DecisionProcess], StochasticSolutionBasis], 
                            scenario_tree_true: ScenarioTree,
                            scenario_tree_proxy: ScenarioTree, 
                            objective_sense: str) -> Dict[str, Union[float, StochasticSolutionBasis]]:
    """ 
    Compute the Value of the Right Distribution (VRD).
    
    Note: VRD is defined as:
        VRD = v(EPD) - v(SP) (for mininization) or v(SP) - v(EPD) (for maximization)
    where:
        v(SP) is the optimal-value of the Two-Stage Problem (SP)
        v(EPD) is the optimal-value of the Expectation of Proxy Distribution Problem (EPD)
    The EPD problem evaluates the value of the Proxy Distribution (PD) solution in the true stochastic problem:
        v(EPD) = v(SP s.t. x0 = x0(PD))
    where x0(PD) is the stage-0 optimal solution obtained using the proxy scenarios.
       
    Arguments:
    ----------
    optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]]
        The optimizer that takes a scenario tree and returns the solution and its corresponding decision process.
        
    evaluate_fct: Callable[[ScenarioTree, DecisionProcess], StochasticSolutionBasis]
        The evaluator that takes a scenario tree and a decision process and returns the solution.
    
    scenario_tree_true: ScenarioTree
        The scenario tree of the true distribution (SP).
        
    scenario_tree_proxy: ScenarioTree
        The scenario tree of the proxy distribution (PD).
                  
    objective_sense: {'max', 'min'}
        
    Returns:
    --------
    Dict[str, Union[float, StochasticSolutionBasis]]: mapping:
         'PD': instance of StochasticSolutionBasis (the PD solution instance)
         'EPD': instance of StochasticSolutionBasis (the EPD solution instance)
         'SP':  instance of StochasticSolutionBasis (the SP solution instance)
         'VRD': float (the Value of Right Distribution)
         'VRD%' float (the Value of Right Distribution in percentage of v(2SP))
    """
    # Problem with proxy scenarios
    print("*** Solve Proxy Distribution Problem (PD) ***")
    proxy_dist_sol, dec_pro = optimize_fct(scenario_tree_proxy)
    
     # True two-stage problem with proxy solution
    print("\n*** Solve Expectation of Proxy Distribution Problem (EPD) ***")
    exp_proxy_dist_sol = evaluate_fct(scenario_tree_true, dec_pro)
    
    # True two-stage problem
    print("\n*** Solve True Stochastic Problem (SP) ***")
    true_stoch_sol, dec_pro = optimize_fct(scenario_tree_true)
    
    # value of proxy distribution
    if objective_sense == 'max':
        vrd = true_stoch_sol.objective_value - exp_proxy_dist_sol.objective_value
    else:
        vrd = exp_proxy_dist_sol.objective_value - true_stoch_sol.objective_value
    return {'PD': proxy_dist_sol,
            'EPD': exp_proxy_dist_sol,
            'SP': true_stoch_sol,
            'VRD': vrd,
            'VRD(%)': 100 * vrd / (10**-10 + abs(true_stoch_sol.objective_value))}
    
    
def compute_wait_and_see(optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]],
                         scenario_tree: ScenarioTree) -> List[Tuple[StochasticSolutionBasis, DecisionProcess]]:
    """ 
    Solve the Wait-and-See (WS) problems. Each WS problem is formulated by taking a single scenario in 
    the scenario tree. 
    
    Arguments:
    ----------
    optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]]
        The optimizer that takes a scenario tree and returns the solution and its corresponding decision process.
        
    scenario_tree: ScenarioTree
        The scenario tree of representing the stochastic problem uncertainty.

    Returns:
    --------
    List[Tuple[StochasticSolutionBasis, DecisionProcess]]
    """
    return [optimize_fct(scenario_tree) for scenario_tree in decompose(scenario_tree)]
    

def compute_opportunity_cost(optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]],
                             evaluate_fct: Callable[[ScenarioTree, DecisionProcess], StochasticSolutionBasis],
                             scenario_tree: ScenarioTree) -> np.ndarray:            
    """
    Compute the square-matrix of opportunity cost. The computation consists in two steps:
        (1) solving the two-stage problem for each scenario independently;
        (2) solving the 2nd-stage problem for each scenario and each 1st-stage solution computed at step (1).
    
    Arguments:
    ----------
    optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess]]
        The optimizer that takes a scenario tree and returns the solution and its corresponding decision process.
        
    evaluate_fct: Callable[[ScenarioTree, DecisionProcess], StochasticSolutionBasis]
        The evaluator that takes a scenario tree and a decision process and returns the solution.
        
    scenario_tree: ScenarioTree
        The scenario tree of representing the stochastic problem uncertainty.

    Returns:
    --------
    2d-array of shape (n_leaves, n_leaves) where n_leaves is the number of leaves in the scenario tree. 
        The element [i, j] is the objective value of the i-th wait-and-see solution when evaluated in the j-th 
        scenario. 
        Formally, it is: F(x_i^*, xi_j) 
        where x_i^* is the solution of the i-th WS problem and xi_j is the j-th scenario of the scenario tree.
    """
    # (1) Solve wait and see problems
    ws_solutions = []
    for k, scenario_tree in enumerate(decompose(scenario_tree)):
        with TimeIt(f"Solve Wait-and-See Problems #{k}...\n", lambda t: f"Finished {t:.3f} sec\n"):
            ws_solutions.append(optimize_fct(scenario_tree))
     
    # (2) Evaluate wait and see solutions
    n_leaves = scenario_tree.width[-1]
    matrix = np.zeros((n_leaves, n_leaves))
    for k, (ws_sol, ws_dp) in enumerate(ws_solutions):
        with TimeIt(f"Evaluate Wait-and-See Solution #{k}...", lambda t: f"Finished {t:.3f} sec\n"):
            eval_sol = evaluate_fct(scenario_tree, ws_dp)
            for j, leaf in enumerate(eval_sol.scenario_tree.leaves):
                matrix[k, j] = leaf.data['v']
    return matrix


