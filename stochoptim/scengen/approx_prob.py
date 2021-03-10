# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import unicode_literals

import os
import datetime
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import copy
import docplex.mp.model
import docplex.mp.context

from stochoptim.util import RedirectStd
from stochoptim.util import TimeIt
from stochoptim.scengen.decision_process import DecisionProcess
#from stochoptim.stochprob.stochastic_problem_basis import StochasticProblemBasis
from stochoptim.scengen.scenario_tree import ScenarioTree

           
class ApproximateProblem:

    def __init__(self, 
                 scenario_tree: ScenarioTree, 
                 stochastic_problem,
                 decision_process: Optional[DecisionProcess] = None, 
                 decomposition=True, 
                 precompute_parameters=True,
                 precompute_decisions=True,
                 relaxation=False, 
                 keep_integer: Dict[int, List[str]] = None,
                 remove_constraints: Optional[List[str]] = None,
                 warmstart: Dict[str, Union[float, int]] = None, 
                 find_only_feasibility=False,
                 with_variable_name=False,
                 objective_type='path',
                 verbose=3,
                 check_sanity=False,
                 check_fixed_constraints=False,
                 check_topology=True,
                 clear_between_trees=True, 
                 fill_scenario_tree=True,
                 timelimit=None, 
                 logfile="",
                 **kwargs):
        
        """ Docplex implementation of the approximate problem.
        
        Arguments:
        ----------
        scenario_tree: instance of ScenarioTree
            The scenario tree must be filled with data values 'scenario' at every level >= 1 
            and weights 'W' at every level >= 0.
            
        stochastic_problem: instance of a subclass of StochasticProblemBasis
            The stochastic problem to be solved on the scenario tree.
            
        decision_process: instance of DecisionProcess (optional).
            The decisions that will be fixed beforehand in the approximate problem (possibly none). 
            Any stage and any scenario can have fixed decisions; the optimizer will then find the optimal 
            decisions given the fixed ones. 
            
        decomposition: bool (default: True)
        
        precompute_parameters: bool (default: True)
        
        precompute_decisions: bool (default: True)
        
        relaxation: bool (default: False): 
            If True, the optimizer solves the continuous relaxation of the problem.
        
        keep_integer: Dict[int, List[str]] or None (default: None)
            All the variable names that should stay integer even if the problem is relaxed.
            
        remove_constraints: List[str] or None (default: None)
            Contains the name of the constraints to be removed from the problem.
            If None, no constraint is removed.
            
        warmstart: Dict[str, Union[float, int]] or None (default: None)
        
        find_only_feasibility: bool (default: False)
            If True, the solver will only look for a feasible solution and not an optimal one.
            
        with_variable_name: bool (default: False)
            If True, the model will refer to the variables by their own names as given in the problem's definition.
            
        verbose: {0, 1, 2, 3} (default: 3)
        
        check_sanity: bool (default: True)
            If True, execute the sanity check specified in the stochastic problem.
            
        check_fixed_constraints: bool (default: False)
            If True, check that the constraints with decisions fixed beforehand in the decision process are satisfied. 
            Note that this may be time consuming.
            
        check_topology: bool (default: True)
            If True, check that all scenario trees have the same topology (which is a necessary condition
            for them to be solved together).
            
        clear_between_trees: bool (default: True)
            If True, the solution of one scenario tree is not used as a warmstart for the next ones.
        
        fill_scenario_tree: bool (default: True)
            If False, only the solve details at the root node will be available (i.e., objective value, best bound, 
            solve status and time).
            
        timelimit: int >= 1 or None (default: None)
            Time limit for the optimization of one stochastic problem. If None, no limit is set.

        logfile: str or None (default: "")
            This string is appended to the log filename. This file is located in a folder 'logfiles' in the 
            current directory (the folder is created if it doesn't already exist). If None, no log is generated. 
        """
        self.scenario_tree = scenario_tree
        self._stochastic_problem = stochastic_problem
        self._decision_process = decision_process
        
        self.decomposition = decomposition
        self.relaxation = relaxation
        self.keep_integer = keep_integer if keep_integer is not None else dict()
        self.remove_constraints = remove_constraints
        self.precompute_parameters = precompute_parameters
        self.precompute_decisions = precompute_decisions
        self.objective_type = objective_type
        self.verbose = verbose
        self.check_sanity = check_sanity
        self.check_fixed_constraints = check_fixed_constraints
        self.warmstart = warmstart 
        self.find_only_feasibility = find_only_feasibility
        self.with_variable_name = with_variable_name
        self.clear_between_trees = clear_between_trees
        self.fill_scenario_tree = fill_scenario_tree
        self.timelimit = timelimit 
        self.logfile = logfile
        
        self._stage_fixed_dvar = 0 # earliest stage for which some decisions are not fixed
        self._stage_subtree_decomposition = None # stage at which the problem is decomposed
        self._n_subroots = None
        self._docplex_models = {}
        self._docplex_solutions = {}
        self._random_cts = {} # to include the random constraints
        self.dvar_dict = {}
        self._fully_fixed_var_name: Dict[Tuple[int], List[str]] = {} # node.address: list of dvar names
        
        self._determine_stage_fixed_dvar()
        self._stage_subtree_decomposition = self._stage_fixed_dvar if self.decomposition else 0
        self._n_subroots = sum(1 for _ in self._subroots(self.scenario_tree))
        if self.decomposition and self._stage_subtree_decomposition != 0:
            self._print(f"Problem is decomposed at stage {self._stage_subtree_decomposition} "
                        f"({self._n_subroots} subtrees).\n", level=1)
            
        self._initialize_docplex_models(**kwargs)
                             
    def _print(self, string, level):
        """Print a string conditionally to the verbose value"""
        if level <= self.verbose:
            print(string, end="")
             
    def _determine_stage_fixed_dvar(self):
        """Determine the earliest stage for which some of the variables of the decision process are not fixed.
        That is, we look for the stage t such that some variables are not fixed at t but all variables are fixed at 
        any t' < t. (Note that some variables may also be fixed at t' > t.). 
        (This stage determines the subtrees from which the problem will be decomposed if self.decomposition is True.)"""     
        for node in self.scenario_tree.forward_nodes:
            for var_name in self._stochastic_problem.map_stage_to_dvar_names[node.level]:
                # if any variable in 'var_name' has no fixed value at the node
                if (self._decision_process(node, var_name) == None).any():                    
                    self._stage_fixed_dvar = node.level
                    return

    def _subroots(self, tree):
        """Returns the effective roots from which the problem is decomposed."""
        return tree.nodes_at_level(self._stage_subtree_decomposition)
                     
    def _initialize_docplex_models(self, **kwargs):
        """Create a docplex model for each subroot."""
        with _timeit(self, "Initialize model... ", 1):
            self._docplex_models = {subroot.address: model for subroot, model in zip(
                                                self._subroots(self.scenario_tree), 
                                                create_docplex_models(self._n_subroots, **kwargs))}
    
    def add_decision_variables(self):
        """Add the decision variables to the scenario tree. 
        Specifically, it adds the keys 'dvar' to the node data dictionary.
        """
        # walk the whole tree and set the variables whose values are fixed
        with _timeit(self, "Initialize variables... ", 2):
            for node in self.scenario_tree.nodes:
                node.data['decision'] = {}
                self._fully_fixed_var_name[node.address] = []
                for var_name in self._stochastic_problem.map_stage_to_dvar_names[node.level]:
                    if (self._decision_process(node, var_name) != None).all():
                        # if all variables in 'var_name' have a given value at the node
                        node.data['decision'][var_name] = self._decision_process(node, var_name)
                        self._fully_fixed_var_name[node.address].append(var_name)
                        
            # set the non-fixed variables in each subtree
            for subroot in self._subroots(self.scenario_tree):
                model = self._docplex_models[subroot.address]
                for node in subroot.nodes:
                    for var_info in self._stochastic_problem.decision_variables_definition(node.level):
                        var_name, indexing_set, lb, ub, var_type = var_info
                        docplex_var_name = None
                        if self.warmstart is not None or self.with_variable_name:
                            docplex_var_name = self._docplex_var_name(var_name, node)
                        if len(indexing_set) == 0:
                            continue # if variable has no indexing set
                        if (self._decision_process(node, var_name) != None).all():
                            continue # if variable is fixed
                        if self.relaxation and var_name in self.keep_integer.get(node.level, list()):
                            node.data['decision'][var_name] = add_variables_by_blocks(indexing_set, lb, ub, var_type, 
                                                                                      model, False, docplex_var_name)  
                        else:    
                            node.data['decision'][var_name] = add_variables_by_blocks(indexing_set, lb, ub, var_type, 
                                                                                      model, self.relaxation, 
                                                                                      docplex_var_name)
            self.dvar_dict = self.scenario_tree.get_data_dict(with_keys=['decision'])
        
    @staticmethod
    def _docplex_var_name(var_name, node):
        return f'{var_name}_{node.address}'
    
    def add_expected_objective(self):
        """Set the expected objective function in the docplex models"""
        for index, subroot in enumerate(self._subroots(self.scenario_tree), 1):
            model = self._docplex_models[subroot.address]
            display_level = 2 if not self._stochastic_problem.is_obj_random else 3
            space = "  " if self._stochastic_problem.is_obj_random else ""
            with _timeit(self, f"\r{space}Define objective function at subtree #{index}... ", display_level, index):
                if self.find_only_feasibility:
                    model.set_objective(self._stochastic_problem.objective_sense, 0) 
                else:       
                    if self.objective_type == 'node':
                        model.set_objective(self._stochastic_problem.objective_sense, 
                                            self._docplex_models[subroot.address].sum((node.data["W"] / subroot.data["W"]) *
                                            self._stochastic_problem.objective_node(node, model) for node in subroot.nodes)) 
                    elif self.objective_type == 'path':
                        model.set_objective(self._stochastic_problem.objective_sense, 
                                            self._docplex_models[subroot.address].sum((leaf.data["W"] / subroot.data["W"]) *
                                            self._stochastic_problem.objective_path(leaf, model) for leaf in subroot.leaves))
                    else:
                        raise ValueError(f"`objective_type` must be either 'path' or 'node', not '{self.objective_type}'")
    
    def add_deterministic_constraints(self):
        """Add the deterministic constraints in the docplex models"""
        for index, subroot in enumerate(self._subroots(self.scenario_tree), 1):
            model = self._docplex_models[subroot.address]
            with _timeit(self, f"\rAdd deterministic constraints at subroot #{index}... ", 2, index):
                # indicator constraints
                try:
                    ct_type = ('deterministic', 'indicator')
                    next(self._gen_constraints(subroot, ct_type, model)) # test whether generator is empty
                    model.add_indicators(*zip(*self._gen_constraints(subroot, ct_type, model)))
                except StopIteration:
                    pass
                # linear constraints (must be added last)
                try:
                    ct_type = ('deterministic', 'linear')
                    next(self._gen_constraints(subroot, ct_type, model)) # test whether generator is empty
                    model.add_constraints(self._gen_constraints(subroot, ct_type, model))
                    model.add_constraints(self._gen_constraints_fixing_variables(subroot))
                except StopIteration:
                    pass

    def add_random_constraints(self):
        """Add the random constraints in the docplex models"""
        for index, subroot in enumerate(self._subroots(self.scenario_tree), 1):
            model = self._docplex_models[subroot.address]
            rand_cts = []
            with _timeit(self, f"\r  Add random constraints at subtree #{index}... ", 3, index):
                # indicator constraints
                try:
                    ct_type = ('random', 'indicator')
                    next(self._gen_constraints(subroot, ct_type, model)) # test whether generator is empty
                    rand_cts += model.add_indicators(*zip(*self._gen_constraints(subroot, ct_type, model)))
                except StopIteration:
                    pass
                # quadratic constraints (must be added last)
                try:
                    ct_type = ('random', 'quadratic')
                    next(self._gen_constraints(subroot, ct_type, model)) # test whether generator is empty
                    for ct in self._gen_constraints(subroot, ct_type, model):
                        rand_cts.append(model.add_constraint(ct)) # one by one
                except StopIteration:
                    pass  
                # linear constraints (must be added last)
                try:
                    ct_type = ('random', 'linear')
                    next(self._gen_constraints(subroot, ct_type, model)) # test whether generator is empty
                    rand_cts += model.add_constraints(self._gen_constraints(subroot, ct_type, model))
                except StopIteration:
                    pass      
            self._random_cts[subroot.address] = rand_cts
              
    def remove_random_constraints(self):
        """Remove the random constraints of the stochatic from the docplex models"""
        for index, subroot in enumerate(self._subroots(self.scenario_tree), 1):
            with _timeit(self, f"\r  Remove random constraints at subtree #{index}... ", 3, index):
                self._docplex_models[subroot.address].remove_constraints(self._random_cts[subroot.address])
                
    # --- Constraints generators ---
    def _gen_constraints_fixing_variables(self, subroot):
        """Yield all the constraints fixing each variable value individually inside a given subtree. Variables have 
        their value fixed individually by an explicit constraint if they belong to an array where some variables are not
        fixed."""
        for node in subroot.nodes:
            for var_name in self._stochastic_problem.map_stage_to_dvar_names[node.level]:
                # if var name not fully (or not at all) fixed at that node
                if var_name not in self._fully_fixed_var_name[node.address]: 
                    is_fixed = (self._decision_process(node, var_name) != None)
                    if is_fixed.any(): # if some variables are fixed
                        for index in is_fixed.nonzero()[0]:
                            fixed_value = self._decision_process(node, var_name)[index]
                            yield node.data['decision'][var_name][index] == fixed_value, \
                                    f"{var_name}_{index}_forced_at_{fixed_value}"
                                
    def _gen_constraints(self, subroot, ct_type, model):
        """Generate all the constraints of a certain type in a subtree.
        Arguments:
        ----------
        ct_type: tuple (a, b) where a in {'deterministic', 'random'} and b in {'linear', 'indicator'}.
        """
        for node in subroot.nodes:
            if node.level >= self._stage_fixed_dvar:
                yield from self._stochastic_problem.generate_node_constraints(node, 
                                                                              ct_type, 
                                                                              model, 
                                                                              self.remove_constraints)
                if ct_type[1] == 'linear':
                    yield from iter(node.data.pop(f'{ct_type[0]}-ct'))

    def check_fixed_deterministic_constraints(self):
        """Yield all the deterministic constraints that involve fixed decisions. This will raise an Exception if one of
        the constraint is not satisfied"""
        with _timeit(self, f"\rCheck feasibility of fixed decisions in deterministic constraints... ", 2):
            for node in self.scenario_tree.nodes:
                if node.level < self._stage_fixed_dvar:
                    for ct_type in ['linear', 'indicator']:
                        yield from self._stochastic_problem.generate_node_constraints(node, 
                                                                                      ('deterministic', ct_type),
                                                                                      None, 
                                                                                      self.remove_constraints)
                
    def check_fixed_random_constraints(self):
        """Yield all the deterministic constraints that involve fixed decisions. This will raise an Exception if one of
        the constraint is not satisfied"""
        with _timeit(self, f"\r  Check feasibility of fixed decisions in random constraints... ", 3):
            for node in self.scenario_tree.nodes:
                if node.parent and node.level < self._stage_fixed_dvar:
                    for ct_type in ['regular', 'indicator']:
                        yield from self._stochastic_problem.generate_node_constraints(node, 
                                                                                      ('random', ct_type),
                                                                                      None, 
                                                                                      self.remove_constraints)
                            
    # --- Precomputation ---
    def make_precomputation(self):
        with _timeit(self, "  Precompute variables and parameters... ", 3):
            # initialize memory
            for node in self.scenario_tree.nodes:
                if node.data.get('memory') is not None:
                    # copy the the reference to the underlying 'memory' dictionary so that
                    # any new key will not be added to it
                    node.data['memory'] = node.data['memory'].copy()
            # parameters
            if self.precompute_parameters:
                for node in self.scenario_tree.nodes:
                    self._stochastic_problem.precompute_parameters_at_node(node)
            # variables
            if self.precompute_decisions:
                # fixed decisions (no model needed)
                for node in self.scenario_tree.nodes:
                    if node.level < self._stage_fixed_dvar:
                        self._stochastic_problem.precompute_decision_variables_at_node(node)
                # not fixed decisions (needs a model to perform operations on dvar)   
                for subroot in self._subroots(self.scenario_tree):
                    model = self._docplex_models[subroot.address]
                    for node in subroot.nodes:
                        if node.level >= self._stage_fixed_dvar:
                            self._stochastic_problem.precompute_decision_variables_at_node(node,
                                                                                           model)

    # --- Main methods ---
    def build_deterministic_part(self):
        """ Build everything not scenario-dependent in the optimization problem.
        
        In particular: 
            (1) Initialize the decision variables at each node
            (2) Add the deterministic constraints
            (3) Define the objective function if it does not contain random parameters.
        """
        # --- add decisions variables ---
        self.add_decision_variables()
        
        # --- check feasiblity of deterministic constraints with fixed variables
        if self.check_fixed_constraints:
            self.check_fixed_deterministic_constraints()
                 
        # --- add deterministic constraints ---
        self.add_deterministic_constraints()
            
        # --- set objective here if not random ---
        if not self._stochastic_problem.is_obj_random:
            self.add_expected_objective()
                
    def build_random_part(self):
        """ Build everything scenario-dependent in the optimization problem (i.e., which changes from one 
        scenario tree to another).

        In particular: 
            (1) Precompute the decision variables and parameters if required
            (2) Check the parameters validity if required
            (2) Add the random constraints
            (3) Define the objective function if it contains random parameters.
        """
        # --- precompute parameters and variables ---
        if self.precompute_decisions or self.precompute_parameters:
            self.make_precomputation()
                    
        # --- check sanity ---
        if self.check_sanity:
            with _timeit(self, "  Check parameters validity... ", 3):
                self._stochastic_problem.check_sanity(self.scenario_tree)
        
        # --- check feasibility of fixed decisions ---
        if self.check_fixed_constraints:
            self.check_fixed_random_constraints()
               
        # --- add random constraints ---
        self.add_random_constraints()                
                
        # --- set objective here if random ---
        if self._stochastic_problem.is_obj_random:
            self.add_expected_objective()
                        
    def solve_docplex_models(self):
        """Solve the docplex model at every subtree."""                
        # set folder and name of log files
        filename = lambda: None
        if self.logfile is not None:
            if not os.path.exists('logfiles'): 
                os.mkdir('logfiles')
            filename = lambda: r'logfiles\solver {}{}.log'.format(str(datetime.datetime.now()), 
                                                                     self.logfile).replace(":", "_")
           
        for index, subroot in enumerate(self._subroots(self.scenario_tree), 1):
            prob = self._docplex_models[subroot.address]
              
            # add warmstart
            if self.warmstart is not None:
                docplex_warmstart = docplex.mp.solution.SolveSolution(prob, self.warmstart)
                prob.add_mip_start(docplex_warmstart)
                
            # set timelimit
            if self.timelimit is not None:
                if self.timelimit / self._n_subroots < 1:
                    self.timelimit = self._n_subroots
                prob.set_time_limit(self.timelimit / self._n_subroots)
             
            # solve approximate problem
            with _timeit(self, f"\r  Solve problem at subtree #{index}... ", 3, index):
                with RedirectStd(filename()): 
                    self._docplex_solutions[subroot.address] = prob.solve(log_output=True, 
                                                                          clean_before_solve=self.clear_between_trees)
        
    def write_docplex_models(self, filename):
        for index, subroot in enumerate(self._subroots(self.scenario_tree), 1):
            if filename is not None:
                prob = self._docplex_models[subroot.address]
                # export problem as lp file
                with _timeit(self, f"\r  Export to file problem at subtree #{index}... ", 3, index):
                    if self._stage_subtree_decomposition > 0:
                        filename = f"{subroot.address}_{filename}" # prepend subroot address if more than one subroot
                    if '.lp' in filename:
                        prob.export_as_lp(filename)
                    elif '.sav' in filename:
                        prob.export_as_sav(filename)
                    
    def fill_decisions_in_tree(self, 
                               tree_filename=None, tree_extension=None, 
                               with_keys=None, without_keys=None):
        """Fill the scenario tree with the optimal decisions and the solve details."""
        with _timeit(self, f"\r  Fill scenario tree... ", 3):

            # (1) fill subroot with solve details         
            for index, subroot in enumerate(self._subroots(self.scenario_tree), 1):
                prob = self._docplex_models[subroot.address]  
                sol = self._docplex_solutions[subroot.address]
                solve_details = prob.solve_details
                # fill subroot with solve details
                subroot.data.update(dict(v=prob.objective_value if sol is not None else np.nan, 
                                         status=solve_details.status_code,
                                         bound=solve_details.best_bound,
                                         time=solve_details.time))
                
            # (2) fill every node in sub-scenario tree with optimal decisions
                if self.fill_scenario_tree:
                    for node in subroot.nodes:
                        
                        if sol is None:
                            node.data['v'] = np.nan
                            continue
                        
                        node.data['decision'] = copy.copy(node.data['decision'])
                        for var_name in self._stochastic_problem.map_stage_to_dvar_names[node.level]:
                            # if decisions wasn't fixed beforehand
                            if (self._decision_process(node, var_name) == None).any():
                                node.data['decision'][var_name] = \
                                        np.array([y.solution_value for y in node.data['decision'][var_name]])
                                                        
                        # re-do precomputation with actual decision values where it wasn't done already
                        if self.precompute_decisions and node.level >= self._stage_fixed_dvar:
                            self._stochastic_problem.precompute_decision_variables_at_node(node)
                        
                        # remove memory from dictionary if empty
                        if node.data.get('memory') == dict():
                            node.data.pop('memory')
                            
                        # compute objective value at leaf if leaf is not a decomposition subroot
                        if node.is_leaf and node.level != self._stage_subtree_decomposition:
                            node.data['v'] = self._stochastic_problem.objective_path(node)
            
            if not self.fill_scenario_tree:
                for node in self.scenario_tree.nodes:
                    node.data.pop('decision', None)
            else:
                # fill every node that is not root and not in a sub-scenario tree
                for node in self.scenario_tree.nodes:
                    if node.is_root:
                        continue
                    if node.data.get('v') is None: # if node does not have data 'v' already
                        try:
                            node.data['v'] = sum(c.data['v'] * c.data['W'] / node.data['W'] for c in node.children)
                        except: 
                            node.data['v'] = sum(m.data['W'] * m.data['v'] for m in node.leaves)
                            
            # (3) fill actual root with solve details
            if self._stage_subtree_decomposition != 0:
                self.scenario_tree.data.update(dict(v=0, bound=0, time=0, status=set()))
                for subroot in self._subroots(self.scenario_tree):
                    self.scenario_tree.data['v'] += subroot.data["W"] * subroot.data['v']
                    self.scenario_tree.data['bound'] += subroot.data["W"] * subroot.data['bound']
                    self.scenario_tree.data['time'] += subroot.data['time']
                    self.scenario_tree.data['status'].add(subroot.data['status'])
                    
            if tree_filename is not None:
                self.scenario_tree.to_file(tree_filename, tree_extension, with_keys, without_keys)
            
def create_docplex_models(n_models, checker='default', mip_emphasis=0, mip_gap=0.01, lp_method=0, order=1, 
                          order_type=0, node_file=1, work_mem=2048., memory_emphasis=0, random_seed=None, 
                          n_threads=0, benders_strategy=0, perturbation_indicator=0, perturbation_constant=1e-6,
                          tolerance_feasibility=1e-06, tolerances_optimality=1e-06, **kwargs):
    """ Return a number of docplex models initialized with the same parameters.
    
    Arguments:
    ---------
    checker: {'default', 'numeric', 'off'} (default: 'default')
    
    mip_emphasis: int (default: 0)
        0: balance optimality and feasibility
        1: emphasize feasibility over optimality
        2: emphasize optimality over feasibility
        3: emphasize moving best bound
        4: emphasize finding hidden feasible solutions
        (see MIP emphasis switch)
        
    mip_gap: float (default: 0.01)
        Limit on the optimality gap percentage (e.g., 0.05 => algo stops when gap <= 5%)
    
    lp_method: {0, 1, 2, 3, 4} (default: 0)
        0: automatic choice of the algorithm
        1: primal simplex
        2: dual simplex
        3: network simplex
        4: barrier method

    order: {0, 1} (default: 1)
        0: do not use priority order.
        1: use priority order, if one exists (a priority order can be specified as .ord files or via the 
        `order_type` parameter for generic orders).
        
    order_type: {0, 1, 2, 3} (default: 0)
        Selects the type of generic priority order to generate when no priority order is present.
        0: do not generate a priority order
        1: Use decreasing cost
        2: Use increasing bound range
        3: Use increasing cost per coefficient count

    node_file: {0, 1, 2, 3} (default: 1)
        0: no node files
        1: node file in memory and compressed
        2: node file on disk (files created in temporary directory)
        3: node file on disk and compressed (files created in temporary directory)
        (see "Use node files for storage" in cplex guide)
        
    work_mem: float (default: 2048.)
        Maximum tree storage size (in MB). 
        Past this point what happens next is decided by the setting of `node_file`.
        (see "Use node files for storage" in cplex guide)
        
    memory_emphasis: {0, 1} (default: 0)
        0: do not conserve memory. 
        1: conserve memory where possible. 
        (see memory reduction switch in cplex guide)
        
    random_seed: int or None (default: None)
        If None, Cplex default random seed is used (which varies depending on the version)
        
    benders_strategy: {-1, 0, 1, 2, 3} (default: 0)
    
    n_threads: int
        Number of threads used by the solver.
        
    perturbation_indicator: {0, 1} (default: 0)
        0: let CPLEX choose whether to perturbate the problem
        1: turn on perturbation from the beginning
            
    perturbation_constant: float >= 1e-8 (default: 1e-6)
        Amount by which CPLEX perturbs the upper and lower bounds or objective coefficients on the variables when 
        the problem is perturbated.
        
    tolerance_feasibility: 1e-9 <= float <= 1e-1 (default: 1e-06)
        Specifies the degree to which the basic variables of a model may violate their bounds.
        
    tolerances_optimality: 1e-9 <= float <= 1e-1 (default: 1e-06)
        Influences the reduced-cost tolerance for optimality.
    """
    context = docplex.mp.context.Context.make_default_context()
    context.cplex_parameters.emphasis.mip = mip_emphasis
    context.cplex_parameters.mip.tolerances.mipgap = mip_gap
    context.cplex_parameters.lpmethod = lp_method
    context.cplex_parameters.mip.strategy.order = order
    context.cplex_parameters.mip.ordertype = order_type
    context.cplex_parameters.emphasis.memory = memory_emphasis
    context.cplex_parameters.mip.strategy.file = node_file
    context.cplex_parameters.benders.strategy = benders_strategy
    context.cplex_parameters.workmem = work_mem
    context.cplex_parameters.simplex.perturbation.indicator = perturbation_indicator
    context.cplex_parameters.simplex.perturbation.constant = perturbation_constant
    context.cplex_parameters.simplex.tolerances.feasibility = tolerance_feasibility
    context.cplex_parameters.simplex.tolerances.optimality = tolerances_optimality
    if random_seed is not None:
        context.cplex_parameters.randomseed = random_seed
    context.cplex_parameters.threads = n_threads
    return [docplex.mp.model.Model(context=context, checker=checker) for _ in range(n_models)] 

            
def add_variables_by_blocks(indexing_set, lb, ub, var_type, model, relaxation=False, docplex_var_name=None):
    if docplex_var_name is None:
        indexing_set = len(indexing_set)
    if relaxation or var_type == "C":
        var_list = model.continuous_var_list(indexing_set, lb, ub, name=docplex_var_name)
    elif var_type == "B":
        var_list = model.binary_var_list(indexing_set, name=docplex_var_name)
    elif var_type == "I":
        var_list = model.integer_var_list(indexing_set, lb, ub, name=docplex_var_name)
    return np.array(var_list)    
    

class _timeit(TimeIt):
    """Context manager to run a procedure and print its runtime."""
    def __init__(self, prob, string, level_enter, index=None):
        if index is not None:
            TimeIt.__init__(self, 
                            string, 
                            lambda t: f"Finished. ({t:.3f} sec).\n", 
                            level_enter, 
                            level_enter if index == prob._n_subroots else 4, 
                            prob.verbose, 
                            True if index == 1 else False)
        else:
            TimeIt.__init__(self, 
                            string, 
                            lambda t: f"Finished. ({t:.3f} sec).\n",
                            level_enter,
                            level_enter, 
                            prob.verbose, 
                            True)
