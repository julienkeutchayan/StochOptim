# -*- coding: utf-8 -*-

from typing import Tuple, Optional, List

import os
import datetime
import itertools
Cartesian = itertools.product
import docplex.mp.model
import numpy as np

from stochoptim.util import RedirectStd
import stochoptim.scenclust.general_partitioning as gp

class CostSpaceScenarioPartitioning:
    """
    Class to perform 'decision-based' scenarios partitioning.
        
    It is an implementation of the Cost Space Scenario Clustering (CSSC) algorithm, which measures the quality 
    of a partition via a score defined in the space of cost values rather than in the space of distributions. 
    It can thus be used as a way to do 'problem-driven' scenario reduction.
    
    For now there are two ways to solve the partitioning problem: one is by solving its mixed-integer programming 
    formulation (typically possible for N smaller than 1000) and the other one is by drawing randomly many partitions
    and picking the best one according to the partitioning score (the lower the better) (possible for any N).
    
    (A heuristic solution method is yet to be developed.)
    """
    
    def __init__(self, opport_cost_matrix: np.ndarray):
        """ 
        Argument:
        ---------
        opport_cost_matrix: 2d-array of shape (N, N) with any N >= 1.
            Square-matrix of opportunity cost values.
            Can be computed via method `compute_opportunity_cost` in module `stochastic_problem_basis`.
        """
        assert len(opport_cost_matrix.shape) == 2 and opport_cost_matrix.shape[0] == opport_cost_matrix.shape[1], \
                "Wrong input dimensions: it should be a square matrix."
        
        self._opport_cost_matrix = opport_cost_matrix
        self._solution_mip = dict()
        self._solution_random = dict()
        
    # --- Properties ---
    @property
    def n_scenarios(self) -> int:
        return len(self._opport_cost_matrix)
    
    @property            
    def solution_mip(self) -> dict:
        return self._solution_mip
        
    @property            
    def solution_random(self) -> dict:
        return self._solution_random
      
    # --- General methods ---
    def best_representatives(self, partition: Tuple[Tuple[int]]) -> Tuple[int]:
        """Returns the best representatives for a given partition (as defined by those minimizing the 
        partition score).
        
        Argument:
        ---------
        partition: tuple containing tuples of integers.
            Integers are in the same tuple if they belong to the same subset. 
            There are as many sub-tuples as there are subsets.
            
        Returns:
        --------
        tuple of integers: the representatives
            Each integer at position k is the representative of the subset at position k in the partition.
        """
        representatives = []
        for subset in partition:
            error = [self.score_subset(subset, p) for p in subset]
            representatives.append(subset[np.argmin(error)])
        return tuple(representatives)
    
    def score_subset(self, subset: Tuple[int], rep: int) -> float:
        """Computes the score of a specific subset in the partition.
        
        Arguments:
        ----------
        subset: tuple of integers
            The elements in the subset.
            
        rep: integer
            The representative of the subset.
            
        Returns:
        --------
        float >= 0: the subset score.
        """
        assert rep in subset, f"The representative {rep} does not belong to the subset {subset}."
        return abs(np.mean(self._opport_cost_matrix[rep, subset]) - self._opport_cost_matrix[rep, rep]) 
    
    def score_partition(self, partition: Tuple[Tuple[int]], representatives: Tuple[int]):
        """Computes the score of the whole partition.
        
        Arguments:
        ----------
        partition: tuple containing tuples of integers
            Integers are in the same tuple if they belong to the same subset. 
            There are as many sub-tuples as there are subsets.
            
        representatives: tuple of integers
            Each integer at position k is the representative of the subset at position k in the partition.
            
        Returns:
        --------
        float >= 0: the score of the whole partition.
        """
        weights = np.array([len(subset) / self.n_scenarios for subset in partition])
        score_partition = [self.score_subset(subset, rep) for subset, rep in zip(partition, representatives)]
        return np.average(score_partition, weights=weights)
   
    # --- MIP exact solver ---
    def solve_mip(self, 
                  cardinality: int, 
                  is_cardinality_fixed: bool = True,
                  timelimit: Optional[int] = None,
                  random_warmstarts: int = 0,
                  warmstart_partition: Tuple[Tuple[int]] = None,
                  logfile: bool = True,
                  **kwargs):
        """Computes the best partition by solving the MIP formulation.
        
        Arguments:
        ----------
        cardinality: int >= 1
            Cardinality of the partition.
             
        is_cardinality_fixed: bool (default: True)
            If True, the cardinality is exactly the one given as argument.
            If False, it is at most the one given as argument.
            
        timelimit: int >= 1 or None (default: None)
            Time limit for the partitioning problem. If None, no limit is set.
            
        random_warmstarts: int >= 0 (default: 0)
            Number of randomly generated partitions from which the best is used as warmstart.
            
        warmstart_partition: tuple containing tuples of integers or None (default: None)
            Partition used as a warmstart. (The representatives of that partition are computed in an optimal way.)

        logfile: bool (default: True)
            If True, the solver's log is saved in a file. This file is located in a folder 'logfiles' in the 
            current directory (the folder is created if it doesn't already exist).
            
        kwargs: 
        -------
        lowerbound: int >= 1 (default: 1)
            The minimum number of elements in each subset of the randomly generated warmstart partitions.
            
        uniform_sampling: bool (default: False)
            If True, the partitions are sampled from the uniform distribution. However, this is much slower. 
            If False, a non-uniform distribution is sampled (this distribution still has a positive probability 
            of selecting any partition).
        """
        if logfile:
            if os.path.exists('logfiles') is False: os.mkdir('logfiles')
            filename = fr'logfiles\solver {str(datetime.datetime.now())}.log'.replace(":", "_")
        else: filename = None

        self._prob = docplex.mp.model.Model()
        self._initialize_decisions()
        self._initialize_constraints(cardinality, is_cardinality_fixed)
        self._prob.minimize(self._objective_function())

        if timelimit is not None:
            self._prob.set_time_limit(timelimit)
         
        if random_warmstarts > 0:
            clustering_score_dict = self._random_partitions_dict(cardinality, random_warmstarts, **kwargs)
            best_partition, best_rep = min(clustering_score_dict.keys(), key=(lambda k: clustering_score_dict[k]))
            dict_warmstart = self._build_warmstart(best_partition, best_rep)
            docplex_warmstart = docplex.mp.solution.SolveSolution(self._prob, dict_warmstart)
            self._prob.add_mip_start(docplex_warmstart)
            
        if warmstart_partition is not None:
            representatives = self.best_representatives(warmstart_partition)
            dict_warmstart = self._build_warmstart(warmstart_partition, representatives)
            docplex_warmstart = docplex.mp.solution.SolveSolution(self._prob, dict_warmstart)
            self._prob.add_mip_start(docplex_warmstart)
            
        with RedirectStd(filename):  
            self._prob.solve(log_output=logfile)
                
        self._build_solution_dict()
        
    def _initialize_decisions(self):
        # t[n] = error of cluster n (possibly 0 if n is not a cluster)
        self._t = self._prob.continuous_var_list(self.n_scenarios, name="t") 
        # x[n1, n2] = 1 if scenario n1 is in cluster n2
        self._x = self._prob.binary_var_matrix(self.n_scenarios, self.n_scenarios, name="x") 
        # u[n] = 1 if scenario n is the representative of a cluster
        self._u = self._prob.binary_var_list(self.n_scenarios, name="u")
        
    def _objective_function(self):
        return self._prob.sum([(1 / self.n_scenarios) * self._t[n] for n in range(self.n_scenarios)])

    def _initialize_constraints(self, cardinality, is_cardinality_fixed=True):
        for n2 in range(self.n_scenarios):
            # error of each cluster (possibly 0 if cluster is 'closed')
            first_sum = self._prob.sum([self._x[n1, n2] * self._opport_cost_matrix[n2, n1] for n1 in range(self.n_scenarios)])
            second_sum = self._prob.sum([self._x[n1, n2] * self._opport_cost_matrix[n2, n2] for n1 in range(self.n_scenarios)])
            self._prob.add_constraint(self._t[n2] >= first_sum - second_sum)
            self._prob.add_constraint(self._t[n2] >= second_sum - first_sum)
        # closed clusters have no scenarios
        self._prob.add_constraints([self._x[n1, n2] <= self._u[n2] for n1 in range(self.n_scenarios)
                                                                    for n2 in range(self.n_scenarios)])
        # a representative must be in the cluster it represents
        self._prob.add_constraints([self._x[n2, n2] == self._u[n2] for n2 in range(self.n_scenarios)])
        # each scenario must be in one cluster
        self._prob.add_constraints([self._prob.sum([self._x[n1, n2] for n2 in range(self.n_scenarios)]) == 1 
                                                        for n1 in range(self.n_scenarios)])
        # limit on the number of clusters
        if is_cardinality_fixed:
            self._prob.add_constraints([self._prob.sum([self._u[n2] 
                                            for n2 in range(self.n_scenarios)]) == cardinality])
        else: 
            self._prob.add_constraints([self._prob.sum([self._u[n2] 
                                            for n2 in range(self.n_scenarios)]) <= cardinality])

    def _build_warmstart(self, partition, representatives):
        """Build a solution dictionary from a partition and its representatives."""
        assert len(representatives) == len(partition), (f"the lengths of the partition {len(partition)} and the "
                                                       f"representatives ({len(representatives)}) do not match.")
        u = {f'u_{n}': 1. if n in representatives else 0 for n in range(self.n_scenarios)}            
        x = {f'x_{n1}_{n2}': 0.  for n1 in range(self.n_scenarios) for n2 in range(self.n_scenarios)}
        for k, n2 in enumerate(representatives):
            for n1 in partition[k]:
                x[f'x_{n1}_{n2}'] = 1.
        first_sum = lambda n2: sum([x[f'x_{n1}_{n2}'] * self._opport_cost_matrix[n2, n1] for n1 in range(self.n_scenarios)])
        second_sum = lambda n2: sum([x[f'x_{n1}_{n2}'] * self._opport_cost_matrix[n2, n2] for n1 in range(self.n_scenarios)])
        t = {f"t_{n2}": abs(first_sum(n2) - second_sum(n2)) for n2 in range(self.n_scenarios)}
        return {**u, **x, **t}

    def _build_solution_dict(self):
        # get partition, representatives, and weights
        representatives = tuple([n for n in range(self.n_scenarios) if self._u[n].solution_value])
        partition = tuple([tuple([n1 for n1 in range(self.n_scenarios) if self._x[n1, n2].solution_value]) 
                            for n2 in representatives])
        score = self._prob.objective_value
        solve_details = self._prob.solve_details
        self._solution_mip = {'partition': partition, 
                              'representatives': representatives, 
                              'weights': self._weights(partition), 
                              'score': score,
                              'time': solve_details.time,
                              'bound': solve_details.best_bound,
                              'status': solve_details.status_code}
         
    def _weights(self, partition):
        return np.array([len(part) / self.n_scenarios for part in partition])
    
    def print_partition(self):        
        print("Clustering score: {:.3g}".format(self._prob.objective_value))
        representatives = tuple([n for n in range(self.n_scenarios) if self._u[n].solution_value])
        partition = tuple([tuple([n1 for n1 in range(self.n_scenarios) if self._x[n1, n2].solution_value]) 
                                            for n2 in representatives])
        for k, rep in enumerate(representatives):
            print(f"{rep} representative of {partition[k]}")
                    
    # --- Random partitioning ---
    def best_random_partition(self, cardinality: int, n_sample: int, **kwargs):
        """ Returns the best partition and representatives out of a number of randomly generated partitions.
        
        Arguments:
        ----------
        cardinality: int
            Cardinality of the partition.
            
        n_sample: int
            The number of partitions randomly generated. 
             
        kwargs: 
        -------
        lowerbound: int >= 1 (default: 1)
            The minimum number of elements in each subset of the partition
            
        uniform_sampling: bool (default: False)
            If True, the partitions are sampled from the uniform distribution. However, this is much slower. 
            If False, a non-uniform distribution is sampled (this distribution still has a positive probability 
            of selecting any partition).
        
        Returns:
        --------
        dictionary: mapping
            'partition': tuple of tuples of ints
            'representatives': tuple of ints
            'score': float
        """
        clustering_score_dict = self._random_partitions_dict(cardinality, n_sample, **kwargs)
        best_partition, best_rep = min(clustering_score_dict.keys(), key=(lambda k: clustering_score_dict[k]))
        best_score = clustering_score_dict[best_partition, best_rep]
        self._solution_random = {'partition': best_partition, 
                                 'representatives': best_rep, 
                                 'weights': self._weights(best_partition),
                                 'score': best_score,
                                 'all scores': np.array(list(clustering_score_dict.values()))}
        
    def _random_partitions_dict(self, cardinality, n_sample, **kwargs):
        """ Returns the best partition and representatives out of a number of randomly generated partitions.
        
        Arguments:
        ----------
        see .get_random_partition()
            
        Returns:
        --------
        dictionary: mapping
            (partition, representatives): clustering score
        """
        clustering_score_dict = {}        
        for index, partition in enumerate(gp.partitioning(list(range(self.n_scenarios)), 
                                                          cardinality, 
                                                          n_sample, 
                                                          **kwargs), 1):
            # pick the cluster representators and their weights  
            representatives = self.best_representatives(partition)
            clustering_score = self.score_partition(partition, representatives)  
            clustering_score_dict[(partition, representatives)] = clustering_score
        return clustering_score_dict    
            
        