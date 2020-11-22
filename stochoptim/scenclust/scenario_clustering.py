# -*- coding: utf-8 -*-

import os
from typing import Dict, Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from time import time

try:
    import sklearn.cluster
except ModuleNotFoundError as e:
    print(e)
    
try:
    import pyclustering.cluster.kmedoids
    import pyclustering.cluster.kmedians
except ModuleNotFoundError as e:
    print(e)
    
try:
    import kmodes.kprototypes
except ModuleNotFoundError as e:
    print(e)
    
try:
    import sklearn_extra.cluster
except ModuleNotFoundError as e:
    print(e)  

from stochoptim.scenclust.cost_space_partition import CostSpaceScenarioPartitioning
from stochoptim.scengen.scenario_tree import ScenarioTree
from stochoptim.scengen.decision_process import DecisionProcess
from stochoptim.stochprob.stochastic_problem_basis import StochasticProblemBasis
from stochoptim.stochprob.stochastic_solution_basis import StochasticSolutionBasis
 
class ScenarioClustering:
    """
    Class that wraps together several algorithms to perform scenario reduction on a set of (equal-weight) scenarios 
    for a two-stage stochastic problem (e.g., K-means, K-medoids, Cost-Space-Scenario-Clustering, Monte Carlo, etc.).
    """
    def __init__(self, 
                 scenario_tree: ScenarioTree, 
                 stochastic_problem: StochasticProblemBasis, 
                 is_categorical: Optional[np.ndarray] = None,
                 opport_cost_matrix: Optional[np.ndarray] = None,
                 reference_solution: Optional[StochasticSolutionBasis] = None,
                 load_one_hot_scenarios: bool = False):        
        """        
        Arguments:
        ----------
        scenario_tree: instance of ScenarioTree
            The uncertainty representation of the stochastic problem.

        stochastic_problem: subclass of StochasticProblemBasis
            The two-stage stochastic problem.
            
        is_categorical: 1d-array of shape (n_features,) or None (default: None)
            (where `n_features` is the number of features in one scenario.)
            Array has True at pos i if the i-th feature is a categorical variable; else False.
            
        opportunity_cost_matrix: 2d-array of shape (n_scenarios, n_scenarios) or None (default: None)
            (where `n_scenarios` is the number of scenarios in the scenario tree, i.e., the number of child nodes)
            This matrix is used for the Cost Space Scenario Clustering (CSSC) algorithm. 
            If not provided, the algorithm is not available but the other clustering methods still are.  
            
        reference_solution: instance of StochasticSolutionBasis (or subclass) or None (default: None)
            A solution of the stochastic problem that will served as reference to compute the error gap.
            
        load_one_hot_scenarios: bool (default: False)
            If True and if the scenarios have some categorical features, the scenarios of one-hot representation of these
            features is created. It may then be used by K-means, K-medoids, K-medians, etc. to perform clustering on
            categorical variables. (Note that these scenarios will be much larger than the original because of the one-hot
            representation.)
        """
        # scenarios
        self._scenario_tree = scenario_tree
        self._scenarios = self._scenario_tree.to_numpy()
        self._n_scenarios = len(self._scenarios)
        self._n_features = self._scenarios.shape[1]
        
        # categorical features
        self._is_categorical = is_categorical
        self._unique_cat = set()
        self._n_unique_cat = None
        self._n_num_features = None
        self._n_cat_features = None
        self._scen_num_bin = None
            
        # stochastic problem
        self._stochastic_problem = stochastic_problem
        self._opport_cost_matrix = opport_cost_matrix
        self._reference_solution = reference_solution
        self._map_rvar_to_nb = self._stochastic_problem.map_rvar_name_to_nb[1]
                
        # clustering methods
        self._methods_type_a = ["CSSC", "Kmedoids", "MC"]    # methods with representatives
        self._methods_type_b = ["Kmeans", "Kmedians"]   # methods without representatives
                      
        # Kprototypes only available if array `is_categorical` is provided
        if self._is_categorical is not None:
            self._methods_type_b.append("Kprototypes") 
            
        self._methods_available = self._methods_type_a + self._methods_type_b
        self._results = {}
        self._solutions = {}
        self._start = None
        
        self._set_categorical_attributes(load_one_hot_scenarios)
        self._check_sanity()
        
    def _set_categorical_attributes(self, load_one_hot_scenarios):
        if self._is_categorical is not None:
            # check that categories are all integers
            unique_cat = np.unique(self._scenarios[:, self._is_categorical])
            assert (unique_cat.astype('int') == unique_cat).all(), "Some categories are not integers."
            # set unique categories
            self._unique_cat = set(int(cat) for cat in unique_cat)
            self._n_unique_cat = len(self._unique_cat)
            # numerical features
            scen_num = self._scenarios[:, ~self._is_categorical]
            self._n_num_features = scen_num.shape[1]
            # categorical categorical 
            scen_cat = self._scenarios[:, self._is_categorical]
            self._n_cat_features = scen_cat.shape[1]
            if load_one_hot_scenarios:
                # categorical to binary with more features (one-hot encoding)
                scen_bin_3d = {cat: np.ones((self._n_scenarios, self._n_cat_features)) for cat in self._unique_cat}
                scen_bin_3d = {cat: (scen_bin_3d[cat] == scen_cat).astype('int') for cat in self._unique_cat}
                scen_bin = np.concatenate([scen_bin_3d[cat] for cat in self._unique_cat], axis=1)
                self._scen_num_bin = np.concatenate([scen_num, scen_bin], axis=1)
        else:
            self._n_unique_cat = 0
            self._n_num_features = self._n_features
            self._n_cat_features = 0
            
    # --- Properties ---
    @property
    def results(self):
        return self._results
    
    @property
    def solutions(self):
        return self._solutions
    
    @property
    def scenarios(self):
        return self._scenarios
    
    @property
    def n_features(self):
        """Total number of features in a scenario (i.e., number of parameters included in a scenario)."""
        return self._n_features

    @property
    def n_runs(self) -> Dict[str, Dict[int, int]]:
        return {method: {card: self.get_n_runs(method, card) for card in self.cardinalities(method)}
                                 for method in self._methods_available}
        
    def cardinalities(self, method):
        return sorted([card for meth, card in self._results.keys() if meth == method])
    
    def n_random_features(self, method=None, cardinality=None, index_sample=0):
        """Number of actually random features in the original set of scenarios. A feature is random if it 
        has a standard deviation > 0."""
        n_rand_features = lambda scenarios: len((np.max(scenarios, axis=0) - np.min(scenarios, axis=0)).nonzero()[0])
        if method is None:
            return n_rand_features(self._scenarios)
        else:
            return n_rand_features(self.get_scenarios(method, cardinality, index_sample))

    # --- Clustering methods ---       
    def run_cssc(self, 
                 cardinality, 
                 is_cardinality_fixed=True, 
                 solve_mip=True, 
                 n_random_warmstarts=10**3, 
                 timelimit=None, 
                 **kwargs):
        """Scenario clustering by the Cost Space Scenario Clustering (CSSC) method (implemented in the module 
        `cost_space_partition`).
        
        Arguments:
        ----------
        cardinality: int >= 1 
            The number of clusters in the partition.
            
        solve_mip: bool (default: True)
            If True, the CSSC problem is tackled via the MIP formulation solved using an exact solver.
            If False, the CSSC problem is tackled by sampling partitions randomly and picking the best one (in that 
            case the `warmstart` argument must be >= 1).
            
        n_random_warmstarts: int >= 0 (default: 10**3)
            Number of randomly generated partitions out of which the one with the lowest clustering score will 
            be picked as the best partition or used to initialize the MIP formulation. 
            
        timelimit: int >= 1 or None (default: None)
            Time limit for the partioning problem. If None, no limit is set.
            
        kwargs:
        -------
        logfile: bool (default: True)
            If True, the solver's log is saved in a file. This file is located in a folder 'logfiles' in the 
            current directory (the folder is created if it doesn't already exist).
                              
        warmstart_partition: tuple of tuples of int, or None (default: None)
            Must be None if warmstart is not None.
            If provided, this partition is used as warmstart (along with the representatives).
            
        lowerbound: int >= 1 (default: 1)
            The minimum number of elements in each subset of the randomly generated warmstart partitions.
            
        uniform_sampling: bool (default: False)
            If True, the partitions are sampled from the uniform distribution. However, this is much slower. 
            If False, a non-uniform distribution is sampled (this distribution still has a positive probability 
            of selecting any partition).
        """
        assert self._opport_cost_matrix is not None, "Provide the opportunity cost matrix to run the CSSC algorithm."
        self._check_opportunity_cost_matrix()
        assert solve_mip or n_random_warmstarts >= 1, "`n_random_warmstarts` should be >= 1 if solve_mip=False"
        self.cssc = CostSpaceScenarioPartitioning(self._opport_cost_matrix)
        self._start = time()
        
        if solve_mip:
            self.cssc.solve_mip(cardinality, is_cardinality_fixed, timelimit, n_random_warmstarts, **kwargs)
            representatives = self.cssc.solution_mip['representatives']
            partition = self.cssc.solution_mip['partition']
            score = self.cssc.solution_mip['score']
            
        else:
            self.cssc.best_random_partition(cardinality, n_random_warmstarts, **kwargs)
            representatives = self.cssc.solution_random['representatives']
            partition = self.cssc.solution_random['partition']
            score = self.cssc.solution_random['score']
        self._set_results('CSSC', len(representatives), representatives, partition, None, score)
    
    def run_kmeans(self, cardinality, **kwargs):
        """Scenario clustering by Kmeans method."""
        self._start = time()
        self.kmeans = sklearn.cluster.KMeans(n_clusters=cardinality, **kwargs)
        
        if self._n_unique_cat > 0: # if some categorical features, clustering done on one-hot encoding of features
            self.kmeans.fit(self._scen_num_bin)
            num_centroids = self.kmeans.cluster_centers_[:, :self._n_num_features]
            bin_centroids = self.kmeans.cluster_centers_[:, -(self._n_cat_features * self._n_unique_cat):]
            cat_centroids = self._one_hot_centroids_to_cat(bin_centroids)
            self.kmeans.scenarios = np.zeros((cardinality, self.n_features))
            self.kmeans.scenarios[:, ~self._is_categorical] = num_centroids
            self.kmeans.scenarios[:, self._is_categorical] = cat_centroids
        else:
            self.kmeans.fit(self._scenarios)
            self.kmeans.scenarios = np.array(self.kmeans.cluster_centers_)
        self.kmeans.partition = tuple(tuple((self.kmeans.labels_ == i).nonzero()[0]) for i in range(cardinality))
        self._set_results('Kmeans', cardinality, None, self.kmeans.partition, 
                          self.kmeans.scenarios, self.kmeans.inertia_)
        
    def run_kmedians(self, cardinality, **kwargs):
        """Scenario clustering by the Kmedian algorithm of 'pyclustering'"""
        self._start = time()
        random_indices = np.random.choice(range(self._n_scenarios), size=cardinality, replace=False)
        
        if self._n_unique_cat > 0: # if some categorical features, clustering done on one-hot encoding of features
            initial_medians = self._scen_num_bin[random_indices]
            self.kmedians = pyclustering.cluster.kmedians.kmedians(self._scen_num_bin, initial_medians, **kwargs)
            self.kmedians.process()
            num_centroids = np.array(self.kmedians.get_medians())[:, :self._n_num_features]
            bin_centroids = np.array(self.kmedians.get_medians())[:, -(self._n_cat_features * self._n_unique_cat):]
            cat_centroids = self._one_hot_centroids_to_cat(bin_centroids)
            self.kmedians.scenarios = np.zeros((cardinality, self.n_features))
            self.kmedians.scenarios[:, ~self._is_categorical] = num_centroids
            self.kmedians.scenarios[:, self._is_categorical] = cat_centroids
        else:
            initial_medians = self._scenarios[random_indices]
            self.kmedians = pyclustering.cluster.kmedians.kmedians(self._scenarios, initial_medians, **kwargs)
            self.kmedians.process()
            self.kmedians.scenarios = np.array(self.kmedians.get_medians())

        self.kmedians.partition = tuple(tuple(part) for part in self.kmedians.get_clusters())
        self.kmedians.score = self.kmedians.get_total_wce()
        self._set_results('Kmedians', cardinality, None, self.kmedians.partition, 
                          self.kmedians.scenarios, self.kmedians.score)

    def run_kmedoids(self, cardinality, which_kmedoids='pyclustering', **kwargs): 
        """Scenario clustering by the Kmedoids algorithm of the 'pyclustering' or 'sklearn_extra' library.
        
        Arguments:
        ----------
        which_kmedoids: {'pyclustering', 'sklearn_extra'}
        """
        param = {**{'which_kmedoids': which_kmedoids}, **kwargs}
        self._start = time()
        if which_kmedoids == 'pyclustering':
            initial_medoids = list(np.random.choice(range(self._n_scenarios), size=cardinality, replace=False))
            if self._n_unique_cat > 0: # if some categorical features, clustering done on one-hot encoding of features
                self.kmedoids = pyclustering.cluster.kmedoids.kmedoids(self._scen_num_bin, initial_medoids, **kwargs)
            else:
                self.kmedoids = pyclustering.cluster.kmedoids.kmedoids(self._scenarios, initial_medoids, **kwargs)
            self.kmedoids.process()
            self.kmedoids.score = None
            self.kmedoids.partition = tuple([tuple(part) for part in self.kmedoids.get_clusters()])
            self.kmedoids.representatives = tuple(self.kmedoids.get_medoids()) 
            
        elif which_kmedoids == 'sklearn_extra':
            self.kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=cardinality, **kwargs)
            if self._n_unique_cat > 0: # if some categorical features, clustering done on one-hot encoding of features
                self.kmedoids.fit(self._scen_num_bin)
            else:
                self.kmedoids.fit(self._scenarios)
            self.kmedoids.score = self.kmedoids.inertia_
            self.kmedoids.representatives = tuple(self.kmedoids.medoid_indices_)
            self.kmedoids.partition = tuple(tuple((self.kmedoids.labels_ == i).nonzero()[0]) for i in range(cardinality))
        else:
             raise ValueError("Wrong `which_kmedoids` argument: should be 'pyclustering' or 'sklearn_extra', "
                              f"not {which_kmedoids}")
        self._set_results('Kmedoids', cardinality, self.kmedoids.representatives, self.kmedoids.partition,
                          score=self.kmedoids.score, param=param)
        
    def run_kprototypes(self, cardinality, init='Huang', **kwargs):
        """Scenario clustering by Kprototypes method.
        
        init: {'Huang', 'Cao', 'random'} (default: 'Huang')
            If 'random', both the centroids of the numerical and categorical variables are initialized by picking a 
            scenario randomly.
            If not 'random', the categorical centroids are initialized by 'Huang' or 'Cao' and the numerical by
            k-means++.
        """
        assert self._is_categorical is not None, \
            "Provide the mask on the categorical variables using `is_categorical`" 
        self._start = time()
        
        if self._n_cat_features < self.n_features: # use Kprototype
            self.kprot = kmodes.kprototypes.KPrototypes(n_clusters=cardinality, init=init, **kwargs)
            categorical_indices = self._is_categorical.nonzero()[0]
            self.kprot.fit(self._scenarios, categorical=list(categorical_indices))
            self.kprot.partition = tuple(tuple((self.kprot.labels_ == i).nonzero()[0]) for i in range(cardinality))
            self.kprot.scenarios = -np.ones((cardinality, self._scenarios.shape[1]))
            self.kprot.scenarios[:, ~self._is_categorical] = self.kprot.cluster_centroids_[0]
            self.kprot.scenarios[:, self._is_categorical] = self.kprot.cluster_centroids_[1]
            
        else: # use Kmodes instead of Kprototype
            self.kprot = kmodes.kmodes.KModes(n_clusters=cardinality, init=init, **kwargs)
            self.kprot.fit(self._scenarios)
            self.kprot.scenarios = np.array(self.kprot.cluster_centroids_)
            self.kprot.partition = tuple(tuple((self.kprot.labels_ == i).nonzero()[0]) for i in range(cardinality))
            
        self._set_results('Kprototypes', cardinality, None, self.kprot.partition, 
                          self.kprot.scenarios, self.kprot.cost_)

    def run_monte_carlo(self, cardinality):   
        """Random scenario clustering by Monte Carlo method."""
        self._start = time()
        self.mc_representatives = np.random.choice(range(len(self._scenarios)), size=cardinality, replace=False)
        self._set_results('MC', cardinality, tuple(self.mc_representatives))
        
    # --- Clustering results ---
    def get_n_runs(self, method, cardinality):
        """Return the number of clustering samples performed"""
        return len(self._results.get((method, cardinality), []))
    
    def get_scenarios(self, method, cardinality, index_sample=0):
        try:
            if method in self._methods_type_a:
                return np.array([self._scenarios[rep] for rep in self.get_representatives(method, cardinality, 
                                                                                             index_sample)])
            elif method in self._methods_type_b:
                return self._results[method, cardinality][index_sample]['scenarios']
        except (KeyError, IndexError):
            return None
        
    def get_weights(self, method, cardinality, index_sample=0):
        try:
            return self._results[method, cardinality][index_sample]['weights']
        except (KeyError, IndexError):
            return None
    
    def get_representatives(self, method, cardinality, index_sample=0):
        try:
            return self._results[method, cardinality][index_sample]['reps']
        except (KeyError, IndexError):
            return None

    def get_partition(self, method, cardinality, index_sample=0):
        try:
            return self._results[method, cardinality][index_sample]['partition']
        except (KeyError, IndexError):
            return None
        
    def get_clust_time(self, method, cardinality, index_sample=0):
        try:
            return self._results[method, cardinality][index_sample]['time']
        except (KeyError, IndexError):
            return None
        
    def get_score(self, method, cardinality, index_sample=0):
        try:
            return self._results[method, cardinality][index_sample]['score']
        except (KeyError, IndexError):
            return None
        
    def get_param(self, method, cardinality, index_sample=0):
        try:
            return self._results[method, cardinality][index_sample]['param']
        except (KeyError, IndexError):
            return dict()
        
    def _set_results(self, method, cardinality, representatives, partition=None, scenarios=None, score=None, param=None):
        if self.get_n_runs(method, cardinality) == 0:
            self._results[method, cardinality] = []
        # check param
        assert isinstance(param, dict) or param is None, f"`param` must be a dictionary or None, not {param}"
        param = param if param is not None else dict()
        if method in self._methods_type_a:
            self._set_results_type_a(method, cardinality, representatives, partition, score, param)
        elif method in self._methods_type_b:
            self._set_results_type_b(method, cardinality, partition, scenarios, score, param)
        else:
            raise ValueError(f"{method} is neither of type 'a' nor of type 'b'.")
            
    def _set_results_type_a(self, method, cardinality, representatives, partition, score, param):
        """Set results for methods of type 'a', i.e., those with representatives (e.g., CSSC, Kmedoids, MC)."""
        if partition is None: # equal weights
            self._results[method, cardinality].append({'reps': representatives,
                                                      'weights': np.ones((cardinality,)) / cardinality,
                                                      'partition': None,
                                                      'score': score,
                                                      'param': param,
                                                      'time': time() - self._start})
        else:
            self._results[method, cardinality].append({'reps': representatives,
                                                      'weights': np.array([len(part) / self._n_scenarios 
                                                                               for part in partition]),
                                                      'partition': partition,
                                                      'score': score,
                                                      'param': param,
                                                      'time': time() - self._start})
        
    def _set_results_type_b(self, method, cardinality, partition, scenarios, score, param):
        """Set results for methods of type 'b', i.e., those with partition but no representatives (e.g., Kmeans)"""
        self._results[method, cardinality].append({'scenarios': scenarios,
                                                  'weights': np.array([len(part) / self._n_scenarios 
                                                                           for part in partition]),
                                                  'partition': partition,
                                                  'score': score,
                                                  'param': param,
                                                  'time': time() - self._start})
        
    def _one_hot_centroids_to_cat(self, binary_centroids: np.ndarray):
        """
        Argument:
        ---------
        binary_centroids: 2d-array of shape (cardinality, n_cat * n_cat_features)
            Centroids of the one-hot encoding of categories (with continuous values inside the interval [0, 1])
         
        Returns:
        --------
        cat_centroids: 2d-array of shape (cardinality, n_cat_feeatures)
            Centroids of the categories (with values in `unique_cat`)
        """
        cardinality = binary_centroids.shape[0]
        assert binary_centroids.shape[1] == self._n_unique_cat * self._n_cat_features, \
            (f"Wrong shape for binary centroids, should be ({cardinality}, {self._n_unique_cat * self._n_cat_features}), not "
             f"{binary_centroids.shape}")
        bin_centroids_3d = np.zeros((self._n_unique_cat, cardinality, self._n_cat_features))
        for k, cat in enumerate(self._unique_cat):
            bin_centroids_3d[k] = binary_centroids[:, :self._n_cat_features]
            binary_centroids = binary_centroids[:, self._n_cat_features:]
        cat_centroids = bin_centroids_3d.argmax(axis=0) 
        # turn category index (k) to actual category (cat)
        for k, cat in enumerate(self._unique_cat):
            cat_centroids[(cat_centroids == k)] = cat
        return cat_centroids
    
    def delete_result(self, method, cardinality, indices_sample):
        """Delete all results that are associated to a specific clustering method, cardinality and sample index.
        
        This deletes the results of the clustering instance as well as all the stochastic problem solutions 
        obtained for that instance.

        Arguments:
        ----------
        indices_sample: int or list of ints
            The sample index (or indices) to be deleted.
        """
        if isinstance(indices_sample, int):
            indices_sample = [indices_sample]
        assert len(set(indices_sample)) == len(indices_sample), "Each index should appear only once."
        for index in reversed(sorted(indices_sample)): # remove from larger to lower
            # delete from results attribute
            self._results[method, cardinality].pop(index)
            # delete from solution attribute
            self._solutions.pop((method, cardinality, index), None)
           
    def delete_solution(self, method, cardinality, index_sample, timelimit):
        """Delete the solution associated to a specific method, cardinality, sample index and time limit.
        
        Arguments:
        ----------
        timelimit: int/float or 2-tuple of ints/floats
        """
        # delete solution with this timelimit (whether it is a clustered or implementation solution)
        self._solutions.get((method, cardinality, index_sample), {}).pop(timelimit, None)
        # if clustered solution, then delete also the implementation solution obtained from it
        if isinstance(timelimit, (float, int)):
            for tlimit in list(self._solutions.get((method, cardinality, index_sample), {}).keys()):
                if isinstance(tlimit, tuple) and tlimit[0] == timelimit:
                    self._solutions[method, cardinality, index_sample].pop(tlimit, None) 
            
    # --- Solve stochastic problems ---
    def inner_solve(self, 
                      cardinality: int,
                      methods: Optional[List[str]] = None, 
                      indices_sample: Optional[Dict[str, List[int]]] = None, 
                      timelimit_opt: Optional[int] = None, 
                      timelimit_eval: Optional[int] = None, 
                      logfile_opt: Optional[str] = None,
                      logfile_eval: Optional[str] = None,
                      **kwargs):
        """Solve the stochastic problem over the clustered set of scenarios and evaluate the output decisions. 
        
        All solutions are done via the `.solve` method of the problem; if this method does not fit the needs, check 
        the `outer_solve` where one can implement their own 'optimizer' and 'evaluator' function. 
        
        Arguments:
        ----------
        cardinality: int
            The cardinality of the clustering for which the stochatic problem will be solved.
            
        methods: list of str or None (default: None)
            The clustering methods for which the stochastic problem will be solved.
            If None, all methods are solved.
        
        indices_sample: Dict[str, List[int]] or None (default: None)
            Mapping from the method to the indices of samples for which the stochastic problem will be solved.
            If None, all samples are solved.
            
        timelimit_opt: int >= 1 or None (default: None)
            Time limit to solve the proxy problem with the clustered scenarios.
            
        timelimit_eval: int >= 1 or None (default: None)
            Time limit to solve the problem that evaluates the solution of the clustered problem.
            
        logfile_opt: str or None (default: "")
            This string is appended to the log filename of the optimizer. This file is located in a folder 'logfiles' 
            in the current directory (the folder is created if it doesn't already exist).
            If None, no log is available. 
            
        logfile_eval: str or None (default: "")
            This string is appended to the log filename of the evaluator. This file is located in a folder 'logfiles' 
            in the current directory (the folder is created if it doesn't already exist). 
            If None, no log is available. 
            
        kwargs:
        -------
        all kwargs of StochasticProblemBasis.solve() except:
            `timelimit`, `logfile`, `clear_between_trees`
        """
        methods = self._check_and_set_methods(methods)
        indices_sample = self._check_and_set_samples(indices_sample, methods, cardinality)
                   
        method_samples = [(method, index) for method in methods for index in indices_sample[method]]
        # create list of reduced scenarios and weights using all the methods 
        scenarios_sets = [self.get_scenarios(method, cardinality, index) for method, index in method_samples]
        weights_sets = [self.get_weights(method, cardinality, index) for method, index in method_samples]
        scenario_trees = [ScenarioTree.twostage_from_scenarios(scenarios, self._map_rvar_to_nb, weights)
                                                    for scenarios, weights in zip(scenarios_sets, weights_sets)]

        # solve the stochastic problem on the reduced sets
        solutions = self._stochastic_problem.solve(*scenario_trees,
                                                  timelimit=timelimit_opt,
                                                  clear_between_trees=True,
                                                  logfile=logfile_opt,
                                                  fill_scenario_tree=True,
                                                  **kwargs)
        if len(scenario_trees) == 1:
            solutions = [solutions]
        
        for i, (method, index) in enumerate(method_samples):
            key = (method, cardinality, index)
            if self._solutions.get(key) is None:
                self._solutions[key] = {}
            self._solutions[key][timelimit_opt] = solutions[i]
            
        # get the implementation solution
        for method, index in method_samples:
            key = (method, cardinality, index)
            decision_process = DecisionProcess(self._stochastic_problem.map_dvar_to_index, 
                                               {0: self._solutions[key][timelimit_opt].x0})            
            self._solutions[key][(timelimit_opt, timelimit_eval)] = \
                    self._stochastic_problem.solve(self._scenario_tree,
                                                  decision_process=decision_process,
                                                  timelimit=timelimit_eval,
                                                  logfile=logfile_eval,
                                                  **kwargs)
            
    def outer_solve(self,
                     cardinality: int,
                     optimize_fct: Callable[[ScenarioTree], Tuple[StochasticSolutionBasis, DecisionProcess, int]],
                     evaluate_fct: Callable[[ScenarioTree, DecisionProcess], Tuple[StochasticSolutionBasis, int]],
                     methods: Optional[List[str]] = None, 
                     indices_sample: Optional[Dict[str, List[int]]] = None,
                     **kwargs):
        """Solve the stochastic problem over the clustered set of scenarios and evaluate the solution using the optimizer 
        and evaluator functions provided as input.
         
        Arguments:
        ----------
        cardinality: int
            The cardinality of the clustering for which the stochatic problem will be solved.
            
        optimize_fct: The optimizer function that takes a scenario tree (ScenarioTree) and 
            returns a solution to the stochastic problem (StochasticSolutionBasis), the decision process 
            (DecisionProcess) to be evaluated in the next step, and the time (int) it took to solve the problem.
            
        evaluate_fct: The evaluator function that takes a scenario tree (ScenarioTree), a decision process 
            (DecisionProcess), and returns a solution to the problem (StochasticSolutionBasis) with 
            the time (int) it took to solve the problem.
            
        methods: list of str or None (default: None)
            The clustering methods for which the stochastic problem will be solved.
            If None, all methods are solved.
        
        indices_sample: Dict[str, List[int]] or None (default: None)
            Mapping from the method to the indices of samples for which the stochastic problem will be solved.
            If None, all samples are solved.
        """
        methods = self._check_and_set_methods(methods)
        indices_sample = self._check_and_set_samples(indices_sample, methods, cardinality)

        for method in methods:            
            for index_sample in indices_sample[method]:
               
                key = (method, cardinality, index_sample)
                if self._solutions.get(key) is None:
                    self._solutions[key] = {}     
                    
                scenarios = self.get_scenarios(*key) 
                weights = self.get_weights(*key)
                scenario_tree = ScenarioTree.twostage_from_scenarios(scenarios, self._map_rvar_to_nb, weights)
                
                sol_opt, dec_pro, time_opt = optimize_fct(scenario_tree)
                sol_eval, time_eval = evaluate_fct(self._scenario_tree, dec_pro)
                
                self._solutions[key][time_opt] = sol_opt
                self._solutions[key][(time_opt, time_eval)] = sol_eval
        
    def _check_and_set_samples(self, indices_sample, methods, cardinality):
        """Check and set indices_sample input"""
        if indices_sample is None:
            indices_sample = {method: range(self.get_n_runs(method, cardinality)) for method in methods}
        else:
            # check methods name
            assert set(indices_sample.keys()).issubset(set(self._methods_available)), \
                f"Methods {set(methods).difference(set(self._methods_available))} in `indices_sample` do not exist."
            # check indices sample
            for method in methods:
                all_samples = range(self.get_n_runs(method, cardinality))
                if indices_sample.get(method) is not None:
                    assert set(indices_sample[method]).issubset(set(all_samples)), \
                        (f"Sample indices {set(indices_sample[method]).difference(set(all_samples))} do not exist "
                         f"for {method} and cardinality {cardinality}.")
                else:
                   indices_sample[method] = all_samples
        return indices_sample
   
    def _check_and_set_methods(self, methods):
        """Check and set methods input"""
        if methods is None:
            methods = self._methods_available
        else:
            assert isinstance(methods, list), f"`methods` must be of type list, not {type(methods)}."
            # check methods name
            assert set(methods).issubset(set(self._methods_available)), \
                f"Methods {set(methods).difference(set(self._methods_available))} do not exist."
        return methods
                
    def get_opt_solution(self, method, cardinality, index_sample=None, timelimit=None):
        try:
            if index_sample is None:
                index_sample = 0
            solutions = self._solutions[method, cardinality, index_sample]
            if timelimit is None:
                timelimit = [t for t in solutions.keys() if isinstance(t, (int, float)) or t is None][0]
            return solutions[timelimit]
        except (KeyError, IndexError):
            return None
        
    def get_eval_solution(self, method, cardinality, index_sample=None, timelimit_tuple=None):
        try:
            if index_sample is None:
                index_sample = 0
            solutions = self._solutions[method, cardinality, index_sample]
            if timelimit_tuple is None:
                timelimit_tuple = [t for t in solutions.keys() if isinstance(t, tuple)][0]
            return solutions[timelimit_tuple]
        except (KeyError, IndexError):
            return None

    def get_gap(self, method, cardinality, index_sample=None, timelimit_tuple=None):
        """Return the gap (in %) from the reference solution"""
        try:
            v_ref = self._reference_solution.objective_value
            v_sol = self.get_eval_solution(method, cardinality, index_sample, timelimit_tuple).objective_value
            gap = round(100 * (v_ref - v_sol) / (10**-10 + abs(v_ref)), 5)
            return gap if self._stochastic_problem.objective_sense == 'max' else -gap
        except (AttributeError, KeyError):
            return None

    # --- Display Results ---
    def plot_method(self, 
                    method: str,
                    to_show: str = 'obj',
                    samples: Optional[Dict[int, List[int]]] = None, 
                    card_fct: Callable[[int], bool] = lambda card: True,
                    x_in_percent: bool = False,
                    show_mean=False,
                    title: Optional[str] = None,
                    figsize=(7,5),
                    ax=None):
        """
        samples: mapping between the cardinality and the list of sample indices to be plotted.
        
        to_show: {'gap', 'obj', 'clust-time', 'solve-time'}
        """        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if samples is None:
            samples = {card: range(n_sample) for card, n_sample in self.n_runs[method].items()}
        if show_mean: 
            y_list, card_list = [], []
        for card in sorted(list(samples.keys())):
            if not card_fct(card):
                continue 
            if show_mean: 
                y_list.append([])
                card_list.append(card)
            for index in samples[card]:
                key = (method, card, index)              
                if self._solutions.get(key) is None:
                    continue
                timelimits = self._solutions[key].keys()
                timelimits_tuple = [time for time in timelimits if isinstance(time, tuple)]
                for timelimit in timelimits_tuple:
                    # pick what to display
                    if to_show == 'gap':
                        assert self._reference_solution is not None, \
                            "Impossible to plot the gap if the reference solution is not provided"
                        y = self.get_gap(method, card, index, timelimit)
                        ax.axhline(0)
                    elif to_show == 'obj':
                        y = self._solutions[key][timelimit].objective_value
                        if self._reference_solution is not None:
                            ax.axhline(self._reference_solution.objective_value)
                    elif to_show == 'clust-time':
                        y = self.get_clust_time(*key)
                    elif to_show == 'solve-time':
                        y = self._solutions[key][timelimit[0]].scenario_tree.data['time']
                        if self._reference_solution is not None:
                            ax.axhline(self._reference_solution.scenario_tree.data['time'])
                    else:
                        raise NotImplementedError("Wrong `to_show` argument: should be 'gap', 'obj', 'clust-time', "
                                                  f"or 'solve-time', not {to_show}")
                    if x_in_percent:
                        ax.scatter(100 * card / self._n_scenarios, y, marker="x")
                    else:
                        ax.scatter(card, y, marker="x")
                    if show_mean:
                        y_list[-1].append(y)
        if show_mean:
            ax.plot(card_list, [np.mean(ys) for ys in y_list])
        if x_in_percent:
            ax.set_xlabel("cardinality (%)")
        else:
            ax.set_xlabel("cardinality")
        ax.set_ylabel(to_show)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{method}")
        return ax

    def plot_results(self, methods=None, sharey=True, sharex=False, figsize=(15,3), **kwargs):
        methods = self._methods_available if methods is None else methods
        methods_with_data = [method for method in methods if self.n_runs[method] != dict()]
        fig, ax = plt.subplots(1, len(methods_with_data), figsize=figsize, sharey=sharey, sharex=sharex)
        if len(methods_with_data) == 1:
            self.plot_method(methods_with_data[0], ax=ax, **kwargs)
        else:
            for i, method in enumerate(methods_with_data):
                self.plot_method(method, ax=ax[i], **kwargs)
        return ax
        
    # --- Save and load results ----
    def to_file(self, wdir, tree_extension='pickle', show_files=True, remove_tree_keys=None):
        """Save the clustering instance (clustering results and clustered solution of the stochastic problem).

        Argument:
        ---------
        wdir: string
        
        tree_extension: {'txt', 'pickle'} (default: 'pickle')
        
        show_file: bool (default: True)
        
        remove_tree_keys: List[str] or None (default: None)
            Data keys that will not be saved in the scenario tree in addition to those that are not saved by default
            (see `_save_tree()`).
        """
        # Create folder if doesn't exist
        if not os.path.exists(wdir): 
            os.mkdir(wdir)
        else:
            print(f"Directory already exists. Files may be overwritten.")
        if show_files:
            print(f"These files have been saved at {wdir}:")
        
        # (1) Save string representation just for readable information
        filename = "clustering summary.txt"
        with open(wdir + filename, "w") as f:
            f.write(self.__repr__())
        if show_files:
            print(f"  {filename}")    
        
         # (2) Save result dictionary
        import pickle
        filename = "results.pickle"
        with open(wdir + filename, "wb") as f:
            pickle.dump(self._results, f)
        if show_files:
            print(f"  {filename}")
        
        # (3) Save scenario trees
        for method, cardinality, index_sample in self._solutions.keys():
            key = (method, cardinality, index_sample)
            for timelimit in self._solutions[key].keys():
                filename = self._save_tree(wdir, tree_extension, method, cardinality, index_sample, 
                                           timelimit, remove_tree_keys)
                if show_files:
                    print(f"  {filename}")

    def _save_tree(self, wdir, tree_extension, method, cardinality, index_sample, timelimit, remove_tree_keys):
        if remove_tree_keys is None:
            remove_tree_keys = []
        key = (method, cardinality, index_sample)
        if isinstance(timelimit, (int, float)) or timelimit is None: # tree from clustered problem
            scenario_tree = self._solutions[key][timelimit].scenario_tree
            filename = f"tree_clust-{method}-K{cardinality}-#{index_sample}-{timelimit}sec"
            scenario_tree.to_file(wdir + filename, 'txt', 
                                  without_keys=['scenario', 'memory', 'decision', 'W', 'w'] + remove_tree_keys)
        elif isinstance(timelimit, tuple): # tree from implementation problem
            scenario_tree = self._solutions[key][timelimit].scenario_tree
            filename = f"tree_impl-{method}-K{cardinality}-#{index_sample}-{timelimit}sec"
            scenario_tree.to_file(wdir + filename, tree_extension, 
                                  without_keys=['scenario', 'memory'] + remove_tree_keys)
        else:
            raise TypeError(f"Wrong type for for `timelimit`. Should be int, float or tuple, not {type(timelimit)}")
        return filename
    
    def from_file(self, wdir, tree_extension='pickle', show_files=False):
        """Load the clustering results.
        
        Specifically, this builds the `results` and `solution` attributes of the instance. This is done by
        loading the 'results.pickle' file and all the scenario-tree files saved in the working directory.
        
        Arguments:
        ---------
        wdir: string
            Path to the working directory that contains all the files.
            
        tree_extension: {'txt', 'pickle'} (default: 'pickle')
            Format in which the scenario trees of the implementation problems are saved. 
            (Note that the format for the scenario trees of the clustered problems are always 'txt'.)
        """
        from os import listdir
        from os.path import isfile, join
        import pickle
        
        # (1) Load result dictionary
        loaded_files = ["results.pickle"]  
        try:
            with open(wdir + "results.pickle", "rb") as f:
                self._results = pickle.load(f)
            if show_files:
                print(f"These files have been loaded from {wdir}: \n  results.pickle\n")
                
        except FileNotFoundError:
            print(f"File '{wdir}results.pickle' not found. Impossible to load the clustering results.")
            return 
        
        # (2) Load scenario-tree solution
        files_in_wdir = [f for f in listdir(wdir) if isfile(join(wdir, f))] # get all file in working directory
        for file in files_in_wdir:
            # if file doesn't contain a clustered or implementation scenario tree
            if 'tree_clust' not in file and 'tree_impl' not in file: 
                continue
            loaded_files.append(file)
            # load tree
            filename = file.split('.')[0] # remove extension
            extension = 'txt' if 'clust' in filename else tree_extension
            scenario_tree = ScenarioTree.from_file(wdir + filename, extension)
            
            # get tuple (method, cardinality, index_sample, timelimit)
            method, cardinality, index_sample, timelimit = ScenarioClustering._parse_filename(filename)
            key = (method, cardinality, index_sample)
            if self._solutions.get(key) is None:
                self._solutions[key] = {}
                
            if 'clust' in filename: 
                self._solutions[key][timelimit] = self._stochastic_problem._solution_class(self._stochastic_problem, 
                                                                                        scenario_tree)
            else:
                 # append scenarios to the scenario tree
                scenario_tree.append_data_dict(self._scenario_tree.get_data_dict(['scenario']))
                self._solutions[key][timelimit] = self._stochastic_problem._solution_class(self._stochastic_problem, 
                                                                                       scenario_tree)
            if show_files:
                print(file)
    
    @staticmethod
    def _parse_filename(filename):
        method, cardinality_str, index_sample_str, timelimit_str = filename.split('-')[1:]
        cardinality = int(cardinality_str[1:]) # remove prefix 'K' and convert to integer
        index_sample = int(index_sample_str[1:]) # remove prefix '#' and convert to integer
        timelimit = eval(timelimit_str[:-3]) # remove suffix 'sec' and convert to tuple or integer via eval()
        return method, cardinality, index_sample, timelimit
      
    # --- Representations ---    
    def __str__(self):
        string = ("Scenario Clustering: \n"
                  f"  Scenarios: {self._n_scenarios} \n"
                  f"  Features: {self.n_features} \n"
                  f"    - Random: {self.n_random_features()} (with std > 0) \n"
                  f"    - Numeric: {self._n_num_features} \n"
                  f"    - Categorical: {self._n_cat_features} \n"
                  f"  Clustering methods: {self._methods_available}\n\n")
        
        string += "Clustering performed:\n"
        for method in self._methods_available:
            string += (f"  {method}: {self.n_runs[method]}\n")
            
        string += ("\nReference solution: \n"
                  f"  {self._reference_solution} \n")
        return string
                  
    def print_results(self, method=None, cardinality=None, index_sample=None, return_string=False):
        # limit print of the weights
        np.set_printoptions(threshold=15, linewidth=120) # max 15 weights
        # get methods, cardinality and indices to print
        methods = self._methods_available if method is None else [method]
        is_card_valid = lambda card: True if cardinality is None else card == cardinality
        is_index_valid = lambda index: True if index_sample is None else index == index_sample
        string = f"Reference solution: \n  {self._reference_solution} \n"
        for method in methods:
            string += f"\n{method}: \n" + "#" * (len(method)+1) + "\n"
            for card in self.cardinalities(method):
                if not is_card_valid(card): continue
                string += f"  Cardinality: {card} \n  " + "-" * len(f"Cardinality: {card}") + "\n"
                for index in range(self.get_n_runs(method, card)):
                    if not is_index_valid(index): continue
                    key = (method, card, index)
                    string += (f"    Weights: {self.get_weights(*key)} \n"
                               f"    Representatives: {self.get_representatives(*key)} \n"
                               f"    Score: {self.get_score(*key)} \n"
                               f"    Param: {self.get_param(*key)} \n"
                                "    Random features: "
                                        f"{self.n_random_features(*key)}/{self.n_random_features()} \n"
                               f"    Time: {self.get_clust_time(*key):.2f} sec\n")                    
                    if self._solutions.get(key) is None:
                        string += (f"    Clustered problem: None \n"
                                   f"    Implement. problem: None \n"
                                   f"    Gap to ref. sol. (%): None\n\n")
                    else:
                        timelimits = self._solutions[key].keys()
                        timelimits_clust = [time for time in timelimits if isinstance(time, (int, float)) or time is None]
                        timelimits_tuple = [time for time in timelimits if isinstance(time, tuple)]
                        string += f"    Clustered problem: \n"
                        for timelimit in timelimits_clust:
                            string += f"      {timelimit} sec: {self.get_opt_solution(*key, timelimit)} \n"
                        string += f"    Implement. problem: \n"
                        for timelimit in timelimits_tuple:
                            string += f"      {timelimit} sec: {self.get_eval_solution(*key, timelimit)} \n"
                        string += f"    Gap to ref. sol. (%): \n"
                        for timelimit in timelimits_tuple:
                            string += f"      {timelimit} sec: {self.get_gap(*key, timelimit)} \n"
                    string += "\n"
        if return_string:
            return string
        else:
            print(string, end="")
            
    # --- Sanity check ---
    def _check_opportunity_cost_matrix(self):
        if self._opport_cost_matrix is None:
            return 
        assert self._opport_cost_matrix.shape == (self._n_scenarios, self._n_scenarios), \
            ("Shape of opportunity cost matrix doesn't match the number of scenarios: "
             f"{self._opport_cost_matrix.shape} != {(self._n_scenarios, self._n_scenarios)}")
            
    def _check_sanity(self):
        self._check_opportunity_cost_matrix()
        assert len(self._scenarios) >= 3, \
            f"Clustering the scenarios makes sense for 3 scenarios or more, not {len(self._scenarios)}."