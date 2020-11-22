# -*- coding: utf-8 -*-

import numpy as np

import copy
import itertools
from time import time
import matplotlib.pyplot as plt

from stochoptim.scengen.tree_structure import Node
from stochoptim.scengen.scenario_tree import ScenarioTree
Cartesian = itertools.product

class TreeSearch:
    
    def __init__(self, scenario_process, variability_process, demerit, nber_stages):
        self.scenario_process = scenario_process
        self.variability_process = variability_process
        self.demerit = demerit
        self.nber_stages = nber_stages
        self.last_stage = nber_stages - 1
        
        self._search_methods = [("VNS", "forward"), ("VNS", "backward"), 
                                ("EXH", "forward"), ("EXH", "backward")]
        self._best_fod = {}
        self._best_tree = {}
        self._fod_sample = {}
        self._initialize()
        
    def _initialize(self, method=None):
        methods = self._search_methods if method is None else [method]
        for method in methods:
            self._best_fod[method] = np.inf
            self._best_tree[method] = None
            self._fod_sample[method] = []
        
    def fod_sample(self, method):
        return np.array(self._fod_sample[method])
    
    def best_tree(self, method):
        return self._best_tree[method]
    
    def best_fod(self, method):
        return self._best_fod[method]
    
    def plot_fod_hist(self, method, bins=10, figsize=(5,5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self._fod_sample[method], bins=bins)
        ax.set_title(f"P10: {np.quantile(self._fod_sample[method], 0.1):.5f} "
                     f"; P50: {np.quantile(self._fod_sample[method], 0.5):.5f}\n"
                     f"min: {min(self._fod_sample[method]):.5f} ;"
                     f"max: {max(self._fod_sample[method]):.5f}")
        plt.show()
        
    def plot_fod_progress(self, method, figsize=(5,5)):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(range(len(self._fod_sample[method])), np.minimum.accumulate(self._fod_sample[method]))
        best_index = np.where(np.minimum.accumulate(self._fod_sample[method]) <= self._best_fod[method] + 10**-10)[0][0]
        ax.scatter(best_index, self._best_fod[method])
        plt.show()
        
    def variable_neighborhood_search(self, 
                                    nber_scenarios, 
                                    initial_tree=None,
                                    optimized='forward', 
                                    max_iteration=np.inf,
                                    max_no_improvement=np.inf,
                                    num_local_samples=10, 
                                    num_neighborhoods=10, 
                                    neighborhood_shrink=0):
        """Explore the space of tree structures via a strategy of 'variable neighborhood' to find the scenario tree 
        of lowest demerit"""
        assert optimized in ["forward", "backward"], ("`optimized` must be either 'forward' or "
        f"'backward', not {optimized}.")
        self._initialize(('VNS', optimized))
        time0 = time()
        if initial_tree is None:
            bushiness = (nber_scenarios,) + (1,) * (self.nber_stages-2)
            initial_tree = ScenarioTree.from_bushiness(bushiness)
            initial_tree.fill(self.scenario_process, optimized, self.variability_process, self.demerit)
            
        self._best_fod[('VNS', optimized)] = initial_tree.get_figure_of_demerit(self.demerit)
        self._best_tree[('VNS', optimized)] = copy.deepcopy(initial_tree)
        
        if initial_tree.depth <= 2:
            return self._best_tree[('VNS', optimized)], self._best_fod[('VNS', optimized)]
    
        iteration, no_improvement_count = 0, 0
        while iteration < max_iteration:
            try:
                iteration += 1
                nbreed = 1 #max(1, int(nber_scenarios / np.log(3*iteration)**neighborhood_shrink))
                candidates = [copy.deepcopy(self._best_tree[('VNS', optimized)]) for i in range(num_local_samples)]
    
                # increase neighborhood distance until improvement
                for neighborhood in range(1, num_neighborhoods + 1):
                    improved = False
        
                    # try multiple samples in the same neighborhood
                    for current_tree in candidates:
                        # increase neighborhood of current candidate
                        for ibreed in range(nbreed):
                            TreeSearch._tree_breed(current_tree) # split or merge
        
                        current_tree.fill(self.scenario_process, optimized, self.variability_process, self.demerit)
                        current_fod = current_tree.get_figure_of_demerit(self.demerit)
                        self._fod_sample[('VNS', optimized)].append(current_fod)
                        
                        if current_fod < self._best_fod[('VNS', optimized)]:
                            improved = True
                            self._best_tree[('VNS', optimized)] = current_tree
                            self._best_fod[('VNS', optimized)] = current_fod
                        
                    # start over if at least one sample provided improvement
                    if improved:
                        no_improvement_count = 0
                        break
        
                if not improved:
                    no_improvement_count += 1
                    if no_improvement_count >= max_no_improvement:
                        break
                    
                print(f"\riteration: {iteration}  demerit: {current_fod:.5f}  "
                      f"best demerit: {self._best_fod[('VNS', optimized)]:.5f}  "
                       f"no improvement count: {no_improvement_count}", end="")  
                
            except KeyboardInterrupt:
                break

        time1 = time()
        print(f"\nTotal number of iterations : {iteration} ({time1-time0:.1f} sec)")

    @staticmethod
    def _merge_nodes(node1, node2):
        assert node2 in node1.parent.children
        node1.parent.children.remove(node2)
        node1.add(*node2.children)

    @staticmethod
    def _split_node(node1, num_children_array):
        nodes = [Node() for nc in num_children_array]
        # redistribute subtrees
        for n, num_children in zip(nodes, num_children_array):
            for nc in node1.children[:num_children]:
                node1.children.remove(nc)
                n.add(nc)
        # remove old node to parent
        node1.parent.children.remove(node1)
        # add new nodes to parent
        node1.parent.add(*nodes)
        
    @staticmethod
    def _tree_breed(tree):
        # randomly pick a valid action
        actions = ['merge']

        splittable_nodes = [node for node in tree.nodes if node.parent and len(node.children) >= 2]
        if len(splittable_nodes) > 0:
            actions.append('split')

        mergeable = [node for node in tree.nodes if not node.is_leaf and node.has_siblings]
        action = 'split' if len(mergeable) == 0 else np.random.choice(actions)

        if action == 'merge':
            node1 = np.random.choice(mergeable)
            node2 = np.random.choice([n for n in node1.parent.children if n is not node1])
            TreeSearch._merge_nodes(node1, node2)

        elif action == 'split':
            node1 = np.random.choice(splittable_nodes)
            k = np.random.randint(len(node1.children) - 1) + 1
            num_children_array = [k, len(node1.children) - k]
            TreeSearch._split_node(node1, num_children_array)
        else:
            raise ValueError("unknown action")
            
    def exhaustive_search(self, 
                          nber_scenarios, 
                          optimized='forward',
                          min_branching_factor=1, 
                          max_iteration=np.inf):
        """Explore exhaustively the space of tree structures to find the scenario tree of lowest demerit"""
        assert optimized in ["forward", "backward"], ("`optimized` must be either 'forward' or "
        f"'backward', not {optimized}.")
        self._initialize(('EXH', optimized))
        time0 = time()
        iteration_count, no_improvement_count = 1, 0

        for structure in TreeSearch._exhaustive_structures(self.nber_stages, nber_scenarios, min_branching_factor):   
            try:
                current_tree = ScenarioTree(structure)
                current_tree.fill(self.scenario_process, optimized, self.variability_process, self.demerit)
                
                current_fod = current_tree.get_figure_of_demerit(self.demerit)
                self._fod_sample[('EXH', optimized)].append(current_fod)
                
                if current_fod < self._best_fod[('EXH', optimized)]:
                    improved = True
                    self._best_tree[('EXH', optimized)] = current_tree
                    self._best_fod[('EXH', optimized)] = current_fod
                else:
                    improved = False
                    
                no_improvement_count += 1 if not improved else 0
                    
                if iteration_count % 10 == 0:
                    print(f"\riteration: {iteration_count}  demerit: {current_fod:.5f}  "
                          f"best demerit: {self._best_fod[('EXH', optimized)]:.5f}  "
                           f"no improvement count: {no_improvement_count}", end="")        
            except KeyboardInterrupt:
                break
            
            iteration_count += 1
            if iteration_count > max_iteration:
                break

        time1 = time()
        print(f"\nTotal number of iterations : {iteration_count-1} ({time1-time0:.1f} sec)")
        
    @staticmethod
    def _exhaustive_structures(depth, N, b):
        """Generates all tree structures with depth T, N scenarios, and a branching lowerbound b""" 
        if depth == 2:
            yield Node.from_bushiness((N,))
            return

        for n in range(b**(depth-1), int(N / b) + 1):            
            for tree in TreeSearch._exhaustive_structures(depth-1, n, b):                
                for new_tree in TreeSearch._extend_structure(tree, N, b):        
                    new_tree.delete_data(["pos", "n"])
                    yield new_tree
                         
    @staticmethod        
    def _pseudo_integer_partitions(cardinality, integer, lowerbounds):
        """Enumeration of all integer partitions (not permutation free) of a fixed cardinality. 
        The partition of an integer k is a tuple (n_1, ..., n_m) such that:
            (i) k = n_1 + ... + n_m
            (ii) n_i is integer >= 1.
            
        Unlike the regular integer partitions (method ._integer_partitions()), here the condition:
            n_1 <= n_2 <= ... <= n_m 
        need not be satisfied.
        
        The lowerbounds adds the condition:
            (iv) n_i >= lowerbound[i], for i = 1, ...,m (componentwise lowerbound).
        
        Arguments:
        ----------
        cardinality: integer >= 1
            The number of elements partitionning the integer.
            
        integer: integer >= 1
            The integer being partitionned.
            
        lowerbounds: tuple of integers >= 1
            The componentwise lowerbound on the elements partitionning the integer.
            
        Returns:
        --------
        iterator on the partitions.
        """
        if cardinality == 1:
            if integer >= lowerbounds[0]:
                yield (integer,)
                return
        for i in range(lowerbounds[0], integer-sum(lowerbounds[1:])+1):
            for t in TreeSearch._pseudo_integer_partitions(cardinality-1, integer-i, lowerbounds[1:]):
                yield (i,) + t
    
    @staticmethod               
    def _integer_partitions(cardinality, integer, lowerbound=1):
        """Enumeration of all integer partitions of a fixed cardinality. A partition of cardinality m of 
        an integer k is a tuple (n_1, ..., n_m) such that:
            (i) k = n_1 + ... + n_m
            (ii) n_i is integer >= 1
            (iii) n_1 <= n_2 <= ... <= n_m.
            
        The lowerbound (optional) adds the condition:
            (iv) n_i >= lowerbound, for i = 1, ...,m.
        
        Arguments:
        ----------
        cardinality: integer >= 1
            The number of elements partitionning the integer.
            
        integer: integer >= 1
            The integer being partitionned.
            
        lowerbound: integer >= 1 (default 1)
            The lowerbound on the elements partitionning the integer.
            
        Returns:
        --------
        iterator on the partitions.
        """
        if cardinality == 1:
            if integer >= lowerbound:
                yield (integer,)
                return
        for i in range(lowerbound, integer-(cardinality-1)*lowerbound+1):
            for t in TreeSearch._integer_partitions(cardinality-1, integer-i, i):
                yield (i,) + t
    
    @staticmethod
    def _extend_structure(tree, N, b):
        """This generator takes any tree structure and generates all 
        tree structure with N scenarios and 1 stage more."""
        partitionP1, partitionP2, P2, lowerboundsP2  = {}, {}, {}, {}
    
        #Create the Partition P1 of leaf nodes
        for leaf in tree.leaves:
            history = tuple([len(n.children) for n in leaf.branch if n != leaf])
        
            if history in partitionP1.keys():
                partitionP1[history] += [leaf] 
            else:
                partitionP1[history] = [leaf] 
            P1 = len(partitionP1.keys())
    
        lowerboundsP1 = [b * len(partitionP1[key]) for key in partitionP1.keys()]
        
        #Create the Partition P2 of leaf nodes
        for i, key in enumerate(partitionP1.keys()):
            partitionP2[key] = list(set([tuple(leaf.parent.children) for leaf in partitionP1[key]]))
            P2[key] = len(partitionP2[key])
            lowerboundsP2[key] = len(partitionP2[key][0])
    
        #Create a data 'pos' (a 3-tuple) to index each leaf in its partition and subpartition
        for i, key in enumerate(partitionP1.keys()):
            for j, leaves in enumerate(partitionP2[key]):
                for k, leaf in enumerate(leaves):
                    leaf.data["pos"] = (i, j, k)
           
        #generate the integer tuples x, y, z to code the tree branching
        indexing_A = TreeSearch._pseudo_integer_partitions(P1, N, lowerboundsP1)
        for x in indexing_A: 
            indexing_B = lambda i, key: list(TreeSearch._integer_partitions(P2[key], 
                                                                                 x[i], 
                                                                                 b*lowerboundsP2[key]))

            for y in Cartesian(*[indexing_B(i, key) for i, key in enumerate(partitionP1.keys())]):
    
                set_B = lambda i, j, key: list(TreeSearch._integer_partitions(lowerboundsP2[key], 
                                                                                   y[i][j], 
                                                                                   b))
                for z in Cartesian(*[Cartesian(*[set_B(i, j, key) for j in range(P2[key])]) 
                                                                for i, key in enumerate(partitionP1.keys())]):
                    new_tree = copy.deepcopy(tree)
                    
                    for leaf in list(new_tree.leaves):
                        (i, j, k) = leaf.data["pos"]
                        leaf.add(*[Node() for i in range(z[i][j][k])])
    
                    yield new_tree
    
                    
                    