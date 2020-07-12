# -*- coding: utf-8 -*-

from typing import Callable, Optional

import numpy as np

from stochoptim.scengen.variability_process import VariabilityProcess
from stochoptim.scengen.tree_structure import Node


class FigureOfDemerit:
    
    def __init__(self, 
                 demerit_fct: Callable[[int, np.ndarray, np.ndarray], float],
                 variability_process: VariabilityProcess):
        """
        Arguments:
        ----------
        demerit_fct: Callable[[int, 2d-array, 1d-array], float]
            The function giving the quality value (so-called 'demerit') of the discretization points and weights.
            See the signature of the `sample_demerit` method.
            
        variability_process: VariabilityProcess
            The variability process used to compute the figure of demerit of a subtree.
        """
        self._demerit_fct = demerit_fct
        self._variability_process = variability_process
        
    def sample_demerit(self, 
                       stage: int, 
                       epsilons: np.ndarray, 
                       weights: np.ndarray) -> float:
        r"""Demerit (quality measure) of the discretization points and weights 
        of the stagewise independent random vector \epsilon_{stage}.
        
        Arguments:
        ----------
        stage: int >= 0
        
        epsilons: 2d-array
            The discretization points given as an array of shape (n_sample, dim_epsilon).
            
        weights: 1d-array
            The discretization weights given as an array of shape (n_sample,).
            
        disc_method: string (optional)
            The method used to generate the discretization points and weights
            
        Returns:
        --------
        float > 0: 
            Positive number representing the quality measure (so-called 'demerit') of the discretization 
            points and weights.
        """
        return self._demerit_fct(stage, epsilons, weights)

    def children_demerit(self, node: Node) -> float:
        """Demerit (quality measure) of the discretization points and weights at the child nodes of 'node'.
        
        Argument:
        ---------
        node: Node
        
        Returns:
        --------
        float > 0: value of the demerit.
        """
        if node.is_leaf:
            return 0
        else:
            return self.sample_demerit(stage=node.level, 
                                       epsilons=np.array([c.data.get("eps") for c in node.children]),
                                       weights=np.array([c.data["w"] for c in node.children]))
        
    def lookback_demerit(self, node: Node) -> float:
        """Figure of demerit as defined in [1, eq.(45)] and computed using the recursion [2, eq.(26)-(27)].
        
        This is the general formula for the figure of demerit, which can be computed only when the scenario tree 
        has been filled (optimally or not) with the scenarios at each node. It uses the path-dependent (lookback)
        variability function. 
        
        [1]: Keutchayan, Munger, Gendreau, Bastin (2018) 'A quality measure for the discretization of probability 
        distributions in multistage stochastic optimization'
        [2]: Keutchayan, Munger, Gendreau (2019) 'On the scenario-tree optimal-value error 
        for stochastic programming problems'
        
        Argument:
        ---------
        node: Node
            If node is the tree root, then it outputs the actual figure of demerit of the scenario tree.
            If node is not the root, then it computes only the subset of the whole sum over the nodes in the subtree
            rooted at 'node'.
        
        Returns:
        --------
        float > 0: value of the demerit.
        """
        if node.is_leaf:
            return 0
        else:
            return self.children_demerit(node) * self._variability_process.node_lookback(node) \
                        + sum(child.data["w"] * self.lookback_demerit(child) for child in node.children)
    
    def looknow_demerit(self, node: Node) -> float:
        """Figure of demerit as defined in [1, eq.(49)] and computed using the recursion [1, eq.(50)-(51)]. 
        
        This is the simplified formula that only features the path-independent (looknow) variability. It coincides 
        with the general formula (given by lookback demerit) if the variability is of the product form accross stages.
        
        [1]: Keutchayan, Munger, Gendreau, Bastin (2018) 'A quality measure for the discretization of probability 
        distributions in multistage stochastic optimization'.
        
        Argument:
        ---------
        node: Node
            If node is the tree root *and* variability is of the product form, then it outputs the actual 
            figure of demerit of the scenario tree (it then coincides with .lookback_demerit(node)).
            If node is not the root *and* variability is of the product form, then it computes figure of demerit 
            of the sub-scenario-tree rooted at 'node'.
        
        Returns:
        --------
        float > 0: value of the demerit.
        """
        if node.is_leaf:
            return 0
        else:
            return self.children_demerit(node) + sum(child.data["w"] 
                                                    * self._variability_process.node_looknow(child)
                                                    * self.looknow_demerit(child) for child in node.children)
    
    def __call__(self, 
                 node: Node, 
                 subtree: bool, 
                 path: Optional[bool] = None) -> float:
        """Returns the demerit of the node it is called on.
        
        Arguments:
        ----------
        node: Node
            The node for which the demerit will be computed.
            
        subtree: bool
            If True, the demerit of the whole subtree rooted at 'node' is computed, 
            otherwise only the demerit of its child nodes.
            
        path: bool or None
            If True, the path-dependent variability is used to compute the subtree demerit, 
            otherwise the path-independent is used. (Makes sense only if the *subtree* demerit is computed, 
            i.e., if subtree is True.)
            
        Returns:
        --------
        float: the quality measure (so-called 'demerit') of a node
        """
        if subtree:
            if path:
                return self.lookback_demerit(node)
            else:
                return self.looknow_demerit(node)
        else:
            return self.children_demerit(node)
            