# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict

from itertools import product
import numpy as np

from stochoptim.scengen.scenario_tree import ScenarioTree

class UncertaintyBasis(ABC):
    """
    Basis for the uncertainty of two-stage stochastic problems from a given set of scenarios.
    """
    
    def __init__(self, name: str, n_scenarios: int, features: Dict[str, int]):
        self.name = name
        self.n_scenarios = n_scenarios
        self.features = features
        self.n_features = sum(self.features.values())
            
    @abstractmethod
    def get_scenario(self, scen_index=None) -> Dict[str, np.ndarray]:
        """Returns a set of scenarios.
        
        Arguments:
        ----------
        scen_indices: list of ints or None (default: None)
            The indices for which the scenarios are returned.
            If None, all scenarios are returned.
            
        Returns:
        --------
        Dict[str, 1d-array]: mapping a variable name to a variable array.
        """  
        pass
    
    def get_scenario_tree(self, scen_indices=None):
        """Returns a set of geological scenarios with all mines and all attributes combined.
        
        Arguments:
        ----------
        scen_indices: list of ints or None (default: None)
            The indices of the scenarios returned. If None, all scenarios are returned.
            
        Returns:
        --------
        instance of ScenarioTree
        """   
        if scen_indices is None:
            scen_indices = range(self.n_scenarios)
        weights = np.ones(len(scen_indices)) / len(scen_indices)        
        data_dict = {(): {'W': 1}}
        for i, scen_index in enumerate(scen_indices):
            data_dict[(i,)] =  {'W': weights[i], 'scenario': self.get_scenario(scen_index)}            
        return ScenarioTree.from_data_dict(data_dict)
    
    def mean_scenario(self, scen_indices=None): # !!! not up to date
        """Returns the mean scenario as a 1d-array of shape (n_features,)."""
        return np.mean(self.get_scenarios(scen_indices), axis=0)
    
    def _check_scen_index(self, scen_index):
        assert scen_index in range(self.n_scenarios), \
            f"Scenario index {scen_index} is not in the range [0, {self.n_scenarios-1}]."
            
    def __repr__(self):
        string = (f"{self.name} \n"
                  f"  - scenarios: {self.n_scenarios} \n"
                  f"  - features: {self.n_features} \n")
        for var_name, n_var in self.features.items():
            string += f"    - {var_name}: {n_var} \n"
        return string
    
    
class CrossUncertainty(UncertaintyBasis):
    
    def __init__(self, *uncertainties, combiner):
        self.uncertainties = uncertainties
        self.combiner = combiner
        self.n_uncertainties = len(self.uncertainties)
        
        self.n_scenarios_per_uncertainty = [uncertainty.n_scenarios for uncertainty in self.uncertainties]
        self.n_features_per_uncertainty = [uncertainty.n_features for uncertainty in self.uncertainties]
        self._set_combiner()
            
        # combine all features together
        features = {}
        for uncertainty in self.uncertainties:
            features = {**features, **uncertainty.features}
            
        UncertaintyBasis.__init__(self, 
                                  name=f"Cross uncertainty ({self.combiner})",
                                  n_scenarios=self.n_scenarios,
                                  features=features)
        
    def _set_combiner(self):
        if self.combiner == "cartesian": 
            self.n_scenarios = np.prod(self.n_scenarios_per_uncertainty)
            self.map_scen_index_to_scen_tuple = list(product(*[range(uncertainty.n_scenarios) 
                                                                    for uncertainty in self.uncertainties]))
        elif self.combiner == "zip":
            assert min(self.n_scenarios_per_uncertainty) == max(self.n_scenarios_per_uncertainty), \
                         "All uncertainties must have the same number of scenarios for combiner 'zip'."
            self.n_scenarios = self.n_scenarios_per_uncertainty[0]
            self.map_scen_index_to_scen_tuple = [tuple([i] * len(self.uncertainties)) 
                                                    for i in range(self.n_scenarios)]
        else:
            raise TypeError(f"Wrong 'combiner' input: must be either 'cartesian' or 'zip', not {self.combiner}")    
          
    def get_scenario(self, scen_index):
        self._check_scen_index(scen_index)
        scen_tuple = self.map_scen_index_to_scen_tuple[scen_index]
        scenario = {}
        for i, uncertainty in enumerate(self.uncertainties):
            scenario = {**scenario, **uncertainty.get_scenario(scen_tuple[i])}
        return scenario
        
    def mean_scenarios(self, uncertainty_indices=None, scen_indices=None): # !!! not up to date
        """ Returns a set of scenarios averaged over certain uncertainties only.
        
        Arguments:
        ----------
        uncertainty_indices: list of ints or None (default: None)
            The index of the uncertainty for which the set of scenarios will be averaged.
            If None, all uncertainties are averaged; the method is then equivalent to `.mean_scenario()` of 
            UncertaintyBasis, expect that the later returns an 1d-array.
            
        scen_indices: list of ints or None (default: None)
            The indices of the cross-uncertainty scenarios over which the average will be done.
            If None, all scenarios are averaged.
            
        Returns:
        --------
        2d-array
        """
        if uncertainty_indices is None:
            return UncertaintyBasis.mean_scenario(self, scen_indices).reshape(1, -1)
        
        assert set(uncertainty_indices).issubset(set(range(self.n_uncertainties))), \
            f"Uncertainty indices {uncertainty_indices} should be in the range [0, {self.n_uncertainties-1}]"
            
        mean_scenarios = self.get_scenarios(scen_indices)
        count = 0
        rows_to_avg = []
        for index in range(self.n_uncertainties):
            if index in uncertainty_indices:
                features_columns = count + np.array(range(self.n_features_per_uncertainty[index]))
                rows_to_avg += list(features_columns)
            count += self.n_features_per_uncertainty[index]
        mean_scenarios[:, rows_to_avg] = np.mean(self.get_scenarios(scen_indices)[:, rows_to_avg], axis=0)
        
        if self.combiner == 'zip':
            if set(uncertainty_indices) == set(range(self.n_uncertainties)): # if all uncertainty are averaged
                # all scenarios are the same so we output the 1st one only 
                return mean_scenarios[[0]] # shape (1, n_features)
            else:
                # all scenarios may be different
                return mean_scenarios # shape (n_scenarios, n_features)
            
        
        # we drop the indentical scenarios by building the scenarios of a new cross uncertainty with 
        # the uncertainties that have not been averaged
        non_avg_uncertainty_indices = [i for i in range(self.n_uncertainties) if i not in uncertainty_indices]
        rows_of_interest = self._rows_sub_cross_uncertainty(non_avg_uncertainty_indices)
        return mean_scenarios[rows_of_interest]
            
    def _rows_sub_cross_uncertainty(self, uncertainty_indices):
        """Returns the rows (scenarios) indices that are required to build a sub cross-uncertainty.
        
        Arguments:
        ----------
        uncertainty_indices: list of ints
            The indices of the uncertainties used to build the sub cross-uncertainty.
            
        Returns:
        --------
        list of ints
        """
        rows = np.linspace(0, self.n_scenarios-1, self.n_scenarios, dtype='int32')
        rows = rows.reshape(*self.n_scenarios_per_uncertainty)
        iterator = product(*[range(n_scenarios) if index in uncertainty_indices else range(1) 
                            for index, n_scenarios in enumerate(self.n_scenarios_per_uncertainty)])
        return [rows[tuple_index] for tuple_index in iterator]
                
    def __repr__(self):
        string = UncertaintyBasis.__repr__(self) + "\nComposed of:\n------------"
        for uncertainty in self.uncertainties:
            string += "\n" + UncertaintyBasis.__repr__(uncertainty)
        return string