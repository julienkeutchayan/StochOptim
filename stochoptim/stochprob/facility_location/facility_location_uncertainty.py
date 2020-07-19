# -*- coding: utf-8 -*-

import numpy as np

from ..uncertainty_basis import UncertaintyBasis

class ClientsPresence(UncertaintyBasis):
    
    def __init__(self, n_scenarios, n_client_locations, p=None, lb=None, ub=None):
        self.n_client_locations = n_client_locations
        self.n_scenarios = n_scenarios
        
        self.p = p
        self.lb = lb
        self.ub = ub
        self._scenarios = None
        
        if self.p is None and self.lb is None and self.ub is None:
            raise TypeError("Provide p or (lb, ub)")
        elif self.p is not None and self.lb is not None and self.ub is not None:
            raise TypeError("Provide p or (lb, ub) but not both")
        
        self.randomize()
        UncertaintyBasis.__init__(self, 
                                  name=f"Presence uncertainty",
                                  n_scenarios=self.n_scenarios,
                                  features={'h': self.n_client_locations})
        
    def randomize(self):
        self._scenarios = np.array([self.bernoulli() for _ in range(self.n_scenarios)])
        
    def get_scenario(self, scen_index):
        assert isinstance(scen_index, int), f"`scen_index should be of type int, not {type(scen_index)}."
        return {'h': self._scenarios[scen_index]}
    
    def bernoulli(self):
        if self.p is not None:
            return np.random.binomial(n=1, p=self.p, size=self.n_client_locations)
        else:
            p = np.random.uniform(self.lb, self.ub)
            return np.random.binomial(n=1, p=p, size=self.n_client_locations)