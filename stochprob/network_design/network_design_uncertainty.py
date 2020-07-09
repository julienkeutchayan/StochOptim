# -*- coding: utf-8 -*-

import numpy as np
from stochoptim.stochprob.uncertainty_basis import UncertaintyBasis

class Demands(UncertaintyBasis):
    
    def __init__(self, n_scenarios, n_commodities, distribution, **kwargs):
        self.n_scenarios = n_scenarios
        self.n_commodities = n_commodities
        self.distribution = distribution
        
        self._scenarios = None
        
        if self.distribution == "lognormal":
            self.mean = kwargs.get("mean", 0)
            self.var = kwargs.get("var", 1)
            self.corr = kwargs.get("corr", 0)
            self.label = "logN({},{})-corr{}".format(self.mean, self.var, self.corr)
            
        elif self.distribution == "uniform":
            self.lb = kwargs.get("lb", 0)
            self.ub = kwargs.get("ub", 1)
            self.corr = kwargs.get("corr", 0)
            self.label = "U[{},{}]-corr{}".format(self.lb, self.ub, self.corr)
            if self.lb >= self.ub:
                raise ValueError("Wrong lower and upper bound")
          
        self.randomize()
        
        UncertaintyBasis.__init__(self, 
                                  name=f"Commodity demand uncertainty: {self.label}", 
                                  n_scenarios=self.n_scenarios,
                                  features={'d': self.n_commodities})
        
    def randomize(self):
        if self.distribution == "lognormal":
            self._scenarios = np.array([self.lognormal() for i in range(self.n_scenarios)])
        elif self.distribution == "uniform":
            self._scenarios = np.array([self.uniform() for i in range(self.n_scenarios)])
    
    def get_scenario(self, scen_index=None):
        assert isinstance(scen_index, int), f"`scen_index should be of type int, not {type(scen_index)}."
        return {'d': self._scenarios[scen_index]}
    
    def lognormal(self):
        mu = [self.mean] * self.n_commodities
        sigma = [[self.corr if i!=j else self.var 
                      for i in range(self.n_commodities)] 
                        for j in range(self.n_commodities)]
        return np.exp(np.random.multivariate_normal(mu, sigma)).astype('int64')
    
    def uniform(self):
        return np.random.randint(self.lb, self.ub+1, size=self.n_commodities)
