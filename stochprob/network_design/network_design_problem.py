# -*- coding: utf-8 -*-

import numpy as np
from itertools import product

from stochoptim.stochprob.network_design.network_design_solution import NetworkDesignSolution
from stochoptim.stochprob.stochastic_problem_basis import StochasticProblemBasis


class NetworkDesign(StochasticProblemBasis):
      
    def __init__(self, param):
        """
        param = {'n_origins': int, 
                 'n_destinations': int, 
                 'n_intermediates': int,
                 'opening_cost': 2d-array,
                 'shipping_cost': 2d-array,
                 'capacity': 2d-array}
        """
        self.param = param        
        self.n_origins = self.param['n_origins']
        self.n_destinations = self.param['n_destinations']
        self.n_intermediates = self.param['n_intermediates']
        
        # vertices
        self.vertices = range(self.n_origins + self.n_destinations + self.n_intermediates) # all vertices
        self.origins = self.vertices[:self.n_origins] # origins
        self.destinations = self.vertices[-self.n_destinations:] # destinations
        
        # arcs
        self.OD_arcs = list(product(self.origins, self.destinations)) # all arcs linking origins to destinations
        self.transport_arcs = [arc for arc in product(self.vertices, self.vertices) 
                                if arc[0] != arc[1] and arc not in self.OD_arcs] # all arcs but OD pairs
        self.all_arcs = self.OD_arcs + self.transport_arcs


        # stochastic problem information
        StochasticProblemBasis.__init__(self, 
                                        name='Network Design Problem',
                                        n_stages=2,
                                        objective_sense='min',
                                        is_obj_random=False,
                                        is_mip=True,
                                        solution_class=NetworkDesignSolution)
                      
    # ---- random parameters ----       
    def d(self, vertex, OD):
        """d_{i}^{k,s}: random demand of commodity OD at vertex i in scenario s"""
        demand = self.get_rvar(1, 'd', OD)
        if vertex == OD[0]: # if the vertex is the origin of the commodity
            return demand
        elif vertex == OD[1]: # if the vertex is the destination of the commodity
            return -demand
        else: # if the vertex is an intermediate step
            return 0
        
    # ---- 1st-stage decision variables ----    
    def y(self, arc=None):
        """y_{i,j} = 1 if arc (i,j) is open, 0 otherwise"""
        return self.get_dvar(0, 'y', arc)
    
    # ---- 2nd-stage decision variables ----    
    def z(self, arc=None, OD=None):
        """z_{i,j}^{k,s}: quantity of commodity OD transported on arc (i,j) in scenario s"""
        if arc is None and OD is None:
            return self.get_dvar(1, 'z')
        else:
            return self.get_dvar(1, 'z', (arc, OD))
 
    def decision_variables_definition(self, stage):
        """For each decision variables of type {y_{t,i} \in A : i \in I} where I is an indexing set 
        (a list of tuples), y_{t,i} is of type A (with A = binary (B), or integer (I), or continuous (C)), 
        and lb <= y_{t,i} <= ub, this function generates a 5-tuple of the form: ('y', I, lb, ub, A) 
        under the conditon: stage == t. (lb and ub can be two lists provided they have the same size as I)"""
        if stage == 0:
            yield 'y', self.transport_arcs, 0, 1, 'B'
        elif stage == 1:
            yield 'z', list(product(self.all_arcs, self.OD_arcs)), 0, None, 'C'

    def random_variables_definition(self, stage):
        """For each random variable of type {xi_{stage,i} : i \in I} where I is an indexing set (an iterable), 
        this method generates a 2-tuple of the form ('xi', I)"""
        if stage == 1:
            yield 'd', self.OD_arcs

    def objective(self):
        """Returns the problem's objective function"""        
        first_stage = self.dot(self.y(), [self.opening_cost(arc) for arc in self.transport_arcs])
        second_stage = self.dot(self.z(), [self.shipping_cost(arc) for (arc, OD) in product(self.all_arcs, self.OD_arcs)])
        return first_stage + second_stage
    
    def deterministic_linear_constraints(self, stage):
        """Generates all the problem's constraints that do not depend on the random parameters"""
        if stage == 1:
            yield self.transport_constraints()
                 
    def random_linear_constraints(self, stage):
        """Generates all the problem's constraints that depend on the random parameters"""
        if stage == 1: 
            yield self.demand_constraints()

    def opening_cost(self, arc):
        return self.param['opening_cost'][arc]
    
    def shipping_cost(self, arc):
        return self.param['shipping_cost'][arc]
    
    def capacity(self, arc):
        return self.param['capacity'][arc]

    def transport_constraints(self):
        # sum commodities on each each <= capacity (if arc is open)
        for arc in self.transport_arcs:
            yield self.sum([self.z(arc, OD) for OD in self.OD_arcs]) \
                                <= self.capacity(arc) * self.y(arc), f"open_{arc}"
                                    
    def demand_constraints(self):
        # [what goes in] - [what goes out] = [demand of the node]
        for vertex, OD in product(self.vertices, self.OD_arcs):
            out_sum = self.sum([self.z(arc, OD) for arc in self.all_arcs if arc[0] == vertex])
            in_sum = self.sum([self.z(arc, OD) for arc in self.all_arcs if arc[1] == vertex])
            yield out_sum - in_sum == self.d(vertex, OD), f"demand_{vertex}_{OD}"
                
    def precompute_decision_variables(self, stage):
        pass
    
    def precompute_parameters(self, stage):
        pass
    
    def sanity_check(self, stage):
        pass
    
    # --- Load and save ---
    @classmethod
    def from_file(cls, path, extension='txt'):
        if extension == 'txt':
            with open(f'{path}.{extension}', "r") as f:
                file_str = f.read()
                file_str = file_str.replace('array', 'np.array')
                file_str = file_str.replace('nan', 'np.nan')
                param = eval(file_str)
        elif extension == 'pickle':
            import pickle
            with open(f'{path}.{extension}', "rb") as f:
                param = pickle.load(f)
        return cls(param)
    
    def to_file(self, path, extension='txt'):
        if extension == 'txt':
            np.set_printoptions(threshold=np.inf) # no limit on the number of elements printed in an array
            with open(f'{path}.{extension}', "w") as f:
                f.write(repr(self.param).replace("),", "),\n"))
        elif extension == 'pickle':
            import pickle
            with open(f'{path}.{extension}', "wb") as f:
                pickle.dump(self.param, f)
        else:
            TypeError(f"Extension should be 'pickle' or 'txt', not {extension}.")
                
    # --- Representation ---
    def __repr__(self):
        string_problem = StochasticProblemBasis.__repr__(self)
        string = ("Network: \n"
                  f"  {len(self.vertices)} nodes\n"      
                  f"  {len(self.transport_arcs)} transportation arcs\n"
                  f"  {len(self.OD_arcs)} commodities")
        return string_problem + "\n" + string
    
from_file = NetworkDesign.from_file

def generate_random_parameters(n_origins, n_destinations, n_intermediates, seed=None):
    """Generate randomly a set of deterministic parameters of the network design problem"""
    if seed is not None:
        np.random.seed(seed)
    n_vertices = n_origins + n_destinations + n_intermediates
    # cost of opening an arc
    opening_cost = np.random.randint(3, 11, size=(n_vertices, n_vertices))
    # cost of shipping one unit on an arc
    shipping_cost = np.random.randint(5, 11, size=(n_vertices, n_vertices))
    shipping_cost[:n_origins, -n_destinations:] = 1000
    # capacity of an arc
    capacity = np.random.randint(10, 41, size=(n_vertices, n_vertices))
    capacity[:n_origins, -n_destinations:] = 10**5
    return {'n_origins': n_origins, 
            'n_destinations': n_destinations, 
            'n_intermediates': n_intermediates,
            'opening_cost': opening_cost,
            'shipping_cost': shipping_cost,
            'capacity': capacity}
    