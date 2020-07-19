# -*- coding: utf-8 -*-

import numpy as np
from itertools import product

from .facility_location_solution import FacilityLocationSolution
from ..stochastic_problem_basis import StochasticProblemBasis


class FacilityLocationProblem(StochasticProblemBasis):
    """ Two-Stage Facility Location Problem.
    
    Arguments:
    ----------
    param: dict with keys and values:
        "pos_client":               2d-array, shape (n_client_locations, 2)
        "pos_facility":             2d-array, shape (n_facility_locations, 2)
        "opening_cost":             1d-array, shape (n_facility_locations,)
        "facility_capacity":        1d-array, shape (n_facility_locations,)
        "penalty":                  1d-array, shape (n_facility_locations,)
        "shipping_cost":            (optional) 2d-array, shape (n_client_locations, n_facility_locations)
        "resources":                (optional) 2d-array, shape (n_client_locations, n_facility_locations)
        "max_facilities":           int
        "min_facilities_in_zone":   1d-array, shape (n_zones,)
        "facility_in_zone":         1d-array, shape (n_facility_locations,) (of int values in [0, n_zones-1])
        
    If the keys "shipping_cost" and/or "resources" are not provided, they are computed 
    from the distances between facilities and clients (see methods: resources() and 
    shipping_cost()).
    """
        
    def __init__(self, param):                
        self.param = param
        self.n_facility_locations = self.param['pos_facility'].shape[0]
        self.n_client_locations = self.param['pos_client'].shape[0]
        self.n_zones = self.param['min_facilities_in_zone'].shape[0]
                
        # sets
        self.client_locations = range(self.n_client_locations)
        self.facility_locations = range(self.n_facility_locations)
        self.zones = range(self.n_zones)
        
        # compute the distances between clients and facilities
        # this distance is the resource consumed by client c served by facility f
        self.d = 30 * np.linalg.norm(self.pos_client()[:, np.newaxis, :] \
                                     - self.pos_facility()[np.newaxis, :, :], axis=2)
        
        # stochastic problem information
        StochasticProblemBasis.__init__(self, 
                                        name='Facility Location Problem',
                                        n_stages=2,
                                        objective_sense='min',
                                        is_obj_random=False,
                                        is_mip=True,
                                        solution_class=FacilityLocationSolution)

    # --- Problem definition ---
    def decision_variables_definition(self, stage):
        """For each decision variables of type {y_{t,i} \in A : i \in I} where I is an indexing set 
        (a list of tuples), y_{t,i} is of type A (with A = binary (B), or integer (I), or continuous (C)), 
        and lb <= y_{t,i} <= ub, this function generates a 5-tuple of the form: ('y', I, lb, ub, A) 
        under the conditon: stage == t. (lb and ub can be two lists provided they have the same size as I)"""
        if stage == 0:
            yield 'x', self.facility_locations, 0, 1, 'B'
        elif stage == 1:
            yield 'y', list(product(self.client_locations, self.facility_locations)), 0, 1, 'B'
            yield 'z', self.facility_locations, 0, None, 'C'

    def random_variables_definition(self, stage):
        """For each random variable of type {xi_{stage,i} : i \in I} where I is an indexing set (an iterable), 
        this method generates a 2-tuple of the form ('xi', I)"""
        if stage == 1:
            yield 'h', self.client_locations
    
    def objective(self):
        """Returns the problem's objective function"""
        return self.dot(self.x(), self.opening_cost()) \
                + self.dot(self.y(), self.shipping_cost().flatten()) \
                + self.dot(self.z(), self.penalty()) 
    
    def deterministic_linear_constraints(self, stage):
        """Generates all the problem's constraints that do not depend on the random parameters"""
        if stage == 0:
            yield self.max_facility_constraint()
            yield self.min_facility_in_zone_constraint()
        if stage == 1:
            yield self.resource_consumption_constraint()

    def random_linear_constraints(self, stage):
        """Generates all the problem's constraints that depend on the random parameters"""
        if stage == 1: 
            yield self.serving_constraint()
                
    # ---- Decision variables ----    
    def x(self, f=None):
        """x_{f} = 1 if facility f is open, 0 otherwise"""
        return self.get_dvar(0, 'x', f)
    
    def y(self, c=None, f=None):
        """y_{c,f} = 1 if facility f ships to client c, 0 otherwise"""
        if c is None and f is None: 
            return self.get_dvar(1, 'y')
        elif c is None and f is not None: 
            return self.get_dvar(1, 'y').reshape(-1, self.n_facility_locations)[:, f]
        elif c is not None and f is None: 
            return self.get_dvar(1, 'y').reshape(self.n_client_locations, -1)[c, :]
        else: 
            return self.get_dvar(1, 'y', (c, f))
         
    def z(self, f=None):
        """z_{f} = penalty for overloading facility f"""
        return self.get_dvar(1, 'z', f)
      
    # ---- Random variables ----       
    def h(self, c=None):
        """h_{c} = 1 if client c is present in scenario s, 0 otherwise"""
        return self.get_rvar(1, 'h', c)
    
    # --- Precomputation ---
    def precompute_decision_variables(self, stage):
        pass
    
    def precompute_parameters(self, stage):
        pass
    
    # --- Sanity check ---
    def sanity_check(self, stage):
        pass
    
    # --- Parameters ---
    def pos_client(self, c=None):
        """Position of client c"""
        pos = np.array(self.param["pos_client"])
        return pos if c is None else pos[c]
        
    def pos_facility(self, f=None):
        """Position of facility f"""
        pos = np.array(self.param["pos_facility"])
        return pos if f is None else pos[f]
        
    def opening_cost(self, f=None):
        """Cost of locating a facility f"""
        cost = np.array(self.param["opening_cost"])
        return cost if f is None else cost[f]
    
    def shipping_cost(self, c=None, f=None):
        if self.param.get('shipping_cost') is None:
            return self.distance(c, f) - 1
        else:
            if c is None and f is None: return self.param.get('shipping_cost')
            elif c is None and f is not None: return self.param.get('shipping_cost')[:, f]
            elif c is not None and f is None: return self.param.get('shipping_cost')[c, :]
            else: self.param.get('shipping_cost')[c, f]
    
    def capacity(self, f=None):
        """Facility capacity"""
        capacity = np.array(self.param["facility_capacity"])
        return capacity if f is None else capacity[f]
    
    def resources(self, c=None, f=None):
        """Resources consumed by facility f shipping to client c"""
        if self.param.get('resources') is None:
            return self.distance(c, f)
        else:
            if c is None and f is None: return self.param.get('resources')
            elif c is None and f is not None: return self.param.get('resources')[:, f]
            elif c is not None and f is None: return self.param.get('resources')[c, :]
            else: self.param.get('resources')[c, f]
    
    def max_facilities(self):
        """Upper bound on the total number of facilities that can be located"""
        return self.param["max_facilities"]
    
    def min_facilities(self, z):
        """Minimum number of facilities to be located in zone z"""
        return self.param["min_facilities_in_zone"][z]
    
    def penalty(self, f=None):
        """Penality for exceeding facility capacity"""
        penalty = np.array(self.param["penalty"])
        return penalty if f is None else penalty[f]
    
    def is_in_zone(self, z):
        """Return a 1d-array with True at pos k if facility k is in zone z"""
        return np.array(self.param["facility_in_zone"]) == z
            
    def distance(self, c=None, f=None):
        """Distance between client location c and facility location f"""
        if c is None and f is None: return self.d
        elif c is None and f is not None: return self.d[:, f]
        elif c is not None and f is None: return self.d[c, :]
        else: self.d[c, f]
        
    # --- constraints ---
    def max_facility_constraint(self):
        yield self.sum(self.x()) <= self.max_facilities(), "max_facilities"
        
    def min_facility_in_zone_constraint(self):
        for zone in self.zones:
            yield self.sum(self.x()[self.is_in_zone(zone)]) >= self.min_facilities(zone), f"min_facilities_zone_{zone}"
             
    def resource_consumption_constraint(self):
        self.big_M = 10**10
        for f in self.facility_locations:
            yield self.dot(self.y(f=f), self.resources(f=f)) - self.z(f) <= self.capacity(f) * self.x(f), \
                    f"resource_facility_{f}"
            yield self.z(f) <= self.big_M * self.x(f), f"big_M_deviation_facility_{f}"
            #for c in self.client_locations:
                #yield self.y(c, f) <= self.x(f)
                
    def serving_constraint(self):
        for c in self.client_locations:
            yield self.sum(self.y(c=c)) == self.h(c), f"assignment_client_{c}"
            
    # --- Load, save ---
    @classmethod
    def from_file(cls, path, extension='txt'):
        if extension == 'txt':
            with open(f'{path}.txt', "r") as f:
                file_str = f.read()
                file_str = file_str.replace('array', 'np.array')
                file_str = file_str.replace('nan', 'np.nan')
                param = eval(file_str)
        elif extension == 'pickle':
            import pickle
            with open(f'{path}.pickle', "rb") as f:
                param = pickle.load(f)
        return cls(param)
    
    def to_file(self, path, extension='txt'):
        if extension == 'txt':
            np.set_printoptions(threshold=np.inf) # no limit on the number of elements printed in an array
            with open(f'{path}.txt', "w") as f:
                f.write(repr(self.param).replace("),", "),\n"))
        elif extension == 'pickle':
            import pickle
            with open(f'{path}.pickle', "wb") as f:
                pickle.dump(self.param, f)
        else:
            TypeError(f"Extension should be 'pickle' or 'txt', not {extension}.")

    # --- Representation ---
    def __repr__(self):
        string_problem = StochasticProblemBasis.__repr__(self)
        string = ("Network: \n"
                  f"  {self.n_facility_locations} facility locations\n"
                  f"  {self.n_client_locations} client locations\n"
                  f"  {self.n_zones} zones")
        return string_problem + "\n" + string
    
    
from_file = FacilityLocationProblem.from_file


def generate_random_parameters(n_facility_locations, n_client_locations, n_zones, seed=None):
    """Generate randomly a set of deterministic parameters of the facility location problem"""
    if seed is not None:
        np.random.seed(seed)
    return {"pos_client": np.random.uniform(0, 1, size=(n_client_locations, 2)),
            "pos_facility": np.random.uniform(0, 1, size=(n_facility_locations, 2)),
            "opening_cost": np.random.randint(40, 81, size=n_facility_locations),
            "facility_capacity": np.random.randint(30, 60, size=n_facility_locations),
            "max_facilities": n_facility_locations,
            "min_facilities_in_zone": np.array([1] * n_zones),
            "facility_in_zone": np.random.choice(range(n_zones), size=n_facility_locations),
            "penalty": 1000 * np.ones(n_facility_locations)}
        