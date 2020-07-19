# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from itertools import product
import numpy as np

from ..stochastic_solution_basis import StochasticSolutionBasis


class FacilityLocationSolution(StochasticSolutionBasis):
    
    def __init__(self, 
                 stochastic_problem,
                 scenario_tree):
        
        StochasticSolutionBasis.__init__(self, 
                                         stochastic_problem,
                                         scenario_tree)

    # --- facilities activity ---
    def is_active(self, facility=None):
        self.set_path_info()
        return self.stochastic_problem.x(facility)
        
    def n_active_facilities(self):
        return np.sum(self.is_active())
    
    # --- facility penalty ---
    def get_penalty_value(self, scen_index, facility=None):
        """Returns the penality receives by a facility in a given scenario"""
        self.set_path_info(scen_index)
        return self.stochastic_problem.z(facility)
    
    def is_penalized(self, scen_index=None, facility=None):
        """Returns True if the facility is penalized in at least one of the scenario, False otherwise"""
        if scen_index is not None:
            return self.get_penalty_value(scen_index, facility) > 0
        else:
            return (np.sum([self.get_penalty_value(scen_index, facility) 
                            for scen_index in range(self.n_scenarios)], axis=0) > 0).astype('bool')
        
    # --- Clients presence ---
    def is_present(self, scen_index, client=None):
        """Returns True if a client location is occupied in a given scenario"""
        self.set_path_info(scen_index)
        return self.stochastic_problem.h(client)
    
    # --- facility/Client assignement ---
    def is_assigned(self, scen_index, client=None, facility=None):
        self.set_path_info(scen_index)
        return self.stochastic_problem.y(client, facility)
    
    # --- Plots ---
    def _plot_facilities(self, scen_index, ax):
        pos_facilities = self.stochastic_problem.pos_facility()
        penalized_mask = self.is_penalized(scen_index)
        active_not_penalized_mask = np.multiply(self.is_active(), ~self.is_penalized(scen_index)).astype('bool')
        not_active_mask = ~(self.is_active().astype('bool'))
        # if facility location is active and penalized in at least one scenario
        if penalized_mask.sum() > 0:
            ax.scatter(*zip(*pos_facilities[penalized_mask]), marker="s", s=30, c="k")
        # if facility location is active but never penalized
        if active_not_penalized_mask.sum() > 0:
            ax.scatter(*zip(*pos_facilities[active_not_penalized_mask]), marker="s", s=30, c="C1")    
        # if facility location is not active
        if not_active_mask.sum() > 0:
            ax.scatter(*zip(*pos_facilities[not_active_mask]), marker="s", s=30, alpha=0.8, 
                       facecolors='none', edgecolors='C1')   
                
    def _plot_clients(self, scen_index, ax):
        is_present_pos = self.stochastic_problem.pos_client()[self.is_present(scen_index).astype('bool')]
        is_not_present_pos = self.stochastic_problem.pos_client()[~(self.is_present(scen_index).astype('bool'))]
        # client is present at location
        if is_present_pos.sum() > 0: 
            ax.scatter(*zip(*is_present_pos), marker="o", c="C0", s=20)
        # client not present at location
        if is_not_present_pos.sum() > 0:
            ax.scatter(*zip(*is_not_present_pos), marker="o", facecolors='none', edgecolors='C0', s=20)
             
    def _plot_clients_facilities_connection(self, scen_index, ax):
        for c, s in product(self.stochastic_problem.client_locations, self.stochastic_problem.facility_locations):
            if self.is_assigned(scen_index, c, s):
                ax.plot(*zip(self.stochastic_problem.pos_client(c), self.stochastic_problem.pos_facility(s)), 
                        color="C{}".format(s % 10))
                        
    def plot_network(self, scen_index=None, show_clients=True, show_facilities=True, show_title=True, 
                     figsize=(5,5), path=None, extension=".pdf", **kwargs):
        
        fig, ax = plt.subplots(figsize=figsize)
        # plot facilities locations
        if show_facilities:
            self._plot_facilities(scen_index, ax)
            
        if scen_index is None:
            if show_clients:
                ax.scatter(*zip(*self.stochastic_problem.pos_client()), marker="o", 
                           facecolors='none', edgecolors='C0', s=20)
        else:
            self._plot_clients(scen_index, ax)
            if show_facilities:
                # plot links between facilities and clients assigned to them
                self._plot_clients_facilities_connection(scen_index, ax)
                        
        if show_title:   
            if scen_index is None:
                obj_to_show = self.objective_value
                ax.set_title(f"$v^*: {obj_to_show:.2f}$")
            else:
                obj_to_show = self.objective_value_at_leaves[scen_index]
                ax.set_title(f"$v^*: {obj_to_show:.2f}$")         
             
        self._set_axes(ax, **kwargs)
        self._save_fig(path, extension)
        plt.show()  
            
    def _save_fig(self, path, extension):
        if path:
            plt.savefig(path + extension)
            print(f"Figure saved at {path + extension}")
                
    def _set_axes(self, ax, xlim=None, ylim=None, show_xaxis=False, show_yaxis=False, show_frame=True):
        # set x and y limits
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)   
        # show frame and axes
        ax.get_xaxis().set_visible(show_xaxis)
        ax.get_yaxis().set_visible(show_yaxis)
        if not show_frame and not show_xaxis and not show_yaxis:
            ax.axis('off')
    
    @classmethod           
    def from_file(cls, 
                  path_tree, 
                  path_problem, 
                  extension_tree='pickle', 
                  extension_prob='txt'):
        from stochoptim.stochprob.facility_location.facility_location_problem import FacilityLocationProblem
        return StochasticSolutionBasis.from_file(FacilityLocationProblem, 
                                                 path_tree, path_problem, 
                                                 extension_tree, extension_prob)
        
from_file = FacilityLocationSolution.from_file