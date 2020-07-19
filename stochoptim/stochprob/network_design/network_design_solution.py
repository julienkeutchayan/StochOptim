# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..stochastic_solution_basis import StochasticSolutionBasis


class NetworkDesignSolution(StochasticSolutionBasis):
    
    def __init__(self, 
                 stochastic_problem,
                 scenario_tree):
        
        StochasticSolutionBasis.__init__(self, 
                                         stochastic_problem,
                                         scenario_tree)
         
    def is_open(self, arc):
        """Returns 1 if the arc has been open at stage 0, otherwise 0"""
        self.set_path_info()
        return self.stochastic_problem.y(arc)
    
    def x0_dict(self, forced_to_integer=False):
        return {arc: self.is_open(arc) for arc in self.stochastic_problem.transport_arcs}

    def n_open_arcs(self):
        return np.sum(self.x0['y'])
    
    def is_penalized(self, od):
        penality = 0
        for scen_index in range(self.n_scenarios):
            self.set_path_info(scen_index)
            penality += self.stochastic_problem.z(od, od)
        return True if penality > 0 else False
         
    def get_transported_amount(self, arc, scen_index):
        """Returns the amount of all commodities transported on an arc in a given scenario"""
        self.set_path_info(scen_index)
        return sum([self.stochastic_problem.z(arc, od) for od in self.stochastic_problem.od_arcs])
    
    def is_used(self, arc, scen_index):
        """True if the arc has some commodities transported on it"""
        return self.get_transported_amount(arc, scen_index) > 0
    
    def n_used_arc(self, scen_index):
        return sum([self.is_used(arc, scen_index) for arc in self.stochastic_problem.transport_arcs])
    
    def get_penalty_value(self, od, scen_index):
        self.set_path_info(scen_index)
        return self.stochastic_problem.z(od, od)
                
    def get_penalty_dict(self):
        dict_penalty_value = {od: 0 for od in self.stochastic_problem.od_arcs} # {od pair: amount transported on od arc}        
        for od in self.stochastic_problem.od_arcs:
            for scen_index in range(self.n_scenarios):
                self.set_path_info(scen_index)
                dict_penalty_value[od] += self.stochastic_problem.z(od, od) 
        return dict_penalty_value
    
    def _vertex_color(self, vertex):
        if vertex in self.stochastic_problem.origins: 
            return "C2"
        elif vertex in self.stochastic_problem.destinations: 
            return "C3"
        else: 
            return "C0"
        
    def _arc_color(self, arc):
        if arc[0] in self.stochastic_problem.origins: 
            return "C2"
        elif arc[1] in self.stochastic_problem.destinations:
            return "C3"
        elif arc[0] in self.stochastic_problem.destinations and arc[1] not in self.stochastic_problem.destinations:
            return "k"
        else:
            return "C0"
            
    def _arc_style(self, arc, scen_index=None):
        arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8" #for FancyArrowPatch
        if scen_index is None:
            if arc in self.stochastic_problem.transport_arcs:
                if self.is_open(arc): 
                    kw = dict(arrowstyle=arrowstyle, color=self._arc_color(arc), alpha=1)
                else: 
                    kw = dict(arrowstyle=arrowstyle, color="k", alpha = 0.05)
            elif arc in self.stochastic_problem.od_arcs:                
                if self.is_penalized(arc):
                    kw = dict(arrowstyle=arrowstyle, color="k", alpha = 1, linestyle = "--")
                else:
                    kw = dict(arrowstyle=arrowstyle, color="k", alpha = 0.05)    
            return kw
        else:
            if arc in self.stochastic_problem.transport_arcs:
                if self.is_used(arc, scen_index): # if open and something is transported on it
                    kw = dict(arrowstyle=arrowstyle, color=self._arc_color(arc), alpha=1)
                elif self.is_open(arc): # if open but nothing is transported on it
                    kw = dict(arrowstyle=arrowstyle, color=self._arc_color(arc), alpha=0.2)
                else: 
                    kw = dict(arrowstyle=arrowstyle, color="k", alpha = 0.05)
            elif arc in self.stochastic_problem.od_arcs:
                if self.get_penalty_value(arc, scen_index) > 0:
                    kw = dict(arrowstyle=arrowstyle, color="k", alpha = 1, linestyle = "--")
                else:
                    kw = dict(arrowstyle=arrowstyle, color="k", alpha = 0.05) 
            return kw
            
    def _vertex_label(self, vertex, scen_index):
        self.set_path_info(scen_index)
        label = ""
        if vertex in self.stochastic_problem.origins: 
            label = "{}: ".format(vertex)
            label += ",".join([str(self.stochastic_problem.d(vertex, (vertex, D)))
                                    for D in self.stochastic_problem.destinations])
        elif vertex in self.stochastic_problem.destinations: 
            label = "{}: ".format(vertex)
            label += ",".join([str(self.stochastic_problem.d(vertex, (O, vertex)))
                                    for O in self.stochastic_problem.origins])
        return label
                
    def plot_network(self, 
                     scen_index=None, 
                     show_title=True, 
                     randomize_txt_box=True,
                     figsize=(6,5), 
                     path=None, 
                     extension=".pdf"):
        fig, ax = plt.subplots(figsize=figsize)
        N = len(self.stochastic_problem.vertices)
        pos = {vertex: np.exp(2*np.pi*1j * k / N) for k, vertex in enumerate(self.stochastic_problem.vertices, 1)}
        pos = {key: (value.real, value.imag) for key, value in pos.items()}
        
        # Plot vertices
        for vertex in self.stochastic_problem.vertices:
            label = self._vertex_label(vertex, scen_index) if scen_index is not None else None
            ax.scatter(*pos[vertex], c=self._vertex_color(vertex), 
                       s=150, alpha=0.8, label=label)
            ax.text(*pos[vertex], vertex, 
                    horizontalalignment='center', verticalalignment='center')   
            
        # Plot arcs
        for arc in self.stochastic_problem.all_arcs:             
            a = patches.FancyArrowPatch(pos[arc[0]], pos[arc[1]], 
                                        connectionstyle="arc3,rad=.05", 
                                        **self._arc_style(arc, scen_index))
            fig.gca().add_patch(a)
            if scen_index is not None:
                self._print_transportated_value(arc, scen_index, randomize_txt_box, pos, ax)
                
        if scen_index is not None:
            ax.legend(bbox_to_anchor=(1, 1.15), loc=1, borderaxespad=1, prop={'size': 7})
        if show_title:
            if scen_index is None:
                obj_to_show = self.objective_value
                ax.set_title(f"$v^*: {obj_to_show:.2f}$")
            else:
                obj_to_show = self.objective_value_at_leaves[scen_index]
                ax.set_title(f"$v^*: {obj_to_show:.2f}$")
        ax.axis('off')
        self._save_fig(path, extension)
        fig.show()
           
    def _print_transportated_value(self, arc, scen_index, randomize_txt_box, pos, ax):  
        if not hasattr(self, 'alpha'):
            # coef of the convex combination giving the position of the box between the two vertices
            self.alpha = {arc: 0.1 for arc in self.stochastic_problem.all_arcs} 
            
        if self.is_used(arc, scen_index): # only arcs with commodity transported on it have a txt box
            # color of the box
            if arc[0] in self.stochastic_problem.origins:
                box_color = 'C2' 
            elif arc[1] in self.stochastic_problem.destinations:
                box_color = 'C3'
            else:
                box_color = 'C0'
            props = dict(boxstyle='round', facecolor=box_color, alpha=0.3)
            # location of the box
            if randomize_txt_box:
                n_adjacent_arcs = sum([self.is_used(a, scen_index) for a in self.stochastic_problem.all_arcs 
                                           if a[0] == arc[0] and a != arc])
                self.alpha[arc] = 0.1 if n_adjacent_arcs == 0 else np.random.uniform(0.05, 0.2)
            x, y = (1 - self.alpha[arc]) * np.array(pos[arc[0]]) + self.alpha[arc] * np.array(pos[arc[1]])
            # plot text in box
            txt_to_print = int(round(self.get_transported_amount(arc, scen_index), 0))
            ax.text(x, y, txt_to_print, fontsize=10, 
                    horizontalalignment='center', verticalalignment='center', bbox=props)
                    
    def _save_fig(self, path, extension):
        if path:
            plt.savefig(path + extension)
            print(f"Figure saved at {path + extension}") 
            
    @classmethod
    def from_file(cls, 
                  path_tree, 
                  path_problem, 
                  extension_tree='pickle', 
                  extension_prob='txt'):
        """ Load network design solution from two files: one for the problem and one for the scenario tree.
        Arguments:
        ----------
        path_tree: string
        extension_tree: {'pickle', 'txt'}, see: Node.from_file().
        path_problem: string
        extension_prob: {'pickle', 'txt'}, see: MinePlanningProblem.from_file()             
        """            
        from stochoptim.stochprob.network_design.network_design_problem import NetworkDesign
        from stochoptim.scengen.scenario_tree import ScenarioTree
        problem = NetworkDesign.from_file(path_problem, extension_prob)
        scenario_tree = ScenarioTree.from_file(path_tree, extension_tree)
        return cls(problem, scenario_tree)

from_file = NetworkDesignSolution.from_file