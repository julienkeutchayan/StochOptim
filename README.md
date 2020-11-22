## StochOptim Description
StochOptim is a Stochastic Optimization package that provides tools for formulating and solving two-stage and 
multi-stage problems.

Three main reasons why you would like to use it:
* If you want an easy way to formulate your stochastic problem, represent its uncertainty via a scenario tree, 
and generate/solve the equivalent deterministic program.
* If you want to build non-standard optimized scenario trees from given probability distributions (**scenario generation**)
* If you want to build non-standard optimized scenario trees directly from historical data (**scenario clustering**)

By *non-standard* we mean scenario trees with non-homogeneous branching structures (which may depend on the stage and 
the scenarios). \
By *optimized* we mean scenario trees with a branching structure that minimizes some criterion measuring its 
suitability to the problem.

The concept of building scenario trees suitable to the problem is gaining popularity over the past years. 
Suitability is measured by the ability of the equivalent deterministic program to provide optimal decisions that are as close as possible from those 
of the original problem. It falls under the umbrella of **problem-driven methods**, which are opposed to 
**distribution-driven methods** where suitability is defined with respect to the uncertainty and not the problem. 

The tools implemented in this package have been developed in the papers: 
* J. Keutchayan, D. Munder, M. Gendreau (2019) [On the Scenario-Tree Optimal-Value Error for Stochastic Programming Problems](https://pubsonline.informs.org/doi/10.1287/moor.2019.1043) (*Mathematics of Operations Research*)
* J. Keutchayan, D. Munder, M. Gendreau, F. Bastin (2018) [The Figure of Demerit: A Quality Measure for the Discretization of Probability Distributions in Multistage Stochastic Optimization](https://www.researchgate.net/profile/Julien_Keutchayan/publication/322644958_The_Figure_of_Demerit_A_Quality_Measure_for_the_Discretization_of_Probability_Distributions_in_Multistage_Stochastic_Optimization/links/5bdcddd14585150b2b9a4b82/The-Figure-of-Demerit-A-Quality-Measure-for-the-Discretization-of-Probability-Distributions-in-Multistage-Stochastic-Optimization.pdf) (*preprint*)
* J. Keutchayan, M. Gendreau, F. Bastin (2018) [Problem-Driven Scenario Trees in Multistage Stochastic Optimization: An Illustration in Option Pricing](https://www.researchgate.net/profile/Julien_Keutchayan/publication/328703934_Problem-Driven_Scenario_Trees_in_Multistage_Stochastic_Optimization_An_Illustration_in_Option_Pricing/links/5bdcde684585150b2b9a4b89/Problem-Driven-Scenario-Trees-in-Multistage-Stochastic-Optimization-An-Illustration-in-Option-Pricing.pdf) (*preprint*)

for the **scenario generation** part, and in:
* J. Keutchayan, J. Ortmann, W. Rei (2020) [Problem-Driven Scenario Clustering in Stochastic Optimization]() (soon available)

for the **scenario clustering** part. 

## Installation

### PyPI

`pip install stochoptim`

### Dependencies

* `docplex` (version >= 2.9.133)

## Basic Usage

Let's consider the following multistage stochastic optimization problem:

![https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Images/multistage_problem.PNG](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Images/multistage_problem.PNG) 

To solve it:

* First, we write the class for our problem and let it inherit from `StochasticProblemBasis`. Our class must define the decision variables, random parameters, objective function and constraints:
```javascript
from stochoptim.stochprob.stochastic_problem_basis import StochasticProblemBasis

class MyMultistageStochasticProblem(StochasticProblemBasis):
    
    def __init__(self, T, A, B, c, d0):
        
        self.T = T         # last stage
        self.A = A         # A = {t: A[t] for t=0,...,T} where A[t] is a 2d-array
        self.B = B         # B = {t: B[t] for t=1,...,T} where B[t] is a 2d-array
        self.c = c         # c = {t: c[t] for t=0,...,T} where c[t] is a 1d-array
        self.d0 = d0       # 1d-array
        self.n = {t: A[t].shape[1] for t in range(self.T + 1)} # number of variables per stage
        self.m = {t: A[t].shape[0] for t in range(self.T + 1)} # number of constraints per stage
      
        StochasticProblemBasis.__init__(self, 
                                        name='Simple Example of Multistage Problem',
                                        n_stages=self.T + 1,        # number of stages 
                                        objective_sense='max',      # are we maximizing or minimizing?
                                        is_obj_random=False,        # does the objective function contain randomness?
                                        is_mip=False)               # does the problem include integer or binary variables?

    def decision_variables_definition(self, t):
        yield 'x', range(self.n[t]), 0, None, 'C'
        
    def random_variables_definition(self, t):
        if t >= 1:
            yield 'd', range(self.m[t])
    
    def objective(self):
        return self.sum([self.dot(self.c[t], self.get_dvar(t, 'x')) 
                         for t in range(self.T + 1)])                           # c[t].x[t] summed over t
    
    def deterministic_linear_constraints(self, t):
        if t == 0:
            yield iter(self.dot(self.A[t][i], self.get_dvar(t, 'x')) \
                       <= self.d0[i] for i in range(self.m[t]))                 # A[0].x[0] <= d[0]
    
    def random_linear_constraints(self, t):
        if t >= 1:
            yield iter(self.dot(self.A[t][i], self.get_dvar(t, 'x')) \
                       + self.dot(self.B[t][i], self.get_dvar(t-1, 'x')) \
                       <= self.get_rvar(t, 'd')[i] for i in range(self.m[t]))    # A[t].x[t] + B[t].x[t-1] <= d[t]
```
We instantiate the problem with the parameters of interest; for example for T = 2, 2 variables and 3 constraints per stage:
```javascript
import numpy as np

T = 2

A = {0: np.array([[4, 1], [6, 4], [5, 9]]), 
     1: np.array([[2, 6], [7, 9], [8, 9]]), 
     2: np.array([[6, 3], [9, 2], [7, 5]])}

B = {1: np.array([[9, 5], [7, 5], [2, 7]]), 
     2: np.array([[9, 9], [2, 7], [1, 5]])}

c = {0: np.array([4, 3]), 
     1: np.array([-5,  4]), 
     2: np.array([-3,  1])}

d0 = np.array([8, 1, 1])

my_stochastic_problem = MyMultistageStochasticProblem(T, A, B, c, d0)
```
* Then, we build the uncertainty of our problem using a scenario tree: (for the sake of simplicity we consider here a standard (non-optimized) tree with 2 branches per stage)

```javascript
from stochoptim.scengen.scenario_tree import ScenarioTree
from stochoptim.scengen.scenario_process import ScenarioProcess

def scenario_fct(stage, epsilon, scenario_path):
    if stage >= 1:
        return {'d': np.random.uniform(1, 10, size=(3,))}
      
scenario_process = ScenarioProcess(scenario_fct, None)      # scenario generator
my_scenario_tree = ScenarioTree.from_bushiness([2,2])       # naked tree structure with 2 branches per stage
my_scenario_tree.fill(scenario_process)                     # tree structure filled with scenarios
```
* Finally, we call the `.solve()` method of the problem on the scenario tree:
```javascript
solution = my_stochastic_problem.solve(my_scenario_tree)
```
* We can now plot the solutions:
```javascript
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=5, figsize=(15, 5))
solution.scenario_tree.plot_scenarios('d', ax=axes[0])
axes[1].axis('off')
solution.scenario_tree.plot(lambda node: np.round(node.data['decision'].get('x'), 3), ax=axes[2])
axes[3].axis('off')
solution.scenario_tree.plot(lambda node: np.round(node.data['v'], 3), ax=axes[4])
```

![https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Images/scenario_tree_solution.png](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Images/scenario_tree_solution.png) 

**Left**: values and probabilities of the 3-dimensional random parameter 'd'. \
**Middle**: optimal decisions 'x'. \
**Right**: optimal objective values (at the root node and conditionally on the scenarios)

If you want to try other scenario trees, see notebook [Basic Example](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/0.%20Basic%20Example.ipynb)

## Tutorials and Examples

**Case studies:**
* [Two-Stage Facility Location Problem](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Two-Stage%20Facility%20Location%20Problem.ipynb)
* [Two-Stage Network Design Problem](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Two-Stage%20Network%20Design%20Problem.ipynb)
* [Multistage Facility Location Problem](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Multistage%20Facility%20Location%20Problem.ipynb)

**Scenario generation:**
* [Tree Structures](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/1.%20Tree%20Structures.ipynb)
* [Scenario Trees](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/2.%20Scenario%20Trees.ipynb)
* [Scenario Trees with Optimized Scenarios](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/3.%20Scenario%20trees%20with%20optimized%20scenarios.ipynb)
* [Scenario Trees with Optimized Scenarios and Structure (method #1)](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/4.%20Scenario%20trees%20with%20optimized%20scenarios%20and%20structure%20(method%20%231).ipynb)
* [Scenario Trees with Optimized Scenarios and Structure (method #2)](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/5.%20Scenario%20trees%20with%20optimized%20scenarios%20and%20structure%20(method%20%232).ipynb)

**Scenario clustering:**
* [Two-Stage Scenario Clustering](https://github.com/julienkeutchayan/StochOptim/blob/master/notebooks/Two-Stage%20Scenario%20Clustering.ipynb)