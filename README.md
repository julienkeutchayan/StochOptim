# StochOptim
StochOptim is a Stochastic Optimization package that provides tools for formulating and solving two-stage and multi-stage problems.

Three main reasons why you would like to use it:
* If you want an easy way to formulate your stochastic problem, represent its uncertainty via a scenario tree, and generate/solve its equivalent deterministic program (EDP).
* If you want to build non-standard optimized scenario trees from given probability distributions (**scenario generation**)
* If you want to build non-standard optimized scenario trees directly from historical data (**scenario reduction**)

By *non-standard* we mean scenario trees with non-homogeneous branching structures (which may depend on the stage and the scenarios). \
By *optimized* we mean scenario trees with a branching structure suitable to your stochastic problem.

The concept of building scenario trees suitable to the problem is gaining popularity over the past years. Suitability is measured by the ability of the EDP to 
provide optimal decisions that are as close as possible from those of the original problem. It falls under the umbrella of **problem-driven methods**, 
which are opposed to **distribution-driven methods** where suitability is defined with respect to the uncertainty and not the problem. 

The tools implemented in this package have been developed in the papers: 
* J. Keutchayan, D. Munder, M. Gendreau (2019) [On the Scenario-Tree Optimal-Value Error for Stochastic Programming Problems](https://pubsonline.informs.org/doi/10.1287/moor.2019.1043) (*Mathematics of Operations Research*)
* J. Keutchayan, D. Munder, M. Gendreau, F. Bastin (2018) [The Figure of Demerit: A Quality Measure for the Discretization of Probability Distributions in Multistage Stochastic Optimization](https://www.researchgate.net/profile/Julien_Keutchayan/publication/322644958_The_Figure_of_Demerit_A_Quality_Measure_for_the_Discretization_of_Probability_Distributions_in_Multistage_Stochastic_Optimization/links/5bdcddd14585150b2b9a4b82/The-Figure-of-Demerit-A-Quality-Measure-for-the-Discretization-of-Probability-Distributions-in-Multistage-Stochastic-Optimization.pdf)
* J. Keutchayan, M. Gendreau, F. Bastin (2018) [Problem-Driven Scenario Trees in Multistage Stochastic Optimization: An Illustration in Option Pricing](https://www.researchgate.net/profile/Julien_Keutchayan/publication/328703934_Problem-Driven_Scenario_Trees_in_Multistage_Stochastic_Optimization_An_Illustration_in_Option_Pricing/links/5bdcde684585150b2b9a4b89/Problem-Driven-Scenario-Trees-in-Multistage-Stochastic-Optimization-An-Illustration-in-Option-Pricing.pdf)

for the **scenario generation** part, and in:
* J. Keutchayan, J. Ortmann, W. Rei (2020) [Problem-Driven Scenario Clustering in Stochastic Optimization]() (soon available) \

for the **scenario reduction** part. 
