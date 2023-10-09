#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from itertools import product

import numpy as np

import pyomo.environ as pe

# TODO: Add more test probelms as needed.
'''
This script has collection of test cases that can be used to enumerate solutions.
That is, simple problems where the alternative solutions can be found manually.

I started on a few problems here. I tired to enumerate all of the solutions for
disrete cases. This should make it easy to find all feasible points, and/or
all points within some percent/value of optimality.

I created some pure continuous problems and found bounds and extreme points for those
but more work is needed to be able to find the bounds and extreme points within some
threshold optimality.

I have not done any mixed cases yet, but an case with those would be useful. 
get_2d_diamond_problem does let make x or y discrete, but I have not found the bounds
and extreme points for these cases yet.

Other cases come to mind? A quadtratic maybe?

'''


def get_2d_diamond_problem(discrete_x=False, discrete_y=False):
    '''Simple 2d problem where the feasible is diamond-shaped.'''
    m = pe.ConcreteModel()
    m.x = pe.Var(within=pe.Integers if discrete_x else pe.Reals)
    m.y = pe.Var(within=pe.Integers if discrete_y else pe.Reals)
    
    m.o = pe.Objective(expr = m.x + m.y, sense=pe.maximize)
    
    m.c1 = pe.Constraint(expr= -4/5 * m.x - 4 <= m.y)
    m.c2 = pe.Constraint(expr=  5/9 * m.x - 5 <= m.y)
    m.c3 = pe.Constraint(expr=  2/9 * m.x + 2 >= m.y)
    m.c4 = pe.Constraint(expr= -1/2 * m.x + 3 >= m.y)

    m.extreme_points = {(0.737704918, -4.590163934),
                        (-5.869565217, 0.695652174),
                        (1.384615385, 2.307692308),
                        (7.578947368, -0.789473684)}

    m.continuous_bounds = pe.ComponentMap()
    m.continuous_bounds[m.x] = (-5.869565217, 7.578947368)
    m.continuous_bounds[m.y] = (-4.590163934, 2.307692308)

    return m


def get_2d_unbounded_problem():
    '''
    Simple 2d problem where the feasible region is unbounded, but the problem
    has an optimal solution.'''
    m = pe.ConcreteModel()
    m.x = pe.Var(within=pe.Reals)
    m.y = pe.Var(within=pe.Reals)
    
    m.o = pe.Objective(expr = m.y - m.x)
    
    m.c1 = pe.Constraint(expr= m.x <= 4)
    m.c2 = pe.Constraint(expr= m.y >= 2)

    m.extreme_points = {(4, 2)}

    m.continuous_bounds = pe.ComponentMap()
    m.continuous_bounds[m.x] = (float('-inf'), 4)
    m.continuous_bounds[m.y] = (2, float('inf'))

    return m

def get_triangle_ip():
    ''' 
    Simple 2d discrete problem where the feasible region looks like a 90-45-45
    right triangle and the optimal solutions fall along the hypotenuse.
    '''
    m = pe.ConcreteModel()
    var_max = 5
    m.x = pe.Var(within=pe.NonNegativeIntegers, bounds=(0,var_max))
    m.y = pe.Var(within=pe.NonNegativeIntegers, bounds=(0,var_max))
    
    m.o = pe.Objective(expr=m.x + m.y, sense=pe.maximize)
    m.c = pe.Constraint(expr= m.x + m.y <= var_max)
    
    feasible_sols = []
    for i in range(var_max + 1):
        for j in range(var_max + 1):
            if i + j <= var_max:
                feasible_sols.append(((i, j), i + j))
    feasible_sols = sorted(feasible_sols, key=lambda sol: sol[1], reverse=True)
    m.feasible_sols = feasible_sols
    
    return m

def get_implied_bound_ip():
    '''
    2d discrete problem where the bounds of z are impled by x and y. This
    facilitate testing cases where the impled bounds are tighter than the
    given bounds for the variable.
    '''
    m = pe.ConcreteModel()
    m.x = pe.Var(within=pe.NonNegativeIntegers, bounds=(0,5))
    m.y = pe.Var(within=pe.NonNegativeIntegers, bounds=(0,5))
    m.z = pe.Var(within=pe.NonNegativeIntegers, bounds=(0,5))
    
    m.o = pe.Objective(expr = m.x + m.z)
    
    m.c1 = pe.Constraint(expr= m.x + m.y == 3)
    m.c2 = pe.Constraint(expr= m.x + m.y + m.z <= 5)

    m.extreme_points = {(4, 2)}

    m.var_bounds = pe.ComponentMap()
    m.var_bounds[m.x] = (0, 3)
    m.var_bounds[m.y] = (0, 3)
    m.var_bounds[m.z] = (0, 2)
    
    return m
            

def get_aos_test_knapsack(var_max, weights, values, capacity_fraction):
    '''
    Creates a knapsack problem, given arrays of weights and values, and
    returns all feasible solutions. The capacity represents the percent of the
    total max weight that can be selected (sum weights * var_max). The var_max
    parameter sets the upper bound on all variables, teh max number of times
    they can be selected.
    '''
    assert len(weights) == len(values), \
        'weights and values must be the same length.'
    assert 0 <= capacity_fraction and capacity_fraction <= 1, \
            'capacity_fraction must be between 0 and 1.'
    
    num_vars = len(weights)
    capacity = sum(weights) * var_max * capacity_fraction
    
    m = pe.ConcreteModel()
    m.i = pe.RangeSet(0,num_vars-1)
    
    if var_max == 1:        
        m.x = pe.Var(m.i, within=pe.Binary)
    else:
        m.x = pe.Var(m.i, within=pe.NonNegativeIntegers, bounds=(0,var_max))

    m.o = pe.Objective(expr=sum(values[i]*m.x[i] for i in m.i), 
                       sense=pe.maximize)

    m.c = pe.Constraint(expr=sum(weights[i]*m.x[i] for i in m.i) <= capacity)
    
    var_domain = range(var_max+1)
    all_combos = product(var_domain, repeat=num_vars)
    
    feasible_sols = []
    for sol in all_combos:
        if np.dot(sol, weights) <= capacity:
            feasible_sols.append((sol, np.dot(sol, values)))
    feasible_sols = sorted(feasible_sols, key=lambda sol: sol[1], reverse=True)
    print(feasible_sols)
    return m