#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import

cvxopt, cvxopt_available = attempt_import( "cvxopt" )
if not cvxopt_available:
    raise ImportError('The CVXOPT solver requires cvxopt')


def solve_LP(structures):

    c = cvxopt.matrix(structures['Linear_Program'][1][0][0])
    objective_shift = structures['Linear_Program'][1][1]
    AG = structures['Linear_Program'][1][2]
    if AG is None:
        raise ValueError('Linear Program with no constraints is unbounded')
    bh = -1*structures['Linear_Program'][1][3]
    operators = structures['Linear_Program'][2]

    A = None
    G = None
    b = None
    h = None

    eqUnscramble = []
    iqUnscramble = []
    for i in range(0,len(AG)):
        if operators[i] == '==':
            eqUnscramble.append(i)
            if A is None:
                A = [AG[i].tolist()]
                b = [bh[i]]
            else:
                A += [AG[i].tolist()]
                b += [bh[i]]
        else:
            iqUnscramble.append(i)
            if G is None:
                G = [AG[i].tolist()]
                h = [bh[i]]
            else:
                G += [AG[i].tolist()]
                h += [bh[i]]

    if A is not None:
        A = cvxopt.matrix(A).T
    if b is not None:
        b = cvxopt.matrix(b)
    if G is not None:
        G = cvxopt.matrix(G).T
    if h is not None:
        h = cvxopt.matrix(h)

    res = cvxopt.solvers.lp(c,G,h,A,b)
    res['objective_shift'] = objective_shift 
    res['inequality_unscramble'] = iqUnscramble
    res['equality_unscramble']   = eqUnscramble

    return res

