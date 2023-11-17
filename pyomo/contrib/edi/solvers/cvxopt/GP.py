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

import copy
from pyomo.common.dependencies import attempt_import

cvxopt, cvxopt_available = attempt_import( "cvxopt" )
if not cvxopt_available:
    raise ImportError('The CVXOPT solver requires cvxopt')


def solve_GP(structures):

    trm_idx = [ structures['Geometric_Program'][1][i][0] for i in range(0,len(structures['Geometric_Program'][1])) ]
    tm_num = 0
    ctr = 0
    K = []
    for i in range(0,len(trm_idx)):
        if tm_num == trm_idx[i]:
            ctr += 1
        else:
            K.append(ctr)
            ctr = 1
            tm_num += 1
    K.append(ctr)

    K_with_equalities = K

    GP_formulation = []
    K = []
    slicer_prev = 0
    unscrambler = []
    conCounter = 0
    for i in range(0,len(K_with_equalities)):
        if K_with_equalities[i] > 1:
            slicer = slicer_prev + K_with_equalities[i]
            for ii in range(slicer_prev,slicer):
                GP_formulation.append(structures['Geometric_Program'][1][ii])
            slicer_prev = slicer
            K.append(K_with_equalities[i])
            unscrambler.append(conCounter)
            conCounter += 1
        else:
            if i > 0 and structures['Geometric_Program'][2][i-1] == '==':
                slicer = slicer_prev + 1
                GP_formulation.append(structures['Geometric_Program'][1][slicer_prev])
                original_row = copy.deepcopy(structures['Geometric_Program'][1][slicer_prev])
                original_row[1] = 1/original_row[1]
                for ii in range(2,len(original_row)):
                    original_row[ii] *= -1
                GP_formulation.append(original_row)
                slicer_prev = slicer
                K.append(1)
                K.append(1)
                unscrambler.append([conCounter, conCounter+1])
                conCounter += 2
            else:
                slicer = slicer_prev + 1
                GP_formulation.append(structures['Geometric_Program'][1][slicer_prev])
                slicer_prev = slicer
                K.append(1)
                unscrambler.append(conCounter)
                conCounter += 1

    GP_formulation = cvxopt.matrix( GP_formulation )
    g = cvxopt.log( GP_formulation[1,:] ).T
    F = GP_formulation[2:,:].T

    res_transformed = cvxopt.solvers.gp(K, F, g)

    res = {}
    res['status']           = res_transformed['status']
    res['primal objective'] = cvxopt.exp(res_transformed['primal objective'])
    res['dual objective']   = cvxopt.exp(res_transformed['dual objective'])
    res['x']                = cvxopt.exp(res_transformed['x'])
    res['y']                = [] # equality
    res['z']                = [] # inequality   
    res['s']                = cvxopt.exp(res_transformed['snl']) - 1

    for i in range(0,len(unscrambler)):
        vl = unscrambler[i]
        if isinstance(vl,int):
            res['z'].append(res_transformed['znl'][vl])
        else: # is list
            y1 = res['z'].append(res_transformed['znl'][vl[0]])
            y2 = res['z'].append(res_transformed['znl'][vl[1]])

            if abs(y1) >= abs(y2):
                y1temp = y1
                y1 = y2
                y2 = y1temp

            if abs(y1) >= cvxopt.solver.options['feastol']:
                raise ValueError('An equality constraint threw an unexpected dual')
            
            res['y'].append(y2)

    res['y'] = cvxopt.matrix(res['y'])
    res['z'] = cvxopt.matrix(res['z'])

    return res






