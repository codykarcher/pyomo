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

import math
import random
import copy
import pyomo.environ as pyo

from pyomo.core.base.block import _BlockData
from pyomo.common.collections.component_map import ComponentMap
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar

from pyomo.contrib.edi.tools.structureWalker import _StructureVisitor

from pyomo.common.dependencies import numpy, numpy_available

if numpy_available:
    import numpy as np
else:
    raise ImportError('The stucture detector requires numpy')

from pyomo.contrib.edi.tools.detectorSupportFunctions import (
     gpRow_add,
     gpRow_subtract,
     # gpRow_multiply,
     gpRow_divide,
     # collapseGProws,
     parseDict_GP,
     checkObjectiveHessian_PD,
     checkLinear,
     unstructured_dict,
)

def structure_detector(pyomo_component):
    # Various setup things

    if not isinstance(pyomo_component, _BlockData):
        raise ValueError(
            "Invalid type %s passed into the convexity detector"
            % (str(type(pyomo_component)))
        )

    variableList = [
        vr
        for vr in pyomo_component.component_objects(
            pyo.Var, descend_into=True, active=True
        )
    ]

    N_bound_cons = 0
    for vr in variableList:
        if vr.domain.name not in ['Reals','NonNegativeReals','NonPositiveReals']:
            return unstructured_dict() | { "message":"A non-continuous variable was detected" } 

        var_lower_bound, var_upper_bound = vr.bounds
        if vr.domain.name == 'NonNegativeReals':
            if var_lower_bound is None:
                var_lower_bound = 0
            else:
                var_lower_bound = max([0,var_lower_bound])

        if vr.domain.name == 'NonPositiveReals':
            if var_upper_bound is None:
                var_upper_bound = 0
            else:
                var_upper_bound = min([0,var_upper_bound])

        if var_lower_bound is not None:
            proposedKey = vr.name + '_lowerBound'
            if proposedKey in list(pyomo_component.__dict__.keys()):
                for i in range(0,10):
                    proposedKey = vr.name + '_lowerBound_randomSeed_' + str(int(math.floor(random.random()*1e6)))
                    if proposedKey not in list(pyomo_component.__dict__.keys()):
                        break
                raise ValueError('Could not found a unique identifier for the lower bound on variable '+vr.name)

            setattr(pyomo_component, proposedKey, pyo.Constraint(expr = vr >= var_lower_bound))
            N_bound_cons += 1

        if var_upper_bound is not None:
            proposedKey = vr.name + '_upperBound'
            if proposedKey in list(pyomo_component.__dict__.keys()):
                for i in range(0,10):
                    proposedKey = vr.name + '_upperBound_randomSeed_' + str(int(math.floor(random.random()*1e6)))
                    if proposedKey not in list(pyomo_component.__dict__.keys()):
                        break
                raise ValueError('Could not found a unique identifier for the upper bound on variable '+vr.name)

            setattr(pyomo_component, proposedKey, pyo.Constraint(expr = vr <= var_lower_bound))
            N_bound_cons += 1

    objectives = [
        obj
        for obj in pyomo_component.component_data_objects(
            pyo.Objective, descend_into=True, active=True
        )
    ]
    constraints = [
        con
        for con in pyomo_component.component_objects(
            pyo.Constraint, descend_into=True, active=True
        )
    ]

    parameterList = [
        pm
        for pm in pyomo_component.component_objects(
            pyo.Param, descend_into=True, active=True
        )
    ]

    variableMap = ComponentMap()
    unwrappedVariables = []
    vrIdx = -1
    for i in range(0, len(variableList)):
        vr = variableList[i]
        vrIdx += 1
        if isinstance(vr, ScalarVar):
            variableMap[vr] = vrIdx
            unwrappedVariables.append(vr)
        elif isinstance(vr, IndexedVar):
            variableMap[vr] = vrIdx
            unwrappedVariables.append(vr)
            for sd in vr.index_set().data():
                vrIdx += 1
                variableMap[vr[sd]] = vrIdx
                unwrappedVariables.append(vr[sd])
        else:
            raise DeveloperError(
                'Variable is not a variable.  Should not happen.  Contact developers'
            )

    N_vars_unwrapped = len(variableMap.keys())

    # Declare a visitor/walker
    visitor = _StructureVisitor()

    # starts building the output dict
    structures = {"Linear_Program"   :[True,[],[]], 
                  "Quadratic_Program":[True,[],[]],
                  "Geometric_Program":[True,[],[]], 
                  "Signomial_Program":[True,[],[]],} # Convex, LogConvex, Convex_QCQP, 

    # Iterate over the objectives and print
    if len(objectives) != 1:
        return unstructured_dict()

    for obj in objectives:
        rv = visitor.walk_expression(obj.sense * obj)
        gpRows = parseDict_GP(0,rv,N_vars_unwrapped,variableMap)

        if not all([rw[0]==0.0 for rw in gpRows]):
            # is sp with fractional objective
            structures['Linear_Program'][0] = False
            structures['Quadratic_Program'][0] = False
            structures['Geometric_Program'][0] = False
            if not all([rw[1]>0.0 for rw in gpRows]):
                # has subtraction in the objective, which is not allowed
                return unstructured_dict()
            else:
                structures['Signomial_Program'][1] += gpRows
        else:
            if not all([rw[1]>0.0 for rw in gpRows]):
                # has subtraction in the objective
                structures['Geometric_Program'][0] = False
                structures['Signomial_Program'][0] = False
            else:
                # has monomial or posynomial objective
                structures['Geometric_Program'][1] += gpRows
                structures['Signomial_Program'][1] += gpRows  

            # check hessian of objective
            quadraticCheck = checkObjectiveHessian_PD(gpRows)
            if quadraticCheck[0]:
                structures['Quadratic_Program'][1] = quadraticCheck[1:] + [None,None]
                structures['Linear_Program'][0] = False
            else:
                structures['Quadratic_Program'][0] = False

                linearCheck = checkLinear(gpRows)
                if linearCheck[0]:
                    structures['Linear_Program'][1] = linearCheck[1:] + [None,None]
                else:
                    structures['Linear_Program'][0] = False  

    # Iterate over the constraints
    N_cons = 0
    if len(constraints) > 0:
        conCounter = 0
        operatorList = []
        for i in range(0, len(constraints)):
            con = constraints[i]
            for c in con.values():
                N_cons += 1
                cexpr = c.expr
                rv = visitor.walk_expression(cexpr)
                for rvv in rv:
                    gpRows_lhs = parseDict_GP(i+1,rvv['lhs'],N_vars_unwrapped,variableMap)
                    gpRows_rhs = parseDict_GP(i+1,rvv['rhs'],N_vars_unwrapped,variableMap)
                    operator = rvv['operator']
                    operatorList.append(operator)

                    if not any([rw[0] < 0 for rw in gpRows_lhs]):
                        lhs_zeroed = gpRow_subtract(copy.deepcopy(gpRows_lhs), copy.deepcopy(gpRows_rhs))
                        # Do LP/QP stuff
                        if not all([rw[0]>=0.0 for rw in lhs_zeroed]):
                            # has fraction
                            structures['Linear_Program'][0] = False
                            structures['Linear_Program'][1] = None
                            structures['Quadratic_Program'][0] = False
                            structures['Quadratic_Program'][1] = None         
                        else:               
                            linearCheck = checkLinear(lhs_zeroed)
                            if linearCheck[0]:
                                if structures['Linear_Program'][0] != False:
                                    if structures['Linear_Program'][1][2] is None:
                                        structures['Linear_Program'][1][2] = linearCheck[1]
                                        structures['Linear_Program'][1][3] = linearCheck[2]
                                    else:
                                        structures['Linear_Program'][1][2] = np.append( structures['Linear_Program'][1][2], linearCheck[1] , axis=0)
                                        structures['Linear_Program'][1][3] = np.append( structures['Linear_Program'][1][3], linearCheck[2] )
                                if structures['Quadratic_Program'][0] != False:
                                    if structures['Quadratic_Program'][1][3] is None:
                                        structures['Quadratic_Program'][1][3] = linearCheck[1]
                                        structures['Quadratic_Program'][1][4] = linearCheck[2]
                                    else:
                                        structures['Quadratic_Program'][1][3] = np.append( structures['Quadratic_Program'][1][3], linearCheck[1] , axis=0)
                                        structures['Quadratic_Program'][1][4] = np.append( structures['Quadratic_Program'][1][4], linearCheck[2] )
                            else:
                                structures['Linear_Program'][0] = False
                                structures['Linear_Program'][1] = None
                                structures['Quadratic_Program'][0] = False
                                structures['Quadratic_Program'][1] = None   
                    else:
                        # signomial fraction present
                        structures['Linear_Program'][0] = False
                        structures['Linear_Program'][1] = None
                        structures['Quadratic_Program'][0] = False
                        structures['Quadratic_Program'][1] = None   


                    # Do GP/SP Stuff
                    if len(gpRows_rhs) == 1 :
                        if gpRows_rhs[0][1] < 0:
                            # constraint is bound to less than a negative number, infeasible
                            structures['Geometric_Program'][0] = False
                            structures['Geometric_Program'][1] = None
                            structures['Signomial_Program'][0] = False
                            structures['Signomial_Program'][1] = None                            
                        else:
                            if gpRows_rhs[0][1] == 0.0:
                                lhs_final = gpRows_lhs
                            else:
                                lhs_final = gpRow_divide(gpRows_lhs, gpRows_rhs)
                    else:
                        lhs_zeroed = gpRow_subtract(copy.deepcopy(gpRows_lhs), copy.deepcopy(gpRows_rhs))
                        unique, counts = numpy.unique([lhz[0] for lhz in lhs_zeroed], return_counts=True)
                        countDict = dict(zip(unique, counts))
                        if len(unique) > 1:
                            # have a fraction, this shouldnt be possible at this stage
                            raise RuntimeError('Encountered an unexpected signomial fraction, shouldnt happen but not sure')
                        negative_monomial_indices = []
                        for ii in range(0,len(lhs_zeroed)):
                            if lhs_zeroed[ii][1] < 0:
                                negative_monomial_indices.append(ii)
                        if len(negative_monomial_indices) == 0 or len(lhs_zeroed)==1 :
                            lhs_final = gpRow_add(lhs_zeroed, [[lhs_zeroed[0][0],1.0]+[0.0]*N_vars_unwrapped])
                        elif len(negative_monomial_indices) == 1:
                            negMonomial = copy.deepcopy(lhs_zeroed[negative_monomial_indices[0]])
                            posMonomial = copy.deepcopy(negMonomial)
                            posMonomial[1] *= -1
                            lhs_inter = gpRow_subtract(lhs_zeroed, [negMonomial])
                            lhs_final = gpRow_divide(lhs_inter, [posMonomial])
                        else:
                            negPosynomial = [ copy.deepcopy(lhs_zeroed[nmi]) for nmi in negative_monomial_indices ]
                            posPosynomial = copy.deepcopy(negPosynomial)
                            for ii in range(0,len(posPosynomial)):
                                posPosynomial[ii][1] *= -1
                            lhs_inter = gpRow_subtract(lhs_zeroed, negPosynomial)
                            lhs_final = gpRow_divide(lhs_inter, posPosynomial)

                    if not all([rw[1]>0.0 for rw in lhs_final]):
                        # has subtraction, which is not allowed under this definition of SP
                        structures['Geometric_Program'][0] = False
                        structures['Geometric_Program'][1] = None
                        structures['Signomial_Program'][0] = False
                        structures['Signomial_Program'][1] = None

                    if not all([rw[0]>=0.0 for rw in lhs_final]):
                        # is sp with fraction
                        structures['Geometric_Program'][0] = False
                        structures['Geometric_Program'][1] = None
                        if structures['Signomial_Program'][0] != False:
                            structures['Signomial_Program'][1] += lhs_final
                    else:
                        # monomial or valid posynomial   
                        if structures['Geometric_Program'][0] != False:
                            structures['Geometric_Program'][1] += lhs_final
                        if structures['Signomial_Program'][0] != False:
                            structures['Signomial_Program'][1] += lhs_final

        if structures['Linear_Program'][0] != False:
            structures['Linear_Program'][2] = operatorList
        if structures['Quadratic_Program'][0] != False:
            structures['Quadratic_Program'][2] = operatorList
        if structures['Geometric_Program'][0] != False:
            structures['Geometric_Program'][2] = operatorList
        if structures['Signomial_Program'][0] != False:
            structures['Signomial_Program'][2] = operatorList

        if structures['Geometric_Program'][0] != False:
            conIxs = [ structures['Geometric_Program'][1][i][0] for i in range(0,len(structures['Geometric_Program'][1])) ]
            unique, counts = numpy.unique(conIxs, return_counts=True)
            countDict = dict(zip(unique, counts))
            for i in range(0,len(operatorList)):
                conIx = i+1
                if operatorList[i] == '==':
                    if countDict[conIx] > 1:
                        structures['Geometric_Program'][0] = False
                        structures['Geometric_Program'][1] = None
                        structures['Geometric_Program'][2] = None
                        break

    structures['info'] = {}
    structures['info']['N_cons_total']    = N_cons
    structures['info']['N_cons_noBounds'] = N_cons - N_bound_cons
    structures['info']['N_cons_bounds']   = N_bound_cons
    return structures








