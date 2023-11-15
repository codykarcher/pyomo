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
import re
import io
import pyomo.environ as pyo
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor

from pyomo.core.expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    MonomialTermExpression,
    LinearExpression,
    SumExpression,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
    Expr_ifExpression,
    ExternalFunctionExpression,
)

from pyomo.core.expr.visitor import identify_components
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.expr.template_expr import (
    GetItemExpression,
    GetAttrExpression,
    TemplateSumExpression,
    IndexTemplate,
    Numeric_GetItemExpression,
    templatize_constraint,
    resolve_template,
    templatize_rule,
)
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.core.base.param import _ParamData, ScalarParam, IndexedParam
from pyomo.core.base.set import _SetData
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from pyomo.common.collections.component_map import ComponentMap
from pyomo.common.collections.component_set import ComponentSet
from pyomo.core.expr.template_expr import (
    NPV_Numeric_GetItemExpression,
    NPV_Structural_GetItemExpression,
    Numeric_GetAttrExpression,
)
from pyomo.core.expr.numeric_expr import (
    NPV_SumExpression, 
    NPV_DivisionExpression, 
    NPV_ProductExpression,
    NPV_PowExpression,
    # DivisionExpression as NE_DivisionExpression,
) 

from pyomo.core.base.block import IndexedBlock

from pyomo.core.base.external import _PythonCallbackFunctionID

from pyomo.core.base.block import _BlockData

from pyomo.repn.util import ExprType

from pyomo.common import DeveloperError

_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import attempt_import

cvxopt, cvxopt_available = attempt_import(
    "cvxopt"
)

if numpy_available:
    import numpy as np
else:
    raise ImportError('The CVXOPT solver requires numpy')

if not cvxopt_available:
    raise ImportError('The CVXOPT solver requires cvxopt')


def unarySignomial():
    return {"leadingCoefficients":[1.0], "bases":[[]], "exponents":[[]]}

def no_structure_dict():
    elementDict= {
            "constant"           : {"status":"no", "value":None },
            "monomial"           : {"status":"no", "leadingConstant":None, "bases":[], "exponents":[]},
            "signomial"          : {"status":"no", "leadingCoefficients":None, "bases":[], "exponents":[]},
            "signomial_fraction" : {"status":"no",   "numerator":{"leadingCoefficients":None, "bases":[[]], "exponents":[[]]},
                                                   "denominator":{"leadingCoefficients":None, "bases":[[]], "exponents":[[]]} },
        }
    return elementDict

def monomial_multiplication(lhm,rhm):
    mon = {}
    mon['leadingConstant'] = lhm["leadingConstant"] * rhm["leadingConstant"]
    lbases = lhm['bases']
    lexps  = lhm['exponents']
    rbases = rhm['bases']
    rexps  = rhm['exponents']
    matches = {}
    rskips  = []
    newBases = []
    newExps  = []
    for iii in range(0,len(lbases)):
        for jjj in range(0,len(rbases)):
            if lbases[iii] is rbases[jjj]:
                matches[iii] = jjj
                rskips.append(jjj)
    lkeys = list(matches.keys())
    for iii in range(0,len(lbases)):
        if iii in lkeys:
            newBases.append(lbases[iii])
            newExps.append(lexps[iii] + rexps[matches[iii]])
        else:
            newBases.append(lbases[iii])
            newExps.append(lexps[iii])
    for jjj in range(0,len(rbases)):
        if jjj not in rskips:
            newBases.append(rbases[jjj])
            newExps.append(rexps[jjj])

    mon['bases'] = newBases
    mon['exponents'] = newExps
    return mon

def signomial_multiplication(lhe,rhe):
    rhe_new = {}
    rhe_new['leadingCoefficients'] = []
    rhe_new['bases'] = []
    rhe_new['exponents'] = []

    for i in range(0,len(lhe['leadingCoefficients'])):
        for j in range(0,len(rhe['leadingCoefficients'])):
            mon = monomial_multiplication({'leadingConstant':lhe['leadingCoefficients'][i], 'bases':lhe['bases'][i], 'exponents':lhe['exponents'][i]},
                                        {'leadingConstant':rhe['leadingCoefficients'][j], 'bases':rhe['bases'][j], 'exponents':rhe['exponents'][j]}  )

            rhe_new['leadingCoefficients'].append(mon["leadingConstant"])
            rhe_new['bases'].append(mon['bases'])
            rhe_new['exponents'].append(mon['exponents'])

    elementDict = no_structure_dict()
    elementDict['signomial'] = {'status':'yes'} | rhe_new
    elementDict['signomial_fraction']['status']      = 'yes'
    elementDict['signomial_fraction']['numerator']   = rhe_new
    elementDict['signomial_fraction']['denominator'] = unarySignomial()
    return elementDict

def signomial_fraction_multiplication(lhf,rhf):
    new_num = signomial_multiplication(lhf['numerator'],rhf['numerator'])['signomial']
    new_dem = signomial_multiplication(lhf['denominator'],rhf['denominator'])['signomial']

    del new_num['status']
    del new_dem['status']

    elementDict = no_structure_dict()
    elementDict['signomial_fraction']['status']      = 'yes'
    elementDict['signomial_fraction']['numerator']   = new_num
    elementDict['signomial_fraction']['denominator'] = new_dem
    return elementDict


def signomial_power_evaluation(coeffs,bases,exponents,expVal):
    if len(coeffs) != len(bases) or len(coeffs) != len(exponents) :
        raise ValueError("Signomial data has inconsistent lengths")

    if expVal == 0:
        return handle_num_node(None,None,1.0)

    if expVal < 0:
        return no_structure_dict()

    lhe = {"leadingCoefficients":coeffs, "bases":bases, "exponents":exponents}
    rhe = {"leadingCoefficients":coeffs, "bases":bases, "exponents":exponents}
    for ii in range(0,int(expVal)-1):
        rhe_new = signomial_multiplication(lhe,rhe)
        rhe = rhe_new

    return rhe

def handle_sumExpression_node(visitor, node, *args):
    if any([ a['signomial_fraction']['status']!='yes' for a in args]):
        return no_structure_dict()

    if any([ a['signomial']['status']!='yes' for a in args]):
        # at least one signomial fraction

        signomial_denominator =  signomial_multiplication(unarySignomial(), unarySignomial()) 
        for a in args:
            signomial_denominator = signomial_multiplication(signomial_denominator['signomial'], a['signomial_fraction']['denominator'])

        signomial_numerator = args[0]
        for a in args[1:]:
            signomial_numerator = signomial_multiplication(signomial_numerator['signomial'],a['signomial_fraction']['denominator'])
        
        for i in range(1,len(args)):
            sn_temp = args[i]
            for j in range(0,len(args)):
                if j != i:
                    sn_temp = signomial_multiplication(sn_temp['signomial_fraction']['numerator'], args[j]['signomial_fraction']['denominator'])

            signomial_numerator = handle_sumExpression_node(visitor, node, signomial_numerator, sn_temp)

        signomial_numerator_sig = signomial_numerator['signomial'] 
        del signomial_numerator_sig['status']

        signomial_denominator_sig = signomial_denominator['signomial'] 
        del signomial_denominator_sig['status']

        nsd = no_structure_dict()
        nsd['signomial_fraction']['status'] = 'yes'
        nsd['signomial_fraction']['numerator'] = signomial_numerator_sig
        nsd['signomial_fraction']['denominator'] = signomial_denominator_sig

        return nsd

    # if any([ a['monomial']['status']!='yes' for a in args]):
    #     # at least one signomial
    #     # not needed, next case catches

    if any([ a['constant']['status']!='yes' for a in args]):
        # at least one monomial
        signomialDict = args[0]['signomial']
        for a_all in args[1:]:
            a = a_all['signomial']
            signomialDict['leadingCoefficients'] += a['leadingCoefficients']
            signomialDict['bases'] += a['bases']
            signomialDict['exponents'] += a['exponents']
        elementDict = no_structure_dict()
        elementDict['signomial'] = {'status':'yes'} | signomialDict
        elementDict['signomial_fraction']['numerator'] = signomialDict
        elementDict['signomial_fraction']['denominator'] = unarySignomial()
        return elementDict

    # is still constant
    vl = sum([a['constant']['value'] for a in args])
    return handle_num_node(visitor,float(vl))


def handle_negation_node(visitor, node, arg1):
    arg2 = handle_num_node(visitor, -1.0)
    return handle_product_node(visitor,node,arg2,arg1)

def handle_product_node(visitor, node, arg1, arg2):
    if arg1["constant"]["status"] == "yes" and arg2["constant"]["status"] == "yes":
        vl = float(arg1['constant']['value']*arg2['constant']['value'])
        return handle_num_node(visitor,vl)

    if arg1["monomial"]["status"] == "yes" and arg2["constant"]["status"] == "yes":
        # swap and delay
        arg2_temp = arg2
        arg2 = arg1
        arg1 = arg2_temp

    if ( (arg1["constant"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") or 
         (arg1["monomial"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") ):

        mon = monomial_multiplication(arg1['monomial'],arg2['monomial'])
        elementDict= {
            "constant"           : {"status":"no", "value":None },
            "monomial"           : {"status":"yes", "leadingConstant":mon['leadingConstant'], "bases":mon['bases'], "exponents":mon['exponents']},
            "signomial"          : {"status":"yes", "leadingCoefficients":[mon['leadingConstant']], "bases":[mon['bases']], "exponents":[mon['exponents']]},
            "signomial_fraction" : {"status":"yes",   "numerator":{"leadingCoefficients":[mon['leadingConstant']], "bases":[mon['bases']], "exponents":[mon['exponents']]}, "denominator":unarySignomial() },
        }
        return elementDict

    if ( (arg1["signomial"]["status"] == "yes" and arg2["constant"]["status"] == "yes") or
         (arg1["signomial"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") ):
        # swap and delay
        arg2_temp = arg2
        arg2 = arg1
        arg1 = arg2_temp

    if ( (arg1["constant"]["status"]  == "yes" and arg2["signomial"]["status"] == "yes") or
         (arg1["monomial"]["status"]  == "yes" and arg2["signomial"]["status"] == "yes") or 
         (arg1["signomial"]["status"] == "yes" and arg2["signomial"]["status"] == "yes") ):

        lhe = {"leadingCoefficients":arg1["signomial"]['leadingCoefficients'], "bases":arg1["signomial"]['bases'], "exponents":arg1["signomial"]['exponents']}
        rhe = {"leadingCoefficients":arg2["signomial"]['leadingCoefficients'], "bases":arg2["signomial"]['bases'], "exponents":arg2["signomial"]['exponents']}
        rs = signomial_multiplication(lhe,rhe)
        return rs

    if ( (arg1["signomial_fraction"]["status"] == "yes" and arg2["constant"]["status"] == "yes") or 
         (arg1["signomial_fraction"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") or
         (arg1["signomial_fraction"]["status"] == "yes" and arg2["signomial"]["status"] == "yes") ):
        # swap and delay
        arg2_temp = arg2
        arg2 = arg1
        arg1 = arg2_temp

    if ( (arg1["constant"]["status"]           == "yes" and arg2["signomial_fraction"]["status"] == "yes") or
         (arg1["monomial"]["status"]           == "yes" and arg2["signomial_fraction"]["status"] == "yes") or 
         (arg1["signomial"]["status"]          == "yes" and arg2["signomial_fraction"]["status"] == "yes") or 
         (arg1["signomial_fraction"]["status"] == "yes" and arg2["signomial_fraction"]["status"] == "yes") ):

        # lhf = {"leadingCoefficients":arg1['leadingCoefficients'], "bases":arg1['bases'], "exponents":arg1['exponents']}
        # rhf = {"leadingCoefficients":arg2['leadingCoefficients'], "bases":arg2['bases'], "exponents":arg2['exponents']}
        rs = signomial_fraction_multiplication(arg1["signomial_fraction"],arg2["signomial_fraction"])
        return rs

    return no_structure_dict()

def handle_division_node(visitor, node, arg1, arg2):
    if arg2["constant"]["status"] == "yes":
        arg2['constant']['value'] = 1/arg2['constant']['value']
        arg2['monomial']['leadingConstant'] = 1/arg2['monomial']['leadingConstant']
        arg2['signomial']['leadingCoefficients'] = [ 1/c for c in arg2['signomial']['leadingCoefficients'] ]
        return handle_product_node(visitor, node, arg1, arg2)
    if arg2["monomial"]["status"] == "yes":
        arg2['monomial']['leadingConstant'] = 1/arg2['monomial']['leadingConstant']
        arg2['monomial']['exponents'] = [ -c for c in arg2['monomial']['exponents'] ]
        arg2['signomial']['leadingCoefficients'] = [ 1/c for c in arg2['signomial']['leadingCoefficients'] ]
        for i in range(0,len(arg2['signomial']['exponents'])):
            arg2['signomial']['exponents'][i] = [-c for c in arg2['signomial']['exponents'][i] ]
        return handle_product_node(visitor, node, arg1, arg2)
    if arg2["signomial"]["status"] == "yes":
        arg2['signomial_fraction']['status'] = 'yes'  
        arg2['signomial_fraction']['numerator'] = unarySignomial()
        arg2['signomial_fraction']['denominator'] = unarySignomial()
        arg2['signomial_fraction']['denominator']['leadingCoefficients'] = arg2["signomial"]['leadingCoefficients']
        arg2['signomial_fraction']['denominator']['bases'] = arg2["signomial"]['bases']
        arg2['signomial_fraction']['denominator']['exponents'] = arg2["signomial"]['exponents']
        nsd = no_structure_dict()
        arg2['signomial'] = nsd['signomial']
        return handle_product_node(visitor, node, arg1, arg2)

    return no_structure_dict()

def handle_pow_node(visitor, node, arg1, arg2):
    if arg2["constant"]["value"] is None:
        return no_structure_dict()
    elif arg2["constant"]["status"]=='no':
        # is a param, substitute value first by setting to value
        arg2 = handle_num_node(visitor,arg2["constant"]["value"])
    else:
        pass

    expVal = arg2["constant"]["value"]
    if arg1["constant"]["status"] == "yes":
        arg1["constant"]["value"] = arg1["constant"]["value"]**expVal
        arg1["monomial"]["leadingConstant"] = arg1["monomial"]["leadingConstant"]**expVal
        arg1["signomial"]["leadingCoefficients"] = [ vl ** expVal for vl in arg1["signomial"]["leadingCoefficients"] ]
        arg1["signomial_fraction"]['numerator']["leadingCoefficients"] = [ vl ** expVal for vl in arg1["signomial_fraction"]['numerator']["leadingCoefficients"] ]
        return arg1
    if arg1["monomial"]["status"] == "yes":
        arg1["monomial"]["leadingConstant"] = arg1["monomial"]["leadingConstant"]**expVal
        arg1["monomial"]["exponents"] = [ vl * expVal for vl in arg1["monomial"]["exponents"] ]
        arg1["signomial"]["leadingCoefficients"] = [ vl ** expVal for vl in arg1["signomial"]["leadingCoefficients"] ]
        arg1['signomial_fraction']['numerator']["leadingCoefficients"] = [ vl ** expVal for vl in arg1["signomial_fraction"]['numerator']["leadingCoefficients"] ]
        for i in range(0,len(arg1["signomial"]["exponents"])):
            arg1["signomial"]["exponents"][i] = [ vl * expVal for vl in arg1["signomial"]["exponents"][i]  ]
        for i in range(0,len(arg1["signomial_fraction"]['numerator']["exponents"])):
            arg1["signomial_fraction"]['numerator']["exponents"][i] = [ vl * expVal for vl in arg1["signomial_fraction"]['numerator']["exponents"][i]  ]
        return arg1
    if arg1["signomial"]["status"] == "yes":
        if float(expVal).is_integer():
            return signomial_power_evaluation(arg1["signomial"]["leadingCoefficients"],arg1["signomial"]["bases"],arg1["signomial"]["exponents"],expVal)
        else:
            return no_structure_dict()   
    if arg1["signomial_fraction"]["status"] == "yes":
        if float(expVal).is_integer():
            num = signomial_power_evaluation(arg1["signomial_fraction"]['numerator']["leadingCoefficients"],
                                                arg1["signomial_fraction"]['numerator']["bases"],
                                                arg1["signomial_fraction"]['numerator']["exponents"],
                                                expVal)
            dem = signomial_power_evaluation(arg1["signomial_fraction"]['denominator']["leadingCoefficients"],
                                                arg1["signomial_fraction"]['denominator']["bases"],
                                                arg1["signomial_fraction"]['denominator']["exponents"],
                                                expVal)

            elementDict = no_structure_dict()
            elementDict['signomial_fraction']['status'] = 'yes'
            elementDict['signomial_fraction']['numerator'] = num['signomial_fraction']['numerator']
            elementDict['signomial_fraction']['denominator'] = dem['signomial_fraction']['numerator']
            return elementDict

        else:
            return no_structure_dict()   

    return no_structure_dict()      

def handle_abs_node(visitor, node, arg1):
    if arg1['constant']['status']=='yes':
        vl = abs(arg1['constant']['value'])
        return handle_num_node(visitor,float(vl))
    else:
        # has no structure
        return no_structure_dict()  

def handle_unary_node(visitor, node, arg1):
    fcn_handle = node.getname()
    if fcn_handle == 'sqrt':
        arg2 = handle_num_node(visitor, 0.5)
        return handle_pow_node(visitor, node, arg1, arg2)
    else:
        # has no structure
        return no_structure_dict()

def handle_var_node(visitor, node):
    elementDict= {
        "constant"           : {"status":"no", "value":None }, # TODO: should exempt the case of a fixed variable
        "monomial"           : {"status":"yes", "leadingConstant":1.0, "bases":[node], "exponents":[1.0]},
        "signomial"          : {"status":"yes", "leadingCoefficients":[1.0], "bases":[[node]], "exponents":[[1.0]]},
        "signomial_fraction" : {"status":"yes",  "numerator":{"leadingCoefficients":[1.0], "bases":[[node]], "exponents":[[1.0]]}, "denominator":unarySignomial() },
    }
    return elementDict

def handle_param_node(visitor, node):
    elementDict= {
        "constant"           : {"status":"no", "value":node.value },
        "monomial"           : {"status":"yes", "leadingConstant":1.0, "bases":[node], "exponents":[1.0]},
        "signomial"          : {"status":"yes", "leadingCoefficients":[1.0], "bases":[[node]], "exponents":[[1.0]]},
        "signomial_fraction" : {"status":"yes",  "numerator":{"leadingCoefficients":[1.0], "bases":[[node]], "exponents":[[1.0]]}, "denominator":unarySignomial()},
    }
    return elementDict

def handle_num_node(visitor, node):
    elementDict= {
        "constant"           : {"status":"yes", "value":float(node) }, 
        "monomial"           : {"status":"yes", "leadingConstant":float(node), "bases":[], "exponents":[]},
        "signomial"          : {"status":"yes", "leadingCoefficients":[float(node)], "bases":[[]], "exponents":[[]]},
        "signomial_fraction" : {"status":"yes",  "numerator":{"leadingCoefficients":[float(node)], "bases":[[]], "exponents":[[]]}, "denominator":unarySignomial() },
    }
    return elementDict

def handle_monomialTermExpression_node(visitor, node, arg1, arg2):
    return handle_product_node(visitor,node,arg1,arg2)

def handle_named_expression_node(visitor, node, arg1):
    # needed to preserve consistencency with the exitNode function call
    # prevents the need to type check in the exitNode function
    return arg1

def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    # has no structure
    return no_structure_dict()

def handle_external_function_node(visitor, node, *args):
    # has no structure
    return no_structure_dict()

def handle_functionID_node(visitor, node, *args):
    # seems to just be a placeholder empty wrapper object
    return handle_external_function_node(visitor, node, *args)

def handle_equality_node(visitor, node, arg1, arg2):
    return [ {'lhs':arg1, 'operator':'==', 'rhs':arg2} ]

def handle_inequality_node(visitor, node, arg1, arg2):
    return [ {'lhs':arg1, 'operator':'<=', 'rhs':arg2} ]

def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    return [ {'lhs':arg1, 'operator':'<=', 'rhs':arg2},
             {'lhs':arg2, 'operator':'<=', 'rhs':arg3}  ]


def processMonomial(ix,coeff,bases,exponents,N_vars_unwrapped,variableMap):
    gpRow = [0.0]*(N_vars_unwrapped+2)
    gpRow[0] = ix
    gpRow[1] = coeff
    for i in range(0,len(bases)):
        bs = bases[i]
        if isinstance(bs,_GeneralVarData):
            vr_ix = variableMap[bs]
            gpRow[2+vr_ix] = exponents[i]
        elif isinstance(bs,_ParamData):
            vl = bs.value
            expt = exponents[i]
            gpRow[1] *= vl**expt
        else:
            raise ValueError('Unexpected type in the base')
    return gpRow

def collapseGProws(gpRows):
    similarTerms = []
    skipList = []
    for i in range(0,len(gpRows)):
        if i not in skipList:
            for j in range(0,len(gpRows)):
                if j>i and j not in skipList:
                    if gpRows[i][2:] == gpRows[j][2:] and gpRows[i][0] == gpRows[j][0]:
                        skipList.append(i)
                        skipList.append(j)
                        doAppend = True
                        for ii in range(0,len(similarTerms)):
                            if i in similarTerms[ii]:
                                similarTerms[ii].append(j)
                                dontAppend = False
                                break
                        if doAppend:
                            similarTerms.append([i,j])
    
    gpRows_new = []
    skipList = []
    for i in range(0,len(gpRows)):
        doAppend = True
        if i in skipList:
            doAppend = False
        for ii in range(0,len(similarTerms)):
            if i in similarTerms[ii] and i not in skipList:
                gpRow_new = gpRows[i]
                gpRow_new[1] = sum([gpRows[c][1] for c in similarTerms[ii]])
                if gpRow_new[1] != 0:
                    gpRows_new.append(gpRow_new)
                skipList += similarTerms[ii]
                doAppend = False
                break
        if doAppend:
            if gpRows[i][1] != 0:
                gpRows_new.append(gpRows[i])

    return gpRows_new


def parseDict_GP(ix,rv,N_vars_unwrapped,variableMap):
    if rv['monomial']['status'] == 'yes':
        lcs = rv['monomial']['leadingConstant']
        bss = rv['monomial']['bases']
        exs = rv['monomial']['exponents']
        gpRow = processMonomial(ix,lcs,bss,exs,N_vars_unwrapped,variableMap)
        return [gpRow]

    if rv['signomial']['status'] == 'yes':
        gpRows = []
        for i in range(0,len(rv['signomial']['leadingCoefficients'])):
            lcs = rv['signomial']['leadingCoefficients'][i]
            bss = rv['signomial']['bases'][i]
            exs = rv['signomial']['exponents'][i]
            gpRow = processMonomial(ix,lcs,bss,exs,N_vars_unwrapped,variableMap)
            gpRows.append(gpRow)
        return collapseGProws(gpRows)

    # if neither other thing flags, it is a signomial fraction
    # Denominators have the index -ix-1, except objective which is -1
    gpRows = []
    for j in [0,1]:
        if j == 0:
            sf_ix = ix
            sf_ele = rv['signomial_fraction']['numerator']
        else:
            sf_ix = -ix-1
            sf_ele = rv['signomial_fraction']['denominator']
        for i in range(0,len(sf_ele['leadingCoefficients'])):
            lcs = sf_ele['leadingCoefficients'][i]
            bss = sf_ele['bases'][i]
            exs = sf_ele['exponents'][i]
            gpRow = processMonomial(sf_ix,lcs,bss,exs,N_vars_unwrapped,variableMap)
            gpRows.append(gpRow)

    return collapseGProws(gpRows)



class _ConvexityVisitor(StreamBasedExpressionVisitor):
    def __init__(self):
        super().__init__()

        self._operator_handles = {
            ScalarVar: handle_var_node,
            int: handle_num_node,
            float: handle_num_node,
            SumExpression: handle_sumExpression_node,
            NegationExpression: handle_negation_node,
            ProductExpression: handle_product_node,
            DivisionExpression: handle_division_node,
            PowExpression: handle_pow_node,
            AbsExpression: handle_abs_node,
            UnaryFunctionExpression: handle_unary_node,
            Expr_ifExpression: handle_exprif_node,
            EqualityExpression: handle_equality_node,
            InequalityExpression: handle_inequality_node,
            RangedExpression: handle_ranged_inequality_node,
            _GeneralExpressionData: handle_named_expression_node,
            ScalarExpression: handle_named_expression_node,
            kernel.expression.expression: handle_named_expression_node,
            kernel.expression.noclone: handle_named_expression_node,
            _GeneralObjectiveData: handle_named_expression_node,
            _GeneralVarData: handle_var_node,
            ScalarObjective: handle_named_expression_node,
            kernel.objective.objective: handle_named_expression_node,
            ExternalFunctionExpression: handle_external_function_node,
            _PythonCallbackFunctionID: handle_functionID_node,
            LinearExpression: handle_sumExpression_node,
            MonomialTermExpression: handle_monomialTermExpression_node,
            IndexedVar: handle_var_node,
            ScalarParam: handle_param_node,
            _ParamData: handle_param_node,
            IndexedParam: handle_param_node,
            NPV_SumExpression: handle_sumExpression_node,
            NPV_DivisionExpression: handle_division_node,
            NPV_ProductExpression: handle_product_node,
            NPV_PowExpression: handle_pow_node,
            # NE_DivisionExpression: handle_division_node,
        }
        if numpy_available:
            self._operator_handles[np.float64] = handle_num_node

    def exitNode(self, node, data):
        try:
            return self._operator_handles[node.__class__](self, node, *data)
        except:
            raise DeveloperError(
                'Convexity walker encountered an error when processing type %s, contact the developers'
                % (node.__class__)
            )

def gpRow_add(gr1, gr2):
    gr1_isFrac = any([g[0] <= 0 for g in gr1])
    gr2_isFrac = any([g[0] <= 0 for g in gr2])
    if gr1_isFrac or gr2_isFrac:
        # if gr1_isFrac and gr2_isFrac:
        raise RuntimeError('Signomial Fraction addition should not be occurring here')
    #     elif gr2_isFrac: 
    #         gr1_temp = copy.deepcopy(gr2)
    #         gr2 = gr1
    #         gr1 = gr1_temp
    #     else:
    #         pass

    #     if len(gr2) > 1 :
    #         raise RuntimeError('Adding a posynomial to a fraction, shouldnt happen but not sure')

    #     for i in range(0,len(gr1)):
    #         gr1[i][1] *= gr2[0][1]
    #         for j in range(2,len(gr1[0])):
    #             gr1[i][j] += gr2[0][j]

    #     return gr1
    # else:
    return collapseGProws(gr1+gr2)

def gpRow_subtract(gr1, gr2):
    for i in range(0,len(gr2)):
        if gr2[i][0] >= 0:
            gr2[i][1] *= -1
    return gpRow_add(gr1,gr2)

def gpRow_multiply(gr1, gr2):
    if len(gr1) == 1 and len(gr2) != 1:
        gr1_temp = gr2
        gr2 = gr1
        gr1 = gr1_temp

    if len(gr2) == 1:
        for i in range(0,len(gr1)):
            if gr1[i][0] >= 0:
                gr1[i][1] *= gr2[0][1]
                for j in range(2, len(gr2[0])):
                    gr1[i][j] += gr2[0][j]
        return collapseGProws(gr1)
    else:
        raise RuntimeError('Posynomial multiplication should have already been performed')

def gpRow_divide(gr1, gr2):
    if len(gr2) == 1:
        for i in range(2, len(gr2[0])):
            gr2[0][i] *= -1
        gr2[0][1] = 1/gr2[0][1]
        return gpRow_multiply(gr1, gr2)
    else:
        denom_ix = -1*gr1[0][0] - 1
        for i in range(0, len(gr2)):
            gr2[i][0] = denom_ix
        return gr1+gr2


def checkObjectiveHessian_PD(gpRows):
    N_vars = len(gpRows[0])-2
    hessian = np.zeros([N_vars,N_vars])
    P = np.zeros([N_vars,N_vars])
    q = np.zeros([N_vars])
    r = 0.0
    onlyZeros = True
    for rw in gpRows:
        if rw[0] == 0:
            exponents = rw[2:]
            if any([vl not in [0.0,1.0,2.0] for vl in exponents]) or sum([abs(vl) for vl in exponents]) > 2:
                return [False,None,None,None]
            elif any([vl == 2.0 for vl in exponents]) :
                ix = exponents.index(2.0)
                hessian[ix,ix] = 2.0 * rw[1]
                P[ix,ix] = rw[1]
                onlyZeros = False
            elif sum([abs(vl) for vl in exponents]) == 2:
                # has 1 and 1
                firstIndex  = exponents.index(1.0)
                secondIndex = exponents.index(1.0, firstIndex + 1)
                hessian[firstIndex,secondIndex] = rw[1]
                hessian[secondIndex,firstIndex] = rw[1]
                P[firstIndex,secondIndex] = rw[1]/2.0
                P[secondIndex,firstIndex] = rw[1]/2.0
                onlyZeros = False
            elif sum([abs(vl) for vl in exponents]) == 1: 
                # has one 1
                ix = exponents.index(1.0)
                q[ix] = rw[1]
            else:
                # is a constant, exps=0
                r = rw[1]

        else:
            break

    if onlyZeros:
        return [False,None,None,None]

    eigenvalues, eigenvectors = np.linalg.eig(hessian)
    if min(eigenvalues) > 0:
        return [True,P,q,r]
    else:
        return [False,None,None,None]

def checkLinear(gpRows):
    N_vars = len(gpRows[0])-2
    ucis = np.unique([rw[0] for rw in gpRows])
    N_rows = len(ucis)

    A = np.zeros([N_rows,N_vars])
    b = np.zeros([N_rows])

    for rw in gpRows:
        uci = rw[0]
        exponents = rw[2:]
        if any([vl not in [0.0,1.0] for vl in exponents]) or sum([abs(vl) for vl in exponents]) > 1:
            return [False,None,None]
        elif sum([abs(vl) for vl in exponents]) == 1: 
            # has one 1
            ix = exponents.index(1.0)
            A[uci-ucis[0]][ix] = rw[1]
        else:
            # is a constant, exps=0
            b[uci-ucis[0]] = rw[1]

    return [True,A,b]


def unstructured_dict():
    return {"Linear_Program"   :[False, None, None], 
            "Quadratic_Program":[False, None, None],
            "Geometric_Program":[False, None, None], 
            "Signomial_Program":[False, None, None],} 

def convexity_detector(pyomo_component):
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

        if var_upper_bound is not None:
            proposedKey = vr.name + '_upperBound'
            if proposedKey in list(pyomo_component.__dict__.keys()):
                for i in range(0,10):
                    proposedKey = vr.name + '_upperBound_randomSeed_' + str(int(math.floor(random.random()*1e6)))
                    if proposedKey not in list(pyomo_component.__dict__.keys()):
                        break
                raise ValueError('Could not found a unique identifier for the upper bound on variable '+vr.name)

            setattr(pyomo_component, proposedKey, pyo.Constraint(expr = vr <= var_lower_bound))

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
    visitor = _ConvexityVisitor()

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
    if len(constraints) > 0:
        conCounter = 0
        operatorList = []
        for i in range(0, len(constraints)):
            con = constraints[i]
            for c in con.values():
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

    return structures

def solve_LP(structures):
    import cvxopt
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 1000
    cvxopt.printing.options['width'] = -1

    # structures = convexity_detector(m)
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

    for i in range(0,len(AG)):
        if operators[i] == '==':
            if A is None:
                A = [AG[i].tolist()]
                b = [bh[i]]
            else:
                A += [AG[i].tolist()]
                b += [bh[i]]
        else:
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

    return res


def solve_QP(structures):
    import cvxopt
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 1000
    cvxopt.printing.options['width'] = -1

    # structures = convexity_detector(m)

    P = cvxopt.matrix( structures['Quadratic_Program'][1][0] )
    q = cvxopt.matrix(structures['Quadratic_Program'][1][1])
    objective_shift = structures['Quadratic_Program'][1][2]
    AG = structures['Quadratic_Program'][1][3]
    bh = structures['Quadratic_Program'][1][4]
    operators = structures['Quadratic_Program'][2]

    A = None
    G = None
    b = None
    h = None

    if AG is not None:
        for i in range(0,len(AG)):
            if operators[i] == '==':
                if A is None:
                    A = [AG[i].tolist()]
                    b = [-1*bh[i]]
                else:
                    A += [AG[i].tolist()]
                    b += [-1*bh[i]]
            else:
                if G is None:
                    G = [AG[i].tolist()]
                    h = [-1*bh[i]]
                else:
                    G += [AG[i].tolist()]
                    h += [-1*bh[i]]

    if A is not None:
        A = cvxopt.matrix(A).T
    if b is not None:
        b = cvxopt.matrix(b)
    if G is not None:
        G = cvxopt.matrix(G).T
    if h is not None:
        h = cvxopt.matrix(h)

    res = cvxopt.solvers.qp(2.0*P,q,G,h,A,b)
    res['objective_shift'] = objective_shift 

    return res


def solve_GP(structures):

    # structures = convexity_detector(m)
    
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
    for i in range(0,len(K_with_equalities)):
        if K_with_equalities[i] > 1:
            slicer = slicer_prev + K_with_equalities[i]
            for ii in range(slicer_prev,slicer):
                GP_formulation.append(structures['Geometric_Program'][1][ii])
            slicer_prev = slicer
            K.append(K_with_equalities[i])
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
            else:
                slicer = slicer_prev + 1
                GP_formulation.append(structures['Geometric_Program'][1][slicer_prev])
                slicer_prev = slicer
                K.append(1)

    GP_formulation = cvxopt.matrix( GP_formulation )
    g = cvxopt.log( GP_formulation[1,:] ).T
    F = GP_formulation[2:,:].T

    res = cvxopt.solvers.gp(K, F, g)

    return res


def evaluate_posynomial(gpRows,x_star):
    N_vars = len(gpRows[0])-2
    ucis = np.unique([rw[0] for rw in gpRows])
    if len(ucis) != 1:
        raise ValueError('A non-posynomial object was detected')
    N_rows = len(ucis)
    subVals = np.zeros([len(gpRows),N_vars+1])
    for i in range(0,len(gpRows)):
        subVals[i][0] = gpRows[i][1]
        for j in range(0,N_vars):
            subVals[i][j+1] = x_star[j]**gpRows[i][j+2]
    val = 0.0
    for i in range(0,len(subVals)):
        val += np.prod(subVals[i])

    grad = np.zeros([N_vars])
    for i in range(0,N_vars):
        for j in range(0,len(subVals)):
            if subVals[j][i+1] == 0:
                subGrad = 0
            else:
                subGrad = np.prod(subVals[j])/subVals[j][i+1]
            subGrad *= gpRows[j][i+2] * x_star[i]**(gpRows[j][i+2] - 1)
            grad[i] += subGrad

    return val, grad

def monomial_approximation(gpRows,x_star):
    N_vars = len(gpRows[0])-2
    ucis = np.unique([rw[0] for rw in gpRows])
    if len(ucis) != 1:
        raise ValueError('A non-posynomial object was detected')

    f, dfdx = evaluate_posynomial(gpRows,x_star)

    lc = f
    a = [x_star[i]/f * dfdx[i] for i in range(0,N_vars)]
    for i in range(0,N_vars):
        lc *= (1/x_star[i])**a[i]

    return [[None, lc] + a]

def pccp_modification(gpRows,N_spcons,penalty_exponent=5.0):
    # split signomial equalities

    # Make the PCCP modification

    # slack variable slots
    slackStartIndex = len(gpRows[0])
    objectiveRows = []
    newGProws = []
    for i in range(0,len(gpRows)):
        gpRows[i] += [0.0]*N_spcons
        if gpRows[i][0] in [-1,0]:
            objectiveRows += gpRows[i]
        else:
            newGProws += gpRows[i]

    slackMonomial = [0.0]*len(gpRows[0])
    for i in range(0,len(slackMonomial)):
        if i >= slackStartIndex:
            slackMonomial[i] = penalty_exponent
    slackMonomial[0] = None
    slackMonomial[1] = 1.0

    # slacks in objective
    newObjectiveRows = gpRow_multiply(objectiveRows, [slackMonomial])
    gpRows = newObjectiveRows + newGProws
    
    # add s>=1
    currentCon = gpRows[-1][0]
    for i in range(0,N_spcons):
        currentCon += 1
        nextRow = [0.0]*len(gpRows[0])
        nextRow[0] = currentCon
        nextRow[1] = 1.0
        nextRow[i+slackStartIndex] = -1
        gpRows.append(nextRow)
        
    # add slacks to each signomial
    




def solve_SP(structures, m, reltol=1e-6, var_reltol = 1e-6, max_iter = 100, use_pccp = True, penalty_exponent=5.0):
    variableList = [ vr for vr in m.component_objects( pyo.Var, descend_into=True, active=True ) ]

    unwrappedVariables = []
    vrIdx = -1
    for i in range(0, len(variableList)):
        vr = variableList[i]
        vrIdx += 1
        if isinstance(vr, ScalarVar):
            unwrappedVariables.append(vr)
        elif isinstance(vr, IndexedVar):
            unwrappedVariables.append(vr)
            for sd in vr.index_set().data():
                vrIdx += 1
                unwrappedVariables.append(vr[sd])
        else:
            raise DeveloperError( 'Variable is not a variable.  Should not happen.  Contact developers' )

    x_star = [uv.value for uv in unwrappedVariables]

    spRows = structures['Signomial_Program'][1]
    operators = structures['Signomial_Program'][2]

    gpRows = []
    numerator_rows = {}
    numerator_row_indices = {}
    denominator_rows = {}

    conIxs,conCounts = np.unique([spRows[i][0] for i in range(0,len(spRows))],return_counts=True)

    N_spcons = 0
    nrix = 0
    for i in range(0,len(spRows)):
        rw = spRows[i]
        if -1*rw[0]-1 in conIxs and rw[0]>=0:
            if operators[rw[0]-1] == '==' and conCounts[conIxs.index(conIxs[rw[0]])]>1:
                # need to account for the fact that a signomial equality becomes a monomial equality
                if rw[0] not in list(numerator_rows.keys()):
                    numerator_rows[rw[0]] = []
                    numerator_row_indices[rw[0]] = [nrix]
                    gpRows.append(None)
                    nrix+=1
                    N_spcons += 2
                numerator_rows[rw[0]].append(rw)
            else:
                if rw[0] not in list(numerator_rows.keys()):
                    numerator_rows[rw[0]] = []
                    numerator_row_indices[rw[0]] = []
                numerator_rows[rw[0]].append(rw)
                numerator_row_indices[rw[0]].append(nrix)
                gpRows.append(None)
                nrix+=1
                N_spcons += 1
        elif rw[0]<0:
            if -1*(rw[0]+1) not in list(denominator_rows.keys()):
                denominator_rows[-1*(rw[0]+1)] = []  
            denominator_rows[-1*(rw[0]+1)].append(rw)
        else:
            gpRows.append(rw)
            nrix+=1

    if use_pccp:
        gpRows, = pccp_modification(gpRows,N_spcons,penalty_exponent=penalty_exponent)
    cody

    for itr in range(0,max_iter):
        for ky, vl in numerator_rows.items():
            ma = monomial_approximation(copy.deepcopy(denominator_rows[ky]),x_star)
            if operators[ky-1] == '==' and len(numerator_rows[ky])>1:
                ma_num = monomial_approximation(copy.deepcopy(numerator_rows[ky]),x_star)
                posy = gpRow_divide(ma_num, ma)
                gpRows[numerator_row_indices[ky][0]] = posy[0]
            else:
                posy = gpRow_divide(copy.deepcopy(numerator_rows[ky]), ma)
                for i in range(0,len(numerator_row_indices[ky])):
                    gpRows[numerator_row_indices[ky][i]] = posy[i]

        pack = {}
        pack['Linear_Program'] = [False]
        pack['Quadratic_Program'] = [False]
        pack['Geometric_Program'] = [True, gpRows, operators]
        
        res = solve_GP(pack)

        new_x_star = np.exp(res['x'])
        var_reltol_check = all([abs(new_x_star[i]-x_star[i])/new_x_star[i] <= var_reltol for i in range(0,len(x_star))])

        if var_reltol_check and reltol_check:
            break
        else:
            x_star = [ nxv[0] for nxv in new_x_star.tolist()]

    return res

def cvxopt_solve(m):
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 1000
    cvxopt.printing.options['width'] = -1

    structures = convexity_detector(m)
    # print(structures)
    if structures['Linear_Program'][0]:
        res = solve_LP(structures)
        res['problem_structure'] = 'linear_program'
    elif structures['Quadratic_Program'][0]:
        res = solve_QP(structures)
        res['problem_structure'] = 'quadratic_program'
    elif structures['Geometric_Program'][0]:
        res = solve_GP(structures)
        res['problem_structure'] = 'geometric_program'
    elif structures['Signomial_Program'][0]:
        res = solve_SP(structures,m)
        res['problem_structure'] = 'signomial_program'
    else:
        # print(structures)
        raise ValueError('Could not convert the formulation to a valid CVXOPT structure (LP,QP,GP,SP)')

    return res






