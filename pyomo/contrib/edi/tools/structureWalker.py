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
if numpy_available:
    import numpy as np

from pyomo.contrib.edi.tools.walkerSupportFunctions import (
    unarySignomial,
    no_structure_dict,
    monomial_multiplication,
    signomial_multiplication,
    signomial_fraction_multiplication,
    signomial_power_evaluation,
    # processMonomial,
)

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


class _StructureVisitor(StreamBasedExpressionVisitor):
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
                'Structure walker encountered an error when processing type %s, contact the developers'
                % (node.__class__)
            )
