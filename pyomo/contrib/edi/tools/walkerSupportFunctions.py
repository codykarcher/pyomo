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

from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.core.base.param import _ParamData, ScalarParam, IndexedParam


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