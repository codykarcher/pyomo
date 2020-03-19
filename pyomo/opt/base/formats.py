#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# The formats that are supported by Pyomo
#
__all__ = ['ProblemFormat', 'ResultsFormat', 'guess_format']

try:
    from enum import Enum
except:
    from enum34 import Enum

#
# pyomo - A pyomo.core.PyomoModel object, or a *.py file that defines such an object
# cpxlp - A CPLEX LP file
# nl - AMPL *.nl file
# mps - A free-format MPS file
# mod - AMPL *.mod file
# lpxlp - A LPSolve LP file
# osil - An XML file defined by the COIN-OR OS project: Instance
# FuncDesigner - A FuncDesigner problem
# colin - A COLIN shell command
# colin_optproblem - A Python object that inherits from
#                   pyomo.opt.colin.OptProblem (this can wrap a COLIN shell
#                   command, or provide a runtime optimization problem)
# bar - A Baron input file
# gams - A GAMS input file
#
class ProblemFormat(Enum):
    colin=1
    pyomo=2
    cpxlp=3
    nl=4
    mps=5
    mod=6
    lpxlp=7
    osil=8
    colin_optproblem=9
    FuncDesigner=10
    bar=11
    gams=12

#
# osrl - osrl XML file defined by the COIN-OR OS project: Result
# results - A Pyomo results object  (reader define by solver class)
# sol - AMPL *.sol file
# soln - A solver-specific solution file  (reader define by solver class)
# yaml - A Pyomo results file in YAML format
# json - A Pyomo results file in JSON format
#
class ResultsFormat(Enum):
    osrl=1
    results=2
    sol=3
    soln=4
    yaml=5
    json=6


def guess_format(filename):
    formats = {}
    formats['py']=ProblemFormat.pyomo
    formats['nl']=ProblemFormat.nl
    formats['bar']=ProblemFormat.bar
    formats['mps']=ProblemFormat.mps
    formats['mod']=ProblemFormat.mod
    formats['lp']=ProblemFormat.cpxlp
    formats['osil']=ProblemFormat.osil
    formats['gms']=ProblemFormat.gams
    formats['gams']=ProblemFormat.gams

    formats['sol']=ResultsFormat.sol
    formats['osrl']=ResultsFormat.osrl
    formats['soln']=ResultsFormat.soln
    formats['yml']=ResultsFormat.yaml
    formats['yaml']=ResultsFormat.yaml
    formats['jsn']=ResultsFormat.json
    formats['json']=ResultsFormat.json
    formats['results']=ResultsFormat.yaml
    if filename:
        return formats.get(filename.split('.')[-1].strip(), None)
    else:
        return None
