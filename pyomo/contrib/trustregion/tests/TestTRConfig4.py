#!/usr/python/env python

import pyutilib.th as unittest

from pyutilib.misc.config import ConfigBlock, ConfigValue, ConfigList
from pyomo.common.config import ( 
    PositiveInt, PositiveFloat, NonNegativeFloat, In)
from pyomo.core import Var, value

from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


class TestTrustRegionConfigBlock(unittest.TestCase):
    def setUp(self):
        
        m = ConcreteModel()
        m.z = Var(range(3), domain=Reals, initialize=2.)
        m.x = Var(range(2), initialize=2.)
        m.x[1] = 1.0
        
        def blackbox(a,b):
            return sin(a-b)
        self.bb = ExternalFunction(blackbox)

        m.obj = Objective(
            expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
                + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
            )
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + self.bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
        m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))

        self.m = m.clone()


    def try_solve(self,**kwds):
        '''
        Wrap the solver call in a try block. It should complete without exception. However
        if it does, at least we can check the values of the trust radius that are being
        used by the algorithm.
        '''
        status = True
        try:
            self.optTRF.solve(self.m, [self.bb], **kwds)
        except Exception as e:
            print('error calling optTRF.solve: %s' % str(e)) 
            status = False
        return status


    def test4(self):

        # Initialized with 1.0  
        self.optTRF = SolverFactory('trustregion')
        self.assertEqual(self.optTRF.config.trust_radius, 1.0)

        # Set persistent value to 4.0;
        self.optTRF.config.trust_radius = 4.0
        self.assertEqual(self.optTRF.config.trust_radius, 4.0)
       
        # Set local to 2.0; persistent should still be 4.0
        solve_status = self.try_solve(trust_radius=2.0)
        self.assertTrue(solve_status)
        self.assertEqual(self.optTRF.config.trust_radius, 4.0)
        self.assertEqual(self.optTRF._local_config.trust_radius, 2.0)
    

if __name__ =='__main__':
    unittest.main()
