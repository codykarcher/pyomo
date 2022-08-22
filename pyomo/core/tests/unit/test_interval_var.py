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


import pyomo.common.unittest as unittest
from pyomo.core.base.interval_var import IntervalVar
from pyomo.environ import (
    ConcreteModel, BooleanVar, Integers, Var, value)

class TestScalarIntervalVar(unittest.TestCase):
    def test_initialize_with_no_data(self):
        m = ConcreteModel()
        m.i = IntervalVar()

        self.assertIsInstance(m.i.start_time, Var)
        self.assertEqual(m.i.start_time.domain, Integers)
        self.assertIsNone(m.i.start_time.lower)
        self.assertIsNone(m.i.start_time.upper)

        self.assertIsInstance(m.i.end_time, Var)
        self.assertEqual(m.i.end_time.domain, Integers)
        self.assertIsNone(m.i.end_time.lower)
        self.assertIsNone(m.i.end_time.upper)

        self.assertIsInstance(m.i.length, Var)
        self.assertEqual(m.i.length.domain, Integers)
        self.assertIsNone(m.i.length.lower)
        self.assertIsNone(m.i.length.upper)

        self.assertIsInstance(m.i.is_present, BooleanVar)

    def test_start_and_end_bounds(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(0,5))

        self.assertEqual(m.i.start_time.lower, 0)
        self.assertEqual(m.i.start_time.upper, 5)

        m.i.end_time.bounds = (12, 14)

        self.assertEqual(m.i.end_time.lower, 12)
        self.assertEqual(m.i.end_time.upper, 14)

    def test_constant_length_and_start(self):
        m = ConcreteModel()
        m.i = IntervalVar(length=7, start=3)

        self.assertEqual(m.i.length.lower, 7)
        self.assertEqual(m.i.length.upper, 7)

        self.assertEqual(m.i.start_time.lower, 3)
        self.assertEqual(m.i.start_time.upper, 3)

    def test_non_optional(self):
        m = ConcreteModel()
        m.i = IntervalVar(length=2, end=(4,9), optional=False)

        self.assertEqual(value(m.i.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i.optional)

        # Should also be true by default
        m.i2 = IntervalVar()
        
        self.assertEqual(value(m.i2.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i2.optional)

    def test_optional(self):
        m = ConcreteModel()
        m.i = IntervalVar(optional=True)

        self.assertFalse(m.i.is_present.fixed)
        self.assertTrue(m.i.optional)

class TestIndexedIntervalVar(unittest.TestCase):
    def test_initialize_with_no_data(self):
        m = ConcreteModel()

        m.i = IntervalVar([1, 2])

        for j in [1, 2]:
            self.assertIsInstance(m.i[j].start_time, Var)
            self.assertEqual(m.i[j].start_time.domain, Integers)
            self.assertIsNone(m.i[j].start_time.lower)
            self.assertIsNone(m.i[j].start_time.upper)

            self.assertIsInstance(m.i[j].end_time, Var)
            self.assertEqual(m.i[j].end_time.domain, Integers)
            self.assertIsNone(m.i[j].end_time.lower)
            self.assertIsNone(m.i[j].end_time.upper)

            self.assertIsInstance(m.i[j].length, Var)
            self.assertEqual(m.i[j].length.domain, Integers)
            self.assertIsNone(m.i[j].length.lower)
            self.assertIsNone(m.i[j].length.upper)

            self.assertIsInstance(m.i[j].is_present, BooleanVar)
        
    def test_constant_length(self):
        m = ConcreteModel()
        m.i = IntervalVar(['a', 'b'], length=45)
        
        for j in ['a', 'b']:
            self.assertEqual(m.i[j].length.lower, 45)
            self.assertEqual(m.i[j].length.upper, 45)

    def test_rule_based_start(self):
        m = ConcreteModel()
        def start_rule(m, i):
            return (1 - i, 13 + i)
        m.act = IntervalVar([1, 2, 3], start=start_rule, length=4)

        for i in [1, 2, 3]:
            self.assertEqual(m.act[i].start_time.lower, 1 - i)
            self.assertEqual(m.act[i].start_time.upper, 13 + i)

            self.assertEqual(m.act[i].length.lower, 4)
            self.assertEqual(m.act[i].length.upper, 4)

            self.assertFalse(m.act[i].optional)
            self.assertTrue(m.act[i].is_present.fixed)
            self.assertEqual(value(m.act[i].is_present), True)

    def test_optional(self):
        m = ConcreteModel()
        m.act = IntervalVar([1, 2], end=[0, 10], optional=True)

        for i in [1, 2]:
            self.assertTrue(m.act[i].optional)
            self.assertFalse(m.act[i].is_present.fixed)

            self.assertEqual(m.act[i].end_time.lower, 0)
            self.assertEqual(m.act[i].end_time.upper, 10)

    def test_optional_rule(self):
        m = ConcreteModel()
        m.idx = Set(initialize=[(4, 2), (5, 2)], dimen=2)
        def optional_rule(m, i, j):
            return i % j == 0
        m.act = IntervalVar(m.idx, optional=optional_rule)
        # TODO
