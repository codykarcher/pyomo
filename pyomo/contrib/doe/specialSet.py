#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation 
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners: 
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC., 
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,  
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin, 
#  University of Toledo, West Virginia University, et al. All rights reserved.
# 
#  NOTICE. This Software was developed under funding from the 
#  U.S. Department of Energy and the U.S. Government consequently retains 
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#  worldwide license in the Software to reproduce, distribute copies to the 
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________
import itertools

class SpecialSet: 
    def __init__(self):
        """This class defines variable names with provided names and indexes.
        """
        self.special_set = []

    def specify(self, self_define_res):
        """
        Used for user to provide defined string names

        Parameter
        ---------
        self_define_res: a ``list`` of ``string``, containing the measurement variable names with indexs, 
            for e.g. "C['CA', 23, 0]". 
            If this is defined, no need to define ``measurement_var``, ``extra_idx``, ``time_idx``.
        """
        self.special_set = self_define_res

    def add_elements(self, var_name, extra_index=None, time_index=None):
        """
        Used for generating string names with indexes. 

        Parameter 
        ---------
        var_name: a ``list`` of measurement var names 
        extra_index: a ``list`` containing extra indexes except for time indexes 
            if default (None), no extra indexes needed for all var in var_name
            if it is a nested list, it is a ``list`` of ``list`` of ``list``, 
            they are different multiple sets of indexes for different var_name
            for e.g., extra_index[0] are all indexes for var_name[0], extra_index[0][0] are the first index for var_name[0]
        time_index: a ``list`` containing time indexes
            default choice is None, means this is a model parameter 
            if it is an algebraic variable, time index should be set up to [0]
            if it is a nested list, it is a ``list`` of ``lists``, they are different time set for different var in var_name
        """

        for i, n in enumerate(var_name):
            name_data = str(n)

            # first combine all indexes into a list 
            all_index_list = [] # contains all index lists
            if extra_index:
                for index_list in extra_index[i]: 
                    all_index_list.append(index_list)
            if time_index:
                all_index_list.append(time_index[i])

            # all idnex list for one variable, such as ["CA", 10, 1]
            all_index_for_var = list(itertools.product(*all_index_list))

            for lst in all_index_for_var:
                name1 = name_data+"["
                for i, idx in enumerate(lst):
                    name1 += str(idx)

                    # if i is the last index, close the []. if not, add a "," for the next index. 
                    if i==len(lst)-1:
                        name1 += "]"
                    else:
                        name1 += ","

                self.special_set.append(name1)
        