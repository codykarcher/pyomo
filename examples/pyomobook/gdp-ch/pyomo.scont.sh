#!/bin/sh

# @cmd:
pyomo solve scont.py --transform gdp.bigm --solver=glpk
# @:cmd
python verify_scont.py results.yml
rm results.yml
