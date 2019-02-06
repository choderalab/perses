from __future__ import print_function
import numpy as np
import copy
import logging
import traceback
from openmmtools.alchemy import AlchemicalState

# make something hyperbolic or something to go from on to off to on
default_hybrid_functions = {
    'lambda_sterics_core' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_sterics_insert' : 'select(step(lambda-0.5), 1.0, 2*lambda)',
    'lambda_sterics_delete' : 'select(step(lambda-0.5), 2.0*(lambda - 0.5), 0.0)',
    'lambda_electrostatics_insert' : 'select(step(lambda-0.5),2.0*(lambda-0.5),0.0)',
    'lambda_electrostatics_delete' : 'select(step(lambda-0.5), 1.0, 2.0*lambda)',
    'lambda_bonds' : 'lambda',
    'lambda_angles' : 'lambda',
    'lambda_torsions' : 'lambda'
    }

python_hybrid_functions = {
    'lambda_sterics_core': lambda x: x,
    'lambda_electrostatics': lambda x: x,
    'lambda_sterics_insert': lambda x: 2.0*x if x< 0.5 else 1.0,
    'lambda_sterics_delete': lambda x: 1.0 if x < 0.5 else 2.0*(1-x),
    'lambda_electrostatics_insert': lambda x: 1.0 if x < 0.5 else 2.0*(1-x),
    'lambda_electrostatics_delete': lambda x: 2.0*x if x< 0.5 else 1.0,
    'lambda_bonds': lambda x: x,
    'lambda_angles': lambda x: x,
    'lambda_torsions': lambda x: x
}

python_reverse_functions = {
    'lambda_sterics_core': lambda x: (1-x),
    'lambda_electrostatics': lambda x: (1-x),
    'lambda_sterics_insert': lambda x: 2.0*(1-x) if x> 0.5 else 1.0,
    'lambda_sterics_delete': lambda x: 1.0 if x > 0.5 else 2.0*x,
    'lambda_electrostatics_insert': lambda x: 1.0 if x > 0.5 else 2.0*x,
    'lambda_electrostatics_delete': lambda x: 2.0*(1-x) if x> 0.5 else 1.0,
    'lambda_bonds': lambda x: (1-x),
    'lambda_angles': lambda x: (1-x),
    'lambda_torsions': lambda x: (1-x)
}

class RelativeAlchemicalState(AlchemicalState):
    """
    Relative AlchemicalState to handle all lambda parameters required for relative perturbations

    Attributes
    ----------
    lambda_sterics_core
    lambda_sterics_insert
    lambda_sterics_delete
    lambda_electrostatics_insert
    lambda_electrostatics_delete
    """

    class _LambdaParameter(AlchemicalState._LambdaParameter):
        pass 

    lambda_sterics_core = _LambdaParameter('lambda_sterics_core') 
    lambda_sterics_insert = _LambdaParameter('lambda_sterics_insert')
    lambda_sterics_delete = _LambdaParameter('lambda_sterics_delete')
    lambda_electrostatics_insert = _LambdaParameter('lambda_electrostatics_insert')
    lambda_electrostatics_delete = _LambdaParameter('lambda_electrostatics_delete')

    def set_alchemical_parameters(self, master_lambda):
       """Set each lambda value according to the python_hybrid_function protocol.
       The undefined parameters (i.e. those being set to None) remain
       undefined.
       Parameters
       ----------
       lambda_value : float
           The new value for all defined parameters.
       """
       for parameter_name in python_hybrid_functions:
           lambda_value = python_hybrid_functions[parameter_name](master_lambda)
           setattr(self, parameter_name, lambda_value)

