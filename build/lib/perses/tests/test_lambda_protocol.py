###########################################
# IMPORTS
###########################################
from perses.annihilation.lambda_protocol import *
from unittest import skipIf
from nose.tools import raises
import os

istravis = os.environ.get('TRAVIS', None) == 'true'

#############################################
# TESTS
#############################################

def test_lambda_protocol():
    """

    Tests LambdaProtocol, ensures that it can be instantiated with defaults, and that it fails if disallowed functions are tried

    """

    # check that it's possible to instantiate a LambdaProtocol for all the default types
    for protocol in ['default','namd','quarters']:
        lp = LambdaProtocol(functions=protocol)

    # check that if we give an incomplete set of parameters it will add in the missing terms
    missing_functions = {'lambda_sterics_delete': lambda x : x}
    lp = LambdaProtocol(functions=missing_functions)
    assert (len(missing_functions) == 1)
    assert(len(lp.get_functions()) == 9)

@raises(AssertionError)
def test_lambda_protocol_failure_ends():
    bad_function = {'lambda_sterics_delete': lambda x : -x}
    lp = LambdaProtocol(functions=bad_function)

@raises(AssertionError)
def test_lambda_protocol_naked_charges():
    naked_charge_functions = {'lambda_sterics_insert':
                  lambda x: 0.0 if x < 0.5 else 2.0 * (x - 0.5),
                  'lambda_electrostatics_insert':
                  lambda x: 2.0 * x if x < 0.5 else 1.0}
    lp = LambdaProtocol(functions=naked_charge_functions)
