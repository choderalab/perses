"""
Settings objects for the different protocols using gufe objects.

This module implements the objects that will be needed to run relative binding free
energy calculations using perses.
"""

from gufe.settings.models import ProtocolSettings, ThermoSettings
from openff.units import unit
from perses.protocols.utils import _serialize_pydantic

# Default settings for the lambda functions
x = 'lambda'
DEFAULT_ALCHEMICAL_FUNCTIONS = {
    'lambda_sterics_core': x,
    'lambda_electrostatics_core': x,
    'lambda_sterics_insert': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    'lambda_sterics_delete': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    'lambda_electrostatics_insert': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    'lambda_electrostatics_delete': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    'lambda_bonds': x,
    'lambda_angles': x,
    'lambda_torsions': x
}


class AlchemicalSettings(ProtocolSettings):
    """Settings for the alchemical protocol

    This describes the lambda schedule and the creation of the
    hybrid system.

    Attributes
    ----------
    lambda_functions : dict of strings, optional
      key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
      string that depends on the variable "lambda".
      If not specified, default alchemical functions will be used specified in DEFAULT_ALCHEMICAL_FUNCTIONS.
    softcore_LJ_v2 : bool
      Whether to use the LJ softcore function as defined by
      Gapsys et al. JCTC 2012 Default True.
    interpolate_old_and_new_14s : bool
      Whether to turn off interactions for new exceptions (not just 1,4s)
      at lambda 0 and old exceptions at lambda 1. If False they are present
      in the nonbonded force. Default False.
    phase : str
      The phase of the calculation. Default 'vacuum'.
    """
    # Lambda settings
    lambda_functions = DEFAULT_ALCHEMICAL_FUNCTIONS
    # lambda_windows = 11
    # alchemical settings
    softcore_LJ_v2 = True
    interpolate_old_and_new_14s = False
    phase = 'vacuum'


class ForceFieldSettings(ProtocolSettings):
    """Settings for the force field

    This describes the force field to use for the system.

    Attributes
    ----------
    forcefield_files : list of strings
        List of force field files to use.
    small_molecule_forcefield : str
        The name of the force field to use for small molecules. Default 'openff-2.0.0'
    """
    forcefield_files = [
            "amber/ff14SB.xml",
            "amber/tip3p_standard.xml",
            "amber/tip3p_HFE_multivalent.xml",
            "amber/phosaa10.xml",
        ]
    small_molecule_forcefield = 'openff-1.0.0'


class IntegratorSettings(ProtocolSettings):
    """Settings for the integrator.

    This describes the integrator parameters to use for the simulation.

    Attributes
    ----------
    timestep : float
        The timestep to use in the integrator. Default 4.0 * unit.femtoseconds.
    """
    timestep = 4.0 * unit.femtoseconds
    neq_splitting = "V R H O R V"
    eq_steps = 1000
    neq_steps = 100


class ThermodynamicSettings(ThermoSettings):
    """Settings for the thermodynamic state.

    This describes the thermodynamic state to use for the simulation.

    Attributes
    ----------
    temperature : float
        The temperature to use in the thermodynamic state. Default 300.0 * unit.kelvin.
    """
    temperature = 300.0 * unit.kelvin


class MiscellaneousSettings(ProtocolSettings):
    """Settings for the miscelaneous parameters.

    This describes the miscelaneous parameters to use for the simulation.

    Attributes
    ----------
    platform : str
        The name of the platform to use. Default 'CUDA'.
    save_frequency : int
        The frequency at which to save the simulation data. Default 100.
    """
    platform = 'CUDA'
    save_frequency = 100


class NonEqCyclingSettings(ProtocolSettings):
    """
    Settings for the relative free energy setup protocol.

    Attributes
    ----------
    ligand_input : str
        The path to the ligand input file.
    ligand_index : int
        The index of the ligand in the ligand input file.
    solvent_padding : float
        The amount of padding to add to the ligand in nanometers.
    forcefield : ForceFieldSettings
        The force field settings to use.
    alchemical : AlchemicalSettings
        The alchemical settings to use.
    """
    class Config:
        arbitrary_types_allowed = True

    alchemical_settings = AlchemicalSettings()
    forcefield_settings = ForceFieldSettings()
    integrator_settings = IntegratorSettings()
    thermodynamic_settings = ThermodynamicSettings()
    miscellaneous_settings = MiscellaneousSettings()
