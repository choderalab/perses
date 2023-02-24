"""
Settings objects for the different protocols using gufe objects.

This module implements the objects that will be needed to run relative binding free
energy calculations using perses.
"""

from gufe.settings.models import ProtocolSettings
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
    # TODO: Add type hints
    class Config:
        arbitrary_types_allowed = True

    # Lambda settings
    lambda_functions = DEFAULT_ALCHEMICAL_FUNCTIONS

    # alchemical settings
    softcore_LJ_v2 = True
    interpolate_old_and_new_14s = False

    # NEQ integration settings
    timestep = 4.0 * unit.femtoseconds
    neq_splitting = "V R H O R V"
    eq_steps = 1000
    neq_steps = 1000

    # platform and serialization
    platform = 'CUDA'
    # TODO: We have different settings
    #  - which atoms should we save (mdtraj selection ex: "not water")
    #  - How often we want to store positions (set that 100 for example)
    #  - works store/save frequency. It's not much more data so can be more frequent.
    save_frequency = 100
    traj_save_frequency = 100
    work_save_frequency = 25

    # Number of cycles to run
    num_replicates: int = 1

    def _gufe_tokenize(self):
        return _serialize_pydantic(self)
