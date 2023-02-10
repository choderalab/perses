# fixtures for chemicalcomponents and chemicalsystems to test protocols with
import gufe
import pytest
import importlib.resources
from rdkit import Chem
from gufe.mapping import LigandAtomMapping


@pytest.fixture
def benzene_modifications():
    with importlib.resources.path('gufe.tests.data',
                                  'benzene_modifications.sdf') as f:
        supp = Chem.SDMolSupplier(str(f), removeHs=False)

        mols = list(supp)

    return {m.GetProp('_Name'): m for m in mols}


# Components fixtures

@pytest.fixture
def solvent_comp():
    yield gufe.SolventComponent(positive_ion="Na", negative_ion="Cl")

@pytest.fixture
def benzene(benzene_modifications):
    return gufe.SmallMoleculeComponent(benzene_modifications["benzene"])


@pytest.fixture
def toluene(benzene_modifications):
    return gufe.SmallMoleculeComponent(benzene_modifications["toluene"])


# Systems fixtures

@pytest.fixture
def benzene_vacuum_system(benzene):
    return gufe.ChemicalSystem(
        {"ligand": benzene}
    )

@pytest.fixture
def benzene_solvent_system(benzene, solvent_comp):
    return gufe.ChemicalSystem(
        {"ligand": benzene, "solvent": solvent_comp}
    )

@pytest.fixture
def toluene_vacuum_system(toluene):
    return gufe.ChemicalSystem(
        {"ligand": toluene}
    )

@pytest.fixture
def toluene_solvent_system(toluene, solvent_comp):
    return gufe.ChemicalSystem(
        {"ligand": toluene, "solvent": solvent_comp}
    )


# Settings fixtures

@pytest.fixture
def short_settings():
    from openff.units import unit
    # Build Settings gufe object
    from gufe.settings.models import (
        Settings,
    )
    from perses.protocols.settings import NonEqCyclingSettings

    settings = Settings.get_defaults()
    settings.thermo_settings.temperature = 300 * unit.kelvin
    settings.protocol_settings = NonEqCyclingSettings(eq_steps=25000, neq_steps=25000, save_frequency=250)

    return settings


@pytest.fixture
def short_settings_multiple_cycles():
    from openff.units import unit
    # Build Settings gufe object
    from gufe.settings.models import (
        Settings,
    )
    from perses.protocols.settings import NonEqCyclingSettings

    settings = Settings.get_defaults()
    settings.thermo_settings.temperature = 300 * unit.kelvin
    # TODO: add validation within settings that save_freq is divisor of total steps
    settings.protocol_settings = NonEqCyclingSettings(eq_steps=25000, neq_steps=25000, save_frequency=250,
                                                      num_replicates=5, platform="CPU")

    return settings

@pytest.fixture
def production_settings(short_settings):
    from perses.protocols.settings import NonEqCyclingSettings
    settings = short_settings
    settings.protocol_settings = NonEqCyclingSettings(eq_steps=250000, neq_steps=250000, save_frequency=2000)

    return settings


# Mappings fixtures

@pytest.fixture
def mapping_benzene_toluene(benzene, toluene):
    """Mapping from toluene to benzene"""
    mapping_toluene_to_benzene = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8, 13: 9, 14: 11}
    mapping_obj = LigandAtomMapping(
        componentA=benzene,
        componentB=toluene,
        componentA_to_componentB=mapping_toluene_to_benzene,
    )
    return mapping_obj


@pytest.fixture
def broken_mapping(benzene, toluene):
    """Broken mapping"""
    # Mapping that doesn't make sense for benzene and toluene
    broken_mapping = {40: 20, 5: 1, 6: 2, 7: 3, 38: 4, 9: 5, 10: 6, 191: 7, 12: 8, 13: 99, 14: 11}
    broken_mapping_obj = LigandAtomMapping(
        componentA=benzene,
        componentB=toluene,
        componentA_to_componentB=broken_mapping,
    )
    return broken_mapping_obj