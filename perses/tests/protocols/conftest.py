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
    from perses.protocols import NonEquilibriumCyclingProtocol
    
    settings = NonEquilibriumCyclingProtocol.default_settings()

    settings.thermo_settings.temperature = 300 * unit.kelvin
    settings.eq_steps=25000
    settings.neq_steps=25000
    settings.traj_save_frequency=250
    settings.platform="CPU"

    return settings


@pytest.fixture
def short_settings_gpu(short_settings):
    settings = short_settings.copy(deep=True)
    settings.platform="CUDA"

    return settings


@pytest.fixture
def short_settings_multiple_cycles():
    from openff.units import unit
    from perses.protocols import NonEquilibriumCyclingProtocol

    settings = NonEquilibriumCyclingProtocol.default_settings()

    settings.thermo_settings.temperature = 300 * unit.kelvin
    settings.eq_steps = 1000
    settings.neq_steps = 1000
    settings.traj_save_frequency=250
    settings.work_save_frequency=50
    settings.num_replicates=5
    settings.platform="CPU"

    return settings


@pytest.fixture
def short_settings_multiple_cycles_gpu(short_settings_multiple_cycles):
    settings = short_settings_multiple_cycles.copy(deep=True)
    settings.platform="CUDA"

    return settings


@pytest.fixture
def production_settings(short_settings):
    settings = short_settings.copy(deep=True)

    settings.eq_steps=250000
    settings.neq_steps=250000
    settings.save_frequency=2000

    return settings


# Mappings fixtures

@pytest.fixture
def mapping_benzene_toluene(benzene, toluene):
    """Mapping from toluene to benzene"""
    mapping_toluene_to_benzene = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 11: 14}
    mapping_obj = LigandAtomMapping(
        componentA=benzene,
        componentB=toluene,
        componentA_to_componentB=mapping_toluene_to_benzene,
    )
    return mapping_obj

@pytest.fixture
def mapping_toluene_toluene(toluene):
    """Mapping from toluene to toluene"""
    mapping_toluene_to_toluene = {i: i for i in range(len(toluene.to_rdkit().GetAtoms()))}
    mapping_obj = LigandAtomMapping(
        componentA=toluene,
        componentB=toluene,
        componentA_to_componentB=mapping_toluene_to_toluene,
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
