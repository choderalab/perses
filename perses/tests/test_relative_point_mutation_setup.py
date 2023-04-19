import pytest
import os

def test_PointMutationExecutor():
    """
    Check that a PointMutationExecutor can be instantiated properly for ALA->ASP dipeptide in solvent and that a
    HybridTopologyFactory object can be generated.
    Also check that counterions are added.

    """
    from pkg_resources import resource_filename
    from simtk import unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor

    pdb_filename = resource_filename("perses", "data/ala_vacuum.pdb")
    solvent_delivery = PointMutationExecutor(
        pdb_filename,
        "1",
        "2",
        "ASP",
        ionic_strength=0.15 * unit.molar,
        flatten_torsions=True,
        flatten_exceptions=True,
        conduct_endstate_validation=False,
    )
    htf = solvent_delivery.get_apo_htf()

    # If there is a counterion, there should be water atoms in the core atom class
    solvent_atoms = set(htf.hybrid_topology.select('water'))
    assert len(solvent_atoms.intersection(htf._atom_classes['core_atoms'])) != 0, "There are no water atoms in the core atom " \
                                                                           "class, which may mean that the counterion was not introduced"

@pytest.mark.skipif(os.getenv("OPENMM", default="7.7").upper() in ["8.0", "DEV"], reason="FastMath is BadMath")
def test_PointMutationExecutor_endstate_validation():
    """
    Check that HybridTopologyFactory, RepartitionedHybridTopologyFactory, and RESTCapableHybridTopologyFactory objects
    can be generated for ALA->ASP dipeptide in solvent and conduct endstate validation to sure that the endstate energies
    match those of the real systems.

    """
    from pkg_resources import resource_filename
    from simtk import unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor

    pdb_filename = resource_filename("perses", "data/ala_vacuum.pdb")
    PointMutationExecutor(
        pdb_filename,
        "1",
        "2",
        "ASP",
        ionic_strength=0.15 * unit.molar,
        flatten_torsions=False,
        flatten_exceptions=False,
        conduct_endstate_validation=True,
        generate_unmodified_hybrid_topology_factory=True,
        generate_repartitioned_hybrid_topology_factory=True,
        generate_rest_capable_hybrid_topology_factory=True
    )


def test_PointMutationExecutor_solvated():
    """
    Check that a PointMutationExecutor can be instantiated properly for ALA->ASP dipeptide in solvent when the input PDB
    is solvated.

    """
    import os
    import tempfile
    from openmm import app, unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from perses.tests.test_topology_proposal import generate_atp

    with tempfile.TemporaryDirectory() as temp_dir:
        ala, system_generator = generate_atp(phase='solvent')
        app.PDBFile.writeFile(ala.topology, ala.positions, open(os.path.join(temp_dir, "ala_solvated.pdb"), "w"), keepIds=True)

        PointMutationExecutor(
            os.path.join(temp_dir, "ala_solvated.pdb"),
            "1",
            "2",
            "ASP",
            is_solvated=True,
            flatten_torsions=False,
            flatten_exceptions=False,
            conduct_endstate_validation=False
        )


def test_PointMutationExecutor_without_counterion():
    """
    Check that a PointMutationExecutor can be instantiated properly for ALA->ASP dipeptide in solvent without a counterion.

    """
    from pkg_resources import resource_filename
    from simtk import unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor

    pdb_filename = resource_filename("perses", "data/ala_vacuum.pdb")
    solvent_delivery = PointMutationExecutor(
        pdb_filename,
        "1",
        "2",
        "ASP",
        ionic_strength=0.15 * unit.molar,
        flatten_torsions=True,
        flatten_exceptions=True,
        conduct_endstate_validation=False,
        transform_waters_into_ions_for_charge_changes=False
    )
    htf = solvent_delivery.get_apo_htf()

    # If there is no counterion, there should be no water atoms in the core atom class
    solvent_atoms = set(htf.hybrid_topology.select('water'))
    assert len(solvent_atoms.intersection(htf._atom_classes['core_atoms'])) == 0, "There are water atoms in the core atom " \
                                                                                  "class, which may mean that a counterion is being introduced"

