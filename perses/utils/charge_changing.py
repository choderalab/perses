"""
utilities for charge-changing procedure
"""
from simtk import unit
from simtk.openmm import app
import simtk.unit as unit
import numpy as np

def modify_atom_classes(water_atoms, topology_proposal):
    """
    Modifies:
    - topology proposal._core_new_to_old_atom_map - add the ion(s) to neutralize
    - topology_proposal._new_environment_atoms - remove the ion(s) to neutralize
    - topology_proposal._old_environment_atoms - remove the ion(s) to neutralize

    Parameters
    ----------
    water_atoms : np.array(int)
        integers corresponding to particle indices to turn into ions
    topology_proposal : perses.rjmc.TopologyProposal
        topology_proposal to modify

    """
    for new_index in water_atoms:
        old_index = topology_proposal._new_to_old_atom_map[new_index]
        topology_proposal._core_new_to_old_atom_map[new_index] = old_index
        topology_proposal._new_environment_atoms.remove(new_index)
        topology_proposal._old_environment_atoms.remove(old_index)

def get_charge_difference(current_oemol, new_oemol):
    """
    return the charge of the old res - charge new res

    Parameters
    ----------
    current_resname : str
        three letter identifier for original residue
    new_resname : str
        three letter identifier for new residue

    Returns
    -------
    chargediff : int
        charge(old_res) - charge(old_res)
    """
    # retrieve the total charge
    old_charge = sum([atom.GetFormalCharge() for atom in current_oemol.GetAtoms()])
    new_charge = sum([atom.GetFormalCharge() for atom in new_oemol.GetAtoms()])
    total_chargediff = old_charge - new_charge
    assert type(total_chargediff) == int
    return total_chargediff

def get_water_indices(charge_diff,
                           new_positions,
                           new_topology,
                           radius=0.8):
    """
    Choose random water(s) (at least `radius` nm away from the protein) to turn into ion(s). Returns the atom indices of the water(s) (index w.r.t. new_topology)

    Parameters
    ----------
    charge_diff : int
        the charge difference between the old_system - new_system
    new_positions : np.ndarray(N, 3)
        positions (nm) of atoms corresponding to new_topology
    new_topology : openmm.Topology
        topology of new system
    radius : float, default 0.8
        minimum distance (in nm) that all candidate waters must be from 'protein atoms'

    Returns
    -------
    ion_indices : np.array(abs(charge_diff)*3)
        indices of water atoms to be turned into ions
    """

    import mdtraj as md
    from mdtraj.core.residue_names import _SOLVENT_TYPES

    # Create trajectory
    traj = md.Trajectory(new_positions[np.newaxis, ...], md.Topology.from_openmm(new_topology))

    # Define water atoms
    water_atoms = traj.topology.select("water")

    # Define solute atoms
    # TODO: Update this once we either (1) get solvent as a keyword into the MDTraj DSL, or (2) transition to MDAnalysis
    solvent_types = list(_SOLVENT_TYPES)
    solute_atoms = [atom.index for atom in traj.topology.atoms if atom.residue.name not in solvent_types]

    # Get water atoms within radius of protein
    neighboring_atoms = md.compute_neighbors(traj, radius, solute_atoms, haystack_indices=water_atoms)[0]

    # Get water atoms outside of radius of protein
    nonneighboring_residues = set([atom.residue.index for atom in traj.topology.atoms if (atom.index in water_atoms) and (atom.index not in neighboring_atoms)])
    assert len(nonneighboring_residues) > 0, "there are no available nonneighboring waters"
    # Choose N random nonneighboring waters, where N is determined based on the charge_diff
    choice_residues = np.random.choice(list(nonneighboring_residues), size=abs(charge_diff), replace=False)

    # Get the atom indices in the water(s)
    choice_indices = np.array([[atom.index for atom in traj.topology.residue(res).atoms] for res in choice_residues])

    return np.ndarray.flatten(choice_indices)

def get_ion_and_water_parameters(system, topology, positive_ion_name="NA", negative_ion_name="CL", water_name="HOH"):
    '''
    Get the charge, sigma, and epsilon for the positive and negative ions. Also get the charge of the water atoms.

    Parameters
    ----------
    system : simtk.openmm.System
        the system from which to retrieve parameters
    topology : app.Topology
        the topology corresponding to the above system from which to retrieve atom indices
    positive_ion_name : str, "NA"
        the residue name of each positive ion
    negative_ion_name : str, "CL"
        the residue name of each negative ion
    water_name : str, "HOH"
        the residue name of each water

    Returns
    -------
    particle_parameter_dict : dict
        parameter dict containing lookup parameters for water/ion nonbonded parameters.
        Keys contain {pos_charge, pos_sigma, pos_epsilon, neg_charge, neg_sigma, neg_epsilon, O_charge, H_charge}

    '''

    # Get the indices
    pos_index = None
    neg_index = None
    O_index = None
    H_index = None
    for atom in topology.atoms():
        if atom.residue.name == positive_ion_name and not pos_index:
            pos_index = atom.index
        elif atom.residue.name == negative_ion_name and not neg_index:
            neg_index = atom.index
        elif atom.residue.name == water_name and (not O_index or not H_index):
            if atom.name == 'O':
                O_index = atom.index
            elif atom.name == 'H1':
                H_index = atom.index
    assert pos_index is not None, f"Error occurred when trying to turn a water into an ion: No positive ions with residue name {positive_ion_name} found"
    assert neg_index is not None, f"Error occurred when trying to turn a water into an ion: No negative ions with residue name {negative_ion_name} found"
    assert O_index is not None, f"Error occurred when trying to turn a water into an ion: No O atoms with residue name {water_name} and atom name O found"
    assert H_index is not None, f"Error occurred when trying to turn a water into an ion: No water atoms with residue name {water_name} and atom name H1 found"

    # Get parameters from nonbonded force
    force_dict = {i.__class__.__name__: i for i in system.getForces()}
    if 'NonbondedForce' in [i for i in force_dict.keys()]:
        nbf = force_dict['NonbondedForce']
        pos_charge, pos_sigma, pos_epsilon = nbf.getParticleParameters(pos_index)
        neg_charge, neg_sigma, neg_epsilon = nbf.getParticleParameters(neg_index)
        O_charge, _, _ = nbf.getParticleParameters(O_index)
        H_charge, _, _ = nbf.getParticleParameters(H_index)

    particle_parameter_dict = {'pos_charge' : pos_charge,
                               'pos_sigma' : pos_sigma,
                               'pos_epsilon' : pos_epsilon,
                               'neg_charge' : neg_charge,
                               'neg_sigma' : neg_sigma,
                               'neg_epsilon' : neg_epsilon,
                               'O_charge' : O_charge,
                               'H_charge' : H_charge}
    return particle_parameter_dict

def transform_waters_into_ions(water_atoms, system, charge_diff, particle_parameter_dict):
    """
    given a system and an array of ints (corresponding to atoms to turn into ions), modify the nonbonded particle parameters in the system such that the Os are turned into the ion of interest and the charges of the Hs are zeroed.

    Parameters
    ----------
    water_atoms : np.array(int)
        integers corresponding to particle indices to neutralize
    system : simtk.openmm.System
        system to modify
    charge_diff : int
        the charge difference between the old_system - new_system
    particle_parameter_dict : dict
        parameter dict containing lookup parameters for water/ion nonbonded parameters.
        Keys contain {pos_charge, pos_sigma, pos_epsilon, neg_charge, neg_sigma, neg_epsilon, O_charge, H_charge}

    Returns
    -------
    modify system in place
    """
    _dict = particle_parameter_dict
    # Determine which ion to turn the water into
    if charge_diff < 0: # Turn water into Cl-
        ion_charge, ion_sigma, ion_epsilon = _dict['neg_charge'], _dict['neg_sigma'], _dict['neg_epsilon']
    elif charge_diff > 0: # Turn water into Na+
        ion_charge, ion_sigma, ion_epsilon = _dict['pos_charge'], _dict['pos_sigma'], _dict['pos_epsilon']

    # Scale the nonbonded terms of the water atoms
    force_dict = {i.__class__.__name__: i for i in system.getForces()}
    if 'NonbondedForce' in [i for i in force_dict.keys()]:
        nbf = force_dict['NonbondedForce']
        for idx in water_atoms:
            idx = int(idx)
            charge, sigma, epsilon = nbf.getParticleParameters(idx)
            if charge == _dict['O_charge']:
                nbf.setParticleParameters(idx, ion_charge, ion_sigma, ion_epsilon)
            elif charge == _dict['H_charge']:
                nbf.setParticleParameters(idx, charge*0.0, sigma, epsilon)
            else:
                raise Exception(f"Trying to modify an atom that is not part of a water residue. Atom index: {idx}")
