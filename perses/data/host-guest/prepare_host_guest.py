import mdtraj as md
import numpy as np
import simtk.openmm as openmm
import openeye.oechem as oechem
import openeye.oeiupac as oeiupac
import tempfile
import os
import openeye.oeomega as oeomega

#used SAMPL6 sampling challenge inputs
#git revision 93f642ef9e2c478985cd14f3084d899ae3ee1c6d
host_guest_path = "/Users/grinawap/SAMPL6/host_guest/SAMPLing/CB8-G3-0/PDB/complex.pdb"
host_vacuum_path = "/Users/grinawap/PycharmProjects/perses/perses/data/host-guest/host.mol2"
guests_vacuum_path = "/Users/grinawap/PycharmProjects/perses/perses/data/host-guest/guests.mol2"

#use this molecule as the other end of the perturbation
alternate_guest = '(5-methoxy-4-quinolyl)-(5-vinylquinuclidin-1-ium-2-yl)methanol'

def extract_molecule(complex_pdb_path: str, resname: str) -> md.Trajectory:
    """
    This method extracts the host from the host-guest solvated complex

    Parameters
    ----------
    complex_pdb_path : str
        The path to the PDB complex
    resname : str
        The name of the residue to save
    
    Returns
    -------
    subset_traj : md.Trajectory
        The subsetted trajectory of interest.
    """
    traj = md.load(complex_pdb_path)

    #Extract the specified resname indices
    selected_indices = traj.top.select("resname=='{}'".format(resname))

    #Subset the topology to contain just the selected residue
    subset_topology = traj.top.subset(selected_indices)

    #subset the positions
    subset_positions = traj.xyz[0, selected_indices]
    
    #create a new Trajectory object with just the selected residue
    subset_traj = md.Trajectory(subset_positions, subset_topology)

    return subset_traj
    
def convert_mdtraj_to_oemol(traj: md.Trajectory) -> oechem.OEMol:
    """
    This method converts an mdtraj Trajectory to an OEMol via saving as a PDBfile
    and reading in with OpenEye. Although this seems hacky, it seems less error-prone
    than trying to manually construct the OEMol.

    Parameters
    ----------
    mdtraj: md.Trajectory
        The trajectory to turn into an OEMol
    
    Returns
    -------
    mol : oechem.OEMol
        The trajectory represented as an OEMol
    """
    #create a temporary file with a PDB suffix and save with MDTraj
    pdb_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    traj.save(pdb_file.name)
    pdb_file.close()
    
    #Now use the openeye oemolistream to read in this file as an OEMol:
    ifs = oechem.oemolistream()
    ifs.open(pdb_file.name)
    ifs.SetFormat(oechem.OEFormat_PDB)
    
    mol = oechem.OEMol()
    oechem.OEReadMolecule(ifs, mol)
    
    #close the stream and delete the temporary pdb file
    ifs.close()
    os.unlink(pdb_file.name)

    return mol

def iupac_to_oemol(iupac_name: str) -> oechem.OEMol:
    """
    Convert an IUPAC name to an OEMol with openeye

    Arguments
    ---------
    iupac_name : str
        The name of the molecule
    
    Returns
    -------
    mol : oechem.OEMol
        The OEMol corresponding to the IUPAC name with all hydrogens
    """
    mol = oechem.OEMol()
    oeiupac.OEParseIUPACName(mol, iupac_name)
    oechem.OEAddExplicitHydrogens(mol)

    #generate conformers:
    omega = oeomega.OEOmega()
    omega.SetStrictStereo(False)
    omega.SetMaxConfs(1)
    omega(mol)

    return mol

def write_guest_molecules(old_molecule: oechem.OEMol, new_molecule: oechem.OEMol, filepath: str):
    """
    Write out a file containing the guest molecules with the "old molecule" first.
    
    Parameters
    ----------
    old_molecule : oechem.OEMol
        The molecule to be written first. It should have coordinates placing it within the guest
    new_molecule : oechem.OEMol
        The molecule to be written next. It does not need to have coordinates, as these will be generated
    filepath : str
        The path to the file that is being written
    """
    ostream = oechem.oemolostream()
    ostream.open(filepath)

    #set molecule names to avoid template name collision:
    old_molecule.SetTitle("oldmol")
    new_molecule.SetTitle("newmol")

    #set tripos atom names
    oechem.OETriposAtomNames(old_molecule)
    oechem.OETriposAtomNames(new_molecule)

    #write the molecules in order:
    oechem.OEWriteMolecule(ostream, old_molecule)
    oechem.OEWriteMolecule(ostream, new_molecule)

    ostream.close()

def write_host(host_oemol : oechem.OEMol, filepath: str):
    """
    Write out the host molecule on its own.

    Parameters
    ----------
    host_oemol : oechem.OEMol
        the OEMol containing the host
    filepath : str
        where to write the OEMol
    """
    ostream = oechem.oemolostream()
    ostream.open(filepath)

    #set title to avoid template name collision:
    host_oemol.SetTitle("host")

    #set tripos atom names
    oechem.OETriposAtomNames(host_oemol)

    oechem.OEWriteMolecule(ostream, host_oemol)

    ostream.close()

def prepare_example():
    """
    Prepare the host-guest example relative calculation.
    """
    #load in the host and guest molecules
    host_traj = extract_molecule(host_guest_path, "HST")
    guest_traj = extract_molecule(host_guest_path, "GST")

    #convert the trajs to mols
    host_mol = convert_mdtraj_to_oemol(host_traj)
    guest_mol = convert_mdtraj_to_oemol(guest_traj)

    #make the other guest molecule:
    new_guest_molecule = iupac_to_oemol(alternate_guest)

    #write out the host molecule:
    write_host(host_mol, host_vacuum_path)

    #write out the guest molecules:
    write_guest_molecules(guest_mol, new_guest_molecule, guests_vacuum_path)

if __name__=="__main__":
    prepare_example()