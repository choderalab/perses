"""
Prepare the reference protein structure

This script is a modified version of make_design_units.py from the OpenEye SpruceTK example.
https://docs.eyesopen.com/toolkits/python/sprucetk/examples_make_design_units.html#spruce-examples-make-design-units

"""

#############################################################################
# Script to prepare proteins into design units
#############################################################################
def prepare_receptor(complex_pdb_filename, receptor_pdb_filename, ligand_sdf_filename, design_unit_filename):
    """
    Prepare a Fragalysis complex PDB file into fully protonated reference ligand SDF and protein with X-ray waters PDB.

    Parameters
    ----------
    complex_pdb_filename : str
        The complex PDB or CIF file to read
    receptor_pdb_filename : str
        The fully protonated receptor PDB file to write
    ligand_sdf_filename : str
        The protonated ligand SDF file to write
    design_unit_filename : str
        Filename of the receptor design unit to write
    """
    import sys
    import os
    from openeye import oechem
    from openeye import oegrid
    from openeye import oespruce

    ifs = oechem.oemolistream()
    if not ifs.open(complex_pdb_filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % complex_pdb_filename)
    if ifs.GetFormat() not in [oechem.OEFormat_PDB, oechem.OEFormat_CIF]:
        oechem.OEThrow.Fatal("Only works for .pdb or .cif input files")
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        oechem.OEIFlavor_PDB_Default
        | oechem.OEIFlavor_PDB_DATA
        | oechem.OEIFlavor_PDB_ALTLOC,
    )  # noqa
    
    mol = oechem.OEGraphMol()
    if not oechem.OEReadMolecule(ifs, mol):
        oechem.OEThrow.Fatal("Unable to read molecule from %s" % complex_pdb_filename)

    # Set SpruceTK options
    metadata = oespruce.OEStructureMetadata()
    opts = oespruce.OEMakeDesignUnitOptions()
    # Use the real biological zwitterionic termini
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetForceCapping(False)
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetAllowTruncate(False)
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetDeleteClashingSolvent(True)
    opts.GetPrepOptions().GetBuildOptions().GetSidechainBuilderOptions().SetDeleteClashingSolvent(True)
    # Generate design units
    design_units = oespruce.OEMakeDesignUnits(mol, metadata, opts)

    # Take only the first design unit
    design_units = list(design_units)
    design_unit = design_units[0]

    # Write design unit (for reference)
    oechem.OEWriteDesignUnit(design_unit_filename, design_unit)

    receptor = oechem.OEGraphMol()
    component_mask = oechem.OEDesignUnitComponents_Protein | oechem.OEDesignUnitComponents_Solvent
    #component_mask = oechem.OEDesignUnitComponents_Protein
    design_unit.GetComponents(receptor, component_mask)
    # Clean up PDB info after adding missing atoms
    oechem.OEPerceiveResidues(mol, oechem.OEPreserveResInfo_ChainID | oechem.OEPreserveResInfo_ResidueNumber | oechem.OEPreserveResInfo_ResidueName)

    ligand = oechem.OEGraphMol()
    design_unit.GetLigand(ligand)    

    # Write receptor
    with oechem.oemolostream(receptor_pdb_filename) as ofs:
        oechem.OEWriteMolecule(ofs, receptor)
    # Write ligand
    with oechem.oemolostream(ligand_sdf_filename) as ofs:
        oechem.OEWriteMolecule(ofs, ligand)

if __name__ == "__main__":    
    complex_pdb_filename = 'Mpro-P0008_0A_bound.pdb'
    receptor_pdb_filename = 'receptor.pdb'
    ligand_sdf_filename = 'reference-ligand.sdf'
    design_unit_filename = 'receptor.oedu'
    prepare(complex_pdb_filename, receptor_pdb_filename, ligand_sdf_filename, design_unit_filename)
    
