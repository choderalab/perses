"""
Prepare all SARS-CoV-2 Mpro structures for docking and simulation in monomer and dimer forms and desired protonation states.

"""

# Configure logging
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
from rich.console import Console
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
log = logging.getLogger("rich")

import rich
import openeye

def read_pdb_file(pdb_file):
    #print(f'Reading receptor from {pdb_file}...')

    from openeye import oechem
    ifs = oechem.oemolistream()
    #ifs.SetFlavor(oechem.OEFormat_PDB, oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA | oechem.OEIFlavor_PDB_ALTLOC)  # Causes extra protons on VAL73 for x1425
    ifs.SetFlavor(oechem.OEFormat_PDB, oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA )

    if not ifs.open(pdb_file):
        oechem.OEThrow.Fatal("Unable to open %s for reading." % pdb_file)

    mol = oechem.OEGraphMol()
    if not oechem.OEReadMolecule(ifs, mol):
        oechem.OEThrow.Fatal("Unable to read molecule from %s." % pdb_file)
    ifs.close()

    return (mol)

def prepare_receptor(complex_pdb_filename, output_basepath, assembly_state='dimer', retain_water=True, receptor_protonation_states=None, loop_db=None):
    """
    Parameters
    ----------
    complex_pdb_filename : str
        The complex PDB file to read in
    output_basepath : str
        Base path for output
    assembly_state : str, optional, default='dimer'
        Assembly state to prepare receptor in.
        allowed values: ['monomer', 'dimer']
    retain_water : bool, optional, default=False
        If True, will retain waters in protein PDB file
    receptor_protonation_states : list, optional, default=None
        If specified, only prepare the specified protonation states.
        e.g. ['His41(+)-Cys145(-)-His163(+)']
        Otherwise, all protonation states are prepared.
    loop_db : str, optional, default=None
        If specified, use SpruceTK loop database
    """
    import os
    basepath, filename = os.path.split(complex_pdb_filename)
    prefix, extension = os.path.splitext(filename)
    filepath_prefix = os.path.join(output_basepath, prefix)

    log.info(f"construction:  Preparing {complex_pdb_filename}...")

    # Read in PDB file, skipping UNK atoms (left over from processing covalent ligands)
    pdbfile_lines = [ line for line in open(complex_pdb_filename, 'r') if 'UNK' not in line ]

    # If monomer is specified, drop crystal symmetry lines
    if assembly_state == 'monomer':
        pdbfile_lines = [ line for line in pdbfile_lines if 'REMARK 350' not in line ]

    # Filter out LINK records to covalent inhibitors so we can model non-covalent complex
    pdbfile_lines = [ line for line in pdbfile_lines if 'LINK' not in line ]

    # Reconstruct PDBFile contents
    pdbfile_contents = ''.join(pdbfile_lines)

    # Append SARS-CoV-2 Mpro SEQRES to all structures if they do not have it
    seqres = """\
SEQRES   1 A  306  SER GLY PHE ARG LYS MET ALA PHE PRO SER GLY LYS VAL
SEQRES   2 A  306  GLU GLY CYS MET VAL GLN VAL THR CYS GLY THR THR THR
SEQRES   3 A  306  LEU ASN GLY LEU TRP LEU ASP ASP VAL VAL TYR CYS PRO
SEQRES   4 A  306  ARG HIS VAL ILE CYS THR SER GLU ASP MET LEU ASN PRO
SEQRES   5 A  306  ASN TYR GLU ASP LEU LEU ILE ARG LYS SER ASN HIS ASN
SEQRES   6 A  306  PHE LEU VAL GLN ALA GLY ASN VAL GLN LEU ARG VAL ILE
SEQRES   7 A  306  GLY HIS SER MET GLN ASN CYS VAL LEU LYS LEU LYS VAL
SEQRES   8 A  306  ASP THR ALA ASN PRO LYS THR PRO LYS TYR LYS PHE VAL
SEQRES   9 A  306  ARG ILE GLN PRO GLY GLN THR PHE SER VAL LEU ALA CYS
SEQRES  10 A  306  TYR ASN GLY SER PRO SER GLY VAL TYR GLN CYS ALA MET
SEQRES  11 A  306  ARG PRO ASN PHE THR ILE LYS GLY SER PHE LEU ASN GLY
SEQRES  12 A  306  SER CYS GLY SER VAL GLY PHE ASN ILE ASP TYR ASP CYS
SEQRES  13 A  306  VAL SER PHE CYS TYR MET HIS HIS MET GLU LEU PRO THR
SEQRES  14 A  306  GLY VAL HIS ALA GLY THR ASP LEU GLU GLY ASN PHE TYR
SEQRES  15 A  306  GLY PRO PHE VAL ASP ARG GLN THR ALA GLN ALA ALA GLY
SEQRES  16 A  306  THR ASP THR THR ILE THR VAL ASN VAL LEU ALA TRP LEU
SEQRES  17 A  306  TYR ALA ALA VAL ILE ASN GLY ASP ARG TRP PHE LEU ASN
SEQRES  18 A  306  ARG PHE THR THR THR LEU ASN ASP PHE ASN LEU VAL ALA
SEQRES  19 A  306  MET LYS TYR ASN TYR GLU PRO LEU THR GLN ASP HIS VAL
SEQRES  20 A  306  ASP ILE LEU GLY PRO LEU SER ALA GLN THR GLY ILE ALA
SEQRES  21 A  306  VAL LEU ASP MET CYS ALA SER LEU LYS GLU LEU LEU GLN
SEQRES  22 A  306  ASN GLY MET ASN GLY ARG THR ILE LEU GLY SER ALA LEU
SEQRES  23 A  306  LEU GLU ASP GLU PHE THR PRO PHE ASP VAL VAL ARG GLN
SEQRES  24 A  306  CYS SER GLY VAL THR PHE GLN
SEQRES   1 B  306  SER GLY PHE ARG LYS MET ALA PHE PRO SER GLY LYS VAL
SEQRES   2 B  306  GLU GLY CYS MET VAL GLN VAL THR CYS GLY THR THR THR
SEQRES   3 B  306  LEU ASN GLY LEU TRP LEU ASP ASP VAL VAL TYR CYS PRO
SEQRES   4 B  306  ARG HIS VAL ILE CYS THR SER GLU ASP MET LEU ASN PRO
SEQRES   5 B  306  ASN TYR GLU ASP LEU LEU ILE ARG LYS SER ASN HIS ASN
SEQRES   6 B  306  PHE LEU VAL GLN ALA GLY ASN VAL GLN LEU ARG VAL ILE
SEQRES   7 B  306  GLY HIS SER MET GLN ASN CYS VAL LEU LYS LEU LYS VAL
SEQRES   8 B  306  ASP THR ALA ASN PRO LYS THR PRO LYS TYR LYS PHE VAL
SEQRES   9 B  306  ARG ILE GLN PRO GLY GLN THR PHE SER VAL LEU ALA CYS
SEQRES  10 B  306  TYR ASN GLY SER PRO SER GLY VAL TYR GLN CYS ALA MET
SEQRES  11 B  306  ARG PRO ASN PHE THR ILE LYS GLY SER PHE LEU ASN GLY
SEQRES  12 B  306  SER CYS GLY SER VAL GLY PHE ASN ILE ASP TYR ASP CYS
SEQRES  13 B  306  VAL SER PHE CYS TYR MET HIS HIS MET GLU LEU PRO THR
SEQRES  14 B  306  GLY VAL HIS ALA GLY THR ASP LEU GLU GLY ASN PHE TYR
SEQRES  15 B  306  GLY PRO PHE VAL ASP ARG GLN THR ALA GLN ALA ALA GLY
SEQRES  16 B  306  THR ASP THR THR ILE THR VAL ASN VAL LEU ALA TRP LEU
SEQRES  17 B  306  TYR ALA ALA VAL ILE ASN GLY ASP ARG TRP PHE LEU ASN
SEQRES  18 B  306  ARG PHE THR THR THR LEU ASN ASP PHE ASN LEU VAL ALA
SEQRES  19 B  306  MET LYS TYR ASN TYR GLU PRO LEU THR GLN ASP HIS VAL
SEQRES  20 B  306  ASP ILE LEU GLY PRO LEU SER ALA GLN THR GLY ILE ALA
SEQRES  21 B  306  VAL LEU ASP MET CYS ALA SER LEU LYS GLU LEU LEU GLN
SEQRES  22 B  306  ASN GLY MET ASN GLY ARG THR ILE LEU GLY SER ALA LEU
SEQRES  23 B  306  LEU GLU ASP GLU PHE THR PRO PHE ASP VAL VAL ARG GLN
SEQRES  24 B  306  CYS SER GLY VAL THR PHE GLN
"""
    has_seqres = 'SEQRES' in pdbfile_contents
    if not has_seqres:
        print('Adding SEQRES')
        pdbfile_contents = seqres + pdbfile_contents

    # Read the receptor and identify design units
    from openeye import oespruce, oechem
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(delete=False, mode='wt', suffix='.pdb') as pdbfile:
        pdbfile.write(pdbfile_contents)
        pdbfile.close()
        complex = read_pdb_file(pdbfile.name)
        # TODO: Clean up

    # Strip protons from structure to allow SpruceTK to add these back
    # See: 6wnp, 6wtj, 6wtk, 6xb2, 6xqs, 6xqt, 6xqu, 6m2n
    #print('Suppressing hydrogens')
    #print(f' Initial: {sum([1 for atom in complex.GetAtoms()])} atoms')
    for atom in complex.GetAtoms():
        if atom.GetAtomicNum() > 1:
            oechem.OESuppressHydrogens(atom)
    #print(f' Final: {sum([1 for atom in complex.GetAtoms()])} atoms')

    # Delete and rebuild C-terminal residue because Spruce causes issues with this
    log.info(f":gear:  Rebuilding C-terminus")
    # See: 6m2n 6lze
    #print('Deleting C-terminal residue O')
    pred = oechem.OEIsCTerminalAtom()
    for atom in complex.GetAtoms():
        if pred(atom):
            for nbor in atom.GetAtoms():
                if oechem.OEGetPDBAtomIndex(nbor) == oechem.OEPDBAtomName_O:
                    complex.DeleteAtom(nbor)

    #pred = oechem.OEAtomMatchResidue(["GLN:306:.*:.*:.*"])
    #for atom in complex.GetAtoms(pred):
    #    if oechem.OEGetPDBAtomIndex(atom) == oechem.OEPDBAtomName_O:
    #        print('Deleting O')
    #        complex.DeleteAtom(atom)

    log.info(f":mag:  Identifying design units")
    # Produce zero design units if we fail to protonate

    # Log warnings
    errfs = oechem.oeosstream() # create a stream that writes internally to a stream
    oechem.OEThrow.SetOutputStream(errfs)
    oechem.OEThrow.Clear()
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Verbose) # capture verbose error output

    opts = oespruce.OEMakeDesignUnitOptions()
    #print(f'ligand atoms: min {opts.GetSplitOptions().GetMinLigAtoms()}, max {opts.GetSplitOptions().GetMaxLigAtoms()}')
    #opts.GetSplitOptions().SetMinLigAtoms(7) # minimum fragment size (in heavy atoms)
    
    mdata = oespruce.OEStructureMetadata();
    opts.GetPrepOptions().SetStrictProtonationMode(True);

    # turn off superposition
    opts.SetSuperpose(False)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment
    opts.GetSplitOptions().SetMinLigAtoms(5)
    # also consider alternate locations outside binding pocket, important for later filtering
    opts.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(False)

    # alignment options, only matches are important
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignMethod(
        oechem.OESeqAlignmentMethod_Identity
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignGapPenalty(-1)
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignExtendPenalty(0)

    # We are in reducing conditions, so do not allow disulfide bonds to be built
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetAllowBuildDisulfideBridges(False)
    
    if loop_db is not None:
        from pathlib import Path
        loop_db = str(Path(loop_db).expanduser().resolve())
        opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(loop_db)

    # Set ligand name and SMILES
    #log.info(f":pill:  Setting ligand")
    #smiles = 'CNC(=O)CN1Cc2ccc(Cl)cc2[C@@]2(CCN(c3cncc4ccccc34)C2=O)C1'
    #het = oespruce.OEHeterogenMetadata()
    #het.SetTitle("LIG")  # real ligand 3 letter code
    #het.SetID("LIG")  # in case you have corporate IDs
    #het.SetSmiles(smiles)
    #het.SetType(oespruce.OEHeterogenType_Ligand)
    #mdata.AddHeterogenMetadata(het)

    # Both N- and C-termini should be zwitterionic
    # Mpro cleaves its own N- and C-termini
    # See https://www.pnas.org/content/113/46/12997
    opts.GetPrepOptions().GetBuildOptions().SetCapNTermini(False);
    opts.GetPrepOptions().GetBuildOptions().SetCapCTermini(False);
    # Don't allow truncation of termini, since force fields don't have parameters for this
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetAllowTruncate(False);
    # Delete clashing solvent if needed
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetDeleteClashingSolvent(True)
    opts.GetPrepOptions().GetBuildOptions().GetSidechainBuilderOptions().SetDeleteClashingSolvent(True)
    # Build loops and sidechains
    opts.GetPrepOptions().GetBuildOptions().SetBuildLoops(True);
    opts.GetPrepOptions().GetBuildOptions().SetBuildSidechains(True);

    # Generate ligand tautomers
    protonate_opts = opts.GetPrepOptions().GetProtonateOptions();
    protonate_opts.SetGenerateTautomers(True)

    # Don't flip Gln189
    pred = oechem.OEAtomMatchResidue(["GLN:189:.*:.*:.*"])
    protonate_opts = opts.GetPrepOptions().GetProtonateOptions();
    place_hydrogens_opts = protonate_opts.GetPlaceHydrogensOptions()
    place_hydrogens_opts.SetNoFlipPredicate(pred)

    # Make design units
    log.info(f":lower_left_fountain_pen:  Making design units...")
    design_units = list(oespruce.OEMakeDesignUnits(complex, mdata, opts))
    print(design_units)

    # Restore error stream
    oechem.OEThrow.SetOutputStream(oechem.oeerr)

    # Capture the warnings to a string
    warnings = errfs.str().decode("utf-8")

    log.info(f":receipt:  There are {len(design_units)} design units.")

    if len(design_units) >= 1:
        design_unit = design_units[0] # TODO: Select appropriate design unit based on desired chain (A or B)
        print('')
        print('')
        print(f'{complex_pdb_filename} : SUCCESS')
        print(warnings)
    elif len(design_units) == 0:
        print('')
        print('')
        print(f'{complex_pdb_filename} : FAILURE')
        print(warnings)
        msg = f'No design units found for {complex_pdb_filename}\n'
        msg += warnings
        msg += '\n'
        raise Exception(msg)

    # Variants to generate
    # (OEAtomMatchResidue, atom_name, formal_charge, implicit_hydrogen_count)
    # OEAtomMatchResidue format is regex for (residue name, residue number, insertion code, chain ID, and fragment number)
    variants = {
        'His41(0)-Cys145(0)-His163(0)' : [ ('HIS:41:.*:.*:.*', 'ND1', 0, 0), ('HIS:41:.*:.*:.*', 'NE2', 1, 1), ('CYS:145:.*:.*:.*', 'SG', 0, 1), ('HIS:163:.*:.*:.*', 'NE2', 0, 0), ('HIS:163:.*:.*:.*', 'ND1', 1, 1) ],
        'His41(0)-Cys145(0)-His163(+)' : [ ('HIS:41:.*:.*:.*', 'ND1', 0, 0), ('HIS:41:.*:.*:.*', 'NE2', 1, 1), ('CYS:145:.*:.*:.*', 'SG', 0, 1), ('HIS:163:.*:.*:.*', 'NE2', 1, 1), ('HIS:163:.*:.*:.*', 'ND1', 1, 1) ],
        'His41(+)-Cys145(-)-His163(0)' : [ ('HIS:41:.*:.*:.*', 'ND1', 1, 1), ('HIS:41:.*:.*:.*', 'NE2', 1, 1), ('CYS:145:.*:.*:.*', 'SG', -1, 0), ('HIS:163:.*:.*:.*', 'NE2', 0, 0), ('HIS:163:.*:.*:.*', 'ND1', 1, 1) ],
        'His41(+)-Cys145(-)-His163(+)' : [ ('HIS:41:.*:.*:.*', 'ND1', 1, 1), ('HIS:41:.*:.*:.*', 'NE2', 1, 1), ('CYS:145:.*:.*:.*', 'SG', -1, 0), ('HIS:163:.*:.*:.*', 'NE2', 1, 1), ('HIS:163:.*:.*:.*', 'ND1', 1, 1) ],
    }
    # Filter variants if desired
    if receptor_protonation_states is not None:
        variants = { name : data for (name, data) in variants.items() if name in receptor_protonation_states }

    def generate_variant(design_unit, modifications):
        from openeye import oedocking
        protein = oechem.OEGraphMol()
        design_unit.GetProtein(protein)

        protonate_opts = opts.GetPrepOptions().GetProtonateOptions();
        place_hydrogens_opts = protonate_opts.GetPlaceHydrogensOptions()

        for (residue_match, atom_name, formal_charge, implicit_hydrogen_count) in modifications:
            pred = oechem.OEAtomMatchResidue(residue_match)
            place_hydrogens_opts.SetBypassPredicate(pred)
            for atom in protein.GetAtoms(pred):
                if oechem.OEGetPDBAtomIndex(atom) == getattr(oechem, f'OEPDBAtomName_{atom_name}'):
                    #print('Modifying CYS 145 SG')
                    oechem.OESuppressHydrogens(atom)
                    atom.SetFormalCharge(formal_charge)
                    atom.SetImplicitHCount(implicit_hydrogen_count)
        # Update the design unit with the modified formal charge for CYS 145 SG
        oechem.OEUpdateDesignUnit(design_unit, protein, oechem.OEDesignUnitComponents_Protein)

        # Don't flip Gln189
        pred = oechem.OEAtomMatchResidue(["GLN:189: :A"])
        protonate_opts = opts.GetPrepOptions().GetProtonateOptions();
        place_hydrogens_opts = protonate_opts.GetPlaceHydrogensOptions()
        place_hydrogens_opts.SetNoFlipPredicate(pred)

        # Adjust protonation states
        #print('Re-optimizing hydrogen positions...') # DEBUG
        #place_hydrogens_opts = oechem.OEPlaceHydrogensOptions()
        #place_hydrogens_opts.SetBypassPredicate(pred)
        #protonate_opts = oespruce.OEProtonateDesignUnitOptions(place_hydrogens_opts)
        success = oespruce.OEProtonateDesignUnit(design_unit, protonate_opts)

    master_design_unit = design_unit
    import copy
    for variant_name, modifications in variants.items():
        print(f'Generating variant {variant_name}')
        design_unit = copy.deepcopy(master_design_unit)
        generate_variant(design_unit, modifications)

        prefix = f'{filepath_prefix}-{variant_name}'

        # Write ligand
        ligand = oechem.OEGraphMol()
        design_unit.GetLigand(ligand)
        oechem.OEPerceiveResidues(ligand, oechem.OEPreserveResInfo_ChainID | oechem.OEPreserveResInfo_ResidueNumber | oechem.OEPreserveResInfo_ResidueName)
        with oechem.oemolostream(f'{prefix}-ligand.mol2') as ofs:
            oechem.OEWriteMolecule(ofs, ligand)
        with oechem.oemolostream(f'{prefix}-ligand.pdb') as ofs:
            oechem.OEWriteMolecule(ofs, ligand)
        with oechem.oemolostream(f'{prefix}-ligand.sdf') as ofs:
            oechem.OEWriteMolecule(ofs, ligand)

        # Write protein
        protein = oechem.OEGraphMol()            
        if retain_water:
            component_mask = oechem.OEDesignUnitComponents_Protein | oechem.OEDesignUnitComponents_Solvent
        else:
            component_mask = oechem.OEDesignUnitComponents_Protein
        design_unit.GetComponents(protein, component_mask)
        oechem.OEPerceiveResidues(protein, oechem.OEPreserveResInfo_ChainID | oechem.OEPreserveResInfo_ResidueNumber | oechem.OEPreserveResInfo_ResidueName)
        with oechem.oemolostream(f'{prefix}-protein.pdb') as ofs:
            oechem.OEWriteMolecule(ofs, protein)

        # Write receptor
        from openeye import oedocking
        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, protein, ligand)
        receptor_filename = f'{prefix}-receptor.oeb.gz'
        oedocking.OEWriteReceptorFile(receptor, receptor_filename)

        # Write DesignUnit
        from openeye import oedocking
        oedocking.OEMakeReceptor(design_unit)
        du_filename = f'{prefix}.oedu'
        oechem.OEWriteDesignUnit(du_filename, design_unit)

if __name__ == '__main__':
    # Load setup.yaml
    import yaml
    with open('setup.yaml', 'r') as infile:
        setup_options = yaml.safe_load(infile)

    # Extract setup options
    # NOTE: Almost no sanity checking is done here.
    fragments = [ entry['fragalysis_id'] for entry in setup_options['source_structures'] ]
    structures_path = setup_options['structures_path'] 
    output_basepath = setup_options['receptors_path']
    spruce_loop_database = setup_options.get('spruce_loop_database', None)
    retain_water = setup_options.get('retain_water', True)
    receptor_protonation_states = setup_options.get('receptor_protonation_states', [])
    assembly_states = setup_options.get('assembly_state', ['dimer'])
    assert set(assembly_states).issubset(['monomer', 'dimer'])

    # Prep all receptors
    import glob, os

    # Be quiet
    from openeye import oechem
    #oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)
    #oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)

    source_pdb_files = [os.path.join(structures_path, f'aligned/Mpro-{fragment}/Mpro-{fragment}_bound.pdb') for fragment in fragments]

    # Create output directory
    os.makedirs(output_basepath, exist_ok=True)

    for assembly_state in assembly_states:
        output_path = os.path.join(output_basepath, assembly_state)

        os.makedirs(output_path, exist_ok=True)

        def prepare_receptor_wrapper(complex_pdb_file):
            try:
                prepare_receptor(complex_pdb_file, output_path, assembly_state=assembly_state, retain_water=retain_water, receptor_protonation_states=receptor_protonation_states, loop_db=spruce_loop_database)
                return True
            except Exception as e:
                print(e)
                import traceback
                import sys
                traceback.print_exception(*sys.exc_info())
                return False

        # Serial (DEBUG ONLY)
        #for source_pdb_file in source_pdb_files:
        #    prepare_receptor_wrapper(source_pdb_file)
        #stop

        # Multiprocessing
        from rich.progress import track
        from multiprocessing import Pool
        from tqdm import tqdm
        import os
        processes = None
        if 'LSB_DJOB_NUMPROC' in os.environ:
            processes = int(os.environ['LSB_DJOB_NUMPROC'])
        pool = Pool(processes=processes)
        for status in track(pool.imap_unordered(prepare_receptor_wrapper, source_pdb_files), total=len(source_pdb_files), description='Sprucing structures...'):
            pass
        pool.close()
        pool.join()
