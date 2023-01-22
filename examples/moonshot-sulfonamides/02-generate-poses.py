#!/usr/bin/env python

"""
Generate poses for relative free energy calculations using fragment structures

"""
from openeye import oechem
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def GetFragments(mol, minbonds, maxbonds):
    from openeye import oegraphsim
    frags = []
    fptype = oegraphsim.OEGetFPType("Tree,ver=2.0.0,size=4096,bonds=%d-%d,atype=AtmNum,btype=Order"
                                    % (minbonds, maxbonds))

    for abset in oegraphsim.OEGetFPCoverage(mol, fptype, True):
        fragatompred = oechem.OEIsAtomMember(abset.GetAtoms())

        frag = oechem.OEGraphMol()
        adjustHCount = True
        oechem.OESubsetMol(frag, mol, fragatompred, adjustHCount)
        oechem.OEFindRingAtomsAndBonds(frag)
        frags.append(oechem.OEGraphMol(frag))

    return frags


def GetCommonFragments(mollist, frags,
                       atomexpr=oechem.OEExprOpts_DefaultAtoms,
                       bondexpr=oechem.OEExprOpts_DefaultBonds):

    corefrags = []

    from rich.progress import track
    #for frag in track(frags, description='Finding common fragments'):
    for frag in frags:
        ss = oechem.OESubSearch(frag, atomexpr, bondexpr)
        if not ss.IsValid():
            logging.warning('Is not valid')
            continue

        validcore = True
        for mol in mollist:
            oechem.OEPrepareSearch(mol, ss)
            validcore = ss.SingleMatch(mol)
            if not validcore:
                break

        if validcore:
            corefrags.append(frag)

    return corefrags


def GetCoreFragment(refmol, mols,
                    minbonds=3, maxbonds=200,
                    atomexpr=oechem.OEExprOpts_DefaultAtoms,
                    bondexpr=oechem.OEExprOpts_DefaultBonds):

    #print("Number of molecules = %d" % len(mols))

    frags = GetFragments(refmol, minbonds, maxbonds)
    if len(frags) == 0:
        oechem.OEThrow.Error("No fragment is enumerated with bonds %d-%d!" % (minbonds, maxbonds))

    commonfrags = GetCommonFragments(mols, frags, atomexpr, bondexpr)
    if len(commonfrags) == 0:
        oechem.OEThrow.Error("No common fragment is found!")

    #print("Number of common fragments = %d" % len(commonfrags))

    core = None
    for frag in commonfrags:
        if core is None or GetFragmentScore(core) < GetFragmentScore(frag):
            core = frag

    return core

def GetFragmentScore(mol):

    score = 0.0
    score += 2.0 * oechem.OECount(mol, oechem.OEAtomIsInRing())
    score += 1.0 * oechem.OECount(mol, oechem.OENotAtom(oechem.OEAtomIsInRing()))

    return score

def expand_stereochemistry_star(mol):
    from openeye import oechem, oeomega
    omegaOpts = oeomega.OEOmegaOptions()
    omega = oeomega.OEOmega(omegaOpts)
    maxcenters = 12
    forceFlip = False
    enumNitrogen = True # WARNING: This creates multiple microstates with same SMILES if True
    warts = True # add suffix for stereoisomers

    compound_title = mol.GetTitle()
    compound_smiles = oechem.OEMolToSmiles(mol)

    enantiomers = list()
    for enantiomer in oeomega.OEFlipper(mol, maxcenters, forceFlip, enumNitrogen, warts):
        enantiomer = oechem.OEMol(enantiomer)
        omega.Build(enantiomer) # a single conformer needs to be built to fully define stereochemistry
        enantiomer_smiles =  oechem.OEMolToSmiles(enantiomer)
        oechem.OESetSDData(enantiomer, 'compound', compound_title)
        oechem.OESetSDData(enantiomer, 'compound_smiles', compound_smiles)
        oechem.OESetSDData(enantiomer, 'enantiomer_smiles', enantiomer_smiles)
        enantiomers.append(enantiomer)

    return enantiomers

def expand_stereochemistry(mols):
    """Expand stereochemistry when uncertain

    Parameters
    ----------
    mols : openeye.oechem.OEGraphMol
        Molecules to be expanded

    Returns
    -------
    expanded_mols : openeye.oechem.OEMol
        Expanded molecules
    """
    expanded_mols = list()

    from rich.progress import track
    from multiprocessing import Pool
    from tqdm import tqdm
    import os
    processes = None
    if 'LSB_DJOB_NUMPROC' in os.environ:
        processes = int(os.environ['LSB_DJOB_NUMPROC'])
    pool = Pool(processes=processes)
    for enantiomers in track(pool.imap_unordered(expand_stereochemistry_star, mols), total=len(mols), description='Enumerating stereocenters...'):
        expanded_mols += enantiomers
    pool.close()
    pool.join()

    return expanded_mols

def expand_stereochemistry_serial(mols):
    """Expand stereochemistry when uncertain

    Parameters
    ----------
    mols : openeye.oechem.OEGraphMol
        Molecules to be expanded

    Returns
    -------
    expanded_mols : openeye.oechem.OEMol
        Expanded molecules
    """
    try:
        nmols = len(mols)
    except TypeError as e:
        mols = [mols]

    expanded_mols = list()
    for mol in mols:
        expanded_mols += expand_stereochemistry_star(mol)

    return expanded_mols

def expand_protonation_states_oemols(mols):
    """Expand protonation states

    Parameters
    ----------
    mols : openeye.oechem.OEGraphMol
        Molecules to be expanded

    Returns
    -------
    expanded_mols : openeye.oechem.OEMol
        Expanded molecules
    """
    import os

    # TODO: Use temporary directory
    logging.info(' Writing to sdf...')
    write_molecules(mols, 'input.sdf')
    logging.info(f'  {len(mols)} molecules written.')

    # Convert to mae
    logging.info(' Converting to mae...')
    command = f'$SCHRODINGER/utilities/structconvert input.sdf input.mae'
    os.system(command)

    # Expand protonation and tautomeric states
    # TODO: This will create multiple entries with the same ligand name
    logging.info(' Expanding protonation states with epik...')
    import numpy as np
    min_population = np.exp(-6)
    command = f'$SCHRODINGER/epik -WAIT -ms 16 -ph 7.3 -p {min_population} -imae input.mae -omae output.mae'
    os.system(command)

    # Convert back to SDF
    logging.info(' Converting back to sdf...')
    command = f'$SCHRODINGER/utilities/structconvert output.mae output.sdf'
    os.system(command)

    # Read files
    # TODO: Fix this, since openeye seems to skip over molecules with the same name
    logging.info(' Reading molecules...')
    expanded_mols = load_molecules('output.sdf', append_warts=True)
    logging.info(f'  {len(expanded_mols)} molecules read.')

    # DEBUG
    for index, mol in enumerate(expanded_mols):
        print(f'{index:8d} : {mol.GetTitle()}')

    # TODO: Extract state penalties

    return expanded_mols

class BumpCheck:
    def __init__(self, prot_mol, cutoff=2.0):
        self.near_nbr = oechem.OENearestNbrs(prot_mol, cutoff)
        self.cutoff = cutoff

    def count(self, lig_mol):
        bump_count = 0
        for nb in self.near_nbr.GetNbrs(lig_mol):
            if (not nb.GetBgn().IsHydrogen()) and (not nb.GetEnd().IsHydrogen()):
                bump_count += np.exp(-0.5 * (nb.GetDist() / self.cutoff)**2)
        return bump_count

def mmff_energy(mol):
    """
    Compute MMFF energy

    """
    from openeye import oechem, oeff
    mol = oechem.OEGraphMol(mol)
    mmff = oeff.OEMMFF()
    if not mmff.PrepMol(mol) or not mmff.Setup(mol):
        oechem.OEThrow.Warning("Unable to process molecule: title = '%s'" % mol.GetTitle())
        return None

    vecCoords = oechem.OEDoubleArray(3*mol.GetMaxAtomIdx())
    mol.GetCoords(vecCoords)
    energy = mmff(vecCoords)
    return energy

def generate_restricted_conformers(receptor, refmol, mol, core_smarts=None):
    """
    Generate and select a conformer of the specified molecule using the reference molecule

    Parameters
    ----------
    receptor : openeye.oechem.OEGraphMol
        Receptor (already prepped for docking) for identifying optimal pose
    refmol : openeye.oechem.OEGraphMol
        Reference molecule which shares some part in common with the proposed molecule
    mol : openeye.oechem.OEGraphMol
        Molecule whose conformers are to be enumerated
    core_smarts : str, optional, default=None
        If core_smarts is specified, substructure will be extracted using SMARTS.
    """
    from openeye import oechem, oeomega

    logging.debug(f'mol: {oechem.OEMolToSmiles(mol)} | refmol: {oechem.OEMolToSmiles(refmol)} : core_smarts: {core_smarts}')

    # Make a copy
    refmol = oechem.OEGraphMol(refmol)

    # Generate unique atom names
    from collections import defaultdict
    element_counts = defaultdict(int)
    for oeatom in mol.GetAtoms():
        symbol = oechem.OEGetAtomicSymbol(oeatom.GetAtomicNum())
        element_counts[symbol] += 1
        oeatom.SetName(symbol + str(element_counts[symbol]))
    # Build a single conformer to check stereochemistry
    # TODO: Check stereochemistry that is undefined at this stage
    builder = oeomega.OEConformerBuilder()
    builder.Build(mol)
    # Make sure we haven't inverted carbon stereochemistry
    initial_chiral_centers = { oeatom.GetName() : oechem.OEPerceiveCIPStereo(mol, oeatom) for oeatom in mol.GetAtoms(oechem.OEIsChiralAtom()) if oechem.OEGetAtomicSymbol(oeatom.GetAtomicNum()) == 'C' }
    for (name, cip) in initial_chiral_centers.items():
        if (cip == oechem.OECIPAtomStereo_UnspecStereo):
            print(f'Molecule has unspecified stereochemistry: {oechem.OECreateCanSmiString(mol)} at atom {name}: {cip}')

    # Be quiet
    from openeye import oechem
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)
    #oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)

    # Get core fragment
    core_fragment = None
    if core_smarts:
        # Truncate refmol to SMARTS if specified
        #print(f'Trunctating using SMARTS {refmol_smarts}')
        ss = oechem.OESubSearch(core_smarts)
        oechem.OEPrepareSearch(refmol, ss)
        for match in ss.Match(refmol):
            core_fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(core_fragment, match)
            logging.debug(f'Truncated refmol to generate core_fragment: {oechem.OEMolToSmiles(core_fragment)}')
            break
        #print(f'refmol has {refmol.NumAtoms()} atoms')
    else:
        core_fragment = GetCoreFragment(refmol, [mol])
        oechem.OESuppressHydrogens(core_fragment)
        #print(f'  Core fragment has {core_fragment.NumAtoms()} heavy atoms')
        MIN_CORE_ATOMS = 6
        if core_fragment.NumAtoms() < MIN_CORE_ATOMS:
            return None

    if core_fragment is None:
        return None

    # Create an Omega instance
    #omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
    #omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
    #omegaOpts.SetMaxConfs(5000)
    #omegaOpts.SetEnergyWindow(100)


    # Set the fixed reference molecule
    omegaFixOpts = oeomega.OEConfFixOptions()
    omegaFixOpts.SetFixMaxMatch(10) # allow multiple MCSS matches
    omegaFixOpts.SetFixDeleteH(True) # only use heavy atoms
    omegaFixOpts.SetFixMol(core_fragment)
    #omegaFixOpts.SetFixSmarts(core_smarts) # DEBUG
    #omegaFixOpts.SetFixRMS(0.5)
    omegaFixOpts.SetFixRMS(1.0)

    # This causes a warning:
    #Warning: OESubSearch::Match() is unable to match unset hybridization in the target (EN300-221518_3_1) for patterns with set hybridization, call OEPrepareSearch on the target first
    #atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_Hybridization

    atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_AtomicNumber
    bondexpr = oechem.OEExprOpts_BondOrder | oechem.OEExprOpts_Aromaticity

    #atomexpr = oechem.OEExprOpts_Aromaticity # DEBUG : needed to introduce SO2 into rings
    #bondexpr = oechem.OEExprOpts_Aromaticity # DEBUG : needed for EDJ-MED-fd8ed875

    #atomexpr = oechem.OEExprOpts_RingMember
    #bondexpr = oechem.OEExprOpts_RingMember

    omegaFixOpts.SetAtomExpr(atomexpr)
    omegaFixOpts.SetBondExpr(bondexpr)
    omegaOpts.SetConfFixOptions(omegaFixOpts)

    molBuilderOpts = oeomega.OEMolBuilderOptions()
    molBuilderOpts.SetStrictAtomTypes(False) # don't give up if MMFF types are not found
    omegaOpts.SetMolBuilderOptions(molBuilderOpts)

    omegaOpts.SetWarts(False) # expand molecule title
    omegaOpts.SetStrictStereo(True) # set strict stereochemistry
    omegaOpts.SetIncludeInput(False) # don't include input
    omegaOpts.SetMaxConfs(1000) # generate lots of conformers
    omegaOpts.SetEnergyWindow(20.0) # allow high energies
    omega = oeomega.OEOmega(omegaOpts)

    # TODO: Expand protonation states and tautomers
    #from openeye import oequacpac
    #if not oequacpac.OEGetReasonableProtomer(mol):
    #    logging.warning('No reasonable protomer found')
    #    return None

    mol = oechem.OEMol(mol) # multi-conformer molecule

    ret_code = omega.Build(mol)
    if (mol.GetDimension() != 3) or (ret_code != oeomega.OEOmegaReturnCode_Success):
        msg = f'\nOmega failure for {mol.GetTitle()} : SMILES {oechem.OEMolToSmiles(mol)} : REFMOL {oechem.OEMolToSmiles(refmol)} : core_smarts {core_smarts} : {oeomega.OEGetOmegaError(ret_code)}\n'
        logging.warning(msg)
        return None
        # Return the molecule with an error code
        #oechem.OESetSDData(mol, 'error', '{oeomega.OEGetOmegaError(ret_code)}')
        #return mol

    # Extract poses
    class Pose(object):
        def __init__(self, conformer):
            self.conformer = conformer
            self.clash_score = None
            self.docking_score = None
            self.overlap_score = None

    poses = [ Pose(conf) for conf in mol.GetConfs() ]

    # Score clashes
    bump_check = BumpCheck(receptor)
    for pose in poses:
        pose.clash_score = bump_check.count(pose.conformer)

    # Score docking poses
    from openeye import oedocking
    score = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
    score.Initialize(receptor)
    for pose in poses:
        pose.docking_score = score.ScoreLigand(pose.conformer)

    # Compute overlap scores
    from openeye import oeshape
    overlap_prep = oeshape.OEOverlapPrep()
    overlap_prep.Prep(refmol)
    shapeFunc = oeshape.OEExactShapeFunc()
    shapeFunc.SetupRef(refmol)
    oeshape_result = oeshape.OEOverlapResults()
    for pose in poses:
        tmpmol = oechem.OEGraphMol(pose.conformer)
        overlap_prep.Prep(tmpmol)
        shapeFunc.Overlap(tmpmol, oeshape_result)
        pose.overlap_score = oeshape_result.GetRefTversky()

    # Filter poses based on top 50% of overlap
    #poses = sorted(poses, key= lambda pose : pose.overlap_score)
    #poses = poses[int(0.5*len(poses)):]

    # Filter poses to eliminate worst clashes
    if len(poses) > 10:
        poses = sorted(poses, key= lambda pose : pose.clash_score)
        poses = poses[:int(0.5*len(poses))]

    # Set original compound name
    oechem.OESetSDData(mol, 'parent_compound', mol.GetTitle().rsplit('_')[0])

    # Select the best docking score
    import numpy as np
    #poses = sorted(poses, key=lambda pose : pose.docking_score)
    # Select the best overlap score
    poses = sorted(poses, key=lambda pose : -pose.overlap_score)
    pose = poses[0]
    mol.SetActive(pose.conformer)
    oechem.OESetSDData(mol, 'clash_score', str(pose.clash_score))
    oechem.OESetSDData(mol, 'docking_score', str(pose.docking_score))
    oechem.OESetSDData(mol, 'overlap_score', str(pose.overlap_score))

    # Convert to single-conformer molecule
    mol = oechem.OEGraphMol(mol)

    # Make sure we haven't inverted carbon stereochemistry
    #from openeye import oeszybki
    #szopts = oeszybki.OESzybkiOptions()
    #sz = oeszybki.OESzybki(opts)
    #szresults = oeszybki.OESzybkiResults()
    #if not sz(mol, results):
    #    return None
    final_chiral_centers = { oeatom.GetName() : oechem.OEPerceiveCIPStereo(mol, oeatom) for oeatom in mol.GetAtoms(oechem.OEIsChiralAtom()) if oechem.OEGetAtomicSymbol(oeatom.GetAtomicNum()) == 'C' }
    if initial_chiral_centers != final_chiral_centers:
        print(f'Rejecting due to stereochemistry inversion: {initial_chiral_centers} -> {final_chiral_centers}')
        return None

    # Compute MMFF energy
    energy = mmff_energy(mol)
    oechem.OESetSDData(mol, 'MMFF_internal_energy', str(energy))
    #if energy > 100.0:
    #    print(f'Rejecting due to positive MMFF internal energy: {energy}')
    #    return None

    # Store SMILES
    docked_smiles = oechem.OEMolToSmiles(mol)
    oechem.OESetSDData(mol, 'docked_smiles', docked_smiles)

    return mol

def has_ic50(mol):
    """Return True if this molecule has fluorescence IC50 data"""
    from openeye import oechem
    if not oechem.OEHasSDData(mol, 'f_avg_pIC50'):
        return False

    try:
        if oechem.OEHasSDData(mol, 'f_avg_pIC50'):
            pIC50 = oechem.OEGetSDData(mol, 'f_avg_pIC50')
            pIC50 = float(pIC50)
            return True
        else:
            return False
    except Exception as e:
        return False

# TODO: import this from https://github.com/postera-ai/COVID_moonshot_submissions/blob/master/lib/utils.py
def get_series(mol):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    series_SMARTS_dict = {
        "chromane-5spiro-isoquinoline" : "c1ccc2c(c1)OCCC23***(c4cncc5ccccc45)*3",
        "tetrahydroisoquinoline-5spiro-isoquinoline" : "c1ccc2c(c1)CNCC23***(c4cncc5ccccc45)*3",
        "5spiro-isoquinoline" : "C3***(c4cncc5ccccc45)*3",
        "isoquinoline" : "c4cncc5ccccc45",
        "P1-biaryl" : "a4anaa5aaaaa45",
        #"chlorobenzene-linker": "NC(=O)Cc3[c;H1][c;H1][c;H1]c(Cl)c3",
        #"3-aminopyridine": "[R1][C,N;R0;!$(NC(=O)CN)]C(=O)[C,N;R0;!$(NC(=O)CN)][c]1cnccc1",
        #"3-aminopyridine-like": "[R1]!@[C,N]C(=O)[C,N]!@[R1]",
        #"3-aminopyridine-strict": "c1ccncc1NC(=O)!@[R1]",
        #"Ugi": "[c,C:1][C](=[O])[N]([c,C,#1:2])[C]([c,C,#1:3])([c,C,#1:4])[C](=[O])[NH1][c,C:5]",
        #"quinolones": "NC(=O)c1cc(=O)[nH]c2ccccc12",
        #"piperazine-chloroacetamide": "O=C(CCl)N1CCNCC1",
        #'benzotriazoles': 'c1ccc(NC(=O)[C,N]n2nnc3ccccc32)cc1',
        #'benzotriazoles': 'a1aaa([C,N]C(=O)[C,N]a2aaa3aaaaa32)aa1',
        #'benzotriazoles': 'a2aaa3aaaaa32',
    }

    smi = oechem.OECreateSmiString(mol)

    # Filter out covalent
    try:
        covalent_warheads = ['acrylamide', 'chloroacetamide']
        for warhead in covalent_warheads:
            if oechem.OEHasSDData(mol, warhead) and oechem.OEGetSDData(mol, warhead)=='True':
                return None
    except Exception as e:
        logging.warning(e)

    def check_if_smi_in_series(
        smi, SMARTS, MW_cutoff=550, num_atoms_cutoff=70, num_rings_cutoff=10
    ):
        mol = Chem.MolFromSmiles(smi)
        MW = Chem.Descriptors.MolWt(mol)
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        patt = Chem.MolFromSmarts(SMARTS)
        if (
            (
                len(
                    Chem.AddHs(Chem.MolFromSmiles(smi)).GetSubstructMatches(
                        patt
                    )
                )
                > 0
            )
            and (MW <= MW_cutoff)
            and (num_heavy_atoms <= num_atoms_cutoff)
            and (num_rings <= num_rings_cutoff)
        ):
            return True
        else:
            return False

    for series in series_SMARTS_dict:
        series_SMARTS = series_SMARTS_dict[series]
        if series == "3-amonipyridine-like":
            if check_if_smi_in_series(
                smi,
                series_SMARTS,
                MW_cutoff=410,
                num_rings_cutoff=3,
                num_atoms_cutoff=28,
            ):
                return series
        else:
            if check_if_smi_in_series(smi, series_SMARTS):
                return series
    return None

#def generate_restricted_conformers_star(args):
#    return generate_restricted_conformers(*args)

def generate_restricted_conformers_star(args):
    # TODO: Read this from setup.yaml and cache it in a process-safe manner
    import yaml
    with open('setup.yaml', 'r') as infile:
        setup_options = yaml.safe_load(infile)
    core_smarts_list = setup_options['core_smarts']

    # Build with all core_smarts options
    mols = [ generate_restricted_conformers(*args, core_smarts=core_smarts) for core_smarts in core_smarts_list ]

    # Prune unsuccessful builds
    mols = [ mol for mol in mols if mol!=None ]

    # DEBUG
    if len(mols) == 0:
        #  No options available
        logging.warning('No core_smarts variations succeeded.')
        return None

    if len(mols) == 1:
        # Return the only choice
        return mols[0]

    try:
        from openeye import oechem
        energies = [ float(oechem.OEGetSDData(mol, 'MMFF_internal_energy')) for mol in mols ]

        # Return the best compound
        import numpy as np
        index = np.argmin(energies)
        return mols[index]

        # Prefer the second option if strain energy of first option is poor
        #if energies[0] > energies[1] + 100:
        #    return mols[1]
        #else:
        #    return mols[0]

    except Exception as e:
        logging.warning(e)
        return None


def generate_restricted_conformers_best(args):
    # Unpack arguments
    receptor, refmol, mols = args

    # TODO: Read this from setup.yaml and cache it in a process-safe manner
    import yaml
    with open('setup.yaml', 'r') as infile:
        setup_options = yaml.safe_load(infile)
    core_smarts_list = setup_options['core_smarts']

    # Expand uncertain stereochemistry
    logging.info('Expanding uncertain stereochemistry...')
    mols = expand_stereochemistry_serial(mols)
    #logging.info(f'  There are {len(target_molecules)} target molecules after expanding uncertain stereochemistry')

    # Expand protonation states 
    #logging.info('Expanding protonation states...')
    #mols = expand_protonation_states(mols)
    #logging.info(f'  There are {len(target_molecules)} target molecules after expanding protonation states')

    # Expand uncertain stereochemistry
    #logging.info('Expanding uncertain stereochemistry...')
    #mols = expand_stereochemistry_serial(mols)
    #logging.info(f'  There are {len(target_molecules)} target molecules after expanding uncertain stereochemistry')

    # Build with all core_smarts options
    mols = [ generate_restricted_conformers(receptor, refmol, mol, core_smarts=core_smarts) for core_smarts in core_smarts_list for mol in mols ]

    # Prune unsuccessful builds
    mols = [ mol for mol in mols if mol!=None ]

    # DEBUG
    if len(mols) == 0:
        #  No options available
        logging.warning('No core_smarts variations succeeded.')
        return None

    try:
        from openeye import oechem

        # filter out mols with large energy
        energies = [ float(oechem.OEGetSDData(mol, 'MMFF_internal_energy')) for mol in mols ]
        print(max(energies))


        ENERGY_CUTOFF = 100.0
        ENERGY_CUTOFF = 200.0 # DEBUG
        mols = [ mol for mol in mols if float(oechem.OEGetSDData(mol, 'MMFF_internal_energy')) < ENERGY_CUTOFF ]
        #energies = [ float(oechem.OEGetSDData(mol, 'MMFF_internal_energy')) for mol in mols ]
        # Return best docking score
        energies = [ float(oechem.OEGetSDData(mol, 'docking_score')) for mol in mols ]

        # Return the best compound
        import numpy as np
        index = np.argmin(energies)
        mol = mols[index]

        # Prefer the second option if strain energy of first option is poor
        #if energies[0] > energies[1] + 100:
        #    return mols[1]
        #else:
        #    return mols[0]

    except Exception as e:
        logging.warning(e)
        return None

    # Remove warts
    #title = mol.GetTitle()
    #title = title.rsplit('_')[0]
    #mol.SetTitle(title)

    return mol


def generate_poses(receptor, refmol, target_molecules, output_filename, expand_protonation_states=False):
    """
    Parameters
    ----------
    receptor : openeye.oechem.OEGraphMol
        Receptor (already prepped for docking) for identifying optimal pose
    refmol : openeye.oechem.OEGraphMol
        Reference molecule which shares some part in common with the proposed molecule
    target_molecules : list of OEMol
        List of molecules to build
    output_filename : str
        Output filename for generated conformers
    expand_protonation_states : bool, optional, default=False
        If True, expand protonation states for all ligands.
    """
    logging.info(f'Generating poses for {len(target_molecules)} molecules...')

    # Expand uncertain stereochemistry
    logging.info('Expanding uncertain stereochemistry...')
    target_molecules = expand_stereochemistry(target_molecules)
    logging.info(f'  There are {len(target_molecules)} target molecules after expanding uncertain stereochemistry')

    # Expand protonation states 
    logging.info('Expanding protonation states...')
    target_molecules = expand_protonation_states_oemols(target_molecules)
    logging.info(f'  There are {len(target_molecules)} target molecules after expanding protonation states')

    # Expand uncertain stereochemistry
    #logging.info('Expanding uncertain stereochemistry...')
    #target_molecules = expand_stereochemistry(target_molecules)
    #logging.info(f'  There are {len(target_molecules)} target molecules after expanding uncertain stereochemistry')

    # Identify optimal conformer for each molecule
    n_poses = 0
    with oechem.oemolostream(output_filename) as ofs:
        from rich.progress import track
        from multiprocessing import Pool
        from tqdm import tqdm

        import os
        processes = None
        if 'LSB_DJOB_NUMPROC' in os.environ:
            processes = int(os.environ['LSB_DJOB_NUMPROC'])
        pool = Pool(processes=processes)
        args = [ (receptor, refmol, mol) for mol in target_molecules ]
        # TODO: Use pool.imap but reassemble docked poses in order
        #for pose in track(pool.imap_unordered(generate_restricted_conformers_star, args), total=len(args), description='Enumerating conformers...'):
        #for pose in track(pool.imap_unordered(generate_restricted_conformers_best, args), total=len(args), description='Enumerating conformers...'):
        for arg in args:
            pose = generate_restricted_conformers_best(arg)
            if pose is not None:
                oechem.OEWriteMolecule(ofs, pose)
                n_poses += 1

        pool.close()
        pool.join()

    logging.info(f'  {n_poses} molecule microstates were successfully docked')


        #for mol in tqdm(target_molecules):
        #    pose = generate_restricted_conformers(receptor, core_fragment, mol)
        #    if pose is not None:
        #        oechem.OEWriteMolecule(ofs, pose)

def filter_by_series(mols, filter_series='3-aminopyridine-like'):
    """
    Filter series and include only those that include the required scaffold

    Parameters
    ----------
    mols : list of OEMol
        List of molecules to be filtered

    Returns
    -------
    filtered_mols : list of OEMol
        Filtered list of molecules that match series
    """
    filtered_mols = [ mol for mol in mols if (get_series(mol) == filter_series) ]
    logging.info(f'Filtering by series {filter_series} retains {len(filtered_mols)} / {len(mols)} molecules.')
    return filtered_mols

def filter_by_IC50s(mols):
    """
    Retain only molecules with IC50s
    """
    filtered_mols = [ mol for mol in mols if has_ic50(mol) ]
    logging.info(f'Filtering only molecules with IC50s retains {len(filtered_mols)} / {len(mols)} molecules.')
    return filtered_mols

def load_molecules(filename, append_warts=False):
    """
    Read molecules from the specified file, appending warts (suffixes) to molecules with identical names if desired

    Parameters
    ----------
    filename : str
        The file name from which to read molecules in a format that OpenEye supports
    append_warts : bool, optional, default=False

    Returns
    -------
    target_molecules : list of OEMol
        The list of molecules read (without conformers if none were provided in file)
    """
    from openeye import oechem
    target_molecules = list()
    from collections import defaultdict
    molecule_names_count = defaultdict(int)
    # TODO: Fix this so it doesn't skip over molecules with the smae name
    # Perhaps we can append unique "warts" to each molecule?
    mol = oechem.OEGraphMol()
    with oechem.oemolistream(filename) as ifs:
        while oechem.OEReadMolecule(ifs, mol):
            # Add wart if requested
            if append_warts:
                title = mol.GetTitle()
                molecule_names_count[title] += 1
                mol.SetTitle(title + f'_{molecule_names_count[title]}')

            # TODO: Normalize
            oechem.OEAddExplicitHydrogens(mol)

            # Store a copy
            target_molecules.append( oechem.OEGraphMol(mol) )

    print(f'{len(target_molecules)} molecules loaded.')

    return target_molecules

def load_fragment(fragid, protonation_state, title=None):
    """
    Load the molecule associated with the given fragment

    Parameters
    ----------
    fragid : str
        The XChem fragment ID to load
    protonation_state : str
        The protonation state
    title : str, optional, default=None
        If not None, replace title with specified string.

    Returns
    -------
    refmol : OEMol
        The loaded fragment molecules as an OEMol (with coordinates)

    """
    #refmol_filename = f'receptors/dimer/Mpro-P0157-ligand.mol2'
    #refmol_filename = f'receptors/dimer/Mpro-x10959-ligand.mol2'
    refmol_filename = f'./receptors/dimer/Mpro-{fragid}_bound-{protonation_state}-ligand.mol2'
    refmol = None
    with oechem.oemolistream(refmol_filename) as ifs:
        for mol in ifs.GetOEGraphMols():
            refmol = mol
            break
    if refmol is None:
        raise Exception(f'Could not read {refmol_filename}')
    logging.info(f'Reference molecule has {refmol.NumAtoms()} atoms')

    if title is not None:
        # Replace title
        refmol.SetTitle(title)

    return refmol

def prepend_fragment_molecules(mols, fragments):
    """
    Prepend fragment molecules if they don't already exist in the list of molecules

    Parameters
    ----------
    mols : list of OEMol
        List of molecules to which fragment molecules are to be prepended
    fragments : dict of str : str
        fragments[fragid] is the PostEra CID of the corresponding fragment
    """
    titles = set([mol.GetTitle() for mol in mols])

    # Insert any molecules that are not found, preserving order
    ninserted = 0
    for fragid, cid in fragments.items():
        if cid not in titles:
            # Add the molecule
            fragment = load_fragment(fragid, title=cid)
            mols.insert(ninserted, fragment)
            ninserted += 1

def write_molecules(mols, filename):
    """
    Write molecules to the specified filename

    Parameters
    ----------
    mols : list of OEMol
        Molecules to write
    filename : str
        The filename to be written to
    """
    with oechem.oemolostream(filename) as ofs:
        for mol in mols:
            oechem.OEWriteMolecule(ofs, oechem.OEGraphMol(mol))

def read_receptor(fragid, assembly_state, protonation_state):
    """
    Read the receptor for the given fragment

    Parameters
    ----------
    fragid : str
        The XChem fragment ID
    assembly_state : str
        Assembly state ['monomer', 'dimer']
    protonation_state : str
        Protonation state ['neutral', 'charged']

    Returns
    -------
    receptor : OEReceptor
        The receptor

    """
    from openeye import oechem
    receptor = oechem.OEGraphMol()
    receptor_filename = f'./receptors/dimer/Mpro-{fragid}_bound-{protonation_state}-receptor.oeb.gz'
    from openeye import oedocking
    oedocking.OEReadReceptorFile(receptor, receptor_filename)
    logging.info(f'  Receptor has {receptor.NumAtoms()} atoms.')
    return receptor

if __name__ == '__main__':
    # TODO: Figure out why this single-threading workaround is needed to prevent issues
    from openeye import oechem
    #oechem.OESetMemPoolMode(oechem.OEMemPoolMode_SingleThreaded |
    #                        oechem.OEMemPoolMode_UnboundedCache)

    # XChem fragment structures to dock to
    # XChem FragID : PostEra CID
    # TODO: Try fragalysis structure without sulfonamide too?
    # Load setup.yaml
    import yaml
    with open('setup.yaml', 'r') as infile:
        setup_options = yaml.safe_load(infile)

    # Extract setup options
    fragments = { entry['fragalysis_id'] : entry['postera_id'] for entry in setup_options['source_structures'] }
    
    # Protonation states to dock to
    receptor_protonation_states = setup_options['receptor_protonation_states']

    # Whether to expand protonation states for ligands
    expand_protonation_states = setup_options.get('expand_protonation_states', False)

    # Molecule sets dock
    # This is just the prefix to pull from f"sorted/{prefix}.csv"
    molecules_path = setup_options['molecules_path']
    molecule_sets_to_dock = setup_options['molecule_sets_to_dock']

    for prefix in molecule_sets_to_dock:
        # Read molecules to dock
        target_molecules_filename = molecules_path + '/' + prefix + f'.csv'
        logging.info(f'Reading molecules to be docked from {target_molecules_filename}...')
        target_molecules = load_molecules(target_molecules_filename, append_warts=False)

        # Ensure fragment molecules are contained in set
        #prepend_fragment_molecules(target_molecules, fragments)

        # Generate list of all compounds
        import os
        os.makedirs('docked', exist_ok=True)
        output_filename = f'docked/{prefix}-compounds.smi'
        logging.info(f'Writing annotated molecules to {output_filename}...')
        write_molecules(target_molecules, output_filename)

        # Generate docked poses from each fragment:receptor pair
        for fragment in fragments:
            #for assembly_state in ['monomer', 'dimer']:
            for assembly_state in ['dimer']:
                #for protonation_state in ['neutral', 'charged']:
                for receptor_protonation_state in receptor_protonation_states:
                    # Read receptor
                    logging.info('Reading receptor...')
                    receptor = read_receptor(fragment, assembly_state, receptor_protonation_state)

                    # Read reference fragment with coordinates
                    refmol = load_fragment(fragment, receptor_protonation_state, title=fragments[fragment])

                    # Prepend reference molecule
                    from openeye import oechem
                    molecules_to_dock = target_molecules

                    write_molecules(molecules_to_dock, 'test.smi')
                    
                    # Generate poses for all molecules
                    output_filename = f'docked/{prefix}-{fragment}-{assembly_state}-{receptor_protonation_state}.sdf'
                    logging.info(f'Writing poses to {output_filename}...')
                    generate_poses(receptor, refmol, molecules_to_dock, output_filename, expand_protonation_states=expand_protonation_states)
