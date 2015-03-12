import numpy as np

import tempfile
import commands
from math import *
from openeye.oechem import *
from openeye.oeomega import *
from openeye.oeiupac import *
from openeye.oeshape import *
try:
   from openeye.oequacpac import * #DLM added 2/25/09 for OETyperMolFunction; replacing oeproton
except:
   from openeye.oeproton import * #GJR temporary fix because of old version of openeye tools
from openeye.oeiupac import *
from openeye.oeszybki import *
import os
import re
import shutil

#=============================================================================================
# METHODS FOR WRITING OR EXPORTING MOLECULES
#=============================================================================================

def totalPartialCharge(ligand):
    """Compute the total partial charge of all atoms in molecule.

    ARGUMENTS
      ligand (OEMol) - the molecule

    RETURNS
      total_charge (float) - the total charge
    """

    total_charge = 0.0
    for atom in ligand.GetAtoms():
        total_charge += atom.GetPartialCharge()

    return total_charge

def totalIntegralPartialCharge(ligand):
    """Compute the total partial charge of all atoms in molecule and return nearest integer.

    ARGUMENTS
      ligand (OEMol) - the molecule

    RETURNS
      total_charge (int) - the total charge rounded to the nearest integer
    """

    total_charge = 0.0
    for atom in ligand.GetAtoms():
        total_charge += atom.GetPartialCharge()

    total_charge = int(round(total_charge))

    return total_charge

def adjustCharges(q_n, target_net_charge):
    """Adjust charges q_n to have desired net integral charge but closely match original charges.

    New charged qnew_n are defined by

      qnew_n = alpha (q_n - mu) + b

    where
      mu = Q/N
      Q = \sum_n q_n

    and alpha and b are chosen to minimize RMS(qnew_n - q_n) while guaranteeing \sum_n qnew_n = net_charge.      

    ARGUMENTS
      q_n (numpy float array of length N) - original charges
        CHANGELOG:
            11/15/10: DLM, switched exception handling from raising a string to using a class, as string exceptions are no longer supported in newer Python versions.target_net_charge (integer or float) - the desired net charge

    RETURNS
      qnew_n (numpy float array of length N) - adjusted charges
    """

    # get number of charges
    N = q_n.size

    # compute original total charge
    Q = sum(q_n)

    # compute average charge
    mu = Q / float(N)

    # determine scaling constant alpha and shift value b to minimize RMS and guarantee new net charge
    alpha = sum((q_n - mu)**2 * (1.0 - mu/q_n)) / sum((q_n-mu)**2)
    b = 0

    # compute new charges
    qnew_n = alpha * (q_n - mu) + b

    # check constraints
    Qnew = sum(qnew_n)
    if (abs(Qnew - target_net_charge) > 0.01):
        print "%6s %6s" % ('old', 'new')
        for n in range(N):
            print "%6.3f %6.3f" % (q_n[n], qnew_n[n])                                  
        raise "adjustCharges failed: original net charge %f, new net charge is %f" % (Q, Qnew)
    
    # compute RMS
    rms = sqrt( (1.0/float(N)) * sum((qnew_n - q_n)**2) )
    print "rms fit = %f" % rms

    # return new charges
    return qnew_n

def determineCommonSubstructure(ligands, verbose = False, min_atoms = 4):
    """Find a common substructure shared by all ligands.

    The atom type name strings and integer bond types are used to obtain an exact match.

    ARGUMENTS
      ligands (list of OEMol) - the set of ligands for which the common substructure is to be determined.

    OPTIONAL ARGUMENTS
      verbose (boolean) - if True, verbose information is printed (default: False)
      min_atoms (int) - minimum number of atoms for substructure match (default: 4)

    RETURN VALUES
      common_substructure (OEMol) - a molecule fragment representing the common substructure

    CHANGELOG
      11/15/10 - DLM: Removing the attempt to convert OEMols to OEMol with OEMol( as this raises an exception in my (newer) OEChem.
    """
    # Determine number of ligands
    nligands = len(ligands)

    # First, initialize with first ligand.
    common_substructure = ligands[0].CreateCopy() #DLM modification 11/15/10 -- this is how copies should now be made

    # VERBOSE: Show original atoms
    if verbose:
        atom_index = 1
        for atom in common_substructure.GetAtoms():
            print "%5d %6s %12s" % (atom_index, atom.GetName(), atom.GetType())
            atom_index += 1
        print ""

    # Now delete bits that don't match every other ligand.
    for ligand in ligands[1:]:
        # get ligand name
        ligand_name = ligand.GetTitle()

        # Create an OEMCSSearch from this molecule.
        mcss = OEMCSSearch(ligand, OEExprOpts_StringType, OEExprOpts_IntType)

        # ignore substructures smaller than 4 atoms
        mcss.SetMinAtoms(min_atoms)

        # This modifies scoring function to prefer keeping cycles complete.
        mcss.SetMCSFunc( OEMCSMaxAtomsCompleteCycles() )

       # perform match
        for match in mcss.Match(common_substructure):
            nmatched = match.NumAtoms()

            if verbose: print "%(ligand_name)s : match size %(nmatched)d atoms" % vars()

            # build list of matched atoms in common substructure
            matched_atoms = list()
            for matchpair in match.GetAtoms():
                atom = matchpair.target
                matched_atoms.append(atom)

            # delete all unmatched atoms from common substructure
            for atom in common_substructure.GetAtoms():
                if atom not in matched_atoms:
                    common_substructure.DeleteAtom(atom)

            if verbose:
                # print all retained atoms
                atom_index = 1
                for atom in common_substructure.GetAtoms():
                    print "%5d %6s %12s" % (atom_index, atom.GetName(), atom.GetType())
                    atom_index += 1
                print ""

            # we only need to consider one match
            break

    # Rename common substructure.
    common_substructure.SetTitle('core')

    # return the common substructure
    return common_substructure

def determineMinimumRMSCharges(common_substructure, ligands, debug = False, min_atoms = 4):
    """Determine charges for common substructure that minimize root-mean-squared (RMS) deviation to all ligands while having same net charge as ligands.

    ARGUMENTS
      common_substructure (OEMol) - the common substructure (partial molecule) that matches the ligands
      ligands (list of OEMol) - the set of ligands for which the common_substructure is a shared component

    OPTIONAL ARGUMENTS
      debug (boolean) - if True, debug information is printed (default: False)
      min_atoms (int) - minimum number of atoms for MCSS match (default: 4)
      
    RETURNS
      charged_substructure (OEMol) - the common substructure with atomic charges updated

    NOTES
      An exception will be thrown if the ligands do not have the same total integral charge.
    
    REFERENCES
      [1] Betts JT. A compact algorithm for computing the stationary point of a quadratic function subject to linear constraints. ACM Trans Math Soft 6(3):391-397, 1980.
      
    CHANGELOG:
        11/15/10: DLM, switched exception handling from raising a string to using a class, as string exceptions are no longer supported in newer Python versions.

"""

    # Make a deep copy.
    ligands = [ ligand.CreateCopy() for ligand in ligands ]

    if debug: print "Determining minimum RMS charges..."

    # maximum allowed total charge deviation before raising an exception
    CHARGE_DEVIATION_TOLERANCE = 0.01

    # determine total integral charge on first ligand
    Q = totalIntegralPartialCharge(ligands[0])
    if debug: print "Q = %f" % Q


    class ChargeError(Exception):
        def __init__(self, msg, expr):
            self.expr = expr
            self.msg = msg
            print msg
            print "Exiting..."
        

    # check to make sure all ligands have same net charge
    for ligand in ligands:
        if abs(totalPartialCharge(ligand) - Q) > CHARGE_DEVIATION_TOLERANCE:
            #raise "Ligand %s has charge (%f) different from target charge (Q = %f) - tolerance is %f." % (ligand.GetTitle(), totalPartialCharge(ligand), Q, CHARGE_DEVIATION_TOLERANCE)
            raise ChargeError( "Ligand %s has charge (%f) different from target charge (Q = %f) - tolerance is %f." % (ligand.GetTitle(), totalPartialCharge(ligand), Q, CHARGE_DEVIATION_TOLERANCE), 'abs(totalPartialCharge(ligand) - Q) > CHARGE_DEVIATION_TOLERANCE' )

    # determine number of ligands
    K = len(ligands)
    if debug: print "K = %d" % K
          
    # determine number of atoms in common substructure
    N = common_substructure.NumAtoms()
    if debug: print "N = %d" % N

    # build index of which atoms correspond to which atom indices
    atom_indices = dict() # atom_indices[atom] is the index (0...(N-1)) of atom 'atom'
    n = 0
    for atom in common_substructure.GetAtoms():
        atomname = atom.GetName()
        atom_indices[atomname] = n
        n += 1

    # extract charge on all ligands for common set
    q_kn = np.zeros([K,N], np.float64) # q_kn[k,n] is the charge on the atom of ligand k corresponding to atom n of the common substructure
    k = 0 # ligand index
    mcss = OEMCSSearch(common_substructure, OEExprOpts_StringType, OEExprOpts_IntType)
    mcss.SetMinAtoms(min_atoms)
    for ligand in ligands:
        print ligand.GetTitle()
        # perform match
        for match in mcss.Match(ligand):
            # only store the first match
            for matchpair in match.GetAtoms():
                atomname = matchpair.pattern.GetName()
                n = atom_indices[atomname]
                q_kn[k,n] += matchpair.target.GetPartialCharge()
            break
        # increment ligand counter
        k += 1

    # compute average charge on common substructure over all ligands
    qavg_n = np.mean(q_kn, 0)
    print qavg_n.shape

    # solve linear programming problem to determine minimum RMS charge set subject to constraint that total charge equals Q
    #
    # See reference [1] for information on how the matrix is set up.
    #
    # We solve the linear programming problem A x = b where
    #
    # A = [I 1; 1' 0]
    # b = [qavg_n ; Q]
    # x = [q_n ; lambda]

    A = np.zeros([N+1,N+1], np.float64)
    A[0:N,0:N] = np.eye(N)
    A[0:N,N] = np.ones([N])
    A[N,0:N] = np.ones([N])

    b = np.zeros([N+1,1], np.float64)
    b[0:N,0] = qavg_n[0:N]
    b[N,0] = Q

    Ainv = np.linalg.pinv(A)
    x = np.matrix(Ainv) * np.matrix(b)

    # extract charges
    q_n = np.array(x[0:N])
    lagrange_multiplier = x[N]
    if debug: print "lagrange multiplier is %f" % lagrange_multiplier

    # DISPLAY CHARGES
    if debug:
        # header
        print "%6s" % 'fit',
        for k in range(K):
            print " %6d" % k,
        print ""
        # charges
        for n in range(N):
            print "%6.3f" % q_n[n],
            for k in range(K):
                print " %6.3f" % q_kn[k,n],
            print ""
        print ""
        print ""
        # print total charges
        print "%6.3f" % q_n[:].sum(),
        for k in range(K):
            print " %6.3f" % q_kn[k,:].sum(),
        print ""

    # make a copy of the common substructure to assign new charges to
    charged_common_substructure = OEMol(common_substructure)

    # assign charges
    n = 0
    for atom in charged_common_substructure.GetAtoms():
        charge = float(q_n[n])
        atom.SetPartialCharge(charge)
        n += 1

    # return copy of common substructure with partial charges assigned
    return charged_common_substructure

def round_charges( mol2file, verbose = True ):
    """OE tools tend to write out mol2 files which have noninteger charges by a significant margin, such as 3e-4 or so. Read in a mol2 file, and ensure that the partial charge is really an integer, writing it back out to the same file. OEMol is used to read the molecule but then charges are modified by directly editing the lines."""

    oemol = readMolecule( mol2file)
    totchg = totalPartialCharge( oemol )
    totintchg = totalIntegralPartialCharge( oemol )
    #Compute charge difference
    diff = totintchg - totchg

    if diff > 1e-6 or diff < -1e-6: 
        if verbose: print "Total charge is %.2g; will be set to %s." % (totchg, totintchg)
        #Compute value to change charges by to ensure zero
        modifier = diff/float( oemol.NumAtoms() )
    else:
        modifier = 0

    newcharge = {}
    for atom in oemol.GetAtoms():
        newcharge[ atom.GetName() ] = atom.GetPartialCharge() + modifier

    #Now modify charges by editing mol2 file
    if modifier > 0 or modifier < 0:
        file = open( mol2file, 'r' )
        text = file.readlines()
        file.close()
        atomstart = text.index('@<TRIPOS>ATOM\n')
        atomend = text.index('@<TRIPOS>BOND\n')
        for linenum in range(atomstart+1, atomend):
            tmp = text[linenum].split()
            atomname = tmp[1]
            strcharge = tmp[8]
            thischg = newcharge[atomname]
            text[linenum] = text[linenum].replace( strcharge, '%.7f' % thischg)
        file = open( mol2file, 'w')
        file.writelines(text)
        file.close()

    return True

def readMolecule(filename, normalize = False):
   """Read in a molecule from a file (such as .mol2).

   ARGUMENTS
     filename (string) - the name of the file containing a molecule, in a format that OpenEye autodetects (such as .mol2)

   OPTIONAL ARGUMENTS
     normalize (boolean) - if True, molecule is normalized (renamed, aromaticity, protonated) after reading (default: False)

   RETURNS
     molecule (OEMol) - OEMol representation of molecule

   EXAMPLES
     # read a mol2 file
     molecule = readMolecule('phenol.mol2')
     # works with any type of file that OpenEye autodetects
     molecule = readMolecule('phenol.sdf')
   """

   # Open input stream.
   istream = oemolistream()
   istream.open(filename)

   # Create molecule.
   molecule = OEMol()   

   # Read the molecule.
   OEReadMolecule(istream, molecule)

   # Close the stream.
   istream.close()

   # Normalize if desired.
   if normalize: normalizeMolecule(molecule)

   return molecule

#=============================================================================================
# METHODS FOR INTERROGATING MOLECULES
#=============================================================================================

def formalCharge(molecule):
   """Report the net formal charge of a molecule.

   ARGUMENTS
     molecule (OEMol) - the molecule whose formal charge is to be determined

   RETURN VALUES
     formal_charge (integer) - the net formal charge of the molecule

   EXAMPLE
     net_charge = formalCharge(molecule)
   """

   # Create a copy of the molecule.
   molecule_copy = OEMol(molecule)

   # Assign formal charges.
   OEFormalPartialCharges(molecule_copy)

   # Compute net formal charge.
   formal_charge = int(round(OENetCharge(molecule_copy)))

   # return formal charge
   return formal_charge

def writeMolecule(molecule, filename, substructure_name = 'MOL', preserve_atomtypes = False):
   """Write a molecule to a file in any format OpenEye autodetects from filename (such as .mol2).
   WARNING: The molecule will be standardized before writing by the high-level OEWriteMolecule function.
   OEWriteConstMolecule is used, to avoid changing the molecule you pass in.

   ARGUMENTS
     molecule (OEMol) - the molecule to be written
     filename (string) - the file to write the molecule to (type autodetected from filename)

   OPTIONAL ARGUMENTS
     substructure_name (String) - if a mol2 file is written, this is used for the substructure name (default: 'MOL')
     preserve_atomtypes (bool) - if True, a mol2 file will be written with atom types preserved

   RETURNS
     None

   NOTES
     Multiple conformers are written.

   EXAMPLES
     # create a molecule
     molecule = createMoleculeFromIUPAC('phenol')
     # write it as a mol2 file
     writeMolecule(molecule, 'phenol.mol2')
   """

   # Open output stream.
   ostream = oemolostream()
   ostream.open(filename)

   # Define internal function for writing multiple conformers to an output stream.
   def write_all_conformers(ostream, molecule):
      # write all conformers of each molecule
      for conformer in molecule.GetConfs():
         if preserve_atomtypes: OEWriteMol2File(ostream, conformer)
         else: OEWriteConstMolecule(ostream, conformer)
      return

   # If 'molecule' is actually a list of molecules, write them all.
   if type(molecule) == type(list()):
      for individual_molecule in molecule:
         write_all_conformers(ostream, individual_molecule)
   else:
      write_all_conformers(ostream, molecule)

   # Close the stream.
   ostream.close()

   # Replace substructure name if mol2 file.
   suffix = os.path.splitext(filename)[-1]
   if (suffix == '.mol2' and substructure_name != None):
      modifySubstructureName(filename, substructure_name)

   return

def modifySubstructureName(mol2file, name):
   """Replace the substructure name (subst_name) in a mol2 file.

   ARGUMENTS
     mol2file (string) - name of the mol2 file to modify
     name (string) - new substructure name

   NOTES
     This is useful becuase the OpenEye tools leave this name set to <0>.
     The transformation is only applied to the first molecule in the mol2 file.

   TODO
     This function is still difficult to read.  It should be rewritten to be comprehensible by humans.
   """

   # Read mol2 file.
   file = open(mol2file, 'r')
   text = file.readlines()
   file.close()

   # Find the atom records.
   atomsec = []
   ct = 0
   while text[ct].find('<TRIPOS>ATOM')==-1:
     ct+=1
   ct+=1
   atomstart = ct
   while text[ct].find('<TRIPOS>BOND')==-1:
     ct+=1
   atomend = ct

   atomsec = text[atomstart:atomend]
   outtext=text[0:atomstart]
   repltext = atomsec[0].split()[7] # mol2 file uses space delimited, not fixed-width

   # Replace substructure name.
   for line in atomsec:
     # If we blindly search and replace, we'll tend to clobber stuff, as the subst_name might be "1" or something lame like that that will occur all over. 
     # If it only occurs once, just replace it.
     if line.count(repltext)==1:
       outtext.append( line.replace(repltext, name) )
     else:
       # Otherwise grab the string left and right of the subst_name and sandwich the new subst_name in between. This can probably be done easier in Python 2.5 with partition, but 2.4 is still used someplaces.
       # Loop through the line and tag locations of every non-space entry
       blockstart=[]
       ct=0
       c=' '
       for ct in range(len(line)):
         lastc = c
         c = line[ct]
         if lastc.isspace() and not c.isspace():
           blockstart.append(ct)
       line = line[0:blockstart[7]] + line[blockstart[7]:].replace(repltext, name, 1)
       outtext.append(line)
       
   # Append rest of file.
   for line in text[atomend:]:
     outtext.append(line)
     
   # Write out modified mol2 file, overwriting old one.
   file = open(mol2file,'w')
   file.writelines(outtext)
   file.close()

   return

def parameterizeForAmber(molecule, topology_filename, coordinate_filename, charge_model = False, judgetypes = None, cleanup = True, show_warnings = True, verbose = False, resname = None, netcharge = None, offfile = None, ligand_obj_name = 'molecule', frcmod_filename = None):
   """Parameterize small molecule with GAFF and write AMBER coordinate/topology files.

   ARGUMENTS
     molecule (OEMol) - molecule to parameterize (only the first configuration will be used if multiple are present)
     topology_filename (string) - name of AMBER topology file to be written
     coordinate_filename (string) - name of AMBER coordinate file to be written

   OPTIONAL ARGUMENTS
     charge_model (string) - if not False, antechamber is used to assign charges (default: False) -- if set to 'bcc', for example, AM1-BCC charges will be used
     judgetypes (integer) - if provided, argument passed to -j of antechamber to judge types (default: None)
     cleanup (boolean) - clean up temporary files (default: True)
     show_warnings (boolean) - show warnings during parameterization (default: True)
     verbose (boolean) - show all output from subprocesses (default: False)
     resname (string) - if set, residue name to use for parameterized molecule (default: None)
     netcharge (integer) -- if set, pass this net charge to calculation in antechamber (with -nc (netcharge)), otherwise assumes zero.
     offfile (string) - name of AMBER off file to be written, optionally.
     ligand_obj_name - name of object to store ligand as (in off file); default 'molecule'.
     frmcmod_filename - name of frcmod file to be saved if desired (default: None)

   REQUIREMENTS
     acpypi.py conversion script (must be in MMTOOLSPATH)
     AmberTools installation (in PATH)

   EXAMPLES
     # create a molecule
     molecule = createMoleculeFromIUPAC('phenol')
     # parameterize it for AMBER, using antechamber to assign AM1-BCC charges
     parameterizeForAmber(molecule, topology_filename = 'phenol.prmtop', coordinate_filename = 'phenol.crd', charge_model = 'bcc')

   """

   molecule = OEMol(molecule)

   # Create temporary working directory and copy ligand mol2 file there.
   working_directory = tempfile.mkdtemp()
   old_directory = os.getcwd()
   os.chdir(working_directory)
   if verbose: print "Working directory is %(working_directory)s" % vars()

   # Write molecule to mol2 file.
   tripos_mol2_filename = os.path.join(working_directory, 'tripos.mol2')
   writeMolecule(molecule, tripos_mol2_filename)
   #Ensure charges are rounded to neutrality
   round_charges( tripos_mol2_filename)

   if resname:
      # Set substructure name (which will become residue name) if desired.
      modifySubstructureName(tripos_mol2_filename, resname)

   # Run antechamber to assign GAFF atom types.
   gaff_mol2_filename = os.path.join(working_directory, 'gaff.mol2')
   if netcharge:
      chargstr = '-nc %d' % netcharge
   else: chargestr=''
   command = 'antechamber -i %(tripos_mol2_filename)s -fi mol2 -o %(gaff_mol2_filename)s -fo mol2 %(chargestr)s' % vars()
   if judgetypes: command += ' -j %(judgetypes)d' % vars()
   if charge_model:
      formal_charge = formalCharge(molecule)
      command += ' -c %(charge_model)s -nc %(formal_charge)d' % vars()
   if verbose: print command
   output = commands.getoutput(command)
   if verbose or (output.find('Warning')>=0): print output

   # Generate frcmod file for additional GAFF parameters.
   frcmod_filename_tmp = os.path.join(working_directory, 'gaff.frcmod')
   commands.getoutput('parmchk2 -i %(gaff_mol2_filename)s -f mol2 -o %(frcmod_filename_tmp)s' % vars())

   # Create AMBER topology/coordinate files using LEaP.
   if offfile:
      offstring = 'saveOff %(ligand_obj_name)s amber.off\n' % vars()
   else:
      offstring = ''
   leapscript = """\
source leaprc.gaff
mods = loadAmberParams %(frcmod_filename_tmp)s
%(ligand_obj_name)s = loadMol2 %(gaff_mol2_filename)s
desc %(ligand_obj_name)s
check %(ligand_obj_name)s
saveAmberParm %(ligand_obj_name)s amber.prmtop amber.crd
%(offstring)s
quit""" % vars()
   leapin_filename = os.path.join(working_directory, 'leap.in')
   outfile = open(leapin_filename, 'w')
   outfile.write(leapscript)
   outfile.close()

   tleapout = commands.getoutput('tleap -f %(leapin_filename)s' % vars())
   if verbose: print tleapout
   tleapout = tleapout.split('\n')
   # Shop any warnings.
   if show_warnings:
      fnd = False
      for line in tleapout:
         tmp = line.upper()
         if tmp.find('WARNING')>-1: 
            print line
            fnd = True
         if tmp.find('ERROR')>-1: 
            print line
            fnd = True
      if fnd:
         print "Any LEaP warnings/errors are above."


   # Restore old directory.
   os.chdir(old_directory)

   # Copy gromacs topology/coordinates to desired output files.
   commands.getoutput('cp %s %s' % (os.path.join(working_directory, 'amber.crd'), coordinate_filename))
   commands.getoutput('cp %s %s' % (os.path.join(working_directory, 'amber.prmtop'), topology_filename))
   if offfile:
        commands.getoutput('cp %s %s' % (os.path.join(working_directory, 'amber.off'), offfile))
   if frcmod_filename:
        commands.getoutput('cp %s %s' % (os.path.join( working_directory, frcmod_filename_tmp), frcmod_filename ) )

   # Clean up temporary files.
   os.chdir(old_directory)
   if cleanup:
      commands.getoutput('rm -r %s' % working_directory)
   else:
      print "Work done in %s..." % working_directory

   return

#=============================================================================================
# METHODS FOR READING, EXTRACTING, OR CREATING MOLECULES
#=============================================================================================

#=============================================================================================
def createMoleculeFromIUPAC(name, verbose = False, charge = None, strictTyping = None, strictStereo = True):
   """Generate a small molecule from its IUPAC name.

   ARGUMENTS
     IUPAC_name (string) - IUPAC name of molecule to generate

   OPTIONAL ARGUMENTS
     verbose (boolean) - if True, subprocess output is shown (default: False)
     charge (int) - if specified, a form of this molecule with the desired charge state will be produced (default: None)
     strictTyping (boolean) -- if set, passes specified value to omega (see documentation for expandConformations)
     strictStereo (boolean) -- if True, require stereochemistry to be specified before running omega. If False, don't (pick random stereoisomer if not specified). If not specified (None), do whatever omega does by default (varies with version). Default: True.

   RETURNS
     molecule (OEMol) - the molecule

   NOTES
     OpenEye LexiChem's OEParseIUPACName is used to generate the molecle.
     The molecule is normalized by adding hydrogens.
     Omega is used to generate a single conformation.
     Also note that atom names will be blank coming from this molecule. They are assigned when the molecule is written, or one can assign using OETriposAtomNames for example.

   EXAMPLES
     # Generate a mol2 file for phenol.
     molecule = createMoleculeFromIUPAC('phenol')

   """

   # Create an OEMol molecule from IUPAC name.
   molecule = OEMol() # create a molecule
   status = OEParseIUPACName(molecule, name) # populate the molecule from the IUPAC name

   # Normalize the molecule.
   normalizeMolecule(molecule)

   # Generate a conformation with Omega
   omega = OEOmega()
   if strictStereo<>None:
        omega.SetStrictStereo(strictStereo)

   #omega.SetVerbose(verbose)
   #DLM 2/27/09: Seems to be obsolete in current OEOmega
   if strictTyping != None:
     omega.SetStrictAtomTypes( strictTyping)

   omega.SetIncludeInput(False) # don't include input
   omega.SetMaxConfs(1) # set maximum number of conformations to 1
   omega(molecule) # generate conformation

   if (charge != None):
      # Enumerate protonation states and select desired state.
      protonation_states = enumerateStates(molecule, enumerate = "protonation", verbose = verbose)
      for molecule in protonation_states:
         if formalCharge(molecule) == charge:
            # Return the molecule if we've found one in the desired protonation state.
            return molecule
      if formalCharge(molecule) != charge:
         print "enumerateStates did not enumerate a molecule with desired formal charge."
         print "Options are:"
         for molecule in protonation_states:
            print "%s, formal charge %d" % (molecule.GetTitle(), formalCharge(molecule))
         raise "Could not find desired formal charge."

   # Return the molecule.
   return molecule

def write_file(filename, contents):
    """Write the specified contents to a file.

    ARGUMENTS
      filename (string) - the file to be written
      contents (string) - the contents of the file to be written

    """

    outfile = open(filename, 'w')

    if type(contents) == list:
        for line in contents:
            outfile.write(line)
    elif type(contents) == str:
        outfile.write(contents)
    else:
        raise "Type for 'contents' not supported: " + repr(type(contents))

    outfile.close()

    return

def read_file(filename):
    """Read contents of the specified file.

    ARGUMENTS
      filename (string) - the name of the file to be read

    RETURNS
      lines (list of strings) - the contents of the file, split by line
    """

    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()

    return lines

def select_off_section(lines, label):
   """Return set of lines associated with a section.

   ARGUMENTS
      lines (list of strings) - the lines in the .off file
      label (string) - the section label to locate

   RETURNS
      section_lines (list of strings) - the lines in that section
   """

   section_lines = list()

   nlines = len(lines)
   for start_index in range(nlines):
       # get line
       line = lines[start_index].strip()
       # split into elements
       elements = line.split()
       # see if keyword is matched
       if (len(elements) > 0):
           if (elements[0] == '!' + label):
               # increment counter to start of section data and abort search
               start_index += 1
               break

   # throw an exception if section not found
   if (start_index == nlines):
       raise "Section %(label)s not found." % vars()

   # Locate end of section.
   for end_index in range(start_index, nlines):
       # get line
       line = lines[end_index].strip()
       # split into elements
       elements = line.split()
       # see if keyword is matched
       if (len(elements) > 0):
           if (elements[0][0]=='!'):
               break

   # return these lines
   return lines[start_index:end_index]

def loadGAFFMolecule(molecule, amber_off_filename, debug=False):
    """
    Replace the atom and bond types in an OEMol molecule with GAFF atom and bond types.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
       The molecule to have GAFF parameters read in.
    amber_off_filename : str
       The name of the AMBER .off file to read parameters from.

    Returns
    -------
    molecule : openeye.oechem.OEMol
       The updated molecule.

    """
    # read .off file
    off_lines = read_file(amber_off_filename)

    # build a dictionary of atoms
    atoms = dict()
    for atom in molecule.GetAtoms():
        name = atom.GetName()
        atoms[name] = atom

    # replace atom names with GAFF atom names
    section_lines = select_off_section(off_lines, 'entry.molecule.unit.atoms')
    off_atom_names = dict() # off_atom_names[i] is name of atom i, starting from 1
    atom_index = 1
    for line in section_lines:
        # parse section
        # !entry.molecule.unit.atoms table  str name  str type  int typex  int resx  int flags  int seq  int elmnt  dbl chg
        # "C1" "c3" 0 1 131072 1 6 -0.073210
        elements = line.split()
        name = elements[0].strip('"')
        type = elements[1].strip('"')
        charge = float(elements[7])
        # store changes
        atom = atoms[name]
        atom.SetType(type)
        atom.SetPartialCharge(charge)
        # store atom ordering
        off_atom_names[atom_index] = name
        atom_index += 1

    # build a dictionary of bonds
    bonds = dict()
    for bond in molecule.GetBonds():
        begin_atom_name = bond.GetBgn().GetName() # name of atom at one end of bond
        end_atom_name = bond.GetEnd().GetName() # name of atom at other end of the bond
        # store bond in both directions
        bonds[(begin_atom_name,end_atom_name)] = bond
        bonds[(end_atom_name,begin_atom_name)] = bond

    # replace bond types with GAFF integral bond types
    section_lines = select_off_section(off_lines, 'entry.molecule.unit.connectivity')
    for line in section_lines:
        # parse section
        elements = line.split()
        i = int(elements[0])
        j = int(elements[1])
        bondtype = int(elements[2])
        # get atom names
        begin_atom_name = off_atom_names[i]
        end_atom_name = off_atom_names[j]
        # store changes
        bond = bonds[(begin_atom_name,end_atom_name)]
        bond.SetIntType(bondtype)

    # DEBUG
#    print "atoms"
#    for atom in molecule.GetAtoms():
#        print '%4s %4s' % (atom.GetName(), atom.GetType())
#    print "\nbonds"
#    for bond in molecule.GetBonds():
#        print "%4s %4s %d" % (bond.GetBgn().GetName(), bond.GetEnd().GetName(), bond.GetIntType())
#    print ""

    return molecule

#=============================================================================================
# METHODS FOR MODIFYING MOLECULES
#=============================================================================================
def normalizeMolecule(molecule):
   """Normalize the molecule by checking aromaticity, adding explicit hydrogens, and renaming by IUPAC name.

   ARGUMENTS
     molecule (OEMol) - the molecule to be normalized.

   EXAMPLES
     # read a partial molecule and normalize it
     molecule = readMolecule('molecule.sdf')
     normalizeMolecule(molecule)
   """
   
   # Find ring atoms and bonds
   # OEFindRingAtomsAndBonds(molecule) 
   
   # Assign aromaticity.
   OEAssignAromaticFlags(molecule, OEAroModelOpenEye)   

   # Add hydrogens.
   OEAddExplicitHydrogens(molecule)

   # Set title to IUPAC name.
   name = OECreateIUPACName(molecule)
   molecule.SetTitle(name)

   return molecule

def expandConformations(molecule, maxconfs = None, threshold = None, include_original = False, torsionlib = None, verbose = False, strictTyping = None, strictStereo = None):   
   """Enumerate conformations of the molecule with OpenEye's Omega after normalizing molecule. 

   ARGUMENTS
   molecule (OEMol) - molecule to enumerate conformations for

   OPTIONAL ARGUMENTS
     include_original (boolean) - if True, original conformation is included (default: False)
     maxconfs (integer) - if set to an integer, limits the maximum number of conformations to generated -- maximum of 120 (default: None)
     threshold (real) - threshold in RMSD (in Angstroms) for retaining conformers -- lower thresholds retain more conformers (default: None)
     torsionlib (string) - if a path to an Omega torsion library is given, this will be used instead (default: None)
     verbose (boolean) - if True, omega will print extra information
     strictTyping (boolean) -- if specified, pass option to SetStrictAtomTypes for Omega to control whether related MMFF types are allowed to be substituted for exact matches.
     strictStereo (boolean) -- if specified, pass option to SetStrictStereo; otherwise use default.

   RETURN VALUES
     expanded_molecule - molecule with expanded conformations

   EXAMPLES
     # create a new molecule with Omega-expanded conformations
     expanded_molecule = expandConformations(molecule)

     
   """
   # Initialize omega
   omega = OEOmega()
   if strictTyping != None:
     omega.SetStrictAtomTypes( strictTyping)
   if strictStereo != None:
     omega.SetStrictStereo( strictStereo )
   #Set atom typing options

   # Set verbosity.
   #omega.SetVerbose(verbose)
   #DLM 2/27/09: Seems to be obsolete in current OEOmega

   # Set maximum number of conformers.
   if maxconfs:
      omega.SetMaxConfs(maxconfs)
     
   # Set whether given conformer is to be included.
   omega.SetIncludeInput(include_original)
   
   # Set RMSD threshold for retaining conformations.
   if threshold:
      omega.SetRMSThreshold(threshold) 
 
   # If desired, do a torsion drive.
   if torsionlib:
      omega.SetTorsionLibrary(torsionlib)

   # Create copy of molecule.
   expanded_molecule = OEMol(molecule)   

   # Enumerate conformations.
   omega(expanded_molecule)


   # verbose output
   if verbose: print "%d conformation(s) produced." % expanded_molecule.NumConfs()

   # return conformationally-expanded molecule
   return expanded_molecule

def assignPartialCharges(molecule, charge_model = 'am1bcc', multiconformer = False, minimize_contacts = False, verbose = False):
   """Assign partial charges to a molecule using OEChem oeproton.

   ARGUMENTS
     molecule (OEMol) - molecule for which charges are to be assigned

   OPTIONAL ARGUMENTS
     charge_model (string) - partial charge model, one of ['am1bcc'] (default: 'am1bcc')
     multiconformer (boolean) - if True, multiple conformations are enumerated and the resulting charges averaged (default: False)
     minimize_contacts (boolean) - if True, intramolecular contacts are eliminated by minimizing conformation with MMFF with all charges set to absolute values (default: False)
     verbose (boolean) - if True, information about the current calculation is printed

   RETURNS
     charged_molecule (OEMol) - the charged molecule with GAFF atom types

   NOTES
     multiconformer and minimize_contacts can be combined, but this can be slow

   EXAMPLES
     # create a molecule
     molecule = createMoleculeFromIUPAC('phenol')
     # assign am1bcc charges
     assignPartialCharges(molecule, charge_model = 'am1bcc')
   """

   #Check that molecule has atom names; if not we need to assign them
   assignNames = False
   for atom in molecule.GetAtoms():
       if atom.GetName()=='':
          assignNames = True #In this case we are missing an atom name and will need to assign
   if assignNames:
      if verbose: print "Assigning TRIPOS names to atoms"
      OETriposAtomNames(molecule)

   # Check input pameters.
   supported_charge_models  = ['am1bcc']
   if not (charge_model in supported_charge_models):
      raise "Charge model %(charge_model)s not in supported set of %(supported_charge_models)s" % vars()

   # Expand conformations if desired.   
   if multiconformer:
      expanded_molecule = expandConformations(molecule)
   else:
      expanded_molecule = OEMol(molecule)
   nconformers = expanded_molecule.NumConfs()
   if verbose: print 'assignPartialCharges: %(nconformers)d conformations will be used in charge determination.' % vars()
   
   # Set up storage for partial charges.
   partial_charges = dict()
   for atom in molecule.GetAtoms():
      name = atom.GetName()
      partial_charges[name] = 0.0

   # Assign partial charges for each conformation.
   conformer_index = 0
   for conformation in expanded_molecule.GetConfs():
      conformer_index += 1
      if verbose and multiconformer: print "assignPartialCharges: conformer %d / %d" % (conformer_index, expanded_molecule.NumConfs())

      # Assign partial charges to a copy of the molecule.
      if verbose: print "assignPartialCharges: determining partial charges..."
      charged_molecule = OEMol(conformation)   
      if charge_model == 'am1bcc':
         OEAssignPartialCharges(charged_molecule, OECharges_AM1BCC)         
      
      # Minimize with positive charges to splay out fragments, if desired.
      if minimize_contacts:
         if verbose: print "assignPartialCharges: Minimizing conformation with MMFF and absolute value charges..." % vars()         
         # Set partial charges to absolute value.
         for atom in charged_molecule.GetAtoms():
            atom.SetPartialCharge(abs(atom.GetPartialCharge()))
         # Minimize in Cartesian space to splay out substructures.
         szybki = OESzybki() # create an instance of OESzybki
         szybki.SetRunType(OERunType_CartesiansOpt) # set minimization         
         szybki.SetUseCurrentCharges(True) # use charges for minimization
         results = szybki(charged_molecule)
         # DEBUG
         writeMolecule(charged_molecule, 'minimized.mol2')
         for result in results: result.Print(oeout)
         # Recompute charges;
         if verbose: print "assignPartialCharges: redetermining partial charges..."         
         OEAssignPartialCharges(charged_molecule, OECharges_AM1BCC)         
         
      # Accumulate partial charges.
      for atom in charged_molecule.GetAtoms():
         name = atom.GetName()
         partial_charges[name] += atom.GetPartialCharge()
   # Compute and store average partial charges in a copy of the original molecule.
   charged_molecule = OEMol(molecule)
   for atom in charged_molecule.GetAtoms():
      name = atom.GetName()
      atom.SetPartialCharge(partial_charges[name] / nconformers)

   # Return the charged molecule
   return charged_molecule

def enumerateStates(molecules, enumerate = "protonation", consider_aromaticity = True, maxstates = 200, verbose = True):
    """Enumerate protonation or tautomer states for a list of molecules.

    ARGUMENTS
      molecules (OEMol or list of OEMol) - molecules for which states are to be enumerated

    OPTIONAL ARGUMENTS
      enumerate - type of states to expand -- 'protonation' or 'tautomer' (default: 'protonation')
      verbose - if True, will print out debug output

    RETURNS
      states (list of OEMol) - molecules in different protonation or tautomeric states

    TODO
      Modify to use a single molecule or a list of molecules as input.
      Apply some regularization to molecule before enumerating states?
      Pick the most likely state?
      Add more optional arguments to control behavior.
    """

    # If 'molecules' is not a list, promote it to a list.
    if type(molecules) != type(list()):
       molecules = [molecules]

    # Check input arguments.
    if not ((enumerate == "protonation") or (enumerate == "tautomer")):
        raise "'enumerate' argument must be either 'protonation' or 'tautomer' -- instead got '%s'" % enumerate

    # Create an internal output stream to expand states into.
    ostream = oemolostream()
    ostream.openstring()
    ostream.SetFormat(OEFormat_SDF)
    
    # Default parameters.
    only_count_states = False # enumerate states, don't just count them

    # Enumerate states for each molecule in the input list.
    states_enumerated = 0
    for molecule in molecules:
        if (verbose): print "Enumerating states for molecule %s." % molecule.GetTitle()
        
        # Dump enumerated states to output stream (ostream).
        if (enumerate == "protonation"): 
            # Create a functor associated with the output stream.
            functor = OETyperMolFunction(ostream, consider_aromaticity, False, maxstates)
            # Enumerate protonation states.
            if (verbose): print "Enumerating protonation states..."
            states_enumerated += OEEnumerateFormalCharges(molecule, functor, verbose)        
        elif (enumerate == "tautomer"):
            # Create a functor associated with the output stream.
            functor = OETautomerMolFunction(ostream, consider_aromaticity, False, maxstates)
            # Enumerate tautomeric states.
            if (verbose): print "Enumerating tautomer states..."
            states_enumerated += OEEnumerateTautomers(molecule, functor, verbose)    
    print "Enumerated a total of %d states." % states_enumerated

    # Collect molecules from output stream into a list.
    states = list()
    if (states_enumerated > 0):    
        state = OEMol()
        istream = oemolistream()
        istream.openstring(ostream.GetString())
        istream.SetFormat(OEFormat_SDF)
        while OEReadMolecule(istream, state):
           states.append(OEMol(state)) # append a copy

    # Return the list of expanded states as a Python list of OEMol() molecules.
    return states

def fitMolToRefmol( fitmol, refmol, maxconfs = None, verbose = False, ShapeColor = False):

    """Fit a multi-conformer target molecule to a reference molecule using OpenEye Shape tookit, and return an OE molecule with the top conformers of the resulting fit. Tanimoto scores also returned.

    ARGUMENTS
      fitmol (OEMol) -- the (multi-conformer) molecule to be fit.
      refmol (OEMol) -- the molecule to fit to

    OPTIONAL ARGUMENTS
      maxconfs -- Limit on number of conformations to return; default return all
      verbose -- Turn verbosity on/off
      ShapeColor -- default False. If True, also do a color search (looking at chemistry) rather than just shape, and return the combined score (now running from 0 to 2).

    RETURNS
      outmol (OEMol) -- output (fit) molecule resulting from fitmol
      scores

    NOTES
      Passing this a multi-conformer fitmol is recommended for any molecule with rotatable bonds as fitting only includes rotations and translations, so one of the provided conformers must already have right bond rotations."""

    #Set up storage for overlay
    best = OEBestOverlay()
    #Set reference molecule
    best.SetRefMol(refmol)

    #Color search too if desired
    if ShapeColor:
        best.SetColorForceField( OEColorFFType_ImplicitMillsDean)
        best.SetColorOptimize(True)

    if verbose:
        print "Reference title: ", refmol.GetTitle()
        print "Fit title: ", fitmol.GetTitle()
        print "Num confs: ", fitmol.NumConfs()


    resCount = 0
    #Each conformer-conformer pair generates multiple scores since there are multiple possible overlays; we only want the best. Load the best score for each conformer-conformer pair into an iterator and loop over it
    scoreiter = OEBestOverlayScoreIter()
    OESortOverlayScores(scoreiter, best.Overlay(fitmol), OEHighestTanimoto())
    tanimotos = [] #Storage for scores
    for score in scoreiter:
        #Get the particular conformation of this match and transform to overlay onto reference structure

        #tmpmol = OEGraphMol(fitmol.GetConf(OEHasConfIdx(score.fitconfidx)))
        tmpmol = OEMol(fitmol.GetConf(OEHasConfIdx(score.fitconfidx)))
        score.Transform(tmpmol)
        #Store to output molecule
        try: #If it already exists
            outmol.NewConf(tmpmol)
        except: #Otherwise
            outmol = tmpmol

        #Print some info
        if verbose:
            print "FitConfIdx: %-4d" % score.fitconfidx,
            print "RefConfIdx: %-4d" % score.refconfidx,
            if not ShapeColor:
                print "Tanimoto: %.2f" % score.tanimoto
            else:
                print score.GetTanimotoCombo()
        #Store score
        if not ShapeColor:
            tanimotos.append(score.tanimoto)
        else:
            tanimotos.append(score.GetTanimotoCombo() )
        resCount+=1

        if resCount == maxconfs: break
    return ( outmol, tanimotos )

def create_openmm_system(molecule, charge_model=False, judgetypes=None, cleanup=True, show_warnings=True, verbose=False, resname=None, netcharge=None, offfile=None, ligand_obj_name='molecule', frcmod_filename=None, gaff_mol2_filename=None, prmtop_filename=None, inpcrd_filename=None):
   """
   Parameterize molecule for OpenMM using GAFF.

   Parameters
   ----------
     molecule (OEMol) - molecule to parameterize (only the first configuration will be used if multiple are present)
     topology_filename (string) - name of AMBER topology file to be written
     coordinate_filename (string) - name of AMBER coordinate file to be written

   OPTIONAL ARGUMENTS
     charge_model (string) - if not False, antechamber is used to assign charges (default: False) -- if set to 'bcc', for example, AM1-BCC charges will be used
     judgetypes (integer) - if provided, argument passed to -j of antechamber to judge types (default: None)
     cleanup (boolean) - clean up temporary files (default: True)
     show_warnings (boolean) - show warnings during parameterization (default: True)
     verbose (boolean) - show all output from subprocesses (default: False)
     resname (string) - if set, residue name to use for parameterized molecule (default: None)
     netcharge (integer) -- if set, pass this net charge to calculation in antechamber (with -nc (netcharge)), otherwise assumes zero.
     offfile (string) - name of AMBER off file to be written, optionally. 
     ligand_obj_name - name of object to store ligand as (in off file); default 'molecule'. 
     frmcmod_filename - name of frcmod file to be saved if desired (default: None)
     gaff_mol2_filename - name of GAFF mol2 file to generate (default: None)

   REQUIREMENTS
     acpypi.py conversion script (must be in MMTOOLSPATH)
     AmberTools installation (in PATH)

   EXAMPLES
     # create a molecule
     molecule = createMoleculeFromIUPAC('phenol')
     # parameterize it for AMBER, using antechamber to assign AM1-BCC charges
     parameterizeForAmber(molecule, topology_filename = 'phenol.prmtop', coordinate_filename = 'phenol.crd', charge_model = 'bcc')

   """

   # Create temporary working directory and copy ligand mol2 file there.
   working_directory = tempfile.mkdtemp()
   old_directory = os.getcwd()
   os.chdir(working_directory)
   if verbose: print "Working directory is %(working_directory)s" % vars()

   # Write molecule to mol2 file.
   tripos_mol2_filename = os.path.join(working_directory, 'tripos.mol2')
   writeMolecule(molecule, tripos_mol2_filename)
   #Ensure charges are rounded to neutrality
   round_charges( tripos_mol2_filename)

   if resname:
      # Set substructure name (which will become residue name) if desired.
      modifySubstructureName(tripos_mol2_filename, resname)

   # Run antechamber to assign GAFF atom types.
   gaff_mol2_tmpfilename = os.path.join(working_directory, 'gaff.mol2')
   if netcharge:
      chargstr = '-nc %d' % netcharge
   else: chargestr=''
   command = 'antechamber -i %(tripos_mol2_filename)s -fi mol2 -o %(gaff_mol2_tmpfilename)s -fo mol2 %(chargestr)s' % vars()
   if judgetypes: command += ' -j %(judgetypes)d' % vars()
   if charge_model:
      formal_charge = formalCharge(molecule)
      command += ' -c %(charge_model)s -nc %(formal_charge)d' % vars()
   if verbose: print command
   output = commands.getoutput(command)
   if verbose or (output.find('Warning')>=0): print output

   # Generate frcmod file for additional GAFF parameters.
   frcmod_filename_tmp = os.path.join(working_directory, 'gaff.frcmod')
   commands.getoutput('parmchk -i %(gaff_mol2_tmpfilename)s -f mol2 -o %(frcmod_filename_tmp)s' % vars())

   # Create AMBER topology/coordinate files using LEaP.
   if offfile:
      offstring = 'saveOff %(ligand_obj_name)s amber.off\n' % vars()
   else:
      offstring = ''
   leapscript = """\
source leaprc.gaff
mods = loadAmberParams %(frcmod_filename_tmp)s
%(ligand_obj_name)s = loadMol2 %(gaff_mol2_tmpfilename)s
desc %(ligand_obj_name)s
check %(ligand_obj_name)s
saveAmberParm %(ligand_obj_name)s amber.prmtop amber.crd
%(offstring)s
quit""" % vars()
   leapin_filename = os.path.join(working_directory, 'leap.in')
   outfile = open(leapin_filename, 'w')
   outfile.write(leapscript)
   outfile.close()

   tleapout = commands.getoutput('tleap -f %(leapin_filename)s' % vars())
   if verbose: print tleapout
   tleapout = tleapout.split('\n')
   # Shop any warnings.
   if show_warnings:
      fnd = False
      for line in tleapout:
         tmp = line.upper()
         if tmp.find('WARNING')>-1:
            print line
            fnd = True
         if tmp.find('ERROR')>-1:
            print line
            fnd = True
      if fnd:
         print "Any LEaP warnings/errors are above."

   # Restore old directory.
   os.chdir(old_directory)

   # Copy gromacs topology/coordinates to desired output files.
   if inpcrd_filename:
      commands.getoutput('cp %s %s' % (os.path.join(working_directory, 'amber.crd'), inpcrd_filename))
   if prmtop_filename:
      commands.getoutput('cp %s %s' % (os.path.join(working_directory, 'amber.prmtop'), prmtop_filename))
   if offfile:
        commands.getoutput('cp %s %s' % (os.path.join(working_directory, 'amber.off'), offfile))
   if frcmod_filename:
        commands.getoutput('cp %s %s' % (os.path.join( working_directory, frcmod_filename_tmp), frcmod_filename ) )
   if gaff_mol2_filename:
        commands.getoutput('cp %s %s' % (gaff_mol2_tmpfilename, gaff_mol2_filename ) )

   # Clean up temporary files.
   os.chdir(old_directory)
   if cleanup:
      commands.getoutput('rm -r %s' % working_directory)
   else:
      print "Work done in %s..." % working_directory

   return [tripos_molecule, gaff_molecule, system, topology, positions]
