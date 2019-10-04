import parmed
import tempfile
import os
import shutil

# Get the paths to the gaff.dat and leaprc files:
leaprc_path = os.path.join(os.getenv("CONDA_PREFIX"), 'dat', 'leap', 'cmd', 'leaprc.gaff')
gaffdat_path = os.path.join(os.getenv("CONDA_PREFIX"), 'dat', 'leap', 'parm', 'gaff.dat')

# Make a temporary directory (otherwise we won't be able to find the gaff.dat)
cwd = os.getcwd()
tmpdir = tempfile.mkdtemp()
os.chdir(tmpdir)
shutil.copy(gaffdat_path, os.getcwd())

# Instantiate the amber parameter set:
amber_params = parmed.amber.AmberParameterSet.from_leaprc(leaprc_path)

# Make an OpenMM parameter set:
openmm_params = parmed.openmm.OpenMMParameterSet.from_parameterset(amber_params, remediate_residues=False)

# Return to the original directory and clean up:
os.chdir(cwd)
shutil.rmtree(tmpdir)

# Save the OpenMM parameter set as gaff.xml:
openmm_params.write("gaff.xml", write_unused=True, improper_dihedrals_ordering='amber')
