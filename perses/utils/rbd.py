import os
import tempfile
import re
from math import floor
from simtk.openmm import app
from simtk import unit

"""
Utility functions for prepping RBD and RBD:ACE2 systems in tleap.
"""

def edit_pdb_for_tleap(input_pdb, output_pdb, is_ace2=False):
    """
    Edit a PDB file so that it can be loaded into tleap.

    Parameters
    ----------
    input_pdb : str
        Path to input PDB
    output_pdb : str
        Path to output (edited) PDB
    is_ace2 : boolean, default False
        Indicates whether the file is for ACE2
    """
    
    # Read lines
    with open(input_pdb, "r") as f:
        lines = f.readlines()

    # Iterate through lines, copying them over to new list of lines
    glycan_residue_names = ['UYB', '4YB', 'VMB', '2MA', '0YB', '0fA', '0LB']
    new_lines = []
    previous_res_id =  0
    previous_res_name = ''
    for line in lines:
        if 'CONECT' in line: # Skip CONECT lines
            continue
        if 'TER' in line and 'NME' in line:
            continue
        if 'TER' not in line and 'END' not in line and 'REMARK' not in line and 'TITLE' not in line and 'CRYST1' not in line:
            current_res_name = line[17:20]
            current_res_id = int(line[23:26])
            if current_res_id != previous_res_id:
                if previous_res_name in glycan_residue_names:
                    new_lines.append("TER\n") # add TER if the previous residue was a glycan residue
                if previous_res_name == "NME":
                    new_lines.append("TER\n") # add TER after the NME and before starting the next residue
                previous_res_id = current_res_id 
                previous_res_name = current_res_name
            if current_res_name == 'NME': # change C atom in NMEs to CH3
                atom = line[13:16]
                if atom == 'C  ':
                    line = line[:13] + 'CH3 ' + line[17:]
                if atom == 'H1 ':
                    line = line[:12] + 'HH31' + line[16:]
                if atom == 'H2 ':
                    line = line[:12] + 'HH32' + line[16:]
                if atom == 'H3 ':
                    line = line[:12] + 'HH33' + line[16:]
            if is_ace2:
                if current_res_name == 'CYS' and current_res_id not in [261, 498]: # change CYS to CYX
                    line = line[:17] + 'CYX' + line[20:]
            else:
                if current_res_name == 'CYS': # change CYS to CYX
                    line = line[:17] + 'CYX' + line[20:]
                
        new_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(new_lines)

def edit_tleap_in_inputs(tleap_in_template, tleap_prefix, debug_dir=None):
    """
    Edit the input and output files in the tleap.in file 

    Parameters
    ----------
    tleap_in_template : str
        Template tleap.in file to edit
    tleap_prefix : str
        Prefix for output tleap.in and output tleap files
    debug_dir : str, default None
        If specified, dir to prepend to path of input files
    """
    
    with open(tleap_in_template, "r") as f:
        lines_in = f.readlines()

    new_lines = []
    for line in lines_in:
        if "mol1 = loadpdb" in line:
            if debug_dir:
                linesplit = line.split(" ")
                line = ' '.join(linesplit[:-1]) + f" {os.path.join(debug_dir, linesplit[-1])}"
        if "mol2 = loadpdb" in line:
            if debug_dir:
                linesplit = line.split(" ")
                line = ' '.join(linesplit[:-1]) + f" {os.path.join(debug_dir, linesplit[-1])}"
        if "mol3 = loadpdb" in line:
            if debug_dir:
                linesplit = line.split(" ")
                line = ' '.join(linesplit[:-1]) + f" {os.path.join(debug_dir, linesplit[-1])}"
        if "savepdb" in line:
            linesplit = line.split(" ")
            line = ' '.join(linesplit[:-1]) + f" {tleap_prefix}.pdb\n"
        if "saveamberparm" in line:
            linesplit = line.split(" ")
            line = ' '.join(linesplit[:-2]) + f" {tleap_prefix}.prmtop {tleap_prefix}.inpcrd\n"
        new_lines.append(line)

    with open(f"{tleap_prefix}.in", 'w') as f:
        f.writelines(new_lines)
        
def edit_tleap_in_ions(tleap_prefix):
    """
    Edit the number of ions in the tleap.in file 

    Parameters
    ----------
    tleap_prefix : str
        Prefix for tleap.in file to edit
        
    """
    
    # Run tleap to determine how many waters will be present in solvent
    with tempfile.TemporaryDirectory() as temp_dir:
        tleap_in_temp = os.path.join(temp_dir, "temp")
        tleap_out_temp = os.path.join(temp_dir, "temp.out")
        edit_tleap_in_inputs(f"{tleap_prefix}.in", tleap_in_temp)
        os.system(f"tleap -s -f {tleap_in_temp}.in > {tleap_out_temp}")
    
        # Retrieve charge and num of waters
        with open(tleap_out_temp, "r") as f:
            lines_out = f.readlines()

        for line in lines_out:
            if "Total unperturbed charge" in line:
                charge = float(line.split(":")[1].strip('\n'))
            if "residues" in line:
                result = re.findall(r"\d*", line)
                result_filtered = [r for r in result if r]
                num_waters = int(result_filtered[0])

    # Compute number of ions (copied from OpenMM)
    numWaters = num_waters
    numPositive = 0
    numNegative = 0 
    totalCharge = charge
    ionicStrength = 0.15

    if totalCharge > 0:
        numNegative += totalCharge
    else:
        numPositive -= totalCharge

    numIons = (numWaters - numPositive - numNegative) * ionicStrength / (55.4)  # Pure water is about 55.4 molar (depending on temperature)
    numPairs = int(floor(numIons + 0.5))
    numPositive += numPairs
    numNegative += numPairs
    print(f"num positive: {numPositive}")
    print(f"num negative: {numNegative}")

    # Edit tleap file
    with open(f"{tleap_prefix}.in", "r") as f:
        lines_in = f.readlines()

    new_lines = []
    for line in lines_in:
        if "addionsrand complex" in line:
            line = f"addionsrand complex Na+ {int(numPositive)} Cl- {int(numNegative)}\n"
        new_lines.append(line)

    with open(f"{tleap_prefix}.in", 'w') as f:
        f.writelines(new_lines)
        
def generate_tleap_system(tleap_prefix, 
                        temperature=300 * unit.kelvin, 
                        nonbonded_method=app.PME, 
                        constraints=app.HBonds, 
                        remove_cm_motion=False, 
                        hydrogen_mass=4.0 * unit.amu):

    """
    Generate a tleap system by 1) running tleap and 2) loading the tleap output prmtop and inpcrd files into openmm

    Parameters
    ----------
    tleap_prefix : str
        Prefix for tleap input and output files
    temperature : unit.kelvin, default 300 * unit.kelvin
        Temperature
    nonbonded_method : simtk.openmm.app.Forcefield subclass object default app.PME
        Nonbonded method
    constraints : simtk.openmm.app.Forcefield subclass object, default app.HBonds
        Bonds that should have constraints
    remove_cm_motion : boolean, default False
        Indicates whether to remove center of mass motion
    hydrogen_mass : unit.amu, default 4.0 * unit.amu
        Hydrogen mass
    Returns
    -------
    prmtop.topology : simtk.openmm.app.Topology object
        Topology loaded from the prmtop file
    inpcrd.positions : np.array
        Positions loaded from the inpcrd file
    system : simtk.openmm.System object
        Tleap generated system as an OpenMM object
    """
    
    # Run tleap
    os.system(f"tleap -s -f {tleap_prefix}.in > {tleap_prefix}.out")

    # Check if tleap was successful
    if not os.path.exists(f"{tleap_prefix}.prmtop"):
        raise Exception(f"tleap parametrization did not complete successfully, check {tleap_prefix}.out for errors")

    # Load prmtop and inpcrd files
    prmtop = app.AmberPrmtopFile(f"{tleap_prefix}.prmtop")
    inpcrd = app.AmberInpcrdFile(f"{tleap_prefix}.inpcrd")

    # Generate system
    system = prmtop.createSystem(
        nonbondedMethod=nonbonded_method,
        constraints=constraints,
        temperature=temperature,
        removeCMMotion=remove_cm_motion,
        hydrogenMass=hydrogen_mass
    )

    return prmtop.topology, inpcrd.positions, system


