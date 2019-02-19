import numpy as np
import os
import yaml
import itertools
from openeye import oechem, oeiupac
import copy


def create_yaml_file(mol_a, mol_b, yaml_dict):
    yaml_dict['molecules'] = [mol_a, mol_b]
    yaml_dict['traj_prefix'] = "{}_{}".format(mol_a, mol_b)
    yaml_dict['lengths'] = [5000]

    return yaml_dict

def create_submit_script(template_script_file, yaml_filename):
    with open(template_script_file, "r") as templatefile:
        template_script = templatefile.read()
        template_script.format(yaml_filename)

if __name__=="__main__":
    template_script_file_eq = "submit-eq.sh"
    template_script_file_neq = "submit-neq.sh"

    substituted_benzene_smilefile = "filtered_database.smi"

    substituted_benzenes_iupac = []

    istream = oechem.oemolistream(substituted_benzene_smilefile)

    for mol in istream.GetOEMols():
        mol_copy = oechem.OEMol(mol)
        substituted_benzenes_iupac.append(oeiupac.OECreateIUPACName(mol_copy))


    with open("rj_neq_template.yaml", "r") as yamlfile:
        template_yamldict = yaml.load(yamlfile)

    for pair in itertools.permutations(substituted_benzenes_iupac, 2):
        if pair[0] == pair[1]:
            continue

        new_yaml_dict = create_yaml_file(pair[0], pair[1], copy.deepcopy(template_yamldict))

        new_yaml_filename = "{}_{}_rjneq.yaml".format(pair[0], pair[1])

        with open(new_yaml_filename, 'w') as yaml_outfile:
            yaml.dump(new_yaml_dict, yaml_outfile)

        with open("submit_{}_{}_eq.sh".format(pair[0], pair[1]), 'w') as submiteqfile:
            submit_eq = create_submit_script(template_script_file_eq, new_yaml_filename)
            submiteqfile.write(submit_eq)

        with open("submit_{}_{}_neq.sh".format(pair[0], pair[1]), 'w') as submitneqfile:
            submit_neq = create_submit_script(template_script_file_neq, new_yaml_filename)
            submitneqfile.write(submit_neq)
