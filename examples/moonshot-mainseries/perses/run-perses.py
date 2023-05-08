#!/usr/bin/env python
"""
Run perses
"""

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
    print(f'Loading molecules from {filename}...')
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


def enumerate_transformations():
    # Load options
    import yaml
    with open('../setup.yaml', 'r') as infile:
        setup_options = yaml.safe_load(infile)

    # Enumerate list of all transformations to run
    import copy
    transformations = list()
    for prefix, transformation_group in setup_options['transformation_groups'].items():
        transformation = dict()

        molecule_set = transformation_group['molecule_set']
        fragalysis_id = transformation_group['fragalysis_id']
            
        from itertools import product
        for assembly_state, receptor_protonation_state in product(setup_options['assembly_states'], setup_options['receptor_protonation_states']):

            ligand_file = f'../docked/{molecule_set}-{fragalysis_id}-{assembly_state}-{receptor_protonation_state}.sdf'
            transformation['ligand_file'] = ligand_file

            transformation['protein_pdb'] = f'../receptors/{assembly_state}/Mpro-{fragalysis_id}_bound-{receptor_protonation_state}-protein.pdb'

            molecules = load_molecules(ligand_file)
            nmolecules = len(molecules)

            receptor_protonation_state_safe = receptor_protonation_state
            #receptor_protonation_state_safe = receptor_protonation_state.replace('(', '[').replace(')',']')

            for old_ligand_index in range(nmolecules):
                for new_ligand_index in range(old_ligand_index+1, nmolecules):
                    transformation['old_ligand_index'] = old_ligand_index
                    transformation['new_ligand_index'] = new_ligand_index
                    transformation['trajectory_directory'] = f'{prefix}-{receptor_protonation_state_safe}-{old_ligand_index}-{new_ligand_index}'

                    
                    transformations.append(copy.deepcopy(transformation))

    return transformations

import click

@click.group()
def cli():
    pass

@cli.command()
def count():
    transformations = enumerate_transformations()

    for index, transformation in enumerate(transformations):
        print(f'{index} {transformation}')

    print(f'There are {len(transformations)} transformations.')


@cli.command()
@click.option("--index", type=int, required=True,
              help='Index of calculation to run.')
def run(index):
    # Load options
    import yaml
    with open('../setup.yaml', 'r') as infile:
        setup_options = yaml.safe_load(infile)
        
    transformations = enumerate_transformations()
    print(f'Running transformation {index} / {len(transformations)}')
    # Select appropriate index
    transformation = transformations[index]
    override_string = [ f'{key}:{value}' for key, value in transformation.items() ]
    from perses.app.setup_relative_calculation import run
    print(override_string)
    run(yaml_filename=setup_options['perses_yaml_template'], override_string=override_string)

if __name__ == '__main__':
    cli()