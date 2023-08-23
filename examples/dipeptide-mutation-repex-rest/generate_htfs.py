import os
import pickle
import argparse
from pathlib import Path

import openmm
from openmm import unit, app
from perses.app.relative_point_mutation_setup import PointMutationExecutor
from perses.utils.smallmolecules import  render_protein_residue_atom_mapping

# Read args
parser = argparse.ArgumentParser(description='run equilibration')
parser.add_argument('input_filename', type=str, help='protein file')
parser.add_argument('resid', type=str, help='residue id')
parser.add_argument('mutant_aa', type=str, help='amino acid to mutate to')
parser.add_argument('outdir', type=str, help='output directory')
parser.add_argument('--ligand_input', type=str, help='ligand input file')
parser.add_argument('--is_vacuum', action='store_true', help='whether to generate a vacuum htf')
parser.add_argument('--old_residue', type=str, help='old residue nonstandard name')
args = parser.parse_args()

# Set parameters for input to `PointMutationExecutor`
forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
forcefield_kwargs = {'removeCMMotion': False, 'constraints' : app.HBonds, 'rigidWater': True, 'hydrogenMass' : 3 * unit.amus}

if not args.is_vacuum:
    is_vacuum = False
    is_solvated = True
    barostat = openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 50)
    periodic_forcefield_kwargs = {'nonbondedMethod': app.PME, 'ewaldErrorTolerance': 0.00025}
    nonperiodic_forcefield_kwargs = None
else:
    is_vacuum = True
    is_solvated = False
    barostat = None
    periodic_forcefield_kwargs = None
    nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff}

conduct_endstate_validation = False
w_lifting = 0.3 * unit.nanometer
generate_unmodified_hybrid_topology_factory = False
generate_rest_capable_hybrid_topology_factory = True

# Generate htfs
solvent_delivery = PointMutationExecutor(args.input_filename,
                        '1',
                        args.resid,
                        args.mutant_aa,
                        old_residue=args.old_residue,
                        is_vacuum=is_vacuum,
                        is_solvated=is_solvated,
                        forcefield_files=forcefield_files,
                        barostat=barostat,
                        forcefield_kwargs=forcefield_kwargs,
                        periodic_forcefield_kwargs=periodic_forcefield_kwargs,
                        nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                        conduct_endstate_validation=conduct_endstate_validation,
                        w_lifting=w_lifting,
                        generate_unmodified_hybrid_topology_factory=generate_unmodified_hybrid_topology_factory,
                        generate_rest_capable_hybrid_topology_factory=generate_rest_capable_hybrid_topology_factory
                       )

# Saving htfs as pickles
print("Saving htfs as pickles")
apo_rest_htf = solvent_delivery.get_apo_rest_htf()
phase = 'vacuum' if args.is_vacuum else 'apo'

results_dir = args.outdir

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
with open(os.path.join(args.outdir, f"htf_{phase}.pickle"), "wb") as f:
    pickle.dump(apo_rest_htf, f)

# Render atom map
atom_map_filename = f'{args.outdir}/atom_map.png'
render_protein_residue_atom_mapping(apo_rest_htf._topology_proposal, atom_map_filename)

# Save pdbs
app.PDBFile.writeFile(apo_rest_htf._topology_proposal.old_topology, apo_rest_htf.old_positions(apo_rest_htf.hybrid_positions), open(os.path.join(results_dir, f"{phase}_old.pdb"), "w"), keepIds=True)
app.PDBFile.writeFile(apo_rest_htf._topology_proposal.new_topology, apo_rest_htf.new_positions(apo_rest_htf.hybrid_positions), open(os.path.join(results_dir, f"{phase}_new.pdb"), "w"), keepIds=True)

