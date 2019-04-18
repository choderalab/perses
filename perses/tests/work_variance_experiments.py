# In[0]
################################################################################
# IMPORTS
################################################################################
import copy
from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from functools import partial
from pkg_resources import resource_filename
from perses.rjmc import experimental_geometry as geometry
from perses.rjmc.topology_proposal import SystemGenerator, TopologyProposal, SmallMoleculeSetProposalEngine
from openeye import oechem
if sys.version_info >= (3, 0):
    from io import StringIO
    from subprocess import getstatusoutput
else:
    from cStringIO import StringIO
    from commands import getstatusoutput
from openmmtools.constants import kB
from openmmtools import alchemy, states
from perses.tests.utils import render_atom_mapping

################################################################################
# CONSTANTS
################################################################################

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
PATH = "/home/dominic/Downloads/"
LOGP_FORWARD_THRESHOLD = 1e3
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("Reference")

#create iid bead system and save
def create_iid_systems(system, positions, num_iterations):
    """
    Function to simulate i.i.d conformations of the initial molecule

    Returns
    -------
    iid_positions_A: numpy.ndarray
        num_iterations of independent initial molecule conformations
    """
    from openmmtools import integrators
    import tqdm
    _platform = REFERENCE_PLATFORM
    _integrator = integrators.LangevinIntegrator(temperature, 1./unit.picoseconds, 0.002*unit.picoseconds)
    _ctx = openmm.Context(system, _integrator)
    _ctx.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(_ctx)

    _iid_positions_A=[]
    rp = []
    for _iteration in tqdm.trange(num_iterations):
        _integrator.step(1000)
        _state=_ctx.getState(getPositions=True)
        _iid_positions_A.append(_state.getPositions(asNumpy=True))
        rp.append(beta*_ctx.getState(getEnergy=True).getPotentialEnergy())

    return _iid_positions_A, rp

def test_work_variance(current_mol_name = 'propane', proposed_mol_name = 'butane', num_iterations = 1):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.  Assert that the logp_forward < 1e3.
    This assertion will fail if the proposal order tool proposed the placement of the a carbon before a previously defined carbon in the alkane.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename, compute_potential

    current_mol, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(current_mol_name)
    proposed_mol = createOEMolFromIUPAC(proposed_mol_name)

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=current_mol_name)
    print("Generated proposal engine.")

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old, current_mol=current_mol, proposed_mol=proposed_mol)
    print("topology_proposal complete.")

    # show atom mapping
    filename = str(current_mol_name)+str(proposed_mol_name)+'.pdf'
    render_atom_mapping(filename,current_mol,proposed_mol,topology_proposal.new_to_old_atom_map)

    #remove non-valence forces
    topology_proposal._old_system.removeForce(3)
    topology_proposal._new_system.removeForce(3)



    print("old_system forces: {}".format(topology_proposal._old_system.getForces()))
    print("new_system forces: {}".format(topology_proposal._new_system.getForces()))


    #run simulation from old_positions and get energies thereof
    import tqdm
    old_pos, old_pos_rp = create_iid_systems(system = topology_proposal._old_system, positions = pos_old, num_iterations = num_iterations)
    print("iid positions generator complete")

    logp_forwards, new_positions, new_positions_rp, atom_placement_list = [], [], [], []

    for iteration in tqdm.trange(num_iterations):
        #generate new positions with geometry engine
        new_pos, logp_forward, atom_placements, final_context_reduced_potential = geometry_engine.propose(topology_proposal, old_pos[iteration], beta)
        new_positions.append(new_pos); atom_placement_list.append(atom_placements)
        new_positions_rp.append(final_context_reduced_potential)
        logp_forwards.append(logp_forward)


    return old_pos, old_pos_rp,  new_positions, new_positions_rp, np.array(atom_placement_list), np.array(logp_forwards), topology_proposal

# In[1]
old_pos, old_pos_rp, new_positions, new_positions_rp_corrected, atom_placement_list, logp_forwards, topology_proposal = test_work_variance(current_mol_name = 'propane', proposed_mol_name = 'butane', num_iterations = 1)
# %%
# this is just to print the types of torsion forces in the system
old_topology, new_topology = topology_proposal._old_topology, topology_proposal._new_topology
atoms = list(old_topology.atoms())
for atom in atoms:
    print(atom)
print()
old_system, new_system = topology_proposal._old_system, topology_proposal._new_system
num_torsions_old, num_torsions_new = old_system.getForce(2).getNumTorsions(), new_system.getForce(2).getNumTorsions()
for force_index in range(num_torsions_old):
    print(force_index, old_system.getForce(2).getTorsionParameters(force_index))

# %%
# now we can probably plot one of these torsions
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set(color_codes = True)

phis = np.linspace(-np.pi, np.pi, 1000)

torsion_of_interest = 0
p1, p2, p3, p4, per, phi0, k = old_system.getForce(2).getTorsionParameters(torsion_of_interest)
V_tor = lambda phi, k_tor, phi_0, periodicity, beta: k_tor*beta*(1.+ np.cos(periodicity*phi - phi_0.value_in_unit_system(unit.md_unit_system)))
V1 = V_tor(phis, k, phi0, per, beta)

# torsion_of_interest = 1
# p1, p2, p3, p4, per, phi0, k = old_system.getForce(2).getTorsionParameters(torsion_of_interest)
# V_tor = lambda phi, k_tor, phi_0, periodicity, beta: k_tor*beta*(1.+ np.cos(periodicity*phi - phi_0.value_in_unit_system(unit.md_unit_system)))
# V2 = V_tor(phis, k, phi0, per, beta)
#
# torsion_of_interest = 24
# p1, p2, p3, p4, per, phi0, k = old_system.getForce(2).getTorsionParameters(torsion_of_interest)
# V_tor = lambda phi, k_tor, phi_0, periodicity, beta: k_tor*beta*(1.+ np.cos(periodicity*phi - phi_0.value_in_unit_system(unit.md_unit_system)))
# V3 = V_tor(phis, k, phi0, per, beta)

#V = V1 + V2 + V3
sns.lineplot(phis, V1)

# %%
# play with a parmed structure
import parmed
structure = parmed.openmm.load_topology(topology_proposal.new_topology, topology_proposal.new_system)
structure.bonds
for bond in structure.bonds:
    print(list(
# %%
np.save(PATH+"atom_placement_list.npy", atom_placement_list)
np.save(PATH+"new_positions_rp_corrected.npy", np.array(new_positions_rp_corrected))
np.save(PATH + "logp_forwards.npy", logp_forwards)

# %%
atom_placement_list = np.load(PATH+"atom_placement_list.npy")
new_positions_rp_corrected = list((np.load(PATH+"new_positions_rp_corrected.npy")))
logp_forwards = np.load(PATH+"logp_forwards.npy")
# In[2]
#compute the total work from the final logp_forwards and the total change in energy
atoms_with_positions_reduced_potential = np.array([proposal_instantiation[0,-1] for proposal_instantiation in atom_placement_list])
total_work_forward = logp_forwards + new_positions_rp_corrected - atoms_with_positions_reduced_potential
print("logp_forwards: {}".format(logp_forwards))
print("energy change forward: {}".format(new_positions_rp_corrected - atoms_with_positions_reduced_potential))


# In[3]
#compute the total work for the sequential logp_forwards and the sequential change in energy
atom_works = []
lnZ_list = []

for iteration_index, iteration in enumerate(atom_placement_list):
    energy_change, logp_forward, lnZ_s = [], [], []
    for index, atom in enumerate(iteration):

        #lnZ_s.append([atom[9], atom[10], atom[6] - atom[1] - atom[2]])
        lnZ_s.append([atom[10], atom[11], atom[7] - atom[1] - atom[2]])

        energy_change.append(atom[9])
        print("reduced potential energy: {}".format(atom[9]))

        logp_r, logp_theta, logp_phi, detJ = atom[5], atom[6], atom[7], atom[8]

        if index == 0:
            logp_choice = np.log(1./3)
        elif index == 1:
            logp_choice = 2*np.log(1./3)
        elif index == 2:
            logp_choice = np.log(1./3) + np.log(1./2)
        elif index == 3:
            logp_choice = np.log(1./3)
        else:
            "this cannot possibly be propane --> butane"

        logp_forward.append(logp_choice + logp_r + logp_theta + logp_phi - detJ)
    print("total logp_forward: {}".format(sum(logp_forward)))
    energy_change = [j if i == 0 else j - energy_change[i-1] for i, j in enumerate(energy_change)]
    lnZ_s = list(np.array(lnZ_s) + np.array([[0,0,ec] for ec in energy_change]))
    #lnZ_s[-1] += energy_change
    lnZ_list.append(lnZ_s)

    assert abs(sum(energy_change) - new_positions_rp_corrected[iteration_index] + atoms_with_positions_reduced_potential[iteration_index]) < 1e3
    per_atom_works = [i+j for i, j in zip(energy_change, logp_forward)]
    atom_works.append(per_atom_works)

per_atom_summation = np.array([sum(i) for i in atom_works])
lnZs = np.array(lnZ_list)

# %%
print(lnZs)
# %%
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set(color_codes = True)

lnZs.shape

lnZ_phi_carbon = lnZs[:,0,2]
lnZ_phi_h1 = lnZs[:,1,2]
lnZ_phi_h2 = lnZs[:,2,2]
lnZ_phi_h3 = lnZs[:,3,2]


ax1 = sns.distplot(lnZ_phi_carbon, label = "carbon", norm_hist = True)
#ax1 = sns.distplot(lnZ_phi_h1, label = "hydrogen 1", norm_hist = True)
ax1 = sns.distplot(lnZ_phi_h2, label = "hydrogen 2", norm_hist = True)
ax1 = sns.distplot(lnZ_phi_h3, label = "hydrogen 3", norm_hist = True)
ax1.set(xlabel = "lnZ_phi", ylabel = 'p(lnZ_phi)')
ax1.legend(loc = 'best')
plt.savefig("/home/dominic/Downloads/propane_to_butane_per_atom_lnZ_phi_dist.pdf")

# In[3]
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set(color_codes = True)

for i, j in zip(total_work_forward, per_atom_summation):
    print(i, j)

# %%
atom_works = np.array(atom_works)
# In[4]
#this is just for propane --> butane
#ax = sns.distplot(total_work_forward, label = "total work forward", norm_hist = True)
#ax = sns.distplot(per_atom_summation, label = "per atom summation", norm_hist = True)

ax = sns.distplot(atom_works[:,0], label = 'carbon', norm_hist = True)
#ax = sns.distplot(atom_works[:,1], label = 'hydrogen 1', norm_hist = True)
ax = sns.distplot(atom_works[:,2], label = 'hydrogen 2', norm_hist = True)
ax = sns.distplot(atom_works[:,3], label = 'hydrogen 3', norm_hist = True)
# ax = sns.distplot(work[:,0] + np.log(1./3), label = 'carbon', norm_hist = True)
# ax = sns.distplot(work[:,1] + np.log(1./3) + np.log(1./3), label = 'hydrogen 1', norm_hist = True)
# ax = sns.distplot(work[:,2] + np.log(1./2) + np.log(1./3), label = 'hydrogen 2', norm_hist = True)
# ax = sns.distplot(work[:,3] + np.log(1./1) + np.log(1./3), label = 'hydrogen 3', norm_hist = True)

# ax = sns.distplot(total_work, norm_hist = True, label = "total work")
# ax = sns.distplot(total_work_method_1, norm_hist = True, label = "total work method 1")

# ax = sns.distplot(lnZ_list[:,0] - np.log(3.), label = 'carbon', norm_hist = True)
# ax = sns.distplot(lnZ_list[:,1] - np.log(9.), label = 'hydrogen 1', norm_hist = True)
# ax = sns.distplot(lnZ_list[:,2] - np.log(6.), label = 'hydrogen 2', norm_hist = True)
# ax = sns.distplot(lnZ_list[:,3] - np.log(3.), label = 'hydrogen 3', norm_hist = True)
ax.set(xlabel = "work", ylabel = 'p(work)')
ax.legend(loc = 'best')
plt.savefig("/home/dominic/Downloads/propane_to_butane_per_atom_work_dist.pdf")


# %%
#we need to plot lnZ_phi against theta...just to check if there is a correlation
#we are particularly keen on hydrogen 2...
thetas_h2 = atom_placement_list[:, 2, 3]
thetas_h1 = atom_placement_list[:, 1, 3]
thetas_h3 = atom_placement_list[:, 3, 3]


phis = atom_placement_list[:, 2, 4]

#perhaps we need to load the phis of the previously placed atoms first
phis_carbon, phis_h1, phis_h2, phis_h3 = atom_placement_list[:,0,4], atom_placement_list[:, 1, 4], atom_placement_list[:,2,4], atom_placement_list[:,3,4]
ax = sns.scatterplot(thetas_h2, lnZ_phi_h2)
ax.set(xlabel = "theta", ylabel = 'lnZ_phi')
