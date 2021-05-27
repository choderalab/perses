from perses.tests.test_topology_proposal import generate_atp, generate_dipeptide_top_pos_sys
from openmmtools.testsystems import AlanineDipeptideExplicit
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from simtk import unit, openmm
from perses.tests.utils import compute_potential_components
from openmmtools.constants import kB
from perses.dispersed.utils import configure_platform
import numpy as np
import copy

#############################################
# CONSTANTS
#############################################
temperature = 298.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CUDA")

# TODO: Write test checking that scaled energy of original system matches scaled energy of hybrid system

def compare_bond_energies(htf, is_old=True, check_scale=False):
	"""
	Given a RxnHybridTopologyFactory, check whether the energies of the HarmonicBondForce (from the original system) 
	match the energies of the CustomBondForce (from the hybrid system)

	htf : RxnHybridTopologyFactory
		htf on which to run the energy validation
	is_old : bool
		whether to validate against the old system (if False, will validate against the new system)
	check_scale : bool
		whether to check that the scale region is able to scale the energy
	"""
	htf_copy = copy.deepcopy(htf)

	# Get harmonic bond force and old/new positions
	system = htf._topology_proposal.old_system if is_old else htf._topology_proposal.new_system
	harmonic_bond_force = system.getForce(0) 
	positions = htf.old_positions(htf.hybrid_positions) if is_old else htf.new_positions(htf.hybrid_positions)

	# Get custom bond force and hybrid positions
	bond_force_index = 1
	hybrid_system = htf.hybrid_system
	custom_bond_force = hybrid_system.getForce(bond_force_index)
	hybrid_positions = htf.hybrid_positions

	# Set global parameters
	lambda_old = 1 if is_old else 0
	lambda_new = 0 if is_old else 1
	for i in range(custom_bond_force.getNumGlobalParameters()):
		if custom_bond_force.getGlobalParameterName(i) == 'lambda_0_bonds_old':
			custom_bond_force.setGlobalParameterDefaultValue(i, lambda_old)
		if custom_bond_force.getGlobalParameterName(i) == 'lambda_0_bonds_new':
			custom_bond_force.setGlobalParameterDefaultValue(i, lambda_new)

	# Zero the unique old/new bonds in the custom bond force
	hybrid_to_bond_indices = htf._hybrid_to_new_bond_indices if is_old else htf._hybrid_to_old_bond_indices
	for hybrid_idx, idx in hybrid_to_bond_indices.items():
		p1, p2, hybrid_params = custom_bond_force.getBondParameters(hybrid_idx)
		hybrid_params = list(hybrid_params)
		index_to_zero = -3 if is_old else -1
		hybrid_params[index_to_zero] *= 0
		custom_bond_force.setBondParameters(hybrid_idx, p1, p2, hybrid_params)

	# Get energy components of standard bond force
	platform = configure_platform(REFERENCE_PLATFORM)
	thermostate_other = ThermodynamicState(system=system, temperature=temperature)
	integrator_other = openmm.VerletIntegrator(1.0*unit.femtosecond)
	context_other = thermostate_other.create_context(integrator_other)
	context_other.setPositions(positions)
	components_other = compute_potential_components(context_other, beta=beta)
	print(components_other)

	# Get energy components of custom bond force
	thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
	integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
	context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
	context_hybrid.setPositions(hybrid_positions)
	components_hybrid = compute_potential_components(context_hybrid, beta=beta)
	print(components_hybrid)

	assert np.isclose([components_other[0][1]], [components_hybrid[0][1]])
	print("Success! Custom bond force and standard bond force energies are equal!")

	if check_scale:

		# Get custom bond force and hybrid positions
		bond_force_index = 1
		hybrid_system = htf_copy.hybrid_system
		custom_bond_force = hybrid_system.getForce(bond_force_index)
		hybrid_positions = htf_copy.hybrid_positions

		# Get energy components of custom bond force
		thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
		integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
		context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
		context_hybrid.setPositions(hybrid_positions)
		components_hybrid = compute_potential_components(context_hybrid, beta=beta)
		print(components_hybrid)

		# Set `scale_lambda_{i}` to 0.5
		for i in range(custom_bond_force.getNumGlobalParameters()):
			if custom_bond_force.getGlobalParameterName(i) == 'scale_lambda_0_bonds':
				custom_bond_force.setGlobalParameterDefaultValue(i, 0.5)
			elif custom_bond_force.getGlobalParameterName(i) == 'interscale_lambda_0_bonds':
				custom_bond_force.setGlobalParameterDefaultValue(i, 0.5)

		# Get energy components of custom bond force iwth scaling
		thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
		integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
		context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
		context_hybrid.setPositions(hybrid_positions)
		components_hybrid_scaled = compute_potential_components(context_hybrid, beta=beta)
		print(components_hybrid_scaled)

		assert not np.isclose([components_hybrid[0][1]], [components_hybrid_scaled[0][1]])

		print("Success! Scaling the bond force changes the energy")

def compare_angle_energies(htf, is_old=True, check_scale=False):
	"""
	Given a RxnHybridTopologyFactory, check whether the energies of the HarmonicAngleForce (from the original system) 
	match the energies of the CustomAngleForce (from the hybrid system)

	htf : RxnHybridTopologyFactory
		htf on which to run the energy validation
	is_old : bool
		whether to validate against the old system (if False, will validate against the new system)
	check_scale : bool
		whether to check that the scale region is able to scale the energy
	"""
	htf_copy = copy.deepcopy(htf)

	# Get harmonic angle force and old/new positions
	system = htf._topology_proposal.old_system if is_old else htf._topology_proposal.new_system
	harmonic_angle_force = system.getForce(1) 
	positions = htf.old_positions(htf.hybrid_positions) if is_old else htf.new_positions(htf.hybrid_positions)

	# Get custom angle force and hybrid positions
	angle_force_index = 2
	hybrid_system = htf.hybrid_system
	custom_angle_force = hybrid_system.getForce(angle_force_index)
	hybrid_positions = htf.hybrid_positions

	# Set global parameters
	lambda_old = 1 if is_old else 0
	lambda_new = 0 if is_old else 1
	for i in range(custom_angle_force.getNumGlobalParameters()):
		if custom_angle_force.getGlobalParameterName(i) == 'lambda_0_angles_old':
			custom_angle_force.setGlobalParameterDefaultValue(i, lambda_old)
		if custom_angle_force.getGlobalParameterName(i) == 'lambda_0_angles_new':
			custom_angle_force.setGlobalParameterDefaultValue(i, lambda_new)

	# Zero the unique old/new angles in the custom angle force
	hybrid_to_angle_indices = htf._hybrid_to_new_angle_indices if is_old else htf._hybrid_to_old_angle_indices
	for hybrid_idx, idx in hybrid_to_angle_indices.items():
		p1, p2, p3, hybrid_params = custom_angle_force.getAngleParameters(hybrid_idx)
		hybrid_params = list(hybrid_params)
		index_to_zero = -3 if is_old else -1
		hybrid_params[index_to_zero] *= 0
		custom_angle_force.setAngleParameters(hybrid_idx, p1, p2, p3, hybrid_params)

	# Get energy components of standard angle force
	platform = configure_platform(REFERENCE_PLATFORM)
	thermostate_other = ThermodynamicState(system=system, temperature=temperature)
	integrator_other = openmm.VerletIntegrator(1.0*unit.femtosecond)
	context_other = thermostate_other.create_context(integrator_other)
	context_other.setPositions(positions)
	components_other = compute_potential_components(context_other, beta=beta)
	print(components_other)

	# Get energy components of custom angle force
	thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
	integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
	context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
	context_hybrid.setPositions(hybrid_positions)
	components_hybrid = compute_potential_components(context_hybrid, beta=beta)
	print(components_hybrid)

	assert np.isclose([components_other[1][1]], [components_hybrid[1][1]])
	print("Success! Custom angle force and standard angle force energies are equal!")

	if check_scale:

		# Get custom bond force and hybrid positions
		angle_force_index = 2
		hybrid_system = htf_copy.hybrid_system
		custom_angle_force = hybrid_system.getForce(angle_force_index)
		hybrid_positions = htf_copy.hybrid_positions

		# Get energy components of custom angle force
		thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
		integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
		context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
		context_hybrid.setPositions(hybrid_positions)
		components_hybrid = compute_potential_components(context_hybrid, beta=beta)
		print(components_hybrid)

		# Set `scale_lambda_{i}` to 0.5
		for i in range(custom_angle_force.getNumGlobalParameters()):
			if custom_angle_force.getGlobalParameterName(i) == 'scale_lambda_0_angles':
				custom_angle_force.setGlobalParameterDefaultValue(i, 0.5)
			elif custom_angle_force.getGlobalParameterName(i) == 'interscale_lambda_0_angles':
				custom_angle_force.setGlobalParameterDefaultValue(i, 0.5)

		# Get energy components of custom angle force iwth scaling
		thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
		integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
		context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
		context_hybrid.setPositions(hybrid_positions)
		components_hybrid_scaled = compute_potential_components(context_hybrid, beta=beta)
		print(components_hybrid_scaled)

		assert not np.isclose([components_hybrid[1][1]], [components_hybrid_scaled[1][1]])

		print("Success! Scaling the angle force changes the energy")

def compare_torsion_energies(htf, is_old=True, check_scale=False):
	"""
	Given a RxnHybridTopologyFactory, check whether the energies of the PeriodicTorsionForce (from the original system) 
	match the energies of the CustomTorsionForce (from the hybrid system)

	htf : RxnHybridTopologyFactory
		htf on which to run the energy validation
	is_old : bool
		whether to validate against the old system (if False, will validate against the new system)
	check_scale : bool
		whether to check that the scale region is able to scale the energy
	"""
	htf_copy = copy.deepcopy(htf)

	# Get periodic torsion force and old/new positions
	system = htf._topology_proposal.old_system if is_old else htf._topology_proposal.new_system
	periodic_torsion_force = system.getForce(2) 
	positions = htf.old_positions(htf.hybrid_positions) if is_old else htf.new_positions(htf.hybrid_positions)

	# Get custom torsion force and hybrid positions
	torsion_force_index = 3
	hybrid_system = htf.hybrid_system
	custom_torsion_force = hybrid_system.getForce(torsion_force_index)
	hybrid_positions = htf.hybrid_positions

	# Set global parameters
	lambda_old = 1 if is_old else 0
	lambda_new = 0 if is_old else 1
	for i in range(custom_torsion_force.getNumGlobalParameters()):
		if custom_torsion_force.getGlobalParameterName(i) == 'lambda_0_torsions_old':
			custom_torsion_force.setGlobalParameterDefaultValue(i, lambda_old)
		elif custom_torsion_force.getGlobalParameterName(i) == 'lambda_0_torsions_new':
			custom_torsion_force.setGlobalParameterDefaultValue(i, lambda_new)

	# Zero the unique old/new torsions in the custom torsion force
	hybrid_to_torsion_indices = htf._hybrid_to_new_torsion_indices if is_old else htf._hybrid_to_old_torsion_indices
	print(hybrid_to_torsion_indices)
	for hybrid_idx, idx in hybrid_to_torsion_indices.items():
		p1, p2, p3, p4, hybrid_params = custom_torsion_force.getTorsionParameters(hybrid_idx)
		hybrid_params = list(hybrid_params)
		hybrid_params[-4] *= 0
		hybrid_params[-1] *= 0
		custom_torsion_force.setTorsionParameters(hybrid_idx, p1, p2, p3, p4, hybrid_params)

	# Get energy components of standard torsion force
	platform = configure_platform(REFERENCE_PLATFORM)
	thermostate_other = ThermodynamicState(system=system, temperature=temperature)
	integrator_other = openmm.VerletIntegrator(1.0*unit.femtosecond)
	context_other = thermostate_other.create_context(integrator_other)
	context_other.setPositions(positions)
	components_other = compute_potential_components(context_other, beta=beta)
	print(components_other)

	# Get energy components of custom torsion force
	thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
	integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
	context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
	context_hybrid.setPositions(hybrid_positions)
	components_hybrid = compute_potential_components(context_hybrid, beta=beta)
	print(components_hybrid)

	assert np.isclose([components_other[2][1]], [components_hybrid[2][1]])

	print("Success! Custom torsion force and standard torsion force energies are equal!")

	if check_scale:
	
		## Get custom torsion force and hybrid positions
		torsion_force_index = 3
		hybrid_system = htf_copy.hybrid_system
		custom_torsion_force = hybrid_system.getForce(torsion_force_index)
		hybrid_positions = htf_copy.hybrid_positions

		## Get energy components of custom torsion force
		thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
		integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
		context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
		context_hybrid.setPositions(hybrid_positions)
		components_hybrid = compute_potential_components(context_hybrid, beta=beta)
		print(components_hybrid)

		# Set `scale_lambda_{i}` to 0.5
		for i in range(custom_torsion_force.getNumGlobalParameters()):
			if custom_torsion_force.getGlobalParameterName(i) == 'scale_lambda_0_torsions':
				custom_torsion_force.setGlobalParameterDefaultValue(i, 0.5)
			elif custom_torsion_force.getGlobalParameterName(i) == 'interscale_lambda_0_torsions':
				custom_torsion_force.setGlobalParameterDefaultValue(i, 0.5)

		## Get energy components of custom torsion force with scaling
		thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
		integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
		context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
		context_hybrid.setPositions(hybrid_positions)
		components_hybrid_scaled = compute_potential_components(context_hybrid, beta=beta)
		print(components_hybrid_scaled)

		assert not np.isclose([components_hybrid[2][1]], [components_hybrid_scaled[2][1]])

		print("Success! Scaling the torsion force changes the energy")

def compare_electrostatics_energies(htf, is_old=True, check_scale=False):
	htf_copy = copy.deepcopy(htf)

	# Get nb force and old/new positions
	system = htf._topology_proposal.old_system if is_old else htf._topology_proposal.new_system
	nb_force = system.getForce(3) 
	positions = htf.old_positions(htf.hybrid_positions) if is_old else htf.new_positions(htf.hybrid_positions)

	# Zero the sterics 
	for i in range(nb_force.getNumParticles()):
		charge, sigma, epsilon = nb_force.getParticleParameters(i)
		nb_force.setParticleParameters(i, charge, sigma, epsilon*0)

	for i in range(nb_force.getNumExceptions()):
		p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
		nb_force.setExceptionParameters(i, p1, p2, chargeProd, sigma, epsilon*0)

	# Get custom nb force and hybrid positions
	custom_nb_force_index = 4
	custom_nb_exceptions_force_index = 6
	hybrid_system = htf.hybrid_system
	hybrid_positions = htf.hybrid_positions
	custom_nb_force = hybrid_system.getForce(custom_nb_force_index)
	custom_nb_exceptions_force = hybrid_system.getForce(custom_nb_exceptions_force_index)

	# Set global parameters
	lambda_old = 1 if is_old else 0
	lambda_new = 0 if is_old else 1
	for i in range(custom_nb_force.getNumGlobalParameters()):
		if custom_nb_force.getGlobalParameterName(i) == 'lambda_0_electrostatics_old':
			custom_nb_force.setGlobalParameterDefaultValue(i, lambda_old)
		elif custom_nb_force.getGlobalParameterName(i) == 'lambda_0_electrostatics_new':
			custom_nb_force.setGlobalParameterDefaultValue(i, lambda_new)

	for i in range(custom_nb_exceptions_force.getNumGlobalParameters()):
		if custom_nb_exceptions_force.getGlobalParameterName(i) == 'lambda_0_electrostatics_exceptions_old':
			custom_nb_exceptions_force.setGlobalParameterDefaultValue(i, lambda_old)
		elif custom_nb_exceptions_force.getGlobalParameterName(i) == 'lambda_0_electrostatics_exceptions_new':
			custom_nb_exceptions_force.setGlobalParameterDefaultValue(i, lambda_new)

	# Get energy components of standard nb force
	platform = configure_platform(REFERENCE_PLATFORM)
	thermostate_other = ThermodynamicState(system=system, temperature=temperature)
	integrator_other = openmm.VerletIntegrator(1.0*unit.femtosecond)
	context_other = thermostate_other.create_context(integrator_other)
	context_other.setPositions(positions)
	components_other = compute_potential_components(context_other, beta=beta)
	print(components_other)

	# Get energy components of custom nb force
	thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
	integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
	context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
	context_hybrid.setPositions(hybrid_positions)
	components_hybrid = compute_potential_components(context_hybrid, beta=beta)
	print(components_hybrid)

	# assert np.isclose([components_other[3][1]], [components_hybrid[2][1]])

	print("Success! Custom nb force and standard nb force electrostatics energies are equal!")

	# if check_scale:

	# 	## Get custom torsion force and hybrid positions
	# 	torsion_force_index = 4
	# 	hybrid_system = htf_copy.hybrid_system
	# 	custom_torsion_force = hybrid_system.getForce(torsion_force_index)
	# 	hybrid_positions = htf_copy.hybrid_positions

	# 	## Get energy components of custom nb force
	# 	thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
	# 	integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
	# 	context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
	# 	context_hybrid.setPositions(hybrid_positions)
	# 	components_hybrid = compute_potential_components(context_hybrid, beta=beta)
	# 	print(components_hybrid)

	# 	# Set `scale_lambda_{i}` to 0.5
	# 	for i in range(custom_torsion_force.getNumGlobalParameters()):
	# 		if custom_torsion_force.getGlobalParameterName(i) == 'scale_lambda_0_electrostatics':
	# 			custom_torsion_force.setGlobalParameterDefaultValue(i, 0.5)
	# 		elif custom_torsion_force.getGlobalParameterName(i) == 'interscale_lambda_0_electrostatics':
	# 			custom_torsion_force.setGlobalParameterDefaultValue(i, 0.5)

	# 	## Get energy components of custom nb force with scaling
	# 	thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
	# 	integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
	# 	context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
	# 	context_hybrid.setPositions(hybrid_positions)
	# 	components_hybrid_scaled = compute_potential_components(context_hybrid, beta=beta)
	# 	print(components_hybrid_scaled)

	# 	assert not np.isclose([components_hybrid[2][1]], [components_hybrid_scaled[2][1]])

	# 	print("Success! Scaling the electrostatics force changes the energy")

def generate_cdk_htf(scale_regions=None):
	"""
	Generate a RxnHybridTopologyFactory for a transformation involving CDK2 ligands.

	scale_regions : list of lists, default None
		list of scale region lists, where each scale region list contains the atom indices 
		(corresponding to the hybrid topology) that should be scaled
	"""

	import os
	from pkg_resources import resource_filename
	from perses.app import setup_relative_calculation
	from perses.app.setup_relative_calculation import getSetupOptions

	# Create a htf
	setup_directory = resource_filename("perses", "data/cdk2-example")

	# Get options
	yaml_filename = os.path.join(setup_directory, "cdk2_setup_repex.yaml")
	setup_options = getSetupOptions(yaml_filename)

	# Update options
	for parameter in ['protein_pdb', 'ligand_file']:
	    setup_options[parameter] = os.path.join(setup_directory, setup_options[parameter])
	    
	setup_options['rxn_field'] = True
	setup_options['phases'] = ['solvent']
	setup_options['scale_regions'] = scale_regions
	setup_options['validate_endstate_energies'] = False
	    
	setup_dict = setup_relative_calculation.run_setup(setup_options, serialize_systems=False, build_samplers=False)
	htf = setup_dict['hybrid_topology_factories']['solvent']
	return htf

def test_bookkeeping():
	## Alanine dipeptide in solvent -- one alchemical region
	# Create a htf
	atp, system_generator = generate_atp(phase='solvent')
	htf = generate_dipeptide_top_pos_sys(atp.topology, 
											'THR', 
											atp.system, 
											atp.positions, 
											system_generator,
											rxn_field=True,
											flatten_torsions=True,
											flatten_exceptions=True,
											validate_endstate_energy=False,
											conduct_htf_prop=True)

	# Make copies for running tests
	htf_copy_1, htf_copy_2, htf_copy_3, htf_copy_4, htf_copy_5, htf_copy_6  = copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf)
	
	# Run tests comparing old system to hybrid system
	compare_bond_energies(htf_copy_1)
	compare_angle_energies(htf_copy_2)
	compare_torsion_energies(htf_copy_3)

	# Run tests comparing new system to hybrid system
	compare_bond_energies(htf_copy_4, is_old=False)
	compare_angle_energies(htf_copy_5, is_old=False)
	compare_torsion_energies(htf_copy_6, is_old=False)

	## Alanine dipeptide in solvent -- one alchemical region, one scale region
	# Create a htf
	htf = generate_dipeptide_top_pos_sys(atp.topology, 
											'THR', 
											atp.system, 
											atp.positions, 
											system_generator,
											rxn_field=True,
											flatten_torsions=True,
											flatten_exceptions=True,
											validate_endstate_energy=False,
											conduct_htf_prop=True, 
											scale_regions=[[10, 11, 12, 13, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564]])
	# Make copies for running tests
	htf_copy_1, htf_copy_2, htf_copy_3, htf_copy_4, htf_copy_5, htf_copy_6  = copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf)
	
	# Run tests comparing old system to hybrid system
	compare_bond_energies(htf_copy_1)
	compare_angle_energies(htf_copy_2)
	compare_torsion_energies(htf_copy_3)

	# Run tests comparing new system to hybrid system
	compare_bond_energies(htf_copy_4, is_old=False, check_scale=True)
	compare_angle_energies(htf_copy_5, is_old=False, check_scale=True)
	compare_torsion_energies(htf_copy_6, is_old=False, check_scale=True)

	## Cdk ligand in solvent -- one alchemical region
	# Create a htf
	htf = generate_cdk_htf()

	# Make copies for running tests
	htf_copy_1, htf_copy_2, htf_copy_3, htf_copy_4, htf_copy_5, htf_copy_6  = copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf)

	# Run tests comparing old system to hybrid system
	compare_bond_energies(htf_copy_1)
	compare_angle_energies(htf_copy_2)
	compare_torsion_energies(htf_copy_3)

	# Run tests comparing new system to hybrid system
	compare_bond_energies(htf_copy_4, is_old=False, check_scale=True)
	compare_angle_energies(htf_copy_5, is_old=False, check_scale=True)
	compare_torsion_energies(htf_copy_6, is_old=False, check_scale=True)

	## Cdk ligand in solvent -- one alchemical region, one scale region
	# Create a htf
	htf = generate_cdk_htf(scale_regions=[[44, 2197]])

	# Make copies for running tests
	htf_copy_1, htf_copy_2, htf_copy_3, htf_copy_4, htf_copy_5, htf_copy_6  = copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf), copy.deepcopy(htf)
	
	# Run tests comparing old system to hybrid system
	compare_bond_energies(htf_copy_1)
	compare_angle_energies(htf_copy_2)
	compare_torsion_energies(htf_copy_3)

	# Run tests comparing new system to hybrid system
	compare_bond_energies(htf_copy_4, is_old=False, check_scale=True)
	compare_angle_energies(htf_copy_5, is_old=False, check_scale=True)
	compare_torsion_energies(htf_copy_6, is_old=False, check_scale=True)

