#!/bin/env python

"""
Example illustrating use of expanded ensembles framework to perform a generic expanded ensembles simulation over chemical species.

This could represent sampling of small molecules, protein mutants, or both.

"""


import numpy as np
import simtk.openmm as openmm
import simtk.openmm.app as app
from perses.rjmc.topology_proposal import Transformation, SamplerState
from perses.bias.bias_engine import BiasEngine
from perses.annihilation.alchemical_engine import AlchemicalEliminationEngine
from perses.rjmc.geometry import GeometryEngine
from perses.annihilation.ncmc_switching import NCMCEngine
from perses.rjmc.system_engine import SystemGenerator

def run():
    # Create initial model system, topology, and positions.
    # QUESTION: Do we want to maintain 'metadata' as well?
    # QUESTION: What if we incorporate the tuple (system, topology, metadata, positions) into a SamplerState object, or a named tuple?

    # Specify alchemical elimination and introduction protocols.
    alchemical_elimination_protocol = dict()
    alchemical_introduction_protocol = dict()

    #Create proposal metadata, such as the list of molecules to sample (SMILES here)
    proposal_metadata = {'molecule_list': ['CC','C=C']}
    transformation = Transformation(proposal_metadata)

    #initialize weight calculation engine, along with its metadata
    bias_metadata = {'type':'solvation free energy'}
    bias_calculator = BiasEngine(bias_metadata)

    #initialize with a log_weight
    current_log_weight = 0.0

    #initialize system generator, along with its metadata
    system_generator_metadata = {'protein_ff':'amber99sbildn.xml'}
    system_generator = SystemGenerator(system_generator_metadata)

    #Initialize AlchemicalEliminationEngine
    alchemical_metadata = {'data':0}
    alchemical_engine = AlchemicalEliminationEngine(alchemical_metadata)

    #Initialize NCMC engines.
    sitching_timestep = 1.0 * unit.femtoseconds
    switching_nsteps = 10
    switching_functions = {
        'alchemical_sterics' : 'lambda',
        'alchemical_electrostatocs' : 'lambda',
        'alchemical_bonds' : 'lambda',
        'alchemical_angles' : 'lambda',
        'alchemical_torsionss' : 'lambda'
        }
    ncmc_engine = NCMCEngine(temperature=temperature, timestep=switching_timestep, nsteps=switching_nsteps, functions=switching_functions)

    #initialize GeometryEngine
    geometry_metadata = {'data': 0}
    geometry_engine = GeometryEngine(geometry_metadata)

    #initialize
    (system, topology, positions, state_metadata) = (openmm.System(), app.Topology(), np.array([0.0,0.0,0.0]), {'mol': 'CC'})
    # Run a anumber of iterations.
    niterations = 10
    for i in range(niterations):
        # Store old (system, topology, positions).

        # Propose a transformation from one chemical species to another.
        # QUESTION: This could depend on the geometry; could it not?
                               # The metadata might contain information about the topology, such as the SMILES string or amino acid sequence.
        # QUESTION: Could we instead initialize a transformation object once as
        # transformation = topology_proposal.Transformation(metametadata)
        # and then do
        # [new_topology, new_system, new_metadata] = transformation.propose(old_topology, old_system, old_metadata)?
        top_proposal = transformation.propose(system, topology, positions, state_metadata) # QUESTION: Is this a Topology object or some other container?

        # QUESTION: What about instead initializing StateWeight once, and then using
        # log_state_weight = state_weight.computeLogStateWeight(new_topology, new_system, new_metadata)?
        log_weight = bias_calculator.generate_bias(top_proposal)

        # Create a new System object from the proposed topology.
        # QUESTION: Do we want to isolate the system creation from the Transformation proposal? This does seem like something that *could* be pretty well isolated.
        # QUESTION: Should we at least create only one SystemGenerator, like
        # system_generator = SystemGenerator(forcefield, etc)
        # new_system = system_generator.createSystem(new_topology)
        new_system = system_generator.new_system(top_proposal)
        print(top_proposal)

        # Perform alchemical transformation.

        # Alchemically eliminate atoms being removed.
        # QUESTION: We need the alchemical transformation to eliminate the atoms being removed, right? So we need old topology/system and intermediate topology/system.
        # What if we had a method that took the old/new (topology, system, metadata) and generated some information about the transformation?
        # At minimum, we need to know what atoms are being eliminated/introduced.  We might also need to identify some System for the "intermediate" with scaffold atoms that are shared,
        # since charges and types may need to be modified?
        old_alchemical_system = alchemical_engine.make_alchemical_system(system, top_proposal)
        print(old_alchemical_system)
        [ncmc_old_positions, ncmc_elimination_logp] = ncmc_engine.integrate(old_alchemical_system, positions, direction='deletion')
        print(ncmc_old_positions)
        print(ncmc_elimination_logp)

        # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
        # QUESTION: Again, maybe we want to have the geometry engine initialized once only?
        geometry_proposal = geometry_engine.propose(top_proposal, new_system, ncmc_old_positions)

        # Alchemically introduce new atoms.
        # QUESTION: Similarly, this needs to introduce new atoms.  We need to know both intermediate toplogy/system and new topology/system, right?
        new_alchemical_system = alchemical_engine.make_alchemical_system(new_system, top_proposal)
        [ncmc_new_positions, ncmc_introduction_logp] = ncmc_engine.integrate(new_alchemical_system, geometry_proposal.new_positions, direction='creation')
        print(ncmc_new_positions)
        print(ncmc_introduction_logp)

        # Compute total log acceptance probability, including all components.
        logp_accept = top_proposal.logp + geometry_proposal.logp + ncmc_elimination_logp + ncmc_introduction_logp + log_weight - current_log_weight
        print(logp_accept)

        # Accept or reject.
        if (logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept)):
            # Accept.
            (system, topology, positions, current_log_weight) = (new_system, top_proposal.new_topology, ncmc_new_positions, log_weight)
        else:
            # Reject.
            pass

#
# MAIN
#

if __name__=="__main__":
    run()
