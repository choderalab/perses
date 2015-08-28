#!/bin/env python

"""
Example illustrating use of expanded ensembles framework to perform a generic expanded ensembles simulation over chemical species.

This could represent sampling of small molecules, protein mutants, or both.

"""


import numpy as np
import simtk.openmm as openmm
import simtk.openmm.app as app


def run():

    # Create initial model system, topology, and positions.
    # QUESTION: Do we want to maintain 'metadata' as well?
    # QUESTION: What if we incorporate the tuple (system, topology, metadata, positions) into a SamplerState object, or a named tuple?
    (system, topology, positions) = [openmm.System(), app.Topology(), np.zeros([1,3], np.float32)]

    # Specify alchemical elimination and introduction protocols.
    alchemical_elimination_protocol = dict()
    alchemical_introduction_protocol = dict()

    # Run a anumber of iterations.
    niterations = 10
    for i in range(niterations):
        # Store old (system, topology, positions).
        (old_system, old_topology, old_positions) = (system, topology, positions)

        # Propose a transformation from one chemical species to another.
        # QUESTION: This could depend on the geometry; could it not?
        metadata = {'data': 0} # QUESTION: Shouldn't this 'metadata' state be maintained throughout, and old and new versions kept?
                               # The metadata might contain information about the topology, such as the SMILES string or amino acid sequence.
        # QUESTION: Could we instead initialize a transformation object once as
        # transformation = topology_proposal.Transformation(metametadata)
        # and then do
        # [new_topology, new_system, new_metadata] = transformation.propose(old_topology, old_system, old_metadata)?
        from topology_proposal import Transformation
        transformation = Transformation(old_topology, old_system, metadata)    
        top_proposal = transformation.propose() # QUESTION: Is this a Topology object or some other container?

        # QUESTION: What about instead initializing StateWeight once, and then using
        # log_state_weight = state_weight.computeLogStateWeight(new_topology, new_system, new_metadata)?
        from weight_calculation import StateWeight
        state_weight_calculator = StateWeight(top_proposal)
        log_weight = state_weight_calculator.log_weight

        # Create a new System object from the proposed topology.
        # QUESTION: Do we want to isolate the system creation from the Transformation proposal? This does seem like something that *could* be pretty well isolated.
        # QUESTION: Should we at least create only one SystemGenerator, like
        # system_generator = SystemGenerator(forcefield, etc)
        # new_system = system_generator.createSystem(new_topology)
        from system_generator import SystemGenerator
	new_system_generator = SystemGenerator(top_proposal)
        new_system = new_system_generator.complex_system
        print(top_proposal)
        
        # Perform alchemical transformation.
        from alchemical_elimination import AlchemicalEliminationEngine
        from ncmc_switching import NCMCEngine

        # Alchemically eliminate atoms being removed.
        # QUESTION: We need the alchemical transformation to eliminate the atoms being removed, right? So we need old topology/system and intermediate topology/system.
        # What if we had a method that took the old/new (topology, system, metadata) and generated some information about the transformation?
        # At minimum, we need to know what atoms are being eliminated/introduced.  We might also need to identify some System for the "intermediate" with scaffold atoms that are shared,
        # since charges and types may need to be modified?
        old_alchemical_engine = AlchemicalEliminationEngine(old_system, top_proposal)
        old_alchemical_system = old_alchemical_engine.alchemical_system
        print(old_alchemical_system)
        ncmc_elimination = NCMCEngine(old_alchemical_system, alchemical_elimination_protocol, old_positions)
        ncmc_elimination.integrate()
        ncmc_old_positions = ncmc_elimination.final_positions
        print(ncmc_old_positions)
        ncmc_elimination_logp = ncmc_elimination.log_ncmc
        print(ncmc_elimination_logp)

        # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
        # QUESTION: Again, maybe we want to have the geometry engine initialized once only?
        from geometry import GeometryEngine
        geometry_engine = GeometryEngine(top_proposal, ncmc_old_positions)
        geometry_proposal = geometry_engine.propose()

        # Alchemically introduce new atoms.
        # QUESTION: Similarly, this needs to introduce new atoms.  We need to know both intermediate toplogy/system and new topology/system, right?
        new_alchemical_engine = AlchemicalEliminationEngine(new_system, top_proposal)
        new_alchemical_system = new_alchemical_engine.alchemical_system
        ncmc_introduction = NCMCEngine(new_alchemical_system, alchemical_introduction_protocol, geometry_proposal.new_positions)
        ncmc_introduction.integrate()
        ncmc_introduction_logp = ncmc_introduction.log_ncmc
        ncmc_new_positions = ncmc_introduction.final_positions
        print(ncmc_new_positions)
        print(ncmc_introduction_logp)

        # Compute total log acceptance probability, including all components.
        logp_accept = top_proposal.logp + geometry_proposal.logp + ncmc_elimination_logp + ncmc_introduction_logp + log_weight
        print(logp_accept)

        # Accept or reject.
        if (logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept)):
            # Accept.
            (system, topology, positions) = (new_system, top_proposal.new_topology, ncmc_new_positions)
        else:
            # Reject.
            pass

#
# MAIN
#

if __name__=="__main__":
    run()
