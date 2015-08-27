import topology_proposal
import weight_calculation
import system_generator
import geometry
import alchemical_elimination
import ncmc_switching
import numpy as np
import simtk.openmm as openmm
import simtk.openmm.app as app


if __name__=="__main__":
    #prepare some dummies representing the "current system"	
    current_system = openmm.System()
    current_topology = app.Topology()
    current_positions = np.array([0.0,0.0,0.0])
    alchemical_elimination_protocol = dict()
    alchemical_introduction_protocol = dict()
    niterations = 10
    for i in range(niterations):
        #initialize with the current topology, system, positoins	    
        old_system = current_system
        old_topology = current_topology
        old_initial_positions = current_positions
	old_log_weight = 0

        #This stage proposes a new topology, first creating a Transformation class
        transformation = topology_proposal.Transformation(old_topology, old_system, {'data': 0})
        top_proposal = transformation.propose() #the method that actually proposes the topology

        #calculate the weight g_k of the newly proposed state by instantiating the calculator class
        state_weight_calculator = weight_calculation.StateWeight(top_proposal)
        log_weight = state_weight_calculator.log_weight

        #create a generator for new ligand and complex systems, retreive the complex
	new_system_generator = system_generator.SystemGenerator(top_proposal)
        new_system = new_system_generator.complex_system
        print(top_proposal)

        #Get an alchemically modified version of the old system (should be cached, actually). Similar semantics as above
        old_alchemical_engine = alchemical_elimination.AlchemicalEliminationEngine(old_system, top_proposal)
        old_alchemical_system = old_alchemical_engine.alchemical_system
        print(old_alchemical_system)

        #Create an NCMCEngine to make an NCMC move from the old system --> core, following the alchemical elimination protocol
        ncmc_elimination = ncmc_switching.NCMCEngine(old_alchemical_system, alchemical_elimination_protocol, old_initial_positions)
        ncmc_elimination.integrate()
        ncmc_old_positions = ncmc_elimination.final_positions
        print(ncmc_old_positions)
        ncmc_elimination_logp = ncmc_elimination.log_ncmc
        print(ncmc_elimination_logp)

        #We're now at a common core. Instantiate GeometryEngine to propose new coordinates
        geometry_engine = geometry.GeometryEngine(top_proposal, ncmc_old_positions)
        geometry_proposal = geometry_engine.propose()

        #Generate an alchemical system for the new topology to introduce the atoms
        new_alchemical_engine = alchemical_elimination.AlchemicalEliminationEngine(new_system, topology_proposal)
        new_alchemical_system = new_alchemical_engine.alchemical_system

        #NCMCEngine to go from core ---> new molecule
        ncmc_introduction = ncmc_switching.NCMCEngine(new_alchemical_system, alchemical_introduction_protocol, geometry_proposal.new_positions)
        ncmc_introduction.integrate()
        ncmc_introduction_logp = ncmc_introduction.log_ncmc
        ncmc_new_positions = ncmc_introduction.final_positions
        print(ncmc_new_positions)
        print(ncmc_introduction_logp)

	#accept/reject
        logp_accept = top_proposal.logp + geometry_proposal.logp + ncmc_elimination_logp + ncmc_introduction_logp + log_weight - old_log_weight
        print(logp_accept)
        if logp_accept>=0.0 or np.exp(logp_accept)>np.random.uniform():
            current_system = new_system
            current_topology = top_proposal.new_topology
            current_positions = ncmc_new_positions
