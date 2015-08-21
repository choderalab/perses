import topology_proposal
import geometry
import alchemical_elimination
import ncmc_switching
import numpy as np
import simtk.openmm as openmm
import simtk.openmm.app as app


if __name__=="__main__":
    old_system = openmm.System()
    old_topology = app.Topology
    alchemical_elimination_protocol = dict()
    old_initial_positions = np.array([0.0,0.0,0.0])
    transformation = topology_proposal.Transformation(old_topology, old_system, {'data': 0})
    top_proposal = transformation.propose()
    new_system = openmm.System()
    introduction_alchemical_protocol = dict()
    print(top_proposal)
    old_alchemical_engine = alchemical_elimination.AlchemicalEliminationEngine(old_system, top_proposal)
    old_alchemical_system = old_alchemical_engine.alchemical_system
    print(old_alchemical_system)
    ncmc_elimination = ncmc_switching.NCMCEngine(old_alchemical_system, alchemical_elimination_protocol, old_initial_positions)
    ncmc_elimination.integrate()
    ncmc_old_positions = ncmc_elimination.final_positions
    print(ncmc_old_positions)
    ncmc_elimination_logp = ncmc_elimination.log_ncmc
    print(ncmc_elimination_logp)
    geometry_engine = geometry.GeometryEngine(top_proposal, ncmc_old_positions)
    geometry_proposal = geometry_engine.propose()
    new_alchemical_engine = alchemical_elimination.AlchemicalEliminationEngine(new_system, topology_proposal)
    new_alchemical_system = new_alchemical_engine.alchemical_system
    ncmc_introduction = ncmc_switching.NCMCEngine(new_alchemical_system, introduction_alchemical_protocol, geometry_proposal.new_positions)
    ncmc_introduction.integrate()
    ncmc_introduction_logp = ncmc_introduction.log_ncmc
    ncmc_new_positions = ncmc_introduction.final_positions
    print(ncmc_new_positions)
    print(ncmc_introduction_logp)
    logpaccept = top_proposal.logp + geometry_proposal.logp + ncmc_elimination_logp + ncmc_introduction_logp + log_weight  
