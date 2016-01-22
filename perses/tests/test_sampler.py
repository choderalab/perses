"""
Write a basic test that ensures that all parts are correctly working together
"""

def test_sampler():
    from perses.rjmc.topology_proposal import ProposalEngine
    from perses.rjmc.geometry import GeometryEngine
    from perses.annihilation.ncmc_switching import BaseNCMCEngine
    from perses.bias import bias_engine

    #create proposal engine
    proposal_engine = ProposalEngine(proposal_metadata={'molecule_list': ['CCC','CCCC','CCCCCC']})

    #create geometry engine
    geometry_engine = GeometryEngine()

    #create NCMC Engine with defaults
    ncmc = BaseNCMCEngine()


