.. _rjmc:

Molecular geometry generation via RJMC
**************************************

Reversible jump Monte Carlo (RJMC) engine for sampling molecular geometries in which atoms are created/destroyed.

Topology proposal engines
-------------------------

.. currentmodule:: perses.rjmc.topology_proposal
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    TopologyProposal
    ProposalEngine
    .. SmallMoleculeAtomMapper
    .. PremappedSmallMoleculeSetProposalEngine
    SmallMoleculeSetProposalEngine
    PolymerProposalEngine
    PointMutationEngine
    PeptideLibraryEngine
    .. JRG: This class does not exist anymore?
    .. ButaneProposalEngine
    .. NaphthaleneProposalEngine
    .. PropaneProposalEngine
    .. NullProposalEngine

OpenMM System generation utilities
----------------------------------

.. currentmodule:: perses.rjmc.topology_proposal
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    .. .. This has moved to openforcefields?
    .. SystemGenerator

Geometry proposal engines
-------------------------

.. currentmodule:: perses.rjmc.geometry
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    GeometryEngine
    FFAllAngleGeometryEngine
    .. JRG: This class does not exist anymore?
    .. OmegaGeometryEngine

Geometry utility classes
------------------------

.. currentmodule:: perses.rjmc.geometry
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    GeometrySystemGenerator
    .. JRG: These classes does not exist anymore?
    .. BootstrapParticleFilter
    .. GeometrySystemGeneratorFast
    .. PredAtomTopologyIndex
    .. PredHBond
    .. .. This one is also dead code in some modules!
    .. ProposalOrderTools
