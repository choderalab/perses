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
    SmallMoleculeAtomMapper
    PremappedSmallMoleculeSetProposalEngine
    SmallMoleculeSetProposalEngine
    PolymerProposalEngine
    PointMutationEngine
    PeptideLibraryEngine
    NullProposalEngine
    NaphthaleneProposalEngine
    ButaneProposalEngine
    PropaneProposalEngine

OpenMM System generation utilities
----------------------------------

.. currentmodule:: perses.rjmc.topology_proposal
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SystemGenerator

Geometry proposal engines
-------------------------

.. currentmodule:: perses.rjmc.geometry
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    GeometryEngine
    FFAllAngleGeometryEngine
    OmegaGeometryEngine

Geometry utility classes
------------------------

.. currentmodule:: perses.rjmc.geometry
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PredAtomTopologyIndex
    BootstrapParticleFilter
    GeometrySystemGenerator
    GeometrySystemGeneratorFast
    PredHBond
    ProposalOrderTools
