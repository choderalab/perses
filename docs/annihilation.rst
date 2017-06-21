.. _annihilation:

Alchemical transformations
**************************

Tools for nonequilibrium alchemical transformations

Nonequilibrium switching
------------------------

.. currentmodule:: perses.annihilation.ncmc_switching
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    NCMCEngine
    NCMCHybridEngine
    NCMCAlchemicalIntegrator
    NCMCGHMCAlchemicalIntegrator

Relative alchemical transformations
-----------------------------------

.. currentmodule:: perses.annihilation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    HybridTopologyFactory
    
Principle of HybridTopologyFactory
----------------------------------
    
HybridTopologyFactory is a class that automates the construction of so-called hybrid topologies and systems. In short, this amounts to using an atom map to merge
two initial systems into a new system that contains a union of the former systems' degrees of freedom, along with a lambda parameter to control the degree to which the hybrid represents the old or new system. The process of creating this system happens in several steps, detailed below. Before proceeding, there are several important caveats:

- Atom maps:
    - Atoms whose constraints change between new and old system may not be mapped
    - Virtual sites are only permitted if their parameters are identical in the old and new systems

- Systems:
    - Only HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, NonbondedForce, and MonteCarloBarostat are supported. The presence of other forces will raise an exception.
    
Assignment of Particles to Appropriate Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to properly assign parameters to the particles in the hybrid system, the particles are first assigned to one of four groups:

- Environment: These atoms are mapped between the new and old systems, and additionally are not part of a residue that differs between new and old systems.
    - Examples: solvent molecules, protein residues outside a changing residue

- Core: These atoms are mapped between the new and old system, but are part of a residue that differs between new and old systems. As such, they may need to change parameters.
    - Example: The carbon atoms in a benzene that is being transformed to a chlorobenzene.
    
- Unique old: These atoms are not in the map, and are present only in the old system
    - Example: an extraneous hydrogen atom in a cyclohexane being transformed into a benzene
    
- Unique new: These atoms are not in the map, and are present only in the new system
    - Example: The chlorine atom that appears when a benzene is transformed into a chlorobenzene.
    
Creation of Force Terms
^^^^^^^^^^^^^^^^^^^^^^^

For each supported force, we have to create at least one custom force, as well as an unmodified force. In general, the interactions are handled as follows:

