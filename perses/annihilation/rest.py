"""
From a standard simtk.openmm.system, make a REST2-amenable system
"""

"""
Imports
"""
import simtk.openmm as openmm
import simtk.unit as unit
import mdtraj as md
import numpy as np
import copy
import enum
from perses.annihilation.relative import HybridTopologyFactory

#######LOGGING#############################
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("REST")
_logger.setLevel(logging.INFO)
###########################################

class RESTTopologyFactory(HybridTopologyFactory):
    """
    This class takes a standard simtk.openmm.system equipped with
        a. `HarmonicBondForce`
        b. `HarmonicAngleForce`
        c. `PeriodicTorsionForce`
        d. `NonbondedForce`
    and will convert to another system equipped with
        a. `CustomBondForce`: rewrite `HarmonicBondForce`
        b. `CustomAngleForce`: rewrite `HarmonicAngleForce`
        c. `CustomTorsionForce`: rewrite `PeriodicTorsionForce`
        d. `NonbondedForce`: solvent-solvent
            solvent sterics and electrostatics and exceptions are treated in standard form (no scaling), but solute terms are _all_ zeroed
        e. `CustomNonbondedForce`: solvent-solute and solute-solute
            creates a solvent and solute interaction group. the solute interacts with itself with a rescaling factor, and the solvent interacts with solute (via separate rescaling factor)
        f. `CustomBondForce`:
            since we cannot appropriately treat exceptions in the solute region or the solute/solvent region, we need to treat them as an exception force
    """
    _known_forces = {'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce', 'MonteCarloBarostat'}

    def __init__(self, system, solute_region, use_dispersion_correction=False):
        """
        arguments
            system : simtk.openmm.system
                system that will be rewritten
            solute_region : simtk.openmm.system
                subset solute region
            use_dispersion_correction : bool, default False
                whether to use a dispersion correction
        """
        self._num_particles = system.getNumParticles()
        self._og_system = system
        self._og_system_forces = {type(force).__name__ : force for force in self._old_system.getForces()}
        self._out_system_forces = {}

        assert set(solute_region).issubset(set(range(self._num_particles))), f"the solute region is not a subset of the system particles"
        self._nonbonded_method = self._old_system_forces['NonbondedForce'].getNonbondedMethod()
        self._out_system = openmm.System()

        for particle_idx in range(self._num_particles):
            particle_mass = self._og_system.getParticleMass(particle_idx)
            hybrid_idx = self._out_system.addParticle(particle_mass)

        if "MonteCarloBarostat" in self._og_system_forces.keys():
            barostat = copy.deepcopy(self._og_system_forces["MonteCarloBarostat"])
            self._out_system.addForce(barostat)
            self._out_system_forces[barostat.__class__.__name__] = barostat
            _logger.info("Added MonteCarloBarostat.")
        else:
            _logger.info("No MonteCarloBarostat added.")

        # Copy over the box vectors:
        box_vectors = self._og_system.getDefaultPeriodicBoxVectors()
        self._new_system.setDefaultPeriodicBoxVectors(*box_vectors)
        _logger.info(f"getDefaultPeriodicBoxVectors added to hybrid: {box_vectors}")

        self._og_system_exceptions = self._generate_dict_from_exceptions(self._og_system_forces['NonbondedForce'])


        # Check that there are no unknown forces in the new and old systems:
        for system_name in ('og'):
            force_names = getattr(self, '_{}_system_forces'.format(system_name)).keys()
            unknown_forces = set(force_names) - set(self._known_forces)
            if len(unknown_forces) > 0:
                raise ValueError("Unkown forces {} encountered in {} system" % (unknown_forces, system_name))
        _logger.info("No unknown forces.")

    def _handle_constraints(self):
        for constraint_idx in range(self._og_system.getNumConstraints()):
            atom1, atom2, length = system.getConstraintParameters(constraint_idx)
            self._out_system.addConstraint(atom1, atom2, length)

    def _add_bond_force_terms(self):
        core_energy_expression = '(K/2)*(r-length)^2;'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomBondForce(core_energy_expression)
        custom_core_force.addPerBondParameter('length')
        custom_core_force.addPerBondParameter('k')
        custom_core_force.addPerBondParameter('identifier')

        custom_core_force.addGlobalParameter('solute_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    @property
    def scaling_expression(self):
        out = "scale_factor = solute_scale*_is_solute + solvent_scale*_is_solvent + inter_scale*_is_inter; \
               solvent_scale = 1.; \
               _is_solute = delta(identifier); _is_solvent = delta(1-identifier); _is_inter = delta(2-identifier); \
               "
        return out

    def _add_angle_force_terms(self):
        core_energy_expression = '(K/2)*(theta-theta0)^2;'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomBondForce(core_energy_expression)
        custom_core_force.addPerBondParameter('theta0')
        custom_core_force.addPerBondParameter('k')
        custom_core_force.addPerBondParameter('identifier')

        custom_core_force.addGlobalParameter('solute_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def _add_torsion_force_terms(self):
        core_energy_expression = 'K*(1+cos(periodicity*theta-phase));'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomTorsionForce(core_energy_expression)
        custom_core_force.addPerTorsionParameter('periodicity') # molecule1 periodicity
        custom_core_force.addPerTorsionParameter('phase') # molecule1 phase
        custom_core_force.addPerTorsionParameter('K') # molecule1 spring constant
        custom_core_force.addPerTorsionParameter('identifier')

        custom_core_force.addGlobalParameter('solute_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def _add_nonbonded_force_terms(self):
        from openmmtools.constants import ONE_4PI_EPS0 # OpenMM constant for Coulomb interactions (implicitly in md_unit_system units)

        standard_nonbonded_force = openmm.NonbondedForce()
        custom_nonbonded_expression = f"(4*epsilon*((sigma/r)^12-(sigma/r)^6) + ONE_4PI_EPS0*chargeProd/r) * scale_factor; \
                                        sigma=0.5*(sigma1+sigma2); \
                                        epsilon=sqrt(epsilon1*epsilon2); \
                                        ONE_4PI_EPS0 = {ONE_4PI_EPS0}; \
                                        chargeProd=q1*q2;"

        custom_nonbonded_expression += self.scaling_expression
        custom_nonbonded_force = openmm.CustomNonbondedForce(custom_nonbonded_expression)

        self._out_system.addForce(standard_nonbonded_force)
        self._out_system_forces['NonbondedForce'] = standard_nonbonded_force

        #set the appropriate parameters
        epsilon_solvent = self._og_system_forces['NonbondedForce'].getReactionFieldDielectric()
        r_cutoff = self._og_system_forces['NonbondedForce'].getCutoffDistance()
        if self._nonbonded_method != openmm.NonbondedForce.NoCutoff
            standard_nonbonded_force.setReactionFieldDielectric(epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
            custom_nonbonded_force.setCutoffDistance(r_cutoff)
        if self._nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            [alpha_ewald, nx, ny, nz] = self._og_system_forces['NonbondedForce'].getPMEParameters()
            delta = self._og_system_forces['NonbondedForce'].getEwaldErrorTolerance()
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
        standard_nonbonded_force.setNonbondedMethod(self._nonbonded_method)
        custom_nonbonded_method.setNonbondedMethod(self._translate_nonbonded_method_to_custom(self._nonbonded_method))

        #translate nonbonded to custom
        if self._og_system_forces['NonbondedForce'].getUseDispersionCorrection():
            self._out_system_forces['NonbondedForce'].setUseDispersionCorrection(True)
            if self._use_dispersion_correction:
                custom_nonbonded_force.setUseLongRangeCorrection(True)
        else:
            custom_nonbonded_force.setUseLongRangeCorrection(False)

        if self._og_system_forces['NonbondedForce'].getUseSwitchingFunction():
            switching_distance = self._out_system_forces['NonbondedForce'].getSwitchingDistance()
            standard_nonbonded_force.setUseSwitchingFunction(True)
            standard_nonbonded_force.setSwitchingDistance(switching_distance)
            custom_nonbonded_force.setUseSwitchingFunction(True)
            custom_nonbonded_force.setSwitchingDistance(switching_distance)
        else:
            standard_nonbonded_force.setUseSwitchingFunction(False)
            custom_nonbonded_force.setUseSwitchingFunction(False)

        custom_nonbonded_force.addPerParticleParameter("q")
        custom_nonbonded_force.addPerParticleParameter("sigma")
        custom_nonbonded_force.addPerParticleParameter("epsilon")
        
