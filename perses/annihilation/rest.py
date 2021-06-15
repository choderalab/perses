"""
From a standard simtk.openmm.system, make a REST2-amenable system
"""

"""
Imports
"""
import simtk.openmm as openmm
import copy
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

    def __init__(self, system, solute_region, use_dispersion_correction=False, **kwargs):
        """
        arguments
            system : simtk.openmm.system
                system that will be rewritten
            solute_region : simtk.openmm.system
                subset solute region
            use_dispersion_correction : bool, default False
                whether to use a dispersion correction

        Properties
        ----------
            REST_system : simtk.openmm.system
                the REST2-implemented system
        """
        self._use_dispersion_correction = use_dispersion_correction
        self._num_particles = system.getNumParticles()
        self._og_system = system
        self._og_system_forces = {type(force).__name__ : force for force in self._og_system.getForces()}
        self._out_system_forces = {}
        self._solute_region = solute_region
        self._solvent_region = list(set(range(self._num_particles)).difference(set(self._solute_region)))
        _logger.debug(f"solvent region: {self._solvent_region}")
        _logger.debug(f"solute region: {self._solute_region}")


        assert set(solute_region).issubset(set(range(self._num_particles))), f"the solute region is not a subset of the system particles"
        self._nonbonded_method = self._og_system_forces['NonbondedForce'].getNonbondedMethod()
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
        self._out_system.setDefaultPeriodicBoxVectors(*box_vectors)
        _logger.info(f"getDefaultPeriodicBoxVectors added to hybrid: {box_vectors}")

        self._og_system_exceptions = self._generate_dict_from_exceptions(self._og_system_forces['NonbondedForce'])


        # Check that there are no unknown forces in the new and old systems:
        for system_name in ['og']:
            force_names = getattr(self, '_{}_system_forces'.format(system_name)).keys()
            unknown_forces = set(force_names) - set(self._known_forces)
            if len(unknown_forces) > 0:
                raise ValueError(f"Unknown forces {unknown_forces} encountered in {system_name} system")
        _logger.info("No unknown forces.")

        self._handle_constraints()

        self._add_bond_force_terms()
        self._add_bonds()

        self._add_angle_force_terms()
        self._add_angles()

        self._add_torsion_force_terms()
        self._add_torsions()

        self._add_nonbonded_force_terms()
        self._add_nonbondeds()


    def _handle_constraints(self):
        for constraint_idx in range(self._og_system.getNumConstraints()):
            atom1, atom2, length = self._og_system.getConstraintParameters(constraint_idx)
            self._out_system.addConstraint(atom1, atom2, length)

    def _add_bond_force_terms(self):
        core_energy_expression = '(K/2)*(r-length)^2;'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression()

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomBondForce(core_energy_expression)
        custom_core_force.addPerBondParameter('length')
        custom_core_force.addPerBondParameter('k')
        custom_core_force.addPerBondParameter('identifier')

        custom_core_force.addGlobalParameter('solute_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def scaling_expression(self, nb=False):
        if not nb:
            last_line = '_is_solute = delta(identifier); _is_solvent = delta(1-identifier); _is_inter = delta(2-identifier);'
        else:
            last_line = '_is_solute = delta(identifier1 + identifier2); _is_solvent = delta(2 - (identifier1 + identifier2)); _is_inter = delta(1 - (identifier1 + identifier2));'

        out = f"scale_factor = solute_scale*_is_solute + solvent_scale*_is_solvent + inter_scale*_is_inter; \
               solvent_scale = 1.; \
               {last_line} \
               "
        return out

    def _add_angle_force_terms(self):
        core_energy_expression = '(K/2)*(theta-theta0)^2;'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression()

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomAngleForce(core_energy_expression)
        custom_core_force.addPerAngleParameter('theta0')
        custom_core_force.addPerAngleParameter('k')
        custom_core_force.addPerAngleParameter('identifier')

        custom_core_force.addGlobalParameter('solute_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def _add_torsion_force_terms(self):
        core_energy_expression = 'K*(1+cos(periodicity*theta-phase));'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression()

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomTorsionForce(core_energy_expression)
        custom_core_force.addPerTorsionParameter('periodicity') # molecule1 periodicity
        custom_core_force.addPerTorsionParameter('phase') # molecule1 phase
        custom_core_force.addPerTorsionParameter('k') # molecule1 spring constant
        custom_core_force.addPerTorsionParameter('identifier')

        custom_core_force.addGlobalParameter('solute_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def _add_nonbonded_force_terms(self):
        standard_nonbonded_force = openmm.NonbondedForce()
        self._out_system.addForce(standard_nonbonded_force)
        self._out_system_forces[standard_nonbonded_force.__class__.__name__] = standard_nonbonded_force

        #set the appropriate parameters
        epsilon_solvent = self._og_system_forces['NonbondedForce'].getReactionFieldDielectric()
        r_cutoff = self._og_system_forces['NonbondedForce'].getCutoffDistance()
        switch_bool = self._og_system_forces['NonbondedForce'].getUseSwitchingFunction()
        standard_nonbonded_force.setUseSwitchingFunction(switch_bool)
        if switch_bool:
            switching_distance = self._og_system_forces['NonbondedForce'].getSwitchingDistance()
            standard_nonbonded_force.setSwitchingDistance(switching_distance)

        if self._nonbonded_method != openmm.NonbondedForce.NoCutoff:
            standard_nonbonded_force.setReactionFieldDielectric(epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        if self._nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            [alpha_ewald, nx, ny, nz] = self._og_system_forces['NonbondedForce'].getPMEParameters()
            delta = self._og_system_forces['NonbondedForce'].getEwaldErrorTolerance()
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
        standard_nonbonded_force.setNonbondedMethod(self._nonbonded_method)

        if self._og_system_forces['NonbondedForce'].getUseDispersionCorrection() and self._use_dispersion_correction:
            self._out_system_forces['NonbondedForce'].setUseDispersionCorrection(True)
        else:
            self._out_system_forces['NonbondedForce'].setUseDispersionCorrection(False)

        #add the global value
        self._out_system_forces['NonbondedForce'].addGlobalParameter('electrostatic_scale', 0.)
        self._out_system_forces['NonbondedForce'].addGlobalParameter('steric_scale', 0.)

    def get_identifier(self, particles):
        if type(particles) == int:
            out = 0 if particles in self._solute_region else 1
            return out

        if all(x in self._solvent_region for x in particles):
            out = 1
        elif all(x in self._solute_region for x in particles):
            out = 0
        else:
            out = 2
        return out


    def _add_bonds(self):
        """
        add bonds
        """
        og_bond_force = self._og_system_forces['HarmonicBondForce']
        for bond_idx in range(og_bond_force.getNumBonds()):
            p1, p2, length, k = og_bond_force.getBondParameters(bond_idx)
            identifier = self.get_identifier([p1, p2])
            self._out_system_forces['CustomBondForce'].addBond(p1, p2, [length, k, identifier])

    def _add_angles(self):
        og_angle_force = self._og_system_forces['HarmonicAngleForce']
        for angle_idx in range(og_angle_force.getNumAngles()):
            p1, p2, p3, theta0, k = og_angle_force.getAngleParameters(angle_idx)
            identifier = self.get_identifier([p1, p2, p3])
            self._out_system_forces['CustomAngleForce'].addAngle(p1, p2, p3, [theta0, k, identifier])

    def _add_torsions(self):
        og_torsion_force = self._og_system_forces['PeriodicTorsionForce']
        for torsion_idx in range(og_torsion_force.getNumTorsions()):
            p1, p2, p3, p4, per, phase, k = og_torsion_force.getTorsionParameters(torsion_idx)
            identifier = self.get_identifier([p1, p2, p3, p4])
            self._out_system_forces['CustomTorsionForce'].addTorsion(p1, p2, p3, p4, [per, phase, k, identifier])

    def _add_nonbondeds(self):
        og_nb_force = self._og_system_forces['NonbondedForce']

        for particle_idx in range(self._num_particles):
            q, sigma, epsilon = og_nb_force.getParticleParameters(particle_idx)
            identifier = self.get_identifier(particle_idx)

            if identifier == 1: #solvent
                self._out_system_forces['NonbondedForce'].addParticle(q, sigma, epsilon)

            else: #solute
                self._out_system_forces['NonbondedForce'].addParticle(q, sigma, epsilon)
                self._out_system_forces['NonbondedForce'].addParticleParameterOffset('electrostatic_scale', particle_idx, q, 0.0*sigma, epsilon*0.0)
                self._out_system_forces['NonbondedForce'].addParticleParameterOffset('steric_scale', particle_idx, q*0.0, 0.0*sigma, epsilon)


        #handle exceptions
        for exception_idx in range(og_nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = og_nb_force.getExceptionParameters(exception_idx)
            identifier = self.get_identifier([p1, p2])
            exc_idx = self._out_system_forces['NonbondedForce'].addException(p1, p2, chargeProd, sigma, epsilon)
            if identifier == 1: #solvent
                pass
            elif identifier == 0: #solute
                self._out_system_forces['NonbondedForce'].addExceptionParameterOffset('steric_scale', exc_idx, chargeProd, 0.0*sigma, epsilon)

            elif identifier == 2: #inter
                self._out_system_forces['NonbondedForce'].addExceptionParameterOffset('electrostatic_scale', exc_idx, chargeProd, 0.0*sigma, epsilon)

    @property
    def REST_system(self):
        return self._out_system

class RESTTopologyFactoryV2(HybridTopologyFactory):
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
        d. `NonbondedForce`: all interactions (not-scaled)
        e. `CustomNonbondedForce`: scaled rest-rest and rest-nonrest interactions
            scale factor is lambda-dependent and subtracts from the original value. use separate scaling factors for rest-rest vs rest-nonrest
        f. `CustomBondForce`: scale rest-rest and rest-nonrest exceptions
            since we cannot appropriately scale exceptions in the rest region or the rest/nonrest region, we need to treat them as an exception force

    Definitions:
        - rest region: protein (or small molecule) atoms in the rest region
        - nonrest region: atoms outside of the rest region
            - is_solvent : solvent atoms outside the rest region
            - not_solvent : protein (or small molecule) atoms outside the rest region
    """
    _known_forces = {'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce', 'MonteCarloBarostat'}

    def __init__(self, system, topology, rest_region, use_dispersion_correction=False, **kwargs):
        """
        arguments
            system : simtk.openmm.system
                system that will be rewritten
            topology : simtk.openmm.app.topology
                topology
            rest_region : simtk.openmm.system
                rest region (must be a subset of particles)
            use_dispersion_correction : bool, default False
                whether to use a dispersion correction
        Properties
        ----------
            REST_system : simtk.openmm.system
                the REST2-implemented system
        """
        self._use_dispersion_correction = use_dispersion_correction
        self._num_particles = system.getNumParticles()
        self._topology = topology
        self._og_system = system
        self._og_system_forces = {type(force).__name__ : force for force in self._og_system.getForces()}
        self._out_system_forces = {}
        self._rest_region = rest_region
        self._nonrest_region = list(set(range(self._num_particles)).difference(set(self._rest_region)))
        _logger.debug(f"rest region: {self._rest_region}")
        _logger.debug(f"nonrest region: {self._nonrest_region}")

        assert set(rest_region).issubset(set(range(self._num_particles))), f"the rest region is not a subset of the system particles"
        self._nonbonded_method = self._og_system_forces['NonbondedForce'].getNonbondedMethod()
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
        self._out_system.setDefaultPeriodicBoxVectors(*box_vectors)
        _logger.info(f"getDefaultPeriodicBoxVectors added to hybrid: {box_vectors}")

        self._og_system_exceptions = self._generate_dict_from_exceptions(self._og_system_forces['NonbondedForce'])

        # Check that there are no unknown forces in the new and old systems:
        for system_name in ['og']:
            force_names = getattr(self, '_{}_system_forces'.format(system_name)).keys()
            unknown_forces = set(force_names) - set(self._known_forces)
            if len(unknown_forces) > 0:
                raise ValueError(f"Unknown forces {unknown_forces} encountered in {system_name} system")
        _logger.info("No unknown forces.")

        self._handle_constraints()

        self._add_bond_force_terms()
        self._add_bonds()

        self._add_angle_force_terms()
        self._add_angles()

        self._add_torsion_force_terms()
        self._add_torsions()

        self._add_nonbonded_force_terms()
        self._add_nonbondeds()

        self._add_custom_nonbonded_force_terms()
        self._add_custom_nonbondeds()

        self._add_custom_bond_force_terms()
        self._add_custom_bonds()

    def _handle_constraints(self):
        for constraint_idx in range(self._og_system.getNumConstraints()):
            atom1, atom2, length = self._og_system.getConstraintParameters(constraint_idx)
            self._out_system.addConstraint(atom1, atom2, length)

    def scaling_expression(self, nb=False):
        if not nb:
            last_line = '_is_rest = delta(identifier); _is_not_rest = delta(1-identifier); _is_inter = delta(2-identifier);'
        else:
            last_line = '_is_rest = delta(identifier1 + identifier2); _is_not_rest = delta(2 - (identifier1 + identifier2)); _is_inter = delta(1 - (identifier1 + identifier2));'

        out = f"scale_factor = rest_scale*_is_rest + not_rest_scale*_is_not_rest + inter_scale*_is_inter; \
               not_rest_scale = 1.; \
               {last_line} \
               "
        return out

    def _add_bond_force_terms(self):
        core_energy_expression = '(K/2)*(r-length)^2;'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression()

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomBondForce(core_energy_expression)
        custom_core_force.addPerBondParameter('length')
        custom_core_force.addPerBondParameter('k')
        custom_core_force.addPerBondParameter('identifier')

        custom_core_force.addGlobalParameter('rest_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def _add_angle_force_terms(self):
        core_energy_expression = '(K/2)*(theta-theta0)^2;'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression()

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomAngleForce(core_energy_expression)
        custom_core_force.addPerAngleParameter('theta0')
        custom_core_force.addPerAngleParameter('k')
        custom_core_force.addPerAngleParameter('identifier')

        custom_core_force.addGlobalParameter('rest_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def _add_torsion_force_terms(self):
        core_energy_expression = 'K*(1+cos(periodicity*theta-phase));'
        core_energy_expression += 'K = k*scale_factor;' # linearly interpolate spring constant
        core_energy_expression += self.scaling_expression()

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomTorsionForce(core_energy_expression)
        custom_core_force.addPerTorsionParameter('periodicity') # molecule1 periodicity
        custom_core_force.addPerTorsionParameter('phase') # molecule1 phase
        custom_core_force.addPerTorsionParameter('k') # molecule1 spring constant
        custom_core_force.addPerTorsionParameter('identifier')

        custom_core_force.addGlobalParameter('rest_scale', 1.0)
        custom_core_force.addGlobalParameter('inter_scale', 1.0)

        self._out_system.addForce(custom_core_force)
        self._out_system_forces[custom_core_force.__class__.__name__] = custom_core_force

    def _add_nonbonded_force_terms(self):
        standard_nonbonded_force = openmm.NonbondedForce()
        self._out_system.addForce(standard_nonbonded_force)
        self._out_system_forces[standard_nonbonded_force.__class__.__name__] = standard_nonbonded_force

        #set the appropriate parameters
        epsilon_solvent = self._og_system_forces['NonbondedForce'].getReactionFieldDielectric()
        r_cutoff = self._og_system_forces['NonbondedForce'].getCutoffDistance()
        switch_bool = self._og_system_forces['NonbondedForce'].getUseSwitchingFunction()
        standard_nonbonded_force.setUseSwitchingFunction(switch_bool)
        if switch_bool:
            switching_distance = self._og_system_forces['NonbondedForce'].getSwitchingDistance()
            standard_nonbonded_force.setSwitchingDistance(switching_distance)

        if self._nonbonded_method != openmm.NonbondedForce.NoCutoff:
            standard_nonbonded_force.setReactionFieldDielectric(epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        if self._nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            [alpha_ewald, nx, ny, nz] = self._og_system_forces['NonbondedForce'].getPMEParameters()
            delta = self._og_system_forces['NonbondedForce'].getEwaldErrorTolerance()
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
        standard_nonbonded_force.setNonbondedMethod(self._nonbonded_method)

        if self._og_system_forces['NonbondedForce'].getUseDispersionCorrection() and self._use_dispersion_correction:
            self._out_system_forces['NonbondedForce'].setUseDispersionCorrection(True)
        else:
            self._out_system_forces['NonbondedForce'].setUseDispersionCorrection(False)

    def _add_custom_nonbonded_force_terms(self):

        from openmmtools.constants import ONE_4PI_EPS0
        expression = ["U_electrostatics + U_sterics;", 

                             f"U_electrostatics = {ONE_4PI_EPS0} * chargeProd / r;", 
                             "chargeProd = charge1 * charge2 * elec_scale;",
                             "elec_scale = rest_nb_scale * (is_rest1 * is_rest2  + is_rest1 * is_solvent2 + is_rest2 * is_solvent1) + inter_nb_scale * (is_rest1 * not_solvent2 + is_rest2 * not_solvent1);" 

                             "U_sterics = 4 * epsilon * x * (x-1.0);",
                             "x = (sigma / r)^6;"
                             "sigma = (sigma1 + sigma2) / 2;",
                             "epsilon = sqrt(epsilon1 * epsilon2) * steric_scale;",
                             "steric_scale = rest_nb_scale * (is_rest1 * is_rest2  + is_rest1 * is_solvent2 + is_rest2 * is_solvent1) + inter_nb_scale * (is_rest1 * not_solvent2 + is_rest2 * not_solvent1);" 
                             ]
        formatted_expression = ' '.join(expression) 
        custom_nonbonded_force = openmm.CustomNonbondedForce(formatted_expression)
        self._out_system.addForce(custom_nonbonded_force)
        self._out_system_forces[custom_nonbonded_force.__class__.__name__] = custom_nonbonded_force

        # Set the appropriate switching function parameters
        switch_bool = self._og_system_forces['NonbondedForce'].getUseSwitchingFunction()
        custom_nonbonded_force.setUseSwitchingFunction(switch_bool)
        if switch_bool:
            switching_distance = self._og_system_forces['NonbondedForce'].getSwitchingDistance()
            custom_nonbonded_force.setSwitchingDistance(switching_distance)

        # Set the appropriate cutoff distance
        r_cutoff = self._og_system_forces['NonbondedForce'].getCutoffDistance()
        if self._nonbonded_method != openmm.NonbondedForce.NoCutoff:
            custom_nonbonded_force.setCutoffDistance(r_cutoff)

        # Set the appropriate nonbonded method and whether to use PBCs
        if self._nonbonded_method == openmm.NonbondedForce.NoCutoff:
            custom_nonbonded_force.setNonbondedMethod(self._nonbonded_method)        
        else:
            custom_nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)

        # Set whether to use dispersion correction
        #if self._og_system_forces['NonbondedForce'].getUseDispersionCorrection() and self._use_dispersion_correction:
        #    custom_nonbonded_force.setUseLongRangeCorrection(True)
        #else:
        custom_nonbonded_force.setUseLongRangeCorrection(False)

        # Add the global parameters
        custom_nonbonded_force.addGlobalParameter('rest_nb_scale', 0.) # scale factor: beta/beta0 - 1
        custom_nonbonded_force.addGlobalParameter('inter_nb_scale', 0.) # scale factor: sqrt(beta/beta0) - 1

        # Add per particle parameters
        custom_nonbonded_force.addPerParticleParameter('is_rest') # protein atoms in the rest region
        custom_nonbonded_force.addPerParticleParameter('not_solvent') # protein atoms outside the rest region
        custom_nonbonded_force.addPerParticleParameter('is_solvent') # solvent atoms outside rest region
        custom_nonbonded_force.addPerParticleParameter('charge')
        custom_nonbonded_force.addPerParticleParameter('sigma')
        custom_nonbonded_force.addPerParticleParameter('epsilon')

    def _add_custom_bond_force_terms(self):

        from openmmtools.constants import ONE_4PI_EPS0
        expression = ["U_electrostatics + U_sterics;", 

                             f"U_electrostatics = {ONE_4PI_EPS0} * chargeProd * elec_scale / r;", 
                             "elec_scale = (rest_nb_scale * is_rest) + (inter_nb_scale * is_inter);" 

                             "U_sterics = 4 * epsilon * steric_scale * x * (x-1.0);",
                             "x = (sigma / r)^6;"
                             "steric_scale = (rest_nb_scale * is_rest) + (inter_nb_scale * is_inter);"
                             ]
        formatted_expression = ' '.join(expression)
        custom_bond_force = openmm.CustomBondForce(formatted_expression)
        self._out_system.addForce(custom_bond_force)
        self._out_system_forces[custom_bond_force.__class__.__name__ + '_exceptions'] = custom_bond_force

        # Set PBCs
        if self._nonbonded_method == openmm.NonbondedForce.NoCutoff:
            custom_bond_force.setUsesPeriodicBoundaryConditions(False)
        else:
            custom_bond_force.setUsesPeriodicBoundaryConditions(True)

        # Add the global parameters
        custom_bond_force.addGlobalParameter('rest_nb_scale', 0.) # scale factor: beta/beta0 - 1
        custom_bond_force.addGlobalParameter('inter_nb_scale', 0.) # scale factor: sqrt(beta/beta0) - 1

        # Add per bond parameters
        custom_bond_force.addPerBondParameter('is_rest') # atom pairs in the rest region
        custom_bond_force.addPerBondParameter('is_inter') # atom pairs straddling the rest region
        custom_bond_force.addPerBondParameter('chargeProd')
        custom_bond_force.addPerBondParameter('sigma')
        custom_bond_force.addPerBondParameter('epsilon')

    def get_identifier(self, particles):
        if type(particles) == int:
            out = 0 if particles in self._rest_region else 1
            return out

        if all(x in self._nonrest_region for x in particles):
            out = 1
        elif all(x in self._rest_region for x in particles):
            out = 0
        else:
            out = 2
        return out

    def _is_solvent(self, particle_index, positive_ion_name="NA", negative_ion_name="CL", water_name="HOH"):
        if 'openmm' in self._topology.__module__:
            atoms = self._topology.atoms()
        elif 'mdtraj' in self._topology.__module__:
            atoms = self._topology.atoms
        else:
            raise Exception("Topology object must be simtk.openmm.app.topology or mdtraj.core.topology")
        for atom in atoms:
            if atom.index == particle_index:
                if atom.residue.name == positive_ion_name:
                    return True
                elif atom.residue.name == negative_ion_name:
                    return True
                elif atom.residue.name == water_name:
                    return True
                else:
                    return False

    def _add_bonds(self):
        """
        add bonds
        """
        og_bond_force = self._og_system_forces['HarmonicBondForce']
        for bond_idx in range(og_bond_force.getNumBonds()):
            p1, p2, length, k = og_bond_force.getBondParameters(bond_idx)
            identifier = self.get_identifier([p1, p2])
            self._out_system_forces['CustomBondForce'].addBond(p1, p2, [length, k, identifier])

    def _add_angles(self):
        og_angle_force = self._og_system_forces['HarmonicAngleForce']
        for angle_idx in range(og_angle_force.getNumAngles()):
            p1, p2, p3, theta0, k = og_angle_force.getAngleParameters(angle_idx)
            identifier = self.get_identifier([p1, p2, p3])
            self._out_system_forces['CustomAngleForce'].addAngle(p1, p2, p3, [theta0, k, identifier])

    def _add_torsions(self):
        og_torsion_force = self._og_system_forces['PeriodicTorsionForce']
        for torsion_idx in range(og_torsion_force.getNumTorsions()):
            p1, p2, p3, p4, per, phase, k = og_torsion_force.getTorsionParameters(torsion_idx)
            identifier = self.get_identifier([p1, p2, p3, p4])
            self._out_system_forces['CustomTorsionForce'].addTorsion(p1, p2, p3, p4, [per, phase, k, identifier])

    def _add_nonbondeds(self):
        og_nb_force = self._og_system_forces['NonbondedForce']

        for particle_idx in range(self._num_particles):
            q, sigma, epsilon = og_nb_force.getParticleParameters(particle_idx)
            self._out_system_forces['NonbondedForce'].addParticle(q, sigma, epsilon)

        #handle exceptions
        for exception_idx in range(og_nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = og_nb_force.getExceptionParameters(exception_idx)
            exc_idx = self._out_system_forces['NonbondedForce'].addException(p1, p2, chargeProd, sigma, epsilon)
            
    def _add_custom_nonbondeds(self):
        og_nb_force = self._og_system_forces['NonbondedForce']

        for particle_idx in range(self._num_particles):
            q, sigma, epsilon = og_nb_force.getParticleParameters(particle_idx)
            identifier = self.get_identifier(particle_idx)

            if identifier == 1: # nonrest
                if self._is_solvent(particle_idx):
                    params = [0, 0, 1, q, sigma, epsilon]
                else:
                    params = [0, 1, 0, q, sigma, epsilon]

            else: # rest
                params = [1, 0, 0, q, sigma, epsilon]
            
            self._out_system_forces['CustomNonbondedForce'].addParticle(params)
 
        # handle exceptions
        for exception_idx in range(og_nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = og_nb_force.getExceptionParameters(exception_idx)
            self._out_system_forces['CustomNonbondedForce'].addExclusion(p1, p2)

    def _add_custom_bonds(self):
        og_nb_force = self._og_system_forces['NonbondedForce']

        for exception_idx in range(og_nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = og_nb_force.getExceptionParameters(exception_idx)
            identifier = self.get_identifier([p1, p2])
            
            if identifier == 1: # nonrest
                continue
            elif identifier == 0: # rest
                params = [1, 0, chargeProd, sigma, epsilon]

            elif identifier == 2: # inter
                params = [0, 1, chargeProd, sigma, epsilon]

            self._out_system_forces['CustomBondForce_exceptions'].addBond(p1, p2, params)

    @property
    def REST_system(self):
        return self._out_system
