"""
From a standard simtk.openmm.system, make a REST2-amenable system
"""

"""
Imports
"""
import simtk.openmm as openmm
import copy
from perses.annihilation.relative import HybridTopologyFactory
import simtk.unit as unit
import numpy as np

# Constants copied from: https://github.com/openmm/openmm/blob/master/platforms/reference/include/SimTKOpenMMRealType.h#L89
M_PI = 3.14159265358979323846
E_CHARGE = (1.602176634e-19)
AVOGADRO = (6.02214076e23)
EPSILON0 = (1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO))
ONE_4PI_EPS0 = (1/(4*M_PI*EPSILON0))

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

class RESTTopologyFactoryV3(HybridTopologyFactory):
    """
    This class takes a standard simtk.openmm.system equipped with
        a. `HarmonicBondForce`
        b. `HarmonicAngleForce`
        c. `PeriodicTorsionForce`
        d. `NonbondedForce`
    and will convert to another system that allows for rest scaling and is equipped with
        a. `CustomBondForce`: rewrite `HarmonicBondForce`
        b. `CustomAngleForce`: rewrite `HarmonicAngleForce`
        c. `CustomTorsionForce`: rewrite `PeriodicTorsionForce`
        d. `CustomNonbondedForce`: handles direct space PME electrostatics interactions (with long range correction off)
        e. `CustomNonbondedForce` handles direct space PME steric interactions (with long range correction on)
        f. `CustomBondForce`: handles direct space PME electrostatics and steric exceptions
        g. `NonbondedForce`: handles reciprocal space PME electrostatics
    """

    _known_forces = {'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce', 'MonteCarloBarostat'}

    _default_electrostatics_expression_list = [

        "U_electrostatics;",

        # Define electrostatics functional form
        f"U_electrostatics = {ONE_4PI_EPS0} * chargeProd  * erfc(alpha * r)/ r;",

        # Define chargeProd (with REST scaling)
        "chargeProd = (charge1 * p1_electrostatics_rest_scale) * (charge2 * p2_electrostatics_rest_scale);",

        # Define rest scale factors (normal rest)
        # "p1_electrostatics_rest_scale = is_rest1 * lambda_rest_electrostatics + is_nonrest_solute1 + is_nonrest_solvent1;",
        # "p2_electrostatics_rest_scale = is_rest2 * lambda_rest_electrostatics + is_nonrest_solute2 + is_nonrest_solvent2;",
        # "p1_sterics_rest_scale = is_rest1 * lambda_rest_sterics + is_nonrest_solute1 + is_nonrest_solvent1;",
        # "p2_sterics_rest_scale = is_rest2 * lambda_rest_sterics + is_nonrest_solute2 + is_nonrest_solvent2;",

        # # Define rest scale factors (scaled water rest)
        # "p1_electrostatics_rest_scale = select(1 - is_both_solvent, is_rest1 * lambda_electrostatics_rest + is_nonrest_solute1 + is_nonrest_solvent1 * lambda_electrostatics_rest, 1);",
        # "p2_electrostatics_rest_scale = select(1 - is_both_solvent, is_rest2 * lambda_electrostatics_rest + is_nonrest_solute2 + is_nonrest_solvent2 * lambda_electrostatics_rest, 1);",
        # "is_both_solvent = is_nonrest_solvent1 * is_nonrest_solvent2;",

        # Define rest scale factors (scaled water rest)
        "p1_electrostatics_rest_scale = lambda_rest_electrostatics * (is_rest1 + is_nonrest_solvent1 * is_rest2) + 1 * (is_nonrest_solute1 + is_nonrest_solvent1 * is_nonrest_solute2 + is_nonrest_solvent1 * is_nonrest_solvent2);",
        "p2_electrostatics_rest_scale = lambda_rest_electrostatics * (is_rest2 + is_nonrest_solvent2 * is_rest1) + 1 * (is_nonrest_solute2 + is_nonrest_solvent2 * is_nonrest_solute1 + is_nonrest_solvent2 * is_nonrest_solvent1);",

        # Define alpha
        "alpha = {alpha_ewald};"

    ]

    _default_sterics_expression_list = [

        "U_sterics;",

        # Define sterics functional form
        "U_sterics = 4 * epsilon * x * (x - 1.0);"
        "x = (sigma / r)^6;"

        # Define sigma
        "sigma = (sigma1 + sigma2) / 2;",

        # Define epsilon (with rest scaling)
        "epsilon = p1_sterics_rest_scale * p2_sterics_rest_scale * sqrt(epsilon1 * epsilon2);",

        # # Define rest scale factors (normal rest)
        # "p1_sterics_rest_scale = is_rest1 * lambda_rest_sterics + is_nonrest_solute1 + is_nonrest_solvent1;",
        # "p2_sterics_rest_scale = is_rest2 * lambda_rest_sterics + is_nonrest_solute2 + is_nonrest_solvent2;",

        # Define rest scale factors (scaled water rest)
        # "p1_sterics_rest_scale = select(1 - is_both_solvent, is_rest1 * lambda_sterics_rest + is_nonrest_solute1 + is_nonrest_solvent1 * lambda_sterics_rest, 1);",
        # "p2_sterics_rest_scale = select(1 - is_both_solvent, is_rest2 * lambda_sterics_rest_ + is_nonrest_solute2 + is_nonrest_solvent2 * lambda_sterics_rest, 1);",
        # "is_both_solvent = is_nonrest_solvent1 * is_nonrest_solvent2;",

        # Define rest scale factors (scaled water rest)
        "p1_sterics_rest_scale = lambda_rest_sterics * (is_rest1 + is_nonrest_solvent1 * is_rest2) + 1 * (is_nonrest_solute1 + is_nonrest_solvent1 * is_nonrest_solute2 + is_nonrest_solvent1 * is_nonrest_solvent2);",
        "p2_sterics_rest_scale = lambda_rest_sterics * (is_rest2 + is_nonrest_solvent2 * is_rest1) + 1 * (is_nonrest_solute2 + is_nonrest_solvent2 * is_nonrest_solute1 + is_nonrest_solvent2 * is_nonrest_solvent1);",

    ]

    _default_exceptions_expression_list = [

        "U_electrostatics * electrostatics_rest_scale + U_sterics * sterics_rest_scale;",

        # Define rest scale
        "electrostatics_rest_scale = is_rest * lambda_rest_electrostatics_exceptions * lambda_rest_electrostatics_exceptions + is_inter * lambda_rest_electrostatics_exceptions + is_nonrest;",
        "sterics_rest_scale = is_rest * lambda_rest_sterics_exceptions * lambda_rest_sterics_exceptions + is_inter * lambda_rest_sterics_exceptions + is_nonrest;",

        # Define electrostatics functional form
        # Note that we need to subtract off the reciprocal space for exceptions
        # See explanation for why here: https://github.com/openmm/openmm/issues/3269#issuecomment-934686324
        f"U_electrostatics = U_electrostatics_direct - U_electrostatics_reciprocal;",
        f"U_electrostatics_direct = {ONE_4PI_EPS0} * chargeProd_exceptions / r;",
        f"U_electrostatics_reciprocal = {ONE_4PI_EPS0} * chargeProd_product * erf(alpha * r) / r;",

        # Define sterics functional form
        "U_sterics = 4 * epsilon * x * (x - 1.0);",
        "x = (sigma / r)^6;",

        # Define alpha
        "alpha = {alpha_ewald};"

    ]

    _default_electrostatics_expression = ' '.join(_default_electrostatics_expression_list)
    _default_sterics_expression = ' '.join(_default_sterics_expression_list)
    _default_exceptions_expression = ' '.join(_default_exceptions_expression_list)

    def __init__(self,
                 system,
                 topology,
                 rest_region,
                 use_dispersion_correction=False, # TODO: remove this?
                 **kwargs):
        """
        arguments
            system : simtk.openmm.system
                system that will be rewritten
            topology : simtk.openmm.app.topology
                topology of the system that will be rewritten
            rest_region : list of ints
                list of atom indices that will be define the rest region
            use_dispersion_correction : bool, default False
                whether to use a dispersion correction
            r_cutoff : value in units of distance
                cutoff distance (in nm) beyond which electrostatics are not calculated
            delta : float
                error tolerance for alpha in electrostatics

        Properties
        ----------
            REST_system : simtk.openmm.system
                the REST2-implemented system
        """

        _logger.info("*** Generating RESTTopologyFactoryV3 ***")

        # Set variables related to original system
        self._topology = topology
        self._num_particles = system.getNumParticles()
        self._og_system = system
        self._og_system_forces = {type(force).__name__ : force for force in self._og_system.getForces()}

        # Validate rest region
        self._validate_rest_region(rest_region)
        self._rest_region = rest_region
        self._nonrest_region = list(set(range(self._num_particles)).difference(set(self._rest_region)))
        _logger.debug(f"rest region: {self._rest_region}")
        _logger.debug(f"nonrest region: {self._nonrest_region}")

        # Check that there are no unknown forces in the original system
        for system_name in ['og']:
            force_names = getattr(self, '_{}_system_forces'.format(system_name)).keys()
            unknown_forces = set(force_names) - set(self._known_forces)
            if len(unknown_forces) > 0:
                raise ValueError(f"Unknown forces {unknown_forces} encountered in {system_name} system")
        _logger.info("No unknown forces.")

        # Set nonbonded parameters
        self._nonbonded_method = self._og_system_forces['NonbondedForce'].getNonbondedMethod()
        if self._nonbonded_method == openmm.NonbondedForce.NoCutoff:
            self._alpha_ewald = 0
        else:
            [alpha_ewald, nx, ny, nz] = self._og_system_forces['NonbondedForce'].getPMEParameters()
            if (alpha_ewald / alpha_ewald.unit) == 0.0:
                # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
                tol = self._og_system_forces['NonbondedForce'].getEwaldErrorTolerance()
                alpha_ewald = (1.0 / self._og_system_forces['NonbondedForce'].getCutoffDistance()) * np.sqrt(-np.log(2.0 * tol))
            self._alpha_ewald = alpha_ewald.value_in_unit_system(unit.md_unit_system)
        _logger.info(f"alpha_ewald is {self._alpha_ewald}")
        self._use_dispersion_correction = use_dispersion_correction

        # Create REST system and add particles to it
        self._out_system = openmm.System()
        self._out_system_forces = {}
        for particle_idx in range(self._num_particles):
            particle_mass = self._og_system.getParticleMass(particle_idx)
            hybrid_idx = self._out_system.addParticle(particle_mass)

        # Add barostat
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

        # Prep look up dict for determining if atom is solvent
        if 'openmm' in self._topology.__module__:
            atoms = self._topology.atoms()
        elif 'mdtraj' in self._topology.__module__:
            atoms = self._topology.atoms
        else:
            raise Exception("Topology object must be simtk.openmm.app.topology or mdtraj.core.topology")
        self._atom_idx_to_object = {atom.index: atom for atom in atoms}

        # Copy constraints, checking to make sure they are not changing
        self._handle_constraints()

        # Create custom bond force and add bonds to it
        self._create_bond_force()
        self._copy_bonds()

        # Create custom angle force and add angles to it
        self._create_angle_force()
        self._copy_angles()

        # Create custom torsion force and add torsions to it
        self._create_torsion_force()
        self._copy_torsions()

        # Create custom nonbonded and custom bond forces and add particles/bonds to them
        # self._add_nonbonded_force_terms()
        # self._add_nonbondeds()
        self._create_nonbonded_electrostatics_force()
        self._create_nonbonded_sterics_force()
        self._copy_nonbondeds()

        self._create_nonbonded_exceptions_force()
        self._copy_nonbondeds_exceptions()

        self._create_nonbonded_reciprocal_space_force()
        self._copy_nonbondeds_reciprocal_space()

    def _validate_rest_region(self, rest_region):
        """
        Check that rest_region was defined correctly

        Parameters
        ----------
        rest_region : lists of ints
            contains the indices of atoms that should be scaled by rest
        """

        # Check that rest_region is a list
        assert type(rest_region) == list

        # Check that the list contains ints
        assert all(type(element) == int for element in rest_region)

        # Check that the rest region is a subset of the system particles
        assert set(rest_region).issubset(
            set(range(self._num_particles))), f"the rest region is not a subset of the system particles"

        # Check that there are no duplicate particles in the rest region
        assert len(rest_region) == len(set(rest_region)), "There are duplicate atom indices in the rest_region"


    def _handle_constraints(self):
        for constraint_idx in range(self._og_system.getNumConstraints()):
            atom1, atom2, length = self._og_system.getConstraintParameters(constraint_idx)
            self._out_system.addConstraint(atom1, atom2, length)

    def get_rest_identifier(self, particles):
        """
        For a given particle or set of particles, get the rest_id which is a list of binary ints that defines which
        region the particle(s) belong to.
        If there is a single particle, the regions are: is_rest, is_nonrest_solute, is_nonrest_solvent
        If there is a set of particles, the regions are: is_rest, is_inter, is_nonrest
        Example: if there is a single particle that is in the nonrest_solute region, the rest_id is [0, 1, 0]
        Arguments
        ---------
        particles : set or int
            a set of hybrid particle indices or single particle
        Returns
        -------
        rest_id : list
            list of binaries indicating which region the particle(s) belong to
        """

        def _is_solvent(particle_index, positive_ion_name="NA", negative_ion_name="CL", water_name="HOH"):
            atom = self._atom_idx_to_object[particle_index]
            if atom.residue.name == positive_ion_name:
                return True
            elif atom.residue.name == negative_ion_name:
                return True
            elif atom.residue.name == water_name:
                return True
            else:
                return False

        assert type(particles) in [type(set()), int], f"`particles` must be an integer or a set, got {type(particles)}."

        if isinstance(particles, int):
            rest_id = [0, 1, 0]  # Set the default scale_id to nonrest solute
            if not self._rest_region:
                return rest_id  # If there are no rest regions, set everything as nonrest_solute bc these atoms are not scaled
            else:
                if particles in self._rest_region:  # Here, particles is a single int
                    rest_id = [1, 0, 0]
                elif _is_solvent(particles):  # If the particle is not in a rest region, check if it is a solvent atom
                    rest_id = [0, 0, 1]
                return rest_id


        elif isinstance(particles, set):
            rest_id = [0, 0, 1]  # Set the default scale_id to nonrest solute
            if not self._rest_region:
                return rest_id  # If there are no scale regions, set everything as nonrest bc these atoms are not scaled
            else:
                if particles.intersection(
                        self._rest_region) != set():  # At least one of the particles is in the idx_th rest region
                    if particles.issubset(self._rest_region):  # Then this term is wholly in the rest region
                        rest_id = [1, 0, 0]
                    else:  # It is inter region
                        rest_id = [0, 1, 0]
                return rest_id

        else:
            raise Exception(f"particles is of type {type(particles)}, but only `int` and `set` are allowable")

    def _create_bond_force(self):

        # Define the custom expression
        bond_expression = "rest_scale * (K / 2) * (r - length)^2;"
        bond_expression += "rest_scale = is_rest * lambda_rest_bonds * lambda_rest_bonds " \
                           "+ is_inter * lambda_rest_bonds " \
                           "+ is_nonrest;"

        # Create custom force
        custom_bond_force = openmm.CustomBondForce(bond_expression)
        self._out_system.addForce(custom_bond_force)
        self._out_system_forces[custom_bond_force.__class__.__name__] = custom_bond_force

        # Add global parameters
        custom_bond_force.addGlobalParameter("lambda_rest_bonds", 1.0)

        # Add per-bond parameters for rest scaling -- these sets are disjoint
        custom_bond_force.addPerBondParameter("is_rest")
        custom_bond_force.addPerBondParameter("is_inter")
        custom_bond_force.addPerBondParameter("is_nonrest")

        # Add per-bond parameters for defining energy
        custom_bond_force.addPerBondParameter('length')
        custom_bond_force.addPerBondParameter('K')

    def _create_angle_force(self):

        # Define the custom expression
        angle_expression = "rest_scale * (K / 2) * (theta - theta0)^2;"
        angle_expression += "rest_scale = is_rest * lambda_rest_angles * lambda_rest_angles " \
                            "+ is_inter * lambda_rest_angles " \
                            "+ is_nonrest;"

        # Create custom force
        custom_angle_force = openmm.CustomAngleForce(angle_expression)
        self._out_system.addForce(custom_angle_force)
        self._out_system_forces[custom_angle_force.__class__.__name__] = custom_angle_force

        # Add global parameters
        custom_angle_force.addGlobalParameter("lambda_rest_angles", 1.0)

        # Add per-angle parameters for rest scaling -- these sets are disjoint
        custom_angle_force.addPerAngleParameter("is_rest")
        custom_angle_force.addPerAngleParameter("is_inter")
        custom_angle_force.addPerAngleParameter("is_nonrest")

        # Add per-angle parameters for defining energy
        custom_angle_force.addPerAngleParameter('theta0')
        custom_angle_force.addPerAngleParameter('K')

    def _create_torsion_force(self):

        # Define the custom expression
        torsion_expression = "rest_scale * (K * (1 + cos(periodicity * theta - phase)));"
        torsion_expression += "rest_scale = is_rest * lambda_rest_torsions * lambda_rest_torsions " \
                              "+ is_inter * lambda_rest_torsions " \
                              "+ is_nonrest;"

        # Create custom force
        custom_torsion_force = openmm.CustomTorsionForce(torsion_expression)
        self._out_system.addForce(custom_torsion_force)
        self._out_system_forces[custom_torsion_force.__class__.__name__] = custom_torsion_force

        # Add global parameters
        custom_torsion_force.addGlobalParameter("lambda_rest_torsions", 1.0)

        # Add per-torsion parameters for rest scaling -- these sets are disjoint
        custom_torsion_force.addPerTorsionParameter("is_rest")
        custom_torsion_force.addPerTorsionParameter("is_inter")
        custom_torsion_force.addPerTorsionParameter("is_nonrest")

        # Add per-torsion parameters for defining energy
        custom_torsion_force.addPerTorsionParameter('periodicity')
        custom_torsion_force.addPerTorsionParameter('phase')
        custom_torsion_force.addPerTorsionParameter('K')

    def _create_nonbonded_electrostatics_force(self):

        # Define the custom expression
        expression = self._default_electrostatics_expression
        formatted_expression = expression.format(alpha_ewald=self._alpha_ewald)

        # Create the custom force
        custom_nb_force = openmm.CustomNonbondedForce(formatted_expression)
        self._out_system.addForce(custom_nb_force)
        self._out_system_forces[custom_nb_force.__class__.__name__ + '_electrostatics'] = custom_nb_force

        # Add global parameters
        custom_nb_force.addGlobalParameter(f"lambda_rest_electrostatics", 1.0)

        # Add per-particle parameters for rest scaling -- these three sets are disjoint
        custom_nb_force.addPerParticleParameter("is_rest")
        custom_nb_force.addPerParticleParameter("is_nonrest_solute")
        custom_nb_force.addPerParticleParameter("is_nonrest_solvent")

        # Add per-particle parameters for defining energy
        custom_nb_force.addPerParticleParameter('charge')

        # Handle some nonbonded attributes
        old_system_nbf = self._og_system_forces['NonbondedForce']
        standard_nonbonded_method = old_system_nbf.getNonbondedMethod()
        if standard_nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.PME,
                                         openmm.NonbondedForce.Ewald]:
            custom_nb_force.setNonbondedMethod(self._translate_nonbonded_method_to_custom(standard_nonbonded_method))
            custom_nb_force.setUseSwitchingFunction(False)
            custom_nb_force.setCutoffDistance(old_system_nbf.getCutoffDistance())
            custom_nb_force.setUseLongRangeCorrection(False) # This should be off for electrostatics, but on for sterics

        elif standard_nonbonded_method == openmm.NonbondedForce.NoCutoff:
            custom_nb_force.setNonbondedMethod(self._translate_nonbonded_method_to_custom(standard_nonbonded_method))

        else:
            raise Exception(f"nonbonded method is not recognized")

    def _create_nonbonded_sterics_force(self):

        # Define the custom expression
        expression = self._default_sterics_expression
        formatted_expression = expression.format()

        # Create the custom force
        custom_nb_force = openmm.CustomNonbondedForce(formatted_expression)
        self._out_system.addForce(custom_nb_force)
        self._out_system_forces[custom_nb_force.__class__.__name__ + '_sterics'] = custom_nb_force

        # Add global parameters
        custom_nb_force.addGlobalParameter(f"lambda_rest_sterics", 1.0)

        # Add per-particle parameters for rest scaling -- these three sets are disjoint
        custom_nb_force.addPerParticleParameter("is_rest")
        custom_nb_force.addPerParticleParameter("is_nonrest_solute")
        custom_nb_force.addPerParticleParameter("is_nonrest_solvent")

        # Add per-particle parameters for defining energy
        custom_nb_force.addPerParticleParameter('sigma')
        custom_nb_force.addPerParticleParameter('epsilon')

        # Handle some nonbonded attributes
        old_system_nbf = self._og_system_forces['NonbondedForce']
        standard_nonbonded_method = old_system_nbf.getNonbondedMethod()
        if standard_nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.PME,
                                         openmm.NonbondedForce.Ewald]:
            custom_nb_force.setNonbondedMethod(self._translate_nonbonded_method_to_custom(standard_nonbonded_method))
            if old_system_nbf.getUseSwitchingFunction(): # This should be copied for sterics force, but not for electrostatics force
                custom_nb_force.setUseSwitchingFunction(True)
                custom_nb_force.setSwitchingDistance(old_system_nbf.getSwitchingDistance())
            custom_nb_force.setCutoffDistance(old_system_nbf.getCutoffDistance())
            custom_nb_force.setUseLongRangeCorrection(True) # This should be on for sterics, but off for electrostatics

        elif standard_nonbonded_method == openmm.NonbondedForce.NoCutoff:
            custom_nb_force.setNonbondedMethod(self._translate_nonbonded_method_to_custom(standard_nonbonded_method))

        else:
            raise Exception(f"nonbonded method is not recognized")

    def _create_nonbonded_exceptions_force(self):

        # Define the custom expression
        expression = self._default_exceptions_expression
        formatted_expression = expression.format(alpha_ewald=self._alpha_ewald)

        # Create the custom force
        custom_bond_force = openmm.CustomBondForce(formatted_expression)
        self._out_system.addForce(custom_bond_force)
        self._out_system_forces[custom_bond_force.__class__.__name__] = custom_bond_force

        # Add global parameters
        custom_bond_force.addGlobalParameter(f"lambda_rest_electrostatics_exceptions", 1.0)
        custom_bond_force.addGlobalParameter(f"lambda_rest_sterics_exceptions", 1.0)

        # Add per-bond parameters for rest scaling -- these sets are disjoint
        custom_bond_force.addPerBondParameter("is_rest")
        custom_bond_force.addPerBondParameter("is_inter")
        custom_bond_force.addPerBondParameter("is_nonrest")

        # Add per-bond parameters for defining energy
        custom_bond_force.addPerBondParameter('chargeProd_exceptions')
        custom_bond_force.addPerBondParameter('sigma')
        custom_bond_force.addPerBondParameter('epsilon')
        custom_bond_force.addPerBondParameter('chargeProd_product')

        # Handle some nonbonded attributes
        old_system_nbf = self._og_system_forces['NonbondedForce']
        standard_nonbonded_method = old_system_nbf.getNonbondedMethod()
        if standard_nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.PME,
                                         openmm.NonbondedForce.Ewald]:
            custom_bond_force.setUsesPeriodicBoundaryConditions(True)

        elif standard_nonbonded_method == openmm.NonbondedForce.NoCutoff:
            custom_bond_force.setUsesPeriodicBoundaryConditions(False)

        else:
            raise Exception(f"nonbonded method is not recognized")

    def _create_nonbonded_reciprocal_space_force(self):

        # Create force
        standard_nonbonded_force = openmm.NonbondedForce()
        self._out_system.addForce(standard_nonbonded_force)
        self._out_system_forces[standard_nonbonded_force.__class__.__name__] = standard_nonbonded_force

        # TODO: For now, assume that this force will not be scaled

        # Set nonbonded method and related attributes
        old_system_nbf = self._og_system_forces['NonbondedForce']
        standard_nonbonded_method = old_system_nbf.getNonbondedMethod()
        standard_nonbonded_force.setNonbondedMethod(standard_nonbonded_method)
        if standard_nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
            epsilon_solvent = old_system_nbf.getReactionFieldDielectric()
            r_cutoff = old_system_nbf.getCutoffDistance()
            standard_nonbonded_force.setReactionFieldDielectric(epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        elif standard_nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            [alpha_ewald, nx, ny, nz] = old_system_nbf.getPMEParameters()
            delta = old_system_nbf.getEwaldErrorTolerance()
            r_cutoff = old_system_nbf.getCutoffDistance()
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        elif standard_nonbonded_method in [openmm.NonbondedForce.NoCutoff]:
            pass
        else:
            raise Exception("Nonbonded method %s not supported yet." % str(self._nonbonded_method))

        # Set the use of dispersion correction
        if old_system_nbf.getUseDispersionCorrection():
            standard_nonbonded_force.setUseDispersionCorrection(True)
        else:
            standard_nonbonded_force.setUseDispersionCorrection(False)

        # Set the use of switching function
        if old_system_nbf.getUseSwitchingFunction():
            switching_distance = old_system_nbf.getSwitchingDistance()
            standard_nonbonded_force.setUseSwitchingFunction(True)
            standard_nonbonded_force.setSwitchingDistance(switching_distance)
        else:
            standard_nonbonded_force.setUseSwitchingFunction(False)

        # Disable direct space interactions
        standard_nonbonded_force.setIncludeDirectSpace(False)

    def _copy_bonds(self):
        """
        add bonds
        """
        og_bond_force = self._og_system_forces['HarmonicBondForce']
        custom_bond_force = self._out_system_forces['CustomBondForce']

        # Set periodicity
        if og_bond_force.usesPeriodicBoundaryConditions():
            custom_bond_force.setUsesPeriodicBoundaryConditions(True)

        # Iterate over bonds in original bond force and copy to the custom bond force
        for bond_idx in range(og_bond_force.getNumBonds()):
            # Get particle indices and bond terms
            p1, p2, length, k = og_bond_force.getBondParameters(bond_idx)

            # Given the atom indices, get rest identifier
            rest_id = self.get_rest_identifier(set([p1, p2]))

            # Add bond
            custom_bond_force.addBond(p1, p2, rest_id + [length, k])

    def _copy_angles(self):
        og_angle_force = self._og_system_forces['HarmonicAngleForce']
        custom_angle_force = self._out_system_forces['CustomAngleForce']

        # Set periodicity
        if og_angle_force.usesPeriodicBoundaryConditions():
            custom_angle_force.setUsesPeriodicBoundaryConditions(True)

        # Iterate over angles in original angle force and copy to the custom angle force
        for angle_idx in range(og_angle_force.getNumAngles()):
            # Get particle indices and angle terms
            p1, p2, p3, theta0, k = og_angle_force.getAngleParameters(angle_idx)

            # Given the atom indices, get rest identifier
            rest_id = self.get_rest_identifier(set([p1, p2, p3]))

            # Add angle
            custom_angle_force.addAngle(p1, p2, p3, rest_id + [theta0, k])

    def _copy_torsions(self):
        og_torsion_force = self._og_system_forces['PeriodicTorsionForce']
        custom_torsion_force = self._out_system_forces['CustomTorsionForce']

        # Set periodicity
        if og_torsion_force.usesPeriodicBoundaryConditions():
            custom_torsion_force.setUsesPeriodicBoundaryConditions(True)

        # Iterate over torsions in original torsion force and copy to the custom torsion force
        for torsion_idx in range(og_torsion_force.getNumTorsions()):
            # Get particle indices and torsion terms
            p1, p2, p3, p4, periodicity, phase, k = og_torsion_force.getTorsionParameters(torsion_idx)

            # Given the atom indices, get rest identifier
            rest_id = self.get_rest_identifier(set([p1, p2, p3, p4]))

            # Add torsion
            custom_torsion_force.addTorsion(p1, p2, p3, p4, rest_id + [periodicity, phase, k])

    def _copy_nonbondeds(self):
        og_nb_force = self._og_system_forces['NonbondedForce']
        custom_electrostatics_force = self._out_system_forces['CustomNonbondedForce_electrostatics']
        custom_sterics_force = self._out_system_forces['CustomNonbondedForce_sterics']

        # Iterate over particles in original nonbonded force and copy to the custom forces
        for particle_idx in range(self._num_particles):

            # Get particle terms
            q, sigma, epsilon = og_nb_force.getParticleParameters(particle_idx)

            # Given atom index, get rest identifier
            rest_id = self.get_rest_identifier(particle_idx)

            # Add particle to electrostatics force
            custom_electrostatics_force.addParticle(rest_id + [q])

            # Add particle to sterics force
            custom_sterics_force.addParticle(rest_id + [sigma, epsilon])

    def _copy_nonbondeds_exceptions(self):
        og_nb_force = self._og_system_forces['NonbondedForce']
        custom_exceptions_force = self._out_system_forces['CustomBondForce']
        custom_electrostatics_force = self._out_system_forces['CustomNonbondedForce_electrostatics']
        custom_sterics_force = self._out_system_forces['CustomNonbondedForce_sterics']

        for exception_idx in range(og_nb_force.getNumExceptions()):

            # Get particle indices and exception terms
            p1, p2, chargeProd, sigma, epsilon = og_nb_force.getExceptionParameters(exception_idx)

            # Given atom indices, get rest identifier
            rest_id = self.get_rest_identifier(set([p1, p2]))

            # Compute chargeProd_product from original particle parameters
            p1_params = custom_electrostatics_force.getParticleParameters(p1)
            p2_params = custom_electrostatics_force.getParticleParameters(p2)
            chargeProd_product = p1_params[-1] * p2_params[-1]

            # Add exception
            custom_exceptions_force.addBond(p1, p2, rest_id + [chargeProd, sigma, epsilon, chargeProd_product])

            # Add exclusions to custom nb forces
            custom_electrostatics_force.addExclusion(p1, p2)
            custom_sterics_force.addExclusion(p1, p2)

    def _copy_nonbondeds_reciprocal_space(self):
        og_nb_force = self._og_system_forces['NonbondedForce']
        nb_force = self._out_system_forces['NonbondedForce']

        # Iterate over particles in original nonbonded force and copy to the new nonbonded force
        for particle_idx in range(self._num_particles):
            # Get particle terms
            q, sigma, epsilon = og_nb_force.getParticleParameters(particle_idx)

            # Add particle
            nb_force.addParticle(q, sigma, epsilon)

        for exception_idx in range(og_nb_force.getNumExceptions()):

            # Get particle indices and exception terms
            p1, p2, chargeProd, sigma, epsilon = og_nb_force.getExceptionParameters(exception_idx)

            # Add exception
            nb_force.addException(p1, p2, chargeProd, sigma, epsilon)

    @property
    def REST_system(self):
        return self._out_system
