import numpy as np
import os
from openmmtools import integrators
from simtk import openmm
from simtk.openmm import app


def run_equilibrium(system, topology, configuration, n_steps, report_interval, equilibration_steps, filename):
    from mdtraj.reporters import HDF5Reporter
    integrator = integrators.LangevinIntegrator()
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(configuration)
    openmm.LocalEnergyMinimizer.minimize(simulation.context)

    #equilibrate:
    integrator.step(equilibration_steps)

    print("equilibration complete")

    reporter = HDF5Reporter(filename, report_interval)
    simulation.reporters.append(reporter)
    simulation.step(n_steps)


if __name__ == "__main__":
    import yaml
    import sys
    import os

    yaml_filename = sys.argv[1]
    index = int(sys.argv[2]) - 1 # Indices start at one

    with open(yaml_filename, "r") as yamlfile:
        options = yaml.load(yamlfile)

    setup_options = options['setup']
    equilibrium_options = options['equilibrium']

    project_prefix = setup_options['project_prefix']
    setup_directory = setup_options['output_directory']

    file_to_read = os.path.join(setup_directory, "{}_{}_initial.npy".format(project_prefix, index))

    positions, topology, system, _ = np.load(file_to_read)

    omm_topology = topology.to_openmm()
    omm_topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    equilibration_steps = equilibrium_options['n_equilibration_steps'] * 1000 # value was in ps
    equilibrium_steps = equilibrium_options['n_equilibrium_steps'] * 1000 # also in ps
    output_directory = equilibrium_options['output_directory']
    report_interval = equilibrium_options['report_interval'] * 1000 # also in ps

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_file = os.path.join(output_directory, "{}_{}.h5".format(project_prefix, index))

    run_equilibrium(system, omm_topology, positions, equilibrium_steps, report_interval, equilibration_steps, output_file)
