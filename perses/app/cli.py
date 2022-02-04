# New cli for testing
import datetime
from pathlib import Path

import click
import openmm
from openmm import unit
from openmmtools import integrators
from openmmtools.testsystems import HarmonicOscillator
from perses.app.setup_relative_calculation import getSetupOptions

percy = """
            _                           _
            ;`.                       ,'/
            |`.`-.      _____      ,-;,'|
            |  `-.\__,-'     `-.__//'   |
            |     `|               \ ,  |
            `.  ```                 ,  .'
              \_`      .     ,   ,  `_/
                \    ^  `   ,   ^ ` /
                 | '  |  ____  | , |
                 |     ,'    `.    |
                 |    (  O' O  )   |
                 `.    \__,.__/   ,'
                   `-._  `--'  _,'
                       `------'

                   PERSES CLI DEMO

        """


def _check_openeye_license():
    import openeye

    assert openeye.oechem.OEChemIsLicensed(), "OpenEye license checks failed!"


def _test_gpu():
    test_system = HarmonicOscillator()
    integrator = integrators.LangevinIntegrator(
        temperature=298.0 * unit.kelvin,
        collision_rate=1.0 / unit.picoseconds,
        timestep=1.0 * unit.femtoseconds,
    )
    platform = openmm.Platform.getPlatformByName("CUDA")
    platform.setPropertyDefaultValue("Precision", "mixed")
    platform.setPropertyDefaultValue("DeterministicForces", "true")
    openmm.Context(test_system.system, integrator, platform)


def _write_out_files(path, options):

    # Convert path to a pathlib object
    yaml_path = Path(path)

    # Generate parsed yaml name
    yaml_name = yaml_path.name
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    yaml_parse_name = f"parsed-{time}-{yaml_name}"

    # First make files in same dir as yaml
    files_next_to_yaml = [
        "debug.png",
        "system.xml",
        yaml_parse_name,
    ]

    for _file in files_next_to_yaml:
        with open(_file, "w") as fp:
            pass

    # Now we make the directory structure
    trajectory_directory = Path(options["trajectory_directory"])
    dirs_to_make = trajectory_directory.joinpath("xml")
    Path(dirs_to_make).mkdir(parents=True, exist_ok=True)

    # Now files that belong in the lower directories
    files_in_lower_dir = [
        "atom_mapping.png",
        "out-complex_checkpoint.nc",
        "out-complex.nc",
        "out-complex.pdb",
        "outhybrid_factory.npy.npz",
        "out-solvent_checkpoint.nc",
        "out-solvent.nc",
        "out-solvent.pdb",
        "out_topology_proposals.pkl",
        "out-vacuum_checkpoint.nc",
        "out-vacuum.nc",
    ]

    # add the dir prefix
    files_in_lower_dir = [
        Path(trajectory_directory).joinpath(_) for _ in files_in_lower_dir
    ]

    for _file in files_in_lower_dir:
        with open(_file, "w") as fp:
            pass

    # Now the files in the 'xml' dir
    files_in_xml_dir = [
        "complex-hybrid-system.gz",
        "complex-new-system.gz",
        "complex-old-system.gz",
        "solvent-hybrid-system.gz",
        "solvent-new-system.gz",
        "solvent-old-system.gz",
        "vacuum-hybrid-system.gz",
        "vacuum-new-system.gz",
        "vacuum-old-system.gz",
    ]

    # add the dir prefix
    files_in_xml_dir = [dirs_to_make.joinpath(_) for _ in files_in_xml_dir]

    for _file in files_in_lower_dir:
        with open(_file, "w") as fp:
            pass


@click.command()
@click.option("--yaml-path", type=click.Path(exists=True, dir_okay=False))
def cli(yaml_path):
    """test"""
    click.echo(click.style(percy, fg="bright_magenta"))
    click.echo("📖\t Fetching simulation options ")
    options = getSetupOptions(yaml_path)
    click.echo("🖨️\t Printing options")
    click.echo(options)
    click.echo("🕵️\t Checking OpenEye license")
    _check_openeye_license()
    click.echo("✅\t OpenEye license good")
    click.echo("🖥️⚡\t Making a test system to check if we can get a GPU")
    _test_gpu()
    click.echo("🎉\t GPU test sucsessful!")
    click.echo("🖨️\t Writing out files")
    trajectory_directory = options["trajectory_directory"]
    _write_out_files(trajectory_directory, options)
    click.echo("🧪\t Simulation over")
