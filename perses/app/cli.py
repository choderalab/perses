# New cli for testing

import click
from perses.app.setup_relative_calculation import getSetupOptions
from openmmtools.testsystems import HarmonicOscillator
from openmmtools import integrators
from openmm import unit
import openmm

def _check_openeye_license():
    import openeye
    assert openeye.oechem.OEChemIsLicensed(), "OpenEye license checks failed!"


def _test_gpu():
    test_system = HarmonicOscillator()
    integrator = integrators.LangevinIntegrator(temperature=298.0*unit.kelvin,
                                            collision_rate=1.0/unit.picoseconds,
                                            timestep=1.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("CUDA")
    platform.setPropertyDefaultValue("Precision", "mixed")
    platform.setPropertyDefaultValue("DeterministicForces", "true")
    openmm.Context(test_system.system, integrator, platform)

@click.command()
@click.option("--yaml-path", type=click.Path(exists=True, dir_okay=False))
def cli(yaml_path):
    """test"""
    click.echo("📖 Fetching simulation options ")
    options = getSetupOptions(yaml_path)
    click.echo("🕵️ Checking OpenEye license")
    _check_openeye_license()
    click.echo("✅ OpenEye license good")
    click.echo("🖥️⚡ Making a test system to check if we can get a GPU")
    _test_gpu()
    click.echo("🎉 Test sucsessful!")
    click.echo(options)
