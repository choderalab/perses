# New cli for testing
import logging
import os
import openmm.testInstallation
from cloudpathlib import AnyPath

import click
import openmmtools.utils

# Setting logging level config
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
_logger = logging.getLogger()
_logger.setLevel(LOGLEVEL)


percy = """
MMMMMMMMMMMMXo:ccldOKNNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMWxcxOkxdodddxxxk0XWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWWMMWNX0Okk0WMMMMMM
MMMMMMMMMMMWxlk0OkdddddxkxdoodONWNXK0OOOOOOOOO0KXNWMMMMN0kxxxxxxxxddoddockWMMMMM
MMMMMMMMMMMMKooOOxooooc;:ldOOxooddddodddddddddodddxxkOkoodkOOkxdodxO000kcxWMMMMM
MMMMMMMMMMMMMKolk0000OdoddllxOOkkO00000O000000000OkxdookOOdl:cloooxO00kloXMMMMMM
MMMMMMMMMMMMMMNklokO0OO000OkkO000000000000O000000000O00OdllollxO0000OdldXMMMMMMM
MMMMMMMMMMMMMMMWXkdodkO00O00O0000OdlxO000000000OxxkO000OxkO00OOO00Odlo0WMMMMMMMM
MMMMMMMMMMMMMMMMMMWKkdldOOOO00000kdxOO000000000OxldO0000OO000OOxdoox0NMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMXdldOOO0000O000kkO0000000O0OOOOO0O00000Odloxk0NMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMKook00000000000Odlk00000000kooO000O00000OxcxNMMMMMWX0KNMMMMMM
MMMMMMMMMMMMMMMMMMKloO00000000000O0OkxdooooodxkxkO000000000O0xlkWMMMNkloook0OkxO
MMMMMMMMMMMMMMMMMNolO000000000000OxlllodddddolloxO00000000O00Odl0MMMOcx0Okooddo:
MMMMMMMMMMMMMMMMMOcx0OO00000O000kocoxxkO000OOkkocoO0000O0O0000kldWMMxck0000000xc
MMMMMMMMMMMMMMMMWdlO00O000000OOOo:dd:,:x000xc;cxd:d0000OO000O0OolXMWxck00OO0Oxlx
MMMMMMMMMMMMMMMMXooOO0000OOOOxkkclkc,,lO000k:,,lxclO00000000000dl0MNdcO0000kolkN
MMMMMMMMMMMMMMMMXolO00000kolccxOlckxook0000Oxlokx:okdkkxkO00000dl0XxlxO000kloXWM
MMMMMMMMMMMMMMMMNdlO00000OOd:lO0xclk00000000O0Oxclkkc;loxO0000OocoookO000OolKMMM
MMMMMMMMMMMMMMMMMOcx00000000xclkOkocldkOOOOOxdlldOOocoOO000000Oc,lxO00000kcxWMMM
MMMMMMMMMMMMMMMMMNdlO000000O0kocokOkdoooooooooxOkdllxO000O000Od:oO00O000OooXMMMM
MMMMMMMMMMMMMMMMMMXolk00000O00OkollodxkkkkkkkxdlllxO00000OOOOd:oO000O00OdcOMMMMM
MMMMMMMMMMMMMMMMMMM0::xO0000000O0OkdooooooooooodkO000000000OocoO000O000xlxWMMMMM
MMMMMMMMMMMMMMMMMNOoccclxO00O000000000OOOOOOO00000000OO00OdclxO00O0000kldNMMMMMM
MMMMMMMMMMMMMMMWKdlxOOkolldkO0000000000000000000000000OxolldO0O000000kldNMMMMMMM
MMMMMMMMMMMMMMWOldO00O00OxollodxOO000000000000000OOkdolloxO000O0000OxlxNMMMMMMMM
MMMMMMMMMMMMMWkldO000O00000OkdoooooooodddddddoooooooodkO000OO0O000OdlOWMMMMMMMMM
MMMMMMMMMMMMWOldO00000000000000OOkxxddooooooooddxkO0000O000000O0kdoxKWMMMMMMMMMM
MMMMMMMMMMMMXooO000000O0000000O0000000000000000OO00000OO000000kooxKWMMMMMMMMMMMM
MMMMMMMMMMMWxck00000000000000O000000000000000000O00OO000000O00dcOMMMMMMMMMMMMMMM
MMMWKkkOKWMXooO000000OkkO000000000000000000000000O000000000O00xckMMMMMMMMMMMMMMM
WMKdlcccldKOcd000000OdcdO0000000000000000000000000000000000000klkMMMMMMMMMMMMMMM
odccocccdllolk000000d:oO00000000000000000000000000000000000000OlxWMMMMMMMMMMMMMM
;:;:c::oxllolk00000kclO000000000000000000000000000000000000000OcdWMMMMMMMMMMMMMM
Oollodxdl;lc:x0000Oo:x0000000000000000000000000000000000000000kcxWMMMMMMMMMMMMMM
MN0kxxo::odl:d0000Olck000000000000000000000000000000000000O000dcOMMMMMMMMMMMMMMM
MMMMMMWKkddo;cO000Olck0000000000000000000000000000000000000O0OloNMMMMMMMMMMMMMMM
MMMMMMMMMWWOcoO000OkclO0000000000000000000000000000000000000Ool0MMMMMMMMMMMMMMMM
MMMMMMMMMMWxck0OOO0Olck00O00000000000000000000000000000O000Ool0WMMMMMMMMMMMMMMMM
MMMMMMMMMMM0loxolool:oO000000O000000000000000000000000000OxldKMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMWKkxkKN0o;ck0000O000000000000000000000000OO0OxooONMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMWxck000O000OkkO000000000000000000Oxo:;xNMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMOcx0000000kloO0000000000000Okxoollolc0MMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMKld000O000d;codddddddddoooclooodkO0olKMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMNolO00000Oll00OOkkkkkkkO0OllO000O0OloNMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMWxck00000kckMMMMMMMMMMMMMMkcx00000kcxWMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMkcx00000dcOMMMMMMMMMMMMMMKld00O00xckMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMWdlk00000kckWMMMMMMMMMMMMM0ld0OOO0kldNMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMKlo0OkkO0OloNMMMMMMMMMMMMWxck0OOOO0dc0MMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMXocdoollxxcdNMMMMMMMMMMMMMkcxOdlloxlc0MMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMKolxXXxc:oKMMMMMMMMMMMMMMKl:::x0dcckNMMMMMMMMMMMMMMMMMMMMMM
"""


def _check_openeye_license():
    import openeye

    assert openeye.oechem.OEChemIsLicensed(), "OpenEye license checks failed!"


def _test_platform(platform_name):
    # If a user asks for a platform, try and see if we can use it
    assert openmmtools.utils.platform_supports_precision(platform_name, "mixed")
    click.echo(f"🎉\t Platform test successful! Perses will be using platform {platform_name}.")


@click.command()
@click.option("--yaml", type=click.Path(path_type=AnyPath), required=True,
              help='Input yaml file with simulation parameters and options.')
@click.option("--platform-name", "--platform", type=str, default=None,
              help='Platform name to be used during the simulation. (i.e. cuda, OpenCL, CPU, Reference).')
@click.option("--override", multiple=True, required=False, default=None,
              help='Override specific option using key:value format.')
def cli(yaml, platform_name, override):
    """Run perses relative free energy calculation."""
    from perses.app.setup_relative_calculation import run
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=LOGLEVEL,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    click.echo(click.style(percy, fg="bright_magenta"))
    if override:
        click.echo("✍️ \t Overrides used")
    click.echo("🕵️\t Checking OpenEye license")
    _check_openeye_license()
    click.echo("✅\t OpenEye license good")
    click.echo("🖥️⚡\t Checking whether requested compute platform is available")
    openmm.testInstallation.main()  # Test openmm installation
    if platform_name:
        _test_platform(platform_name)  # test platform compatibility
        override = list(override)  # make override (click.Tuple) a list
        override.append(f"platform:{platform_name}")  # override platform option
    click.echo("🏃\t Running simulation")
    run(yaml_filename=yaml, override_string=override)
    click.echo("🧪\t Simulation over")


if __name__ == "__main__":
    cli()
