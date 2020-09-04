"""
Visualization tools for perses (protein mutations or small molecule transformations).
Requires PDB and trajectory (.dcd or .xtc) files for old and new atoms.
Note: Not tested on small molecules yet

Pre-requisites:
conda install -c schrodinger pymol (+ PyMOL license)
conda install -c conda-forge moviepy

TODO
----
* Test small molecule transformation

"""

__author__ = 'Ivy Zhang'

################################################################################
# IMPORTS
################################################################################

import sys, os
from simtk.openmm import app

try:
    import pymol
    from pymol import cmd
    _FOUND_PYMOL = True
except:
    _FOUND_PYMOL = False

try:
    import moviepy.editor as mpy
    _FOUND_MPY = True
except:
    _FOUND_MPY = False

def _check_pymol():
    """Check whether pymol was imported, if not raise exception. """
    if _FOUND_PYMOL:
        return
    else:
        raise ImportError("PyMOL is required for this module. Please `conda install -c schrodinger pymol`.")

def _check_mpy():
    """Check whether moviepy was imported, if not raise exception. """
    if _FOUND_MPY:
        return
    else:
        raise ImportError(
            "moviepy is required for this module. Please `conda install -c conda-forge moviepy` or `pip install moviepy`.")


################################################################################
# LOGGER
################################################################################

import logging
logging.basicConfig(level=logging.NOTSET)

_logger = logging.getLogger("visualization")
_logger.setLevel(logging.INFO)

################################################################################
# VISUALIZATION
################################################################################

class Visualization(object):
    """
    Visualization tools for perses non-equilibrium switching.

    from perses.analysis.visualization import Visualization

    # Protein mutation example:
    v = Visualization("old.pdb", "new.pdb", "dcd", "A", mutated_residue="667", ligand_chain="C")
    v.load()
    v.format(zoom_distance=3, rotate_x_angle=270, rotate_y_angle=180, rotate_z_angle=270)
    v.save_frames(outfile_prefix="output/frame")
    frames = ["output/frame-%05d.png" % (i) for i in range(200)] # Create list of file paths for the pngs
    v.save_mp4(frames)

    # Small molecule transformation example:
    v = Visualization("old.pdb", "new.pdb", "dcd", "C")
    v.load()
    v.format(zoom_distance=3, rotate_x_angle=90)
    v.save_frames(outfile_prefix="output/frame")
    frames = ["output/frame-%05d.png" % (i) for i in range(200)] # Create list of file paths for the pngs
    v.save_mp4(frames)

    """

    def __init__(self, old_pdb,
                        new_pdb,
                        traj_type,
                        mutated_chain,
                        is_protein=True,
                        mutated_residue=None,
                        ligand_chain=None,):
        """
        Note: the pdb and trajectory names should be the same, only differing in file extension.

        Parameters
        ----------
        old_pdb : str
            Path to pdb file for old system. Used only to load the topology into PyMOL if traj_type is provided.
        new_pdb : str
            Path to pdb file for new system. Used only to load the topology into PyMOL if traj_type is provided.
        traj_type : str
            Trajectory type. Can be "xtc" or "dcd". Note PDB trajectory files are not supported, as read/write
            times are VERY slow.
        mutated_chain : str
            One letter string representing the chain id that contains the mutation/chemical transformation.
        is_protein : bool, default True
            If True, indicates that the trajectory involves a protein mutation.
            If False, indicates that the trajectory involves a small molecule transformation.
        mutated_residue : str, default None
            Residue id of the protein mutation
            Leave as None if the trajectory involves a small molecule transformation.
        ligand_chain : str, default None
            One letter string representing the chain id that contains the ligand.
            Leave as None if the trajectory involves a small molecule transformation.

        """
        _check_pymol()
        _check_mpy()

        self._old_pdb = old_pdb
        self._new_pdb = new_pdb
        self._old_traj = f"{os.path.splitext(old_pdb)[0]}.{traj_type.lower()}"
        self._new_traj = f"{os.path.splitext(new_pdb)[0]}.{traj_type.lower()}"
        self._is_protein = is_protein
        self._mutated_chain = mutated_chain
        self._mutated_residue = mutated_residue
        self._ligand_chain = ligand_chain

        if self._is_protein:
            if self._mutated_residue is None:
                raise Exception("Need to specify the mutated residue id!")
            if self._ligand_chain is None:
                raise Exception("Need to specify the ligand chain!")
            self._old_selection = f"(old and chain {self._mutated_chain} and resi {self._mutated_residue})"
            self._new_selection = f"(new and chain {self._mutated_chain} and resi {self._mutated_residue})"
            self._both_selection = f"(chain {self._mutated_chain} and resi {self._mutated_residue})"
        else:
            self._old_selection = f"(old and chain {self._mutated_chain})"
            self._new_selection = f"(new and chain {self._mutated_chain})"
            self._both_selection = f"chain {self._mutated_chain}"

    def load(self,
             color_complex="green",
             background_color="white"):
        """
        Load the perses trajectory into PyMOL.

        Parameters
        ----------
        color_complex : str, default "green"
            Color to set for the whole complex structure.
        background_color : str, default "white"
            Color to set the background.
        """
        _logger.info("Loading trajectory...")

        # Launch pymol session
        pymol.pymol_argv = ["pymol", "-qc"] + sys.argv[1:]
        pymol.finish_launching()

        # Load trajectories
        cmd.set("defer_builds_mode", 3)  # Improve performance for handling long trajectories
        cmd.load(self._old_pdb)
        cmd.load(self._new_pdb)
        if self._old_traj is not None and self._new_traj is not None:
            cmd.load_traj(self._old_traj, state=1)
            cmd.load_traj(self._new_traj, state=1)
        cmd.set_name(os.path.splitext(os.path.basename(self._old_pdb))[0], "old")
        cmd.set_name(os.path.splitext(os.path.basename(self._new_pdb))[0], "new")
        cmd.intra_fit("old")
        cmd.intra_fit("new")

        # Check that the trajectories are the same length
        old_states = cmd.count_states("old")
        new_states = cmd.count_states("new")
        if old_states != new_states:
            raise Exception(f"The old and new trajectories are not the same length! old: {old_states}, new: {new_states}")

        # Remove solvent
        _logger.info("Removing solvent...")
        cmd.remove("resn hoh")
        cmd.remove("resn na")
        cmd.remove("resn cl")

        # Format overall appearance
        _logger.info("Formatting overall appearance, color, and background color...")
        cmd.space("cmyk")
        cmd.color(f"{color_complex}")
        cmd.bg_color(background_color)

        # Get PyMOL indices of unique old and new atoms as strings
        _logger.info("Getting unique old and new atom indices...")
        old = app.PDBFile(self._old_pdb)
        new = app.PDBFile(self._new_pdb)

        if self._is_protein:
            old_atoms = [atom for residue in old.topology.residues()
             if residue.id == self._mutated_residue and residue.chain.id == self._mutated_chain
             for atom in residue.atoms()]

            new_atoms = [atom for residue in new.topology.residues()
                 if residue.id == self._mutated_residue and residue.chain.id == self._mutated_chain
                 for atom in residue.atoms()]
        else:
            old_atoms = [atom for residue in old.topology.residues()
                         if residue.chain.id == self._mutated_chain
                         for atom in residue.atoms()]

            new_atoms = [atom for residue in new.topology.residues()
                         if residue.chain.id == self._mutated_chain
                         for atom in residue.atoms()]

        core = []
        for new_atom in new_atoms:
            for old_atom in old_atoms:
                if new_atom.name == old_atom.name and new_atom.index == old_atom.index:
                    core.append(new_atom)
                    core.append(old_atom)

        unique_old_str = "+".join([str(atom.index + 1) for atom in old_atoms if atom not in core]) # Add one to match PyMOL index
        unique_new_str = "+".join([str(atom.index + 1) for atom in new_atoms if atom not in core]) # Add one to match PyMOL index
        self._unique_old = f"old and index {unique_old_str}"
        self._unique_new = f"new and index {unique_new_str}"

    def format(self,
               color_residue="yellow",
               color_ligand="green",
               sphere_radius=1,
               zoom_distance=None,
               rotate_x_angle=0,
               rotate_y_angle=0,
               rotate_z_angle=0,
               smooth=True
               ):

        """
        Format the appearance of the trajectory already loaded into PyMOL.
        Note: this function must be called after load().

        Parameters
        ----------
        color_residue : str, default "yellow"
            Color to set the mutated amino acid.
        color_ligand : str, default "green"
            Color to set the ligand.
        sphere_radius : int, default 0.5
            Radius of the spheres shown for the mutated residue (if trajectory contains protein mutation)
            and small molecule.
        zoom_distance : int, default None
            The distance in angstroms around the mutated residue (or altered ligand) on which to zoom.
            The frames will be saved based on this view.
        rotate_x_angle : int, default 0
            The angle in degrees for which to rotate the complex in the x direction.
            The frames will be saved based on this view.
        rotate_y_angle : int, default 0
            The angle in degrees for which to rotate the complex in the y direction.
            The frames will be saved based on this view.
        rotate_z_angle : int, default 0
            The angle in degrees for which to rotate the complex in the z direction.
            The frames will be saved based on this view.
        smooth : bool, default True
            Whether to perform smoothing on the trajectory to reduce high frequency vibrations.
        """
        color_dict = {"yellow": cmd.util.cbay,
                         "green": cmd.util.cbag,
                         "cyan": cmd.util.cbac,
                         "light magenta": cmd.util.cbam,
                         "salmon": cmd.util.cbam,
                         "white": cmd.util.cbaw,
                         "slate": cmd.util.cbab,
                         "orange": cmd.util.cbao,
                         "purple": cmd.util.cbap,
                         "pink": cmd.util.cbak}

        _logger.info("Aligning the mutated residue/ligand...")
        cmd.align(self._old_selection, self._new_selection)

        # Format color and shape of mutated residue
        _logger.info("Coloring the mutated residue/ligand...")
        color_dict[color_residue](self._both_selection)

        # Format spheres and sticks of mutated residue
        _logger.info("Showing spheres...")
        cmd.show("spheres", self._old_selection)
        cmd.set("sphere_scale", sphere_radius)
        cmd.set("sphere_transparency", 1, self._new_selection)

        # Format small molecule
        _logger.info("Coloring the small molecule...")
        color_dict[color_ligand](f"chain {self._ligand_chain}")

        # Fade out the protein and small molecule
        _logger.info("Fading out the protein...")
        cmd.select("not " + self._both_selection)
        cmd.set("cartoon_transparency", 0.8, "sele")
        if self._is_protein:
            _logger.info("Fading out the small molecule...")
            cmd.set("stick_transparency", 0.8, "sele")
            # Remove duplicate small molecule
            cmd.select(f"new and chain {self._ligand_chain}")
            cmd.remove("sele")

        # Set the desired view for the movie
        _logger.info("Setting the view...")
        cmd.rotate("x", rotate_x_angle, "all", 0)
        cmd.rotate("y", rotate_y_angle, "all", 0)
        cmd.rotate("z", rotate_z_angle, "all", 0)
        if zoom_distance:
            cmd.select(f"old within {zoom_distance} of {self._both_selection}")
            cmd.zoom("sele")
            cmd.deselect()

        # Format the speed and smoothness of the trajectory
        if smooth:
            cmd.smooth()

    def _set_transparency(self, old_sphere, new_sphere, old_selection, new_selection):
        """
        Set sphere transparencies.

        Parameters
        ----------
        old_sphere : int
            Sphere transparency of old residue/small molecule
        new_sphere : int
            Sphere transparency of new residue/small molecule

        """
        cmd.show("spheres", self._both_selection)
        cmd.set("sphere_transparency", old_sphere, old_selection)
        cmd.set("sphere_transparency", new_sphere, new_selection)

    def save_frames(self,
                    equilibration_frames=25,
                    outfile_prefix="frame",
                    background_color="white",
                    width=10,
                    dpi=400):
        """
        Save frames as pngs.
        Note: this function must be called after format().

        Parameters
        ----------
        equilibration_frames : int, default 25
            Number of frames to use as "equilibration," aka the frames in the beginning and end where there are no changes
            in transparency and we are just showing the old or new atoms.
        outfile_prefix : str, default "frame"
            Prefiix for the output files at which to save the frames.
        background_color : str, default "white"
            Color to set the background
        width : int, default 10
            Width, in cm, of the png
        dpi : int, default 1500
            Dots per inch. Controls resolution of the png.

        """

        states = int(cmd.count_states("old"))

        for step in range(states):
            cmd.frame(step)
            if step <= equilibration_frames:
                self._set_transparency(0, 1, self._old_selection, self._new_selection)
            elif step >= (states - equilibration_frames):
                self._set_transparency(1, 0, self._old_selection, self._new_selection)
            else:
                lam = (step - equilibration_frames) / (states - 2*(equilibration_frames))
                self._set_transparency(lam, 1-lam, self._unique_old, self._unique_new)

            cmd.set("ray_opaque_background", 1)
            cmd.refresh()
            cmd.png(f"{outfile_prefix}-{step:05d}.png", width=f"{width}cm", dpi=dpi)
            _logger.info(f"Saved frame {step}...")

    @staticmethod
    def save_mp4(frames, outfile="movie.mp4", fps=25):
        """
        Generate a mp4 file from the png frames.

        Parameters
        ----------
        frames : list of strings
            Names of files corresponding to the frames to be used for generating the mp4.
        outfile : str, default "movie.mp4"
            Path to save output mp4 file
        fps : int, default 25
            Frames per second to be used by moviepy
        """
        _logger.info("Generating mp4 file...")
        image_clip = mpy.ImageSequenceClip(frames, fps=fps)
        image_clip.write_videofile(outfile, fps=fps, codec='mpeg4')
