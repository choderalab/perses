from perses.analysis.visualization import Visualization
from pkg_resources import resource_filename
import os
from unittest import skipIf

running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

@skipIf(running_on_github_actions, "Skip helper function on GH Actions")
def test_protein_mutation():
    input_directory = resource_filename("perses", "data/visualization/protein-mutation/")

    v = Visualization(os.path.join(input_directory, "old.pdb"),
                      os.path.join(input_directory, "new.pdb"),
                      "dcd",
                      "A",
                      mutated_residue="667",
                      ligand_chain="C")
    v.load()
    v.format(zoom_distance=3, rotate_x_angle=270, rotate_y_angle=180, rotate_z_angle=270)
    v.save_frames(outfile_prefix=os.path.join(input_directory, "frame"), dpi=300, equilibration_frames=2)
    frames = [os.path.join(input_directory, "frame-%05d.png" % (i)) for i in range(10)]
    v.save_mp4(frames, outfile=os.path.join(input_directory, "movie.mp4"), fps=5)
    for i in range(10):
        os.remove(os.path.join(input_directory, "frame-%05d.png" % (i)))
    os.remove(os.path.join(input_directory, "movie.mp4"))

@skipIf(running_on_github_actions, "Skip helper function on GH Actions")
def test_small_molecule():
    input_directory = resource_filename("perses", "data/visualization/small-molecule")

    v = Visualization(os.path.join(input_directory, "old.pdb"),
                      os.path.join(input_directory, "new.pdb"),
                      "dcd",
                      "C",
                      is_protein=False,
                      unique_old_selection="old and name H14",
                      unique_new_selection="new and (name C17 or name O2 or name C25 or name H20 or name H19 or name C26 or name H21 or name H22 or name C23 or name H18 or name C19 or name H10 or name H11 or name C16 or name O1 or name N2 or name H23 or name C21 or name H14 or name H15)")
    v.load()
    v.format(zoom_distance=3, rotate_x_angle=180, rotate_y_angle=180, rotate_z_angle=270)
    v.save_frames(outfile_prefix=os.path.join(input_directory, "frame"), dpi=300, equilibration_frames=2)
    frames = [os.path.join(input_directory, "frame-%05d.png" % (i)) for i in range(10)]
    v.save_mp4(frames, outfile=os.path.join(input_directory, "movie.mp4"), fps=5)
    for i in range(10):
        os.remove(os.path.join(input_directory, "frame-%05d.png" % (i)))
    os.remove(os.path.join(input_directory, "movie.mp4"))
