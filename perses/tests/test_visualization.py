from perses.analysis.visualization import Visualization
from pkg_resources import resource_filename
import os

running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

@skipIf(running_on_github_actions, "Skip helper function on GH Actions")
def test_visualization():
    input_directory = resource_filename("perses", "data/visualization")

    v = Visualization(os.path.join(input_directory, "old.pdb"),
                      os.path.join(input_directory, "new.pdb"),
                      "dcd",
                      "A",
                      mutated_residue="667",
                      ligand_chain="C")
    v.load()
    v.format(zoom_distance=3, rotate_x_angle=270, rotate_y_angle=180, rotate_z_angle=270)
    v.save_frames(outfile_prefix=os.path.join(input_directory, "frame"), dpi=800, equilibration_frames=2)
    frames = [os.path.join(input_directory, "frame-%05d.png" % (i)) for i in range(10)]
    v.save_mp4(frames, outfile=os.path.join(input_directory, "movie.mp4"), fps=5)
    for i in range(10):
        os.remove(os.path.join(input_directory, "frame-%05d.png" % (i)))
    os.remove(os.path.join(input_directory, "movie.mp4"))