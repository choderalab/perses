from perses.analysis.visualization import Visualization

def test_visualization():
    v = Visualization("examples/visualization/old.pdb", "examples/visualization/new.pdb", "dcd", "A", mutated_residue="667",
                      ligand_chain="C")
    v.load()
    v.format(zoom_distance=3, rotate_x_angle=270, rotate_y_angle=180, rotate_z_angle=270)
    v.save_frames(outfile_prefix="example/frame", dpi=800, equilibration_frames=2)
    frames = ["example/frame-%05d.png" % (i) for i in range(10)]
    v.save_mp4(frames, outfile="example/movie.mp4", fps=5)
