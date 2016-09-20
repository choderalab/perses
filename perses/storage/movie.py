"""
Generate a movie from atom-by-atom proposal PMF and placement PDB files.
Run with `run movie.py` in PyMOL
"""

iteration = 6

from pymol import cmd, movie
#cmd.mclear()
#cmd.mset('1')
cmd.set('sphere_scale', 0.02)
cmd.rewind()
cmd.delete('all')
cmd.load('geometry-%d-forward-proposal.pdb' % iteration)
cmd.hide('all')
cmd.show('spheres', 'name Ar')
cmd.spectrum(expression='b', selection='name Ar', palette='blue_white_red', minimum=0, maximum=6)
cmd.load('geometry-%d-forward-stages.pdb' % iteration)
nframes = 30
specification = ''
for i in range(15):
   specification += '%d x%d ' % (i, nframes)
cmd.mset(specification)
frame = 0
for i in range(15):
    movie.rock(frame, frame+nframes)
    frame += nframes
