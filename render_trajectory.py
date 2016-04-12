#!/opt/local/bin/python2.5

#=============================================================================================
# Render a replica trajectory in PyMOL
#=============================================================================================

#=============================================================================================
# REQUIREMENTS
#
# This code requires the NetCDF module, available either as Scientific.IO.NetCDF or standalone through pynetcdf:
# http://pypi.python.org/pypi/pynetcdf/
# http://sourceforge.net/project/showfiles.php?group_id=1315&package_id=185504
#=============================================================================================

#=============================================================================================
# TODO
#=============================================================================================

#=============================================================================================
# CHAGELOG
#=============================================================================================

#=============================================================================================
# VERSION CONTROL INFORMATION
# * 2009-08-01 JDC
# Created file.
#=============================================================================================

#=============================================================================================
# IMPORTS
#=============================================================================================

import numpy
from numpy import *
import os
import os.path
from pymol import cmd
from pymol import util

#=============================================================================================
# PARAMETERS
#=============================================================================================

pdb_filename = 'ncmc-insert-1.pdb'
pkl_filename = 'ncmc-insert-1-atomindices.pkl'

#=============================================================================================
# MAIN
#=============================================================================================

import __main__
__main__.pymol_argv = [ 'pymol', '-qc']
import pymol
#pymol.finish_launching()

# Delete everything
cmd.rewind()
cmd.reset()
cmd.delete('all')

# Retain order and IDs for imported atoms
cmd.set('retain_order', 1)
cmd.set('pdb_retain_ids', 1)

# Load PDB file
cmd.load(pdb_filename, 'complex')

# Recognize helices and sheets
cmd.dss()

# Align everything
cmd.intra_fit('all')

# Set up display
cmd.remove('resn WAT') # remove waters
cmd.select('receptor', '(not resn MOL) and (not resn WAT) and (not hydrogen)')
cmd.select('ligand', 'resn MOL')
cmd.deselect()
cmd.hide('all')
cmd.show('cartoon', 'receptor')
cmd.show('spheres', 'ligand')
util.cbay('ligand')
cmd.color('green', 'receptor')

# Show surface
#cmd.show('surface', 'receptor')
#cmd.set('surface_color', 'white')
#cmd.set('surface_mode', 3)
#cmd.set('transparency', 0.65)
#cmd.set('surface_quality', 1)

# speed up builds
cmd.set('defer_builds_mode', 3)
cmd.set('cache_frames', 0)
cmd.set('antialias', 2)

# Rewind
cmd.rewind()

# Set viewport size
cmd.viewport(800,600)

# Create one-to-one mapping between states and frames.
cmd.mset("1 -%d" % cmd.count_states())

# Zoom viewport
cmd.zoom('complex')

# Get number of steps
nsteps = cmd.count_states()

import cPickle as pickle
atom_indices = pickle.load(open(pkl_filename, 'r'))
selection = '(id %d)' % (atom_indices[0]+1)
for index in atom_indices[1:]:
    selection += ' or (id %d)' % (index+1)
cmd.select('ncmc', selection)
print(atom_indices)

# Render movie
frame_prefix = 'frames/frame'
cmd.set('ray_trace_frames', 1)
nskip = 10
frame_number = 0
for step in range(0,nsteps,nskip):
    print "rendering frame %04d / %04d" % (step+1, nsteps)
    cmd.frame(step+1)
    cmd.set('sphere_transparency', (1 - (float(step) / float(nsteps-1))), 'ncmc')
    cmd.png(frame_prefix + '%04d.png' % (frame_number), ray=True)
    frame_number += 1

cmd.set('ray_trace_frames', 0)
