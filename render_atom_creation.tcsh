#!/bin/tcsh

# Render the movie
rm -rf atom-creation-frames
mkdir atom-creation-frames

echo Running PyMOL...
/Applications/MacPyMOL.app/Contents/MacOS/MacPyMOL -qc render_atom_creation.py

# Compile into a movie
ffmpeg -r 30 -i atom-creation-frames/frame%04d.png -r 15 -b:v 5000000 -c:v wmv1 -y atom-creation.wmv
