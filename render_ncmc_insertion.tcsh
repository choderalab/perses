#!/bin/tcsh

# Render the movie
rm -rf ncmc-insertion-frames
mkdir ncmc-insertion-frames

echo Running PyMOL...
/Applications/MacPyMOL.app/Contents/MacOS/MacPyMOL -qc render_ncmc_insertion.py

# Compile into a movie
ffmpeg -r 30 -i ncmc-insertion-frames/frame%04d.png -r 15 -b:v 5000000 -c:v wmv1 -y ncmc-insertion.wmv
