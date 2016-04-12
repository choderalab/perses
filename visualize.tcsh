#!/bin/tcsh

# Render the movie
rm -rf frames
mkdir frames

echo Running PyMOL...
/Applications/MacPyMOL.app/Contents/MacOS/MacPyMOL -qc render_trajectory.py

# Compile into a movie
ffmpeg -r 30 -i frames/frame%04d.png -r 15 -b:v 5000000 -c:v wmv1 -y movie.wmv
