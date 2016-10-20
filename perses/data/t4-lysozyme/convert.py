#!/usr/bin/env python
#############################################################################
# Copyright (C) 2003-2015 OpenEye Scientific Software, Inc.
#############################################################################
# Program to convert from one molecule format to another
#############################################################################
import sys
from openeye.oechem import *


def main(argv=sys.argv):
    if len(argv) != 3:
        OEThrow.Usage("%s <infile> <outfile>" % argv[0])

    ifs = oemolistream()
    if not ifs.open(argv[1]):
        OEThrow.Fatal("Unable to open %s for reading" % argv[1])

    ofs = oemolostream()
    if not ofs.open(argv[2]):
        OEThrow.Fatal("Unable to open %s for writing" % argv[2])

    for mol in ifs.GetOEMols():
        # Assign aromaticity.
        OEAssignAromaticFlags(mol, OEAroModelOpenEye)

        # Add hydrogens.
        OEAddExplicitHydrogens(mol)

        OEWriteMolecule(ofs, mol)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
