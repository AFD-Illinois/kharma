#!/usr/bin/env python3

# Fix an (any) KHARMA restart file so that KHARMA can restart from it
# this works around a bug in Parthenon w.r.t. mesh sizes

import sys
import numpy as np
import h5py

outf = h5py.File(sys.argv[2], "r+")

# Parthenon records the full size here,
# but pretty clearly expects the size without ghost zones.
# TODO running this script twice will cause errors
outf['Info'].attrs.modify('MeshBlockSize',
  np.maximum(outf['Info'].attrs['MeshBlockSize'][()] - 2*outf['Info'].attrs['IncludesGhost'][()]*outf['Info'].attrs['NGhost'][()],
             np.ones_like(outf['Info'].attrs['MeshBlockSize'][()])))
outf.close()