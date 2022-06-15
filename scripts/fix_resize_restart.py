#!/usr/bin/env python3

# Fix an (any) KHARMA restart file so that KHARMA can restart from it
# this works around a bug in Parthenon w.r.t. mesh sizes

import sys
import numpy as np
import h5py

outf = h5py.File(sys.argv[1], "r+")

# Restart files from resizing lack a bunch of during-the-run output
for attr in ('Time', 'dt'):
    if not attr in outf['Info'].attrs:
        outf['Info'].attrs.create(attr, 0.)

outf['Info'].attrs['NCycle'] = np.array(0, dtype=np.int32)

for attr in ('NCycle',):
    if not attr in outf['Info'].attrs:
        outf['Info'].attrs.create(attr, 0, dtype=np.int32)

outf.close()
