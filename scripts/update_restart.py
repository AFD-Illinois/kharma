#!/usr/bin/env python3

# Take the contents of one KHARMA restart file, and
# transplant them into another
# Files must have the same mesh structure!

import sys
import numpy as np
import h5py

inf = h5py.File(sys.argv[1], "r")
outf = h5py.File(sys.argv[2], "r+")

# When including ghost zones, Parthenon records the full size here,
# but pretty clearly expects the size without ghost zones when actually restarting.
# Fix this. (Note ghost zones *are* still restored)
# TODO running this script twice will cause errors
outf['Info'].attrs.modify('MeshBlockSize',
  np.maximum(outf['Info'].attrs['MeshBlockSize'][()] - 2*outf['Info'].attrs['IncludesGhost'][()]*outf['Info'].attrs['NGhost'][()],
             np.ones_like(outf['Info'].attrs['MeshBlockSize'][()])))

if 'c.c.bulk.cons' in inf:
    # Restore from old-style files
    outf['cons.rho'][:,:,:,:] = inf['c.c.bulk.cons'][:,:,:,:,0:1]
    outf['cons.u'][:,:,:,:] = inf['c.c.bulk.cons'][:,:,:,:,1:2]
    outf['cons.uvec'][:,:,:,:] = inf['c.c.bulk.cons'][:,:,:,:,2:5]
    outf['cons.B'][:,:,:,:] = inf['c.c.bulk.cons'][:,:,:,:,5:8]

    outf['prims.rho'][:,:,:,:] = inf['c.c.bulk.prims'][:,:,:,:,0:1]
    outf['prims.u'][:,:,:,:] = inf['c.c.bulk.prims'][:,:,:,:,1:2]
    outf['prims.uvec'][:,:,:,:] = inf['c.c.bulk.prims'][:,:,:,:,2:5]
    #outf['prims.B'][:,:,:,:] = inf['c.c.bulk.prims'][:,:,:,:,5:8]

else:
    # Restore from new-style files
    outf['cons.rho'][:,:,:,:] = inf['cons.rho'][()]
    outf['cons.u'][:,:,:,:] = inf['cons.u'][()]
    outf['cons.uvec'][:,:,:,:] = inf['cons.uvec'][()]
    outf['cons.B'][:,:,:,:] = inf['cons.B'][()]

    outf['prims.rho'][:,:,:,:] = inf['prims.rho'][()]
    outf['prims.u'][:,:,:,:] = inf['prims.u'][()]
    outf['prims.uvec'][:,:,:,:] = inf['prims.uvec'][()]
    #outf['prims.B'][:,:,:,:] = inf['prims.B'][()]

# Restore *to* old-style restart files
# outf['c.c.bulk.cons'][()] = inf['c.c.bulk.cons'][:,:,:,:,:5]
# outf['c.c.bulk.B_con'][()] = inf['c.c.bulk.cons'][:,:,:,:,5:]
# outf['c.c.bulk.prims'][()] = inf['c.c.bulk.prims'][:,:,:,:,:5]
# outf['c.c.bulk.B_prim'][()] = inf['c.c.bulk.prims'][:,:,:,:,5:]

inf.close()
outf.close()
