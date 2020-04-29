#=========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================
# Adapted to compute L1 norms by Ben Prather, Illinois AFD


from __future__ import print_function
#****************************************************************
# Note: reader import occurs after we fix the path at the bottom
#****************************************************************

#**************
# other imports
import os
import sys
import numpy as np
import argparse

def addPath():
    """ add the vis/python directory to the pythonpath variable """
    myPath = os.path.realpath(os.path.dirname(__file__))

if __name__ == "__main__":
    addPath()

    #**************
    # import Reader
    #**************
    from phdf import phdf

    files = sys.argv[1:]

    if len(files) != 2:
        print("Usage: l1norm.py file1.pndf file2.phdf")
        exit(1)

    # Load first file and print info
    try:
        f0 = phdf(files[0])
    except:
        print("""
        *** ERROR: Unable to open %s as phdf file
        """%files[0])
        exit(2)

    # Load second file and print info
    try:
        f1 = phdf(files[1])
    except:
        print("""
        *** ERROR: Unable to open %s as phdf file
        """%files[1])
        exit(2)

    # Now go through all variables in first file
    # and hunt for them in second file.
    #
    # Note that indices don't match when blocks
    # are different
    # TODO make sure norm works even between differently refined meshes
    no_diffs = True

    otherLocations = [None]*f0.TotalCells
    for idx in range(f0.TotalCells):
        if f0.isGhost[idx%f0.CellsPerBlock]:
            # don't map ghost cells
            continue

        otherLocations[idx] = f0.findIndexInOther(f1,idx)

    for var in f0.Variables:
        if var == 'Locations' or var == 'Timestep':
            continue
        #initialize info values
        same = True

        # Get values from file
        val0 = f0.Get(var)
        val1 = f1.Get(var)

        norm = np.zeros(val0.shape[1]) # 1D vectors of variables only/at most
        nnorm = 0
        for idx,v in enumerate(val0):
            idx1, _, _, _, _, _ = otherLocations[idx]
            norm += np.abs(val1[idx1] - v)
            nnorm += 1
        norm /= nnorm

        # Print name and norm on a single line for parsing
        print("{} norm: {}".format(var, norm).replace('\n', ''))
        if np.any(norm) > 1e-2:
            no_diffs = False

    if no_diffs:
      exit(0)
    else:
      exit(4)
