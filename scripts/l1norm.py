"""
 File: l1norm.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020, AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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
