#!/bin/bash
set -euo pipefail

BASE=../..

# Extended MHD modes convergence in 2D to exercise basic EMHD source terms

conv_2d() {
    for res in 32 64 128 256
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/emhdmodes.par debug/verbose=1 \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 $2
        mv mhdmodes.out0.00000.phdf mhd_2d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${res}_end_${1}.phdf
    done
}

# 2D modes use small blocks, could pick up some problems at MPI ranks >> 1
# Just one default mode
conv_2d emhd2d_vl "GRMHD/reconstruction=linear_vl"
conv_2d emhd2d_mc "GRMHD/reconstruction=linear_mc"
conv_2d emhd2d_weno "GRMHD/reconstruction=weno5"
