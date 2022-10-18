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
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 $2 \
                      b_field/implicit=false
        mv emhdmodes.out0.00000.phdf emhd_2d_${res}_start_${1}.phdf
        mv emhdmodes.out0.final.phdf emhd_2d_${res}_end_${1}.phdf
    done
}

# 2D modes use small blocks, could pick up some problems at MPI ranks >> 1
# Just one default mode
conv_2d emhd2d_mc "GRMHD/reconstruction=linear_mc"
conv_2d emhd2d_weno "GRMHD/reconstruction=weno5"
# Test that higher-order terms don't mess anything up
conv_2d emhd2d_higher_order "emhd/higher_order_terms=true"
