#!/bin/bash
set -euo pipefail

BASE=../..

conv_2d() {
    for res in 32 48 64 96 128
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/bondi.par parthenon/output0/dt=1000 debug/verbose=1 \
                                           parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                                           parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                                           $2 >log_${1}_${res}.txt 2>&1
        mv bondi.out0.00000.phdf bondi_2d_${res}_start_${1}.phdf
        mv bondi.out0.final.phdf bondi_2d_${res}_end_${1}.phdf
    done
}

# Test coordinates (raw ks?)
conv_2d fmks coordinates/transform=fmks
conv_2d mks coordinates/transform=mks
conv_2d eks coordinates/transform=eks
# Recon
conv_2d linear_mc GRMHD/reconstruction=linear_mc
conv_2d linear_vl GRMHD/reconstruction=linear_vl
# And the GRIM/classic driver
conv_2d imex driver/type=imex
conv_2d imex_im "driver/type=imex driver/step=implicit"
