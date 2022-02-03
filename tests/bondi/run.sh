#!/bin/bash
set -euo pipefail

BASE=../..

conv_2d() {
    for res in 32 48 64 96 128
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/bondi.par parthenon/output0/dt=1000 \
                                           parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                                           parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                                           $2
        mv bondi.out0.00000.phdf bondi_2d_${res}_start_${1}.phdf
        mv bondi.out0.final.phdf bondi_2d_${res}_end_${1}.phdf
    done
}

conv_2d fmks coordinates/transform=fmks
conv_2d mks coordinates/transform=mks
conv_2d linear_mc GRMHD/reconstruction=linear_mc
conv_2d linear_vl GRMHD/reconstruction=linear_vl
