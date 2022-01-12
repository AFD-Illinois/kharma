#!/bin/bash
set -euo pipefail

BASE=../..

# Most of the point of this one is exercising all 3D of transport
# TODO any interesting 2d/1d tests?

conv_3d() {
    for res in 16 24 32 48
    do
      # Eight blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/mhdmodes.par parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=$res \
                                           parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=$half \
                                           $2
        mv mhdmodes.out0.00000.phdf mhd_3d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_3d_${res}_end_${1}.phdf
    done
}
conv_2d() {
    for res in 32 64 128 256
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/mhdmodes.par parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                                           parthenon/meshblock/nx1=16 parthenon/meshblock/nx2=16 parthenon/meshblock/nx3=1 \
                                           mhdmodes/dir=3 $2
        mv mhdmodes.out0.00000.phdf mhd_2d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${res}_end_${1}.phdf
    done
}
conv_1d() {
    for res in 64 128 256 512
    do
      # Eight blocks
      eighth=$(( $res / 8 ))
      $BASE/run.sh -i $BASE/pars/mhdmodes.par parthenon/mesh/nx1=$res parthenon/mesh/nx2=1 parthenon/mesh/nx3=1 \
                                           parthenon/meshblock/nx1=$eighth parthenon/meshblock/nx2=1 parthenon/meshblock/nx3=1 \
                                           mhdmodes/dir=3 $2
        mv mhdmodes.out0.00000.phdf mhd_1d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_1d_${res}_end_${1}.phdf
    done
}

# These 3 double as a demo of why WENO is great
conv_3d entropy mhdmodes/nmode=0
conv_3d entropy_mc "mhdmodes/nmode=0 GRMHD/reconstruction=linear_mc"
conv_3d entropy_vl "mhdmodes/nmode=0 GRMHD/reconstruction=linear_vl"
# Other modes don't benefit, exercise WENO most since we use it
conv_3d slow mhdmodes/nmode=1
conv_3d alfven mhdmodes/nmode=2
conv_3d fast mhdmodes/nmode=3

# 2D modes use small blocks, could pick up some problems at MPI ranks >> 1
# Currently very slow, plus modes are incorrect
#conv_2d fast2d mhdmodes/nmode=3
