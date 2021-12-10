#!/bin/bash

BASE=../..

conv_3d() {
    for res in 8 16 32 64
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
                                           parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                                           mhdmodes/dir=3 $2
        mv mhdmodes.out0.00000.phdf mhd_2d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${res}_end_${1}.phdf
    done
}
conv_1d() {
    for res in 32 64 128 256 512 1024
    do
      # Two blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/mhdmodes.par parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                                           parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                                           mhdmodes/dir=3 $2
        mv mhdmodes.out0.00000.phdf mhd_2d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${res}_end_${1}.phdf
    done
}

conv_3d entropy mhdmodes/nmode=0
python plot_convergence_modes.py 8,16,32,64 "entropy mode in 3D" entropy
conv_3d slow mhdmodes/nmode=1
python plot_convergence_modes.py 8,16,32,64 "slow mode in 3D" slow
conv_3d alfven mhdmodes/nmode=2
python plot_convergence_modes.py 8,16,32,64 "Alfven mode in 3D" alfven
conv_3d fast mhdmodes/nmode=3
python plot_convergence_modes.py 8,16,32,64 "fast mode in 3D" fast

conv_3d entropy_mc "mhdmodes/nmode=0 GRMHD/reconstruction=linear_mc"
python plot_convergence_modes.py 8,16,32,64 "entropy mode in 3D, linear with MC limiter" entropy_mc
conv_3d entropy_vl "mhdmodes/nmode=0 GRMHD/reconstruction=linear_vl"
python plot_convergence_modes.py 8,16,32,64 "entropy mode in 3D, linear with VL limiter" entropy_vl

conv_2d fast_weno "mhdmodes/nmode=3"
python plot_convergence_modes.py 32,64,128,256 "fast mode in 2D, WENO5" fast_weno 2d
conv_2d fast_mc "mhdmodes/nmode=3 GRMHD/reconstruction=linear_mc"
python plot_convergence_modes.py 32,64,128,256 "fast mode in 2D, linear with MC limiter" fast_mc 2d
conv_2d fast_vl "mhdmodes/nmode=3 GRMHD/reconstruction=linear_vl"
python plot_convergence_modes.py 32,64,128,256 "fast mode in 2D, linear with VL limiter" fast_vl 2d
