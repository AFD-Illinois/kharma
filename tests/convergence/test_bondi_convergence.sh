#!/bin/bash

BASE=../..

conv_2d() {
    for res in 32 64 128 256
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/bondi.par parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                                           parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                                           $2
        mv bondi.out0.00000.phdf bondi_2d_${res}_start_${1}.phdf
        mv bondi.out0.00010.phdf bondi_2d_${res}_end_${1}.phdf
    done
}

conv_2d fmks coordinates/transform=fmks
python plot_convergence_bondi.py 32,64,128,256 "in 2D, FMKS coordinates" fmks
conv_2d mks coordinates/transform=mks
python plot_convergence_bondi.py 32,64,128,256 "in 2D, MKS coordinates" mks
conv_2d linear_mc GRMHD/reconstruction=linear_mc
python plot_convergence_bondi.py 32,64,128,256 "in 2D, linear recon with MC limiter" linear_mc
conv_2d linear_vl GRMHD/reconstruction=linear_vl
python plot_convergence_bondi.py 32,64,128,256 "in 2D, linear recon with VL limiter" linear_vl
