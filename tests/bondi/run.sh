#!/bin/bash

BASE=../..

exit_code=0

conv_2d() {
    ALL_RES="32,48,64,96,128"
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
    check_code=0
    python check.py $ALL_RES "$3" $1 || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Bondi test $3 FAIL: $check_code
        exit_code=1
    else
        echo Bondi test $3 success
    fi
}

# Test coordinates (raw ks?)
conv_2d fmks coordinates/transform=fmks "in 2D, FMKS coordinates"
conv_2d mks coordinates/transform=mks "in 2D, MKS coordinates"
# TODO fix this: converges at 2.3!!
#conv_2d eks coordinates/transform=eks "in 2D, EKS coordinates"

# Recon
conv_2d linear_mc GRMHD/reconstruction=linear_mc "in 2D, linear recon with MC limiter"
conv_2d linear_vl GRMHD/reconstruction=linear_vl "in 2D, linear recon with VL limiter"

# And the GRIM/classic driver
# TODO these crash, likely an implicit w/o B field thing
#conv_2d imex driver/type=imex "in 2D, with Imex driver"
#conv_2d imex_im "driver/type=imex GRMHD/implicit=true" "in 2D, semi-implicit stepping"

exit $exit_code
