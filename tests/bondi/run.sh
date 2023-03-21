#!/bin/bash

BASE=../..

exit_code=0

conv_2d() {
    ALL_RES="16,32,48,64"
    for res in 16 32 48 64
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/bondi.par debug/verbose=1 debug/flag_verbose=2 parthenon/time/tlim=50 \
                                           parthenon/output0/dt=1000 parthenon/output0/single_precision_output=false \
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

# Test coordinates
conv_2d fmks coordinates/transform=fmks "in 2D, FMKS coordinates"
conv_2d mks coordinates/transform=mks "in 2D, MKS coordinates"
conv_2d eks coordinates/transform=eks "in 2D, EKS coordinates"
# TODO broken
#conv_2d ks coordinates/transform=null "in 2D, KS coordinates"

# Recon
conv_2d linear_mc GRMHD/reconstruction=linear_mc "in 2D, linear recon with MC limiter"
conv_2d linear_vl GRMHD/reconstruction=linear_vl "in 2D, linear recon with VL limiter"

# And the GRIM/classic driver
conv_2d imex driver/type=imex "in 2D, with Imex driver"
conv_2d imex_im "driver/type=imex GRMHD/implicit=true" "in 2D, semi-implicit stepping"

exit $exit_code
