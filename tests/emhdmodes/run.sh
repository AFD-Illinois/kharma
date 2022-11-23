#!/bin/bash
set -euo pipefail

BASE=../..

# Extended MHD modes convergence in 2D to exercise basic EMHD source terms

exit_code=0

conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/emhdmodes.par debug/verbose=1 \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                      $2 >log_${1}_${res}.txt 2>&1
        mv emhdmodes.out0.00000.phdf emhd_2d_${res}_start_${1}.phdf
        mv emhdmodes.out0.final.phdf emhd_2d_${res}_end_${1}.phdf
    done
    check_code=0
    python check.py $ALL_RES "$3" $1 2d || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo EMHD modes test $3 FAIL: $check_code
        exit_code=1
    else
        echo EMHD modes test $3 success
    fi
}

# 2D modes use small blocks, could pick up some problems at MPI ranks >> 1
# Just one default mode
ALL_RES="32,64,128"
conv_2d emhd2d_weno GRMHD/reconstruction=weno5 "EMHD mode in 2D, WENO5"
ALL_RES="32,64,128,256"
conv_2d emhd2d_mc GRMHD/reconstruction=linear_mc "EMHD mode in 2D, linear/MC reconstruction"

# Test that higher-order terms don't mess anything up
conv_2d emhd2d_higher_order emhd/higher_order_terms=true "EMHD mode in 2D, higher order terms enabled"

exit $exit_code
