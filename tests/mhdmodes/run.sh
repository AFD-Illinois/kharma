#!/bin/bash

BASE=../..

# This test confirms that all of the many transport options in KHARMA
# can converge when modeling each of the basic linearized modes:
# slow, fast, and alfven waves

# It tests:
# 1. different reconstructions WENO vs linear
# 2. different drivers, simple, KHARMA, & ImEx
# 3. different B field transports, Flux-CT and Face-CT

exit_code=0

conv_3d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
      # Eight blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/tests/mhdmodes.par debug/verbose=2 mhdmodes/dir=0 \
                      parthenon/output0/single_precision_output=false parthenon/output0/dt=100. \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=$res \
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=$half \
                      $2 >log_3d_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_3d_${1}_${res}_start.phdf
        mv mhdmodes.out0.final.phdf mhd_3d_${1}_${res}_end.phdf
    done
    check_code=0
    python check.py $ALL_RES "$3" $1 3d 3 || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo MHD modes test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo MHD modes test \"$3\" success
    fi
}
conv_2d() {
    IFS=',' read -ra RES_LIST <<< "$ALL_RES"
    for res in "${RES_LIST[@]}"
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/tests/mhdmodes.par debug/verbose=2 mhdmodes/dir=3 \
                      parthenon/output0/single_precision_output=false parthenon/output0/dt=100. \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=1 \
                      $2 >log_2d_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_2d_${1}_${res}_start.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${1}_${res}_end.phdf
    done
    check_code=0
    python check.py $ALL_RES "$3" $1  2d || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo MHD modes test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo MHD modes test \"$3\" success
    fi
}

# Normal MHD modes, 2D, defaults
ALL_RES="16,24,32,48,64"
conv_2d slow mhdmodes/nmode=1 "slow mode in 2D"
conv_2d alfven mhdmodes/nmode=2 "Alfven mode in 2D"
conv_2d fast mhdmodes/nmode=3 "fast mode in 2D"

# Entropy mode as reconstruction demo
conv_2d entropy_nob "mhdmodes/nmode=0 b_field/solver=none" "entropy mode in 2D, no B field"
conv_2d entropy mhdmodes/nmode=0 "entropy mode in 3D, WENO reconstruction"
conv_2d entropy_mc "mhdmodes/nmode=0 driver/reconstruction=linear_mc" "entropy mode in 2D, linear/MC reconstruction"
#conv_2d entropy_vl "mhdmodes/nmode=0 driver/reconstruction=linear_vl" "entropy mode in 2D, linear/VL reconstruction"
# TODO doesn't converge?
#conv_2d entropy_donor "mhdmodes/nmode=0 driver/reconstruction=donor_cell" "entropy mode in 2D, Donor Cell reconstruction"

# KHARMA driver
conv_2d slow_kharma   "mhdmodes/nmode=1 driver/type=kharma" "slow mode in 2D, KHARMA driver"
conv_2d alfven_kharma "mhdmodes/nmode=2 driver/type=kharma" "Alfven mode in 2D, KHARMA driver"
conv_2d fast_kharma   "mhdmodes/nmode=3 driver/type=kharma" "fast mode in 2D, KHARMA driver"
# ImEx driver
conv_2d slow_imex   "mhdmodes/nmode=1 driver/type=imex" "slow mode in 2D, ImEx explicit"
conv_2d alfven_imex "mhdmodes/nmode=2 driver/type=imex" "Alfven mode in 2D, ImEx explicit"
conv_2d fast_imex   "mhdmodes/nmode=3 driver/type=imex" "fast mode in 2D, ImEx explicit"
# B field totally explicit
conv_2d slow_imex_semi   "mhdmodes/nmode=1 driver/type=imex GRMHD/implicit=true b_field/implicit=false" "slow mode 3D, ImEx semi-implicit"
conv_2d alfven_imex_semi "mhdmodes/nmode=2 driver/type=imex GRMHD/implicit=true b_field/implicit=false" "Alfven mode 3D, ImEx semi-implicit"
conv_2d fast_imex_semi   "mhdmodes/nmode=3 driver/type=imex GRMHD/implicit=true b_field/implicit=false" "fast mode 3D, ImEx semi-implicit"

# KHARMA driver
conv_2d slow_kharma_ct   "mhdmodes/nmode=1 driver/type=kharma b_field/solver=face_ct" "slow mode in 2D, KHARMA driver w/face CT"
conv_2d alfven_kharma_ct "mhdmodes/nmode=2 driver/type=kharma b_field/solver=face_ct" "Alfven mode in 2D, KHARMA driver w/face CT"
conv_2d fast_kharma_ct   "mhdmodes/nmode=3 driver/type=kharma b_field/solver=face_ct" "fast mode in 2D, KHARMA driver w/face CT"
# ImEx driver
conv_2d slow_imex_ct   "mhdmodes/nmode=1 driver/type=imex b_field/solver=face_ct" "slow mode in 2D, ImEx explicit w/face CT"
conv_2d alfven_imex_ct "mhdmodes/nmode=2 driver/type=imex b_field/solver=face_ct" "Alfven mode in 2D, ImEx explicit w/face CT"
conv_2d fast_imex_ct   "mhdmodes/nmode=3 driver/type=imex b_field/solver=face_ct" "fast mode in 2D, ImEx explicit w/face CT"


# simple driver, high res
ALL_RES="16,24,32,48,64,96,128,192,256"
conv_2d slow_highres   "mhdmodes/nmode=1 driver/type=imex" "slow mode in 2D, simple driver"
conv_2d alfven_highres "mhdmodes/nmode=2 driver/type=imex" "Alfven mode in 2D, simple driver"
conv_2d fast_highres   "mhdmodes/nmode=3 driver/type=imex" "fast mode in 2D, simple driver"

# 3D versions, basics only
ALL_RES="16,24,32"
conv_3d slow "mhdmodes/nmode=1 mhdmodes/dir=3" "slow mode in 3D"
conv_3d alfven "mhdmodes/nmode=2 mhdmodes/dir=3" "Alfven mode in 3D"
conv_3d fast "mhdmodes/nmode=3 mhdmodes/dir=3" "fast mode in 3D"

exit $exit_code
