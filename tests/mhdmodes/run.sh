#!/bin/bash

BASE=../..

# Most of the point of this one is exercising all 3D of transport
# TODO restore 2D test, use for codepath equivalence stuff (faster).

exit_code=0

conv_3d() {
    ALL_RES="16,24,32,48"
    for res in 16 24 32 48
    do
      # Eight blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/mhdmodes.par debug/verbose=2 \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=$res \
                      parthenon/meshblock/nx1=$half parthenon/meshblock/nx2=$half parthenon/meshblock/nx3=$half \
                      $2 >log_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_3d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_3d_${res}_end_${1}.phdf
    done
    check_code=0
    python check.py $ALL_RES "$3" $1 || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo MHD modes test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo MHD modes test \"$3\" success
    fi
}
conv_2d() {
    ALL_RES="32,64,128,256"
    for res in 32 64 128 256
    do
      # Four blocks
      half=$(( $res / 2 ))
      $BASE/run.sh -i $BASE/pars/mhdmodes.par debug/verbose=1 mhdmodes/dir=3 \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=16 parthenon/meshblock/nx2=16 parthenon/meshblock/nx3=1 \
                      $2 >log_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_2d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_2d_${res}_end_${1}.phdf
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
conv_1d() {
    ALL_RES="64 128 256 512"
    for res in 64 128 256 512
    do
      # Eight blocks
      eighth=$(( $res / 8 ))
      $BASE/run.sh -i $BASE/pars/mhdmodes.par debug/verbose=1 mhdmodes/dir=3 \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=1 parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=$eighth parthenon/meshblock/nx2=1 parthenon/meshblock/nx3=1 \
                      $2 >log_${1}_${res}.txt 2>&1
        mv mhdmodes.out0.00000.phdf mhd_1d_${res}_start_${1}.phdf
        mv mhdmodes.out0.final.phdf mhd_1d_${res}_end_${1}.phdf
    done
}

# These 3 double as a demo of why WENO is great
conv_3d entropy mhdmodes/nmode=0 "entropy mode in 3D"
conv_3d entropy_mc "mhdmodes/nmode=0 GRMHD/reconstruction=linear_mc" "entropy mode in 3D, linear/MC reconstruction"
conv_3d entropy_vl "mhdmodes/nmode=0 GRMHD/reconstruction=linear_vl" "entropy mode in 3D, linear/VL reconstruction"
# Other modes don't benefit, exercise WENO most since we use it
conv_3d slow mhdmodes/nmode=1 "slow mode in 3D"
conv_3d alfven mhdmodes/nmode=2 "Alfven mode in 3D"
conv_3d fast mhdmodes/nmode=3 "fast mode in 3D"
# And we've got to test classic/GRIM stepping
conv_3d slow_imex   "mhdmodes/nmode=1 driver/type=imex" "slow mode in 3D, ImEx explicit"
conv_3d alfven_imex "mhdmodes/nmode=2 driver/type=imex" "Alfven mode in 3D, ImEx explicit"
conv_3d fast_imex   "mhdmodes/nmode=3 driver/type=imex" "fast mode in 3D, ImEx explicit"
# B field totally explicit
conv_3d slow_imex_semi   "mhdmodes/nmode=1 driver/type=imex GRMHD/implicit=true" "slow mode 3D, ImEx semi-implicit"
conv_3d alfven_imex_semi "mhdmodes/nmode=2 driver/type=imex GRMHD/implicit=true" "Alfven mode 3D, ImEx semi-implicit"
conv_3d fast_imex_semi   "mhdmodes/nmode=3 driver/type=imex GRMHD/implicit=true" "fast mode 3D, ImEx semi-implicit"
# All variables semi-implicit
conv_3d slow_imex_im   "mhdmodes/nmode=1 driver/type=imex GRMHD/implicit=true b_field/implicit=true implicit/use_qr=false" "slow mode 3D, ImEx implicit"
conv_3d alfven_imex_im "mhdmodes/nmode=2 driver/type=imex GRMHD/implicit=true b_field/implicit=true implicit/use_qr=false" "Alfven mode 3D, ImEx implicit"
conv_3d fast_imex_im   "mhdmodes/nmode=3 driver/type=imex GRMHD/implicit=true b_field/implicit=true implicit/use_qr=false" "fast mode 3D, ImEx implicit"

# 2D modes use small blocks, could pick up some problems at MPI ranks >> 1
# Currently very slow, plus modes are incorrect
#conv_2d fast2d mhdmodes/nmode=3

exit $exit_code
