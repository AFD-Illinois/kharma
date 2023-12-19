#!/bin/bash
set -euo pipefail

# Bash script testing initialization vs restart of a torus problem
# Require similarity to round-off after 5 steps

# Set paths
KHARMADIR=../..

exit_code=0

test_restart() {
    $KHARMADIR/run.sh -i $KHARMADIR/pars/tori_3d/sane.par parthenon/time/nlim=5 \
    $2 >log_restart_${1}_first.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_first.phdf

    sleep 1

    $KHARMADIR/run.sh -r torus.out1.00000.rhdf >log_restart_${1}_second.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_second.phdf

    check_code=0
    # Compare to some high degree of accuracy
    # TODO this was formerly 1e-11, we may need to clean up restarting & sequencing of the first steps
    pyharm diff --rel_tol 1e-9 restart_${1}_first.phdf restart_${1}_second.phdf -o compare_restart || check_code=$?
    # Compare binary. Sometimes works but not worth keeping always
    #h5diff --exclude-path=/Info \
    #       --exclude-path=/Input \
    #       --exclude-path=/divB \
    #       torus.out0.final.init.phdf torus.out0.final.restart.phdf
    if [[ $check_code != 0 ]]; then
        echo Restart test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo Restart test \"$3\" success
    fi
}
test_restart_smr() {
    $KHARMADIR/run.sh -i $KHARMADIR/pars/smr/sane2d_refined.par parthenon/time/nlim=5 \
    $2 >log_restart_${1}_first.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_first.phdf

    sleep 1

    $KHARMADIR/run.sh -r torus.out1.00000.rhdf >log_restart_${1}_second.txt 2>&1

    mv torus.out0.final.phdf restart_${1}_second.phdf

    check_code=0
    # Compare to some high degree of accuracy
    # TODO this was formerly 1e-11, we may need to clean up restarting & sequencing of the first steps
    pyharm diff --rel_tol 1e-9 restart_${1}_first.phdf restart_${1}_second.phdf -o compare_restart || check_code=$?
    # Compare binary. Sometimes works but not worth keeping always
    #h5diff --exclude-path=/Info \
    #       --exclude-path=/Input \
    #       --exclude-path=/divB \
    #       torus.out0.final.init.phdf torus.out0.final.restart.phdf
    if [[ $check_code != 0 ]]; then
        echo Restart test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo Restart test \"$3\" success
    fi
}

test_restart kharma "driver/type=kharma" "KHARMA driver"
test_restart imex "driver/type=imex" "ImEx driver"
#test_restart imex_emhd "driver/type=imex emhd/on=true" "ImEx driver, EMHD"
test_restart kharma_face "driver/type=kharma b_field/solver=face_ct" "KHARMA driver, face CT"
test_restart imex_face "driver/type=imex b_field/solver=face_ct" "ImEx driver, face CT"
test_restart kharma_face_2d "driver/type=kharma b_field/solver=face_ct parthenon/mesh/nx3=1 parthenon/meshblock/nx3=1" "KHARMA driver, face CT, 2D"
test_restart imex_face_2d "driver/type=imex b_field/solver=face_ct parthenon/mesh/nx3=1 parthenon/meshblock/nx3=1" "ImEx driver, face CT, 2D"
# SMR
test_restart_smr kharma_face_smr "driver/type=kharma b_field/solver=face_ct" "KHARMA driver, face CT, SMR"
test_restart_smr imex_face_smr "driver/type=imex b_field/solver=face_ct" "ImEx driver, face CT, SMR"

exit $exit_code
