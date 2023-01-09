#!/bin/bash

# Bash script to run 1D (Noh) shock test

# Set paths
KHARMADIR=../..

exit_code=0

noh_test() {
    ALL_RES="64,128,256,512,1024,2048"
    for res in 64 128 256 512 1024 2048
    do
        eighth=$(($res / 8))
        $KHARMADIR/run.sh -i $KHARMADIR/pars/noh.par parthenon/output0/dt=1000 debug/verbose=1 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$eighth \
                            >log_noh_${res}.txt 2>&1

        cp noh.out0.final.phdf noh.out0.final.res$res.phdf
    done
    pyharm-convert *.phdf
    check_code=0
    python check.py . . $ALL_RES 1.666667 || check_code=$?
    if [[ $check_code != 0 ]]; then
        echo Noh shock test FAIL: $check_code
        exit_code=1
    else
        echo Noh shock test success
    fi
    rm *.phdf
}

noh_test

exit $exit_code
