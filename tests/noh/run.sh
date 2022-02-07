#!/bin/bash

# Bash script to run 1D (Noh) shock test

# Set paths
KHARMADIR=../..

noh_test() {
    for res in 64 128 256 512 1024 2048 4096
    do
        eighth=$(($res / 8))
        $KHARMADIR/run.sh -i $KHARMADIR/pars/noh.par parthenon/output0/dt=1000 debug/verbose=1 \
                            parthenon/mesh/nx1=$res parthenon/meshblock/nx1=$eighth

        cp noh.out0.final.phdf noh.out0.final.res$res.phdf
    done
}

noh_test
