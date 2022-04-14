#!/bin/bash
set -euo pipefail

BASE=~/kharma

# Extended MHD shock test convergence to exercise higher order terms
# We'll use just 1 MPI rank to circumvent the somewhat annoying BVP initialization

conv_1d() {
    for res in 256 512 1024 2048
    do
        cp -r ${BASE}/kharma/prob/emhd/shock_soln_${res}_default/*txt ./
        $BASE/run.sh -i $BASE/pars/emhdshock.par debug/verbose=1 \
                      parthenon/mesh/nx1=$res parthenon/mesh/nx2=1 parthenon/mesh/nx3=1 \
                      parthenon/meshblock/nx1=$res parthenon/meshblock/nx2=1 parthenon/meshblock/nx3=1
        mv emhdshock.out0.00000.phdf emhd_1d_${res}_start.phdf
        mv emhdshock.out0.final.phdf emhd_1d_${res}_end.phdf
        rm ./shock_soln*.txt
    done
}

conv_1d
