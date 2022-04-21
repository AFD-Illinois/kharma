#!/bin/bash
#set -euo pipefail

BASE=~/kharma

# Extended MHD atmosphere test convergence to exercise geometrical terms
# We'll use just 1 MPI rank to circumvent the somewhat annoying ODE initialization

conv_2d() {
	for res in 64 128 256 512
	do
		cp -r ${BASE}/kharma/prob/emhd/conducting_atmosphere_${res}_default/*txt ./
		$BASE/run.sh -i $BASE/pars/conducting_atmosphere.par debug/verbose=1 \
									parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
									parthenon/meshblock/nx1=$res parthenon/meshblock/nx2=$res parthenon/meshblock/nx3=1
		mv conducting_atmosphere.out0.00000.phdf emhd_2d_${res}_start.phdf
		mv conducting_atmosphere.out0.final.phdf emhd_2d_${res}_end.phdf
		rm ./atmosphere*.txt
	done
}

conv_2d
