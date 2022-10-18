#!/bin/bash
#set -euo pipefail

BASE=~/kharma

# Viscous bondi inflow convergence to exercise all terms in the evolution equation of dP

conv_2d() {
	for res in 32 64 128 256
	do
		$BASE/run.sh -i $BASE/pars/bondi_viscous.par debug/verbose=1 \
									parthenon/mesh/nx1=$res parthenon/mesh/nx2=$res parthenon/mesh/nx3=1 \
									parthenon/meshblock/nx1=$res parthenon/meshblock/nx2=$res parthenon/meshblock/nx3=1 \
									b_field/implicit=false
		if [[ -d $res ]]; then
			echo -e "Resolution directory exists. Clearing existing files in there and copying new files\n"
			rm -r ${res}
		else
			mkdir $res
		fi
		. /home/vdhruv2/anaconda3/etc/profile.d/conda.sh
		conda activate pyharm
		pyharm-convert --double *.phdf
		conda deactivate
		cp -r ./bondi_viscous.out0*.h5 $res
		mv bondi_viscous.out0.00000.h5 emhd_2d_${res}_start.h5
		mv bondi_viscous.out0.final.h5 emhd_2d_${res}_end.h5
		rm -r ./bondi_viscous*
	done
}

conv_2d
