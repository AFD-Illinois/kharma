#!/bin/bash

# Bash script testing initialization vs restart of a torus problem

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyHARM

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -i $KHARMADIR/pars/sane.par debug/archive_parameters=false perturbation/u_jitter=0 parthenon/time/nlim=5

mv torus.out1.final.rhdf torus.out1.final.first.rhdf

#$KHARMADIR/run.sh -r torus.out1.00000.rhdf parthenon/time/nlim=5
$KHARMADIR/run.sh -i $KHARMADIR/pars/sane.par debug/archive_parameters=false perturbation/u_jitter=0 parthenon/time/nlim=5

mv torus.out1.final.rhdf torus.out1.final.second.rhdf
