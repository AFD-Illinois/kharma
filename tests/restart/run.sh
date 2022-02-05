#!/bin/bash

# Bash script testing initialization vs restart of a torus problem

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyHARM

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -i $KHARMADIR/pars/sane.par parthenon/time/nlim=5

mv torus.out0.final.phdf torus.out0.final.init.phdf

python3 $KHARMADIR/scripts/fix_restart.py torus.out1.00000.rhdf

$KHARMADIR/run.sh -r torus.out1.00000.rhdf parthenon/time/nlim=5

mv torus.out0.final.phdf torus.out0.final.restart.phdf
