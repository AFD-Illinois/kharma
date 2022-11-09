#!/bin/bash

# Bash script testing initialization vs restart of a torus problem
# TODO this *really* should be binary now.

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -i $KHARMADIR/pars/sane.par parthenon/time/nlim=5 >log_restart_1.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.init.phdf

sleep 1

$KHARMADIR/run.sh -r torus.out1.00000.rhdf parthenon/time/nlim=5 >log_restart_2.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.restart.phdf

# compare.py allows for small (5e-10) difference
pyharm-diff torus.out0.final.init.phdf torus.out0.final.restart.phdf -o compare_restart
