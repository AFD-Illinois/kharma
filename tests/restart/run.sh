#!/bin/bash
set -euo pipefail

# Bash script testing initialization vs restart of a torus problem
# Require binary similarity after 5 steps

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -i $KHARMADIR/pars/tori_3d/sane.par parthenon/time/nlim=5 >log_restart_1.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.init.phdf

sleep 1

$KHARMADIR/run.sh -r torus.out1.00000.rhdf parthenon/time/nlim=5 >log_restart_2.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.restart.phdf

# compare.py allows for small (5e-10) difference
#pyharm-diff torus.out0.final.init.phdf torus.out0.final.restart.phdf -o compare_restart
# Compare binary
h5diff --exclude-path=/Info \
       --exclude-path=/Input \
       --exclude-path=/divB \
       torus.out0.final.init.phdf torus.out0.final.restart.phdf
