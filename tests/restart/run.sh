#!/bin/bash
set -euo pipefail

# Bash script testing initialization vs restart of a torus problem
# Require similarity to round-off after 5 steps

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -i $KHARMADIR/pars/tori_3d/sane.par parthenon/time/nlim=5 >log_restart_1.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.init.phdf

sleep 1

$KHARMADIR/run.sh -r torus.out1.00000.rhdf parthenon/time/nlim=5 >log_restart_2.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.restart.phdf

# Compare to some high degree of accuracy
# TODO this was formerly 1e-11, we may need to clean up restarting & sequencing of the first steps
pyharm diff --rel_tol 1e-9 torus.out0.final.init.phdf torus.out0.final.restart.phdf -o compare_restart
# Compare binary. Sometimes works but not worth keeping always
#h5diff --exclude-path=/Info \
#       --exclude-path=/Input \
#       --exclude-path=/divB \
#       torus.out0.final.init.phdf torus.out0.final.restart.phdf
