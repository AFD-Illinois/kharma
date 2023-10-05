#!/bin/bash
set -euo pipefail

# Bash script testing starting a simulation, then resizing it up

# Set paths
KHARMADIR=../..

# This at least stirs up the field slightly vs initialization
$KHARMADIR/run.sh -i $KHARMADIR/pars/tori_3d/sane.par parthenon/time/nlim=5 >log_resize_1.txt 2>&1

# We can only resize/restart from iharm3d-format files
pyharm convert --to_restart torus.out0.final.phdf

sleep 1

$KHARMADIR/run.sh -i $KHARMADIR/pars/restarts/resize_restart.par resize_restart/fname=torus.out0.final.h5 \
                  b_cleanup/always_solve=1 parthenon/time/nlim=5 \
                  >log_resize_2.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.restart.phdf

# Check divB on the re-meshed output
pyharm check-basics torus.out0.final.restart.phdf
