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

# Tolerance is generous to make test short, while testing solver is at least approaching correct vals
# Real simulation resizes should use tighter abs_tolerance than this
$KHARMADIR/run.sh -i $KHARMADIR/pars/restarts/resize_restart.par resize_restart/fname=torus.out0.final.h5 \
                  b_cleanup/abs_tolerance=1e-7 b_cleanup/always_solve=1 parthenon/time/nlim=1 \
                  parthenon/output0/single_precision_output=false >log_resize_2.txt 2>&1

# Check divB on the re-meshed output.  Tolerate some divB as we set the tolerance loosely above for speed
pyharm check-basics --allowed_divb=1e-9 resize_restart.out0.00000.phdf
