#!/bin/bash

# Bash script testing determinism of problem initialization and first steps

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -i $KHARMADIR/pars/sane.par debug/archive_parameters=false perturbation/u_jitter=0 parthenon/time/nlim=5 \
                    >log_reinit_1.txt 2>&1

mv torus.out1.final.rhdf torus.out1.final.first.rhdf

#$KHARMADIR/run.sh -r torus.out1.00000.rhdf parthenon/time/nlim=5
$KHARMADIR/run.sh -i $KHARMADIR/pars/sane.par debug/archive_parameters=false perturbation/u_jitter=0 parthenon/time/nlim=5 \
                    >log_reinit_1.txt 2>&1

mv torus.out1.final.rhdf torus.out1.final.second.rhdf

# This one's a clear case.  Binary or bust
h5diff --exclude-path=/Info torus.out1.final.first.rhdf torus.out1.final.second.rhdf
# And that's the exit code.  One and done.