#!/bin/bash

# Bash script testing starting a simulation, then resizing it up

# Set paths
KHARMADIR=../..

$KHARMADIR/run.sh -i $KHARMADIR/pars/sane.par parthenon/time/nlim=5 >log_resize_1.txt 2>&1

pyharm convert --to_restart torus.out0.final.phdf

sleep 1

$KHARMADIR/run.sh -i ../../pars/resize_restart >log_resize_2.txt 2>&1

mv torus.out0.final.phdf torus.out0.final.restart.phdf

# Check divB on the re-meshed output
pyharm-check-basics torus.out0.final.restart.phdf
