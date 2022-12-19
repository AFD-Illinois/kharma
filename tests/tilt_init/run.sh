#!/bin/bash

# Run default tilted problem to 5 steps
../../run.sh -i ../../pars/mad_tilt.par parthenon/time/nlim=5 debug/verbose=1 \
                parthenon/output0/single_precision_output=false \
                >log_tilt_init.txt 2>&1

# Image the first dump
python ./check.py

# Check some basics (divB) of the first dump
pyharm check-basics torus.out0.00000.phdf
