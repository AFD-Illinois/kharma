#!/bin/bash

# Run default tilted problem to 5 steps
../../run.sh -i ../../pars/mad_tilt.par parthenon/time/nlim=5 debug/verbose=1 \
                >log_tilt_init.txt 2>&1

# Image the first dump so we can visually ensure it looks okay. NO checks/fails!
# TODO should gold or gen with pyharm...
python ./check.py