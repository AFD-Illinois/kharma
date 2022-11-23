#!/bin/bash

# Run checks against analytic result for specified tests

. /home/vdhruv2/anaconda3/etc/profile.d/conda.sh

RES2D="32,64,128,256"

conda activate base

fail=0

python3 check.py $RES2D "Bondi viscous" emhd2d || fail=1

exit $fail
