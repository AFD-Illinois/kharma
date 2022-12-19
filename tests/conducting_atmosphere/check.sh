#!/bin/bash

# Run checks against analytic result for specified tests

. /home/vdhruv2/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

# Very small amplitude by default, preserve double precision
~/pyHARM/scripts/pyharm-convert --double *.phdf

RES2D="64,128,256,512"

conda activate base

fail=0

python3 check.py $RES2D "Conducting atmosphere" emhd2d || fail=1

exit $fail
