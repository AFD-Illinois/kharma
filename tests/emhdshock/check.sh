#!/bin/bash

# Run checks against analytic result for specified tests

. /home/vdhruv2/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

# Very small amplitude by default, preserve double precision
~/pyHARM/scripts/pyharm-convert --double *.phdf

RES1D="256,512,1024,2048"

conda activate base

fail=0

python3 check.py $RES1D "EMHD shock" emhd1d || fail=1

exit $fail
