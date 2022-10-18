#!/bin/bash

# Run checks against analytic result for specified tests

# . ~/libs/anaconda3/etc/profile.d/conda.sh
. /home/vdhruv2/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

# Very small amplitude by default, preserve double precision
pyharm-convert --double *.phdf

RES2D="32,64,128,256"

fail=0
python3 check.py $RES2D "EMHD mode in 2D, linear/MC reconstruction" emhd2d_mc 2d || fail=1
python3 check.py $RES2D "EMHD mode in 2D, WENO5" emhd2d_weno 2d || fail=1

python3 check.py $RES2D "EMHD mode in 2D, higher order terms enabled" emhd2d_higher_order || fail=1

exit $fail
