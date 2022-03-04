#!/bin/bash

# Run checks against analytic result for specified tests

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

pyharm-convert *.phdf

RES3D="16,24,32,48"
RES2D="16,24,32,48"

fail=0
python3 check.py $RES2D "EMHD mode in 2D, WENO5" emhd2d_weno 2d || fail=1
python3 check.py $RES2D "EMHD mode in 2D, linear/MC reconstruction" emhd2d_mc 2d || fail=1
python3 check.py $RES2D "EMHD mode in 2D, linear/VL reconstruction" emhd2d_vl 2d || fail=1

exit $fail
