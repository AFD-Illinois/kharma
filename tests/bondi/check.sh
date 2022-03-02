#!/bin/bash

# Run checks against analytic result for specified tests

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

res="32,48,64,96,128"
python check.py $res "in 2D, FMKS coordinates" fmks || fail=1
python check.py $res "in 2D, MKS coordinates" mks || fail=1
# TODO EKS in pyHARM
#python check.py $res "in 2D, EKS coordinates" eks || fail=1
python check.py $res "in 2D, linear recon with MC limiter" linear_mc || fail=1
python check.py $res "in 2D, linear recon with VL limiter" linear_vl || fail=1

python check.py $res "in 2D, with Imex driver" imex || fail=1
python check.py $res "in 2D, with implicit stepping" imex_im || fail=1

exit $fail
