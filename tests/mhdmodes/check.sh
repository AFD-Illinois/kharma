#!/bin/bash

# Run checks against analytic result for specified tests

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

RES3D="16,24,32,48"
RES2D="32,64,128,256"

fail=0
python3 check.py $RES3D "entropy mode in 3D" entropy || fail=1
python3 check.py $RES3D "slow mode in 3D" slow || fail=1
python3 check.py $RES3D "Alfven mode in 3D" alfven || fail=1
python3 check.py $RES3D "fast mode in 3D" fast || fail=1

python3 check.py $RES3D "entropy mode in 3D, linear/MC reconstruction" entropy_mc || fail=1
python3 check.py $RES3D "entropy mode in 3D, linear/VL reconstruction" entropy_vl || fail=1

python3 check.py $RES3D "slow mode in 3D, classic algo" slow_imex || fail=1
python3 check.py $RES3D "Alfven mode in 3D, classic algo" alfven_imex || fail=1
python3 check.py $RES3D "fast mode in 3D, classic algo" fast_imex || fail=1

python3 check.py $RES3D "slow mode in 3D, classic algo" slow_imex_im || fail=1
python3 check.py $RES3D "Alfven mode in 3D, classic algo" alfven_imex_im || fail=1
python3 check.py $RES3D "fast mode in 3D, classic algo" fast_imex_im || fail=1

#python3 check.py $RES2D "fast mode in 2D, WENO5" fast2d 2d || fail=1
#python3 check.py $RES2D "fast mode in 2D, linear/MC reconstruction" fast_mc 2d || fail=1
#python3 check.py $RES2D "fast mode in 2D, linear/VL reconstruction" fast_vl 2d || fail=1

exit $fail
