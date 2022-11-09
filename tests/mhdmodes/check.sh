#!/bin/bash

# Run checks against analytic result for specified tests

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

RES3D="16,24,32,48"
RES2D="32,64,128,256"

fail=0
#python check.py $RES3D "entropy mode in 3D" entropy || exit_code=$?
python check.py $RES3D "slow mode in 3D" slow || exit_code=$?
python check.py $RES3D "Alfven mode in 3D" alfven || exit_code=$?
python check.py $RES3D "fast mode in 3D" fast || exit_code=$?

#python check.py $RES3D "entropy mode in 3D, linear/MC reconstruction" entropy_mc || exit_code=$?
#python check.py $RES3D "entropy mode in 3D, linear/VL reconstruction" entropy_vl || exit_code=$?

#python check.py $RES3D "slow mode 3D, ImEx Explicit" slow_imex || exit_code=$?
#python check.py $RES3D "Alfven mode 3D, ImEx Explicit" alfven_imex || exit_code=$?
#python check.py $RES3D "fast mode 3D, ImEx Explicit" fast_imex || exit_code=$?

#python check.py $RES3D "slow mode in 3D, ImEx Semi-Implicit" slow_imex_semi || exit_code=$?
#python check.py $RES3D "Alfven mode in 3D, ImEx Semi-Implicit" alfven_imex_semi || exit_code=$?
#python check.py $RES3D "fast mode in 3D, ImEx Semi-Implicit" fast_imex_semi || exit_code=$?

#python check.py $RES3D "slow mode in 3D, ImEx Implicit" slow_imex_im || exit_code=$?
#python check.py $RES3D "Alfven mode in 3D, ImEx Implicit" alfven_imex_im || exit_code=$?
#python check.py $RES3D "fast mode in 3D, ImEx Implicit" fast_imex_im || exit_code=$?

# 2D MODES
#python check.py $RES2D "fast mode in 2D, WENO5" fast2d 2d || exit_code=$?
#python check.py $RES2D "fast mode in 2D, linear/MC reconstruction" fast_mc 2d || exit_code=$?
#python check.py $RES2D "fast mode in 2D, linear/VL reconstruction" fast_vl 2d || exit_code=$?

exit $exit_code
