#!/bin/bash
set -euo pipefail

exit_code=0

tilt_init() {
    # Run default tilted problem to 5 steps
    ../../run.sh -i ../../pars/tori_3d/mad_tilt.par parthenon/time/nlim=5 debug/verbose=1 \
                    parthenon/output0/single_precision_output=false \
                    parthenon/output0/variables=prims,jcon,fflag,pflag,divB \
                    $2 >log_tilt_init_${1}.txt 2>&1

    # Image the first dump
    python ./check.py

    check_code=0
    # Check some basics (divB) of the first dump
    pyharm check-basics torus.out0.final.phdf || check_code=$?

    if [[ $check_code != 0 ]]; then
        echo Tilt init test \"$3\" FAIL: $check_code
        exit_code=1
    else
        echo Tilt init test \"$3\" success
    fi
}

tilt_init cell "b_field/solver=flux_ct" "Cell-centered B"
tilt_init face "b_field/solver=face_ct" "Face-centered B"

exit $exit_code
