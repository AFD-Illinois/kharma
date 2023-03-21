#!/bin/bash
set -euo pipefail

BASE=../..

# Full run to test stability to completion
$BASE/run.sh -i $BASE/pars/bz_monopole.par debug/verbose=1 parthenon/output0/single_precision_output=false >log_bz_monopole_full.txt 2>&1

# At *least* check divB
pyharm-check-basics bz_monopole.out0.final.phdf

# Take 1 step to look for early signs of non-fatal instabilities
$BASE/run.sh -i $BASE/pars/bz_monopole.par parthenon/time/nlim=1 parthenon/output0/dt=0.0 parthenon/output0/single_precision_output=false >log_bz_monopole_step.txt 2>&1

# Check is for plots only!
python ./check.py
