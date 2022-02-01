#!/bin/bash
set -euo pipefail

BASE=../..

# Full run to test stability to completion
$BASE/run.sh -i $BASE/pars/bz_monopole.par

# Take 1 step to look for early signs of non-fatal instabilities
$BASE/run.sh -i $BASE/pars/bz_monopole.par parthenon/time/nlim=1 parthenon/output0/dt=0.0
