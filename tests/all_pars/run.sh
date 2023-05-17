#!/bin/bash
set -euo pipefail

for fil in ../../pars/*.par
do
  ../../run.sh -n 1 -i $fil parthenon/time/nlim=2
  rm *.{hst,phdf,rhdf,xdmf}
done
