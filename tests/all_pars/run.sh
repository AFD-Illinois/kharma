#!/bin/bash
set -euo pipefail

# Skip testing the restarting & benchmark scripts
for folder in bondi electrons emhd shocks smr tests tori_2d tori_3d
do
  for fil in ../../pars/$folder/*.par
  do
    ../../run.sh -n 1 -i $fil parthenon/time/nlim=2
    rm -f *.{hst,phdf,rhdf,xdmf}
  done
done
