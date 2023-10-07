#!/bin/bash
set -euo pipefail

# Skip testing the restarting & benchmark scripts
for folder in bondi electrons emhd shocks smr tests tori_2d tori_3d
do
  for fil in ../../pars/$folder/*.par
  do
    exit_code=0
    par=$(basename $fil)
    prob=${par%.*}
    ../../run.sh -n 1 -i $fil parthenon/time/nlim=2 &>log_${prob}.txt || exit_code=$?
    rm -f *.{hst,phdf,rhdf,xdmf}
    if [ $exit_code -ne 0 ]; then
      echo $par FAIL
    else
      echo $par PASS
    fi
  done
done
