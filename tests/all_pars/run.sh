#!/bin/bash
set -euo pipefail

for fil in ../../pars/*.par
do
  ../../run.sh -i $fil parthenon/time/nlim=2
done
