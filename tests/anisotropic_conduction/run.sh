#!/bin/bash
set -euo pipefail

../../run.sh -i ../../pars/emhd/anisotropic_conduction.par parthenon/time/tlim=5

python make_plots.py .
