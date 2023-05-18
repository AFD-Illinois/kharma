#!/bin/bash

../../run.sh -i ../../pars/anisotropic_conduction.par

python make_plots.py .
