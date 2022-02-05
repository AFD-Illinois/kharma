#!/bin/bash

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyHARM

# Set paths
KHARMADIR=../..

python3 $KHARMADIR/scripts/compare.py torus.out0.final.init.phdf torus.out0.final.restart.phdf init_vs_restart