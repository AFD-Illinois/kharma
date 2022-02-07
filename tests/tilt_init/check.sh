#!/bin/bash

# Image the first dump to ensure tilted disk is created properly

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyHARM

python3 ./check.py
