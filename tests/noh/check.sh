#!/bin/bash

BASEDIR=.
PYHARMDIR=$HOME/Code/pyHARM

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyHARM

python3 $PYHARMDIR/scripts/kharma_convert.py *.phdf
python3 $BASEDIR/check.py . . 64,128,256,512,1024,2048,4096 1.666667
