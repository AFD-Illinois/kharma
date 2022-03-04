#!/bin/bash

BASEDIR=.
PYHARMDIR=$HOME/Code/pyharm

. ~/libs/anaconda3/etc/profile.d/conda.sh
conda activate pyharm

pyharm-convert *.phdf
python3 $BASEDIR/check.py . . 64,128,256,512,1024,2048,4096 1.666667
