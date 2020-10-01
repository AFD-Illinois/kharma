#!/bin/bash

NAME=$1
VAR=$2
mkdir -p frames_${NAME}_${VAR}
cd frames_${NAME}_${VAR}

parallel -P 8 python ../scripts/quick_plot.py {} ../pars/${NAME}.par $VAR frame_{#} ::: ../${NAME}.*.phdf
