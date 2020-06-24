#!/bin/bash

OUT_DIR=results_$1
BASEDIR=../../..

# Make plot dir
PLOT_DIR=$OUT_DIR/plots
mkdir $PLOT_DIR

# Prep plot dir
# TODO keep ana scripts there?
cp plot_convergence_$1.py $PLOT_DIR
cp $BASEDIR/script/analysis/*.py $PLOT_DIR

# CD and run script
cd $PLOT_DIR

python3 plot_convergence_$1.py

