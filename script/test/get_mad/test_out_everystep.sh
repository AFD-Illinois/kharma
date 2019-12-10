#!/bin/bash

source ../test_common.sh

# Must be just a name for now
OUT_DIR=results_torus
RESOURCE_DIR=~/test-resources

SZ=288
# Restart file
RESTART=$RESOURCE_DIR/restart_${SZ}_gold.h5

# Calculate stop time based on desired run length
TI=$(h5ls -dS $RESTART/t | tail -1)
TIME=0.06
TF=$( echo "$TI + $TIME" | bc -l )
echo "Starting at " $TI ", Ending at " $TF

# Initial clean and make of work area
rm -rf build_archive param.dat harm
make_harm_here torus

set_problem_size $SZ 128 128
# Output every step until TF
set_run_dbl DTd 0.0
set_run_dbl tf $TF

# Prep the output directory
rm -rf $OUT_DIR
mkdir -p $OUT_DIR/restarts
cp $RESTART $OUT_DIR/restarts/
cd $OUT_DIR/restarts/
ln -s $(basename $RESTART) restart.last
cd -

# FIRST RUN: 1 proc
set_cpu_topo 1 1 1
make_harm_here torus

echo "Restarting with 1 proc..."

sleep 1

run_harm $OUT_DIR firsttime
echo "Done!"

# Save dumps

cd $OUT_DIR
mv dumps dumps_one
cd ..

sleep 1

# SECOND RUN: 4 procs
set_cpu_topo 2 2 1
make_harm_here torus

echo "Restarting with 4 procs..."
run_harm $OUT_DIR secondtime
echo "Done!"
