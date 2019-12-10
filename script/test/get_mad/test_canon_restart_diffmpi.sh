#!/bin/bash

source ../test_common.sh

SZ=$1
# Must be just a name for now
OUT_DIR=results_torus_$SZ
RESOURCE_DIR=~/test-resources

# Restart file
RESTART=$RESOURCE_DIR/restart_${SZ}_gold.h5
GRID=$RESOURCE_DIR/grid_${SZ}_gold.h5

# Calculate stop time based on desired run length
TI=$(h5ls -dS $RESTART/t | tail -1)
TIME=1.0
TF=$( echo "$TI + $TIME" | bc -l )
echo "Starting at " $TI ", Ending at " $TF

# Initial clean and make of work area
rm -rf build_archive param.dat harm
make_harm_here torus

set_problem_size $SZ 128 128
# Output every step until TF
set_run_dbl DTd $TIME
set_run_dbl tf $TF

# Prep the output directory
rm -rf $OUT_DIR
mkdir -p $OUT_DIR/restarts $OUT_DIR/dumps
cp $RESTART $OUT_DIR/restarts/
cd $OUT_DIR/restarts/
ln -s $(basename $RESTART) restart.last
cd -

# FIRST RUN: 1 proc
set_cpu_topo 1 1 1
make_harm_here torus
sleep 1

echo "Restarting with 1 proc..."
run_harm $OUT_DIR firsttime
echo "Done!"

# Save output dump.  Might be whatever number so we save it the dumb way
cd $OUT_DIR
for file in dumps/dump_*.h5
do
  mv $file ./last_dump_gold.h5
done
rm -rf dumps
cd ..

sleep 1

# SECOND RUN: 16 procs
set_cpu_topo 2 2 4
make_harm_here torus

echo "Restarting with 4 procs..."
run_harm $OUT_DIR secondtime
echo "Done!"

# Ensure naming and grid for analysis
cp $GRID $OUT_DIR/dumps/grid.h5
mv $OUT_DIR/dumps/dump_*.h5 $OUT_DIR/dumps/dump_00000001.h5

verify torus_$SZ
