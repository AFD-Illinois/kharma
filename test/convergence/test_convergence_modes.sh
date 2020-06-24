#!/bin/bash

source ../test_common.sh

# Must be just a name for now
OUT_DIR=results_modes

# Initial clean and make of work area
BASEDIR=../../..
rm -rf build_archive param.dat harm
make_harm_here mhdmodes

# In case we didn't clean up after test_restart_diffmpi
set_cpu_topo 2 2 4

rm -rf $OUT_DIR
mkdir -p $OUT_DIR

for n in 16 32 64
do

  set_problem_size $n $n $n

  sleep 1

  make_harm_here mhdmodes

  for i in 1 2 3
  do

    set_run_int nmode $i

    sleep 1

    echo "Running size $n mode $i..."
    run_harm $OUT_DIR "${n}_${i}"
    echo "Done!"

    mv $OUT_DIR/dumps $OUT_DIR/dumps_${n}_${i}
    rm -rf $OUT_DIR/restarts

  done
done

# Run analysis automatically
./ana_convergence.sh modes
