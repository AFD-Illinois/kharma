__='
   This is the default license template.
   
   File: test_convergence_bondi2d.sh
   Author: bprather
   Copyright (c) 2020 bprather
   
   To edit this license information: Press Ctrl+Shift+P and press 'Create new License Template...'.
'

#!/bin/bash

# Note this script now assumes executable has been built,
# as there are no relevant compile-time variables.

source ../test_common.sh

# Must be just a name
OUT_DIR=results_bondi

rm -rf $OUT_DIR
mkdir -p $OUT_DIR

for n in 32 64 128 256 512
do
  echo "Running $n-squared 2D bondi problem"
  run_harm bondi $n $n 1 fmks
  echo "Done!"

  mkdir $OUT_DIR/dumps_${n}
  mv *.h5 *.xdmf $OUT_DIR/dumps_${n}/
done

# Run analysis automatically
./ana_convergence.sh bondi
