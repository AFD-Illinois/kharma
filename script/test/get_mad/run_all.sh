#!/bin/bash

# Only run if the correct restart files are present
RESOURCE_DIR=~/test-resources

if [[ -f $RESOURCE_DIR/restart_192_gold.h5 && -f $RESOURCE_DIR/grid_192_gold.h5 ]]; then
  ./test_canon_restart_diffmpi.sh 192
fi
if [[ -f $RESOURCE_DIR/restart_288_gold.h5 && -f $RESOURCE_DIR/grid_288_gold.h5 ]]; then     
  ./test_canon_restart_diffmpi.sh 288
fi
