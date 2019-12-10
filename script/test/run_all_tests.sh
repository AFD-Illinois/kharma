#!/bin/bash

#!/bin/bash

RESULTS_DIR=$PWD/test-results

# Keep a separate folder of just results
rm -rf $RESULTS_DIR
mkdir $RESULTS_DIR

for test_problem in *
do
    # Skip non-directories, otherwise cd in
    [ ! -d $test_problem ]&& continue
    cd $test_problem

    bash run_all.sh > test_output.txt

    mkdir $RESULTS_DIR/$test_problem
    cp test_output.txt $RESULTS_DIR/$test_problem
    cp -r results_*/plots $RESULTS_DIR/$test_problem
    cp results_*/*.txt $RESULTS_DIR/$test_problem
    cp results_*/*.png $RESULTS_DIR/$test_problem

    cd ..
done