# Common functions for test scripts
# Mostly changing compile-time and runtime parameters

# TODO move more common restart-test functions here

# Usage: set_compile_int <param> <value>
set_compile_int () {
  sed -i -e "s/$1 [0-9]\+/$1 $2/g" build_archive/parameters.h
}

# TODO accommodate more values/spaces/etc.
# Usage: set_run_dbl <param> <value>
set_run_dbl () {
  sed -i -e "s/$1 = [0-9]\+\\.[0-9]\+/$1 = $2/g" param.dat
}

# Usage: set_run_int <param> <value>
set_run_int () {
  sed -i -e "s/$1 = [0-9]\+/$1 = $2/g" param.dat
}

set_problem_size () {
  set_compile_int N1TOT $1
  set_compile_int N2TOT $2
  set_compile_int N3TOT $3
}

set_cpu_topo () {
  set_compile_int N1CPU $1
  set_compile_int N2CPU $2
  set_compile_int N3CPU $3
  export HARM_NPROC=$(( $1 * $2 * $3 ))
  export IBRUN_TASKS_PER_NODE=$HARM_NPROC
  export OMP_NUM_THREADS=$(( $(nproc --all) / $HARM_NPROC ))
  #echo "Exporting $OMP_NUM_THREADS threads"
}

# This allows a separate make target if we want
make_harm_here () {
  [ -z ${HARM_BASE_DIR+x} ] && HARM_BASE_DIR=../../..
  [ -z ${HARM_MAKE_JOBS+x} ] && HARM_MAKE_JOBS=$(nproc --all)
  make -f $HARM_BASE_DIR/makefile -j$HARM_MAKE_JOBS PROB=$1 debug
  # Use default param.dat if none is present in test dir
  if [ ! -f param.dat ]; then
    [ -f $HARM_BASE_DIR/prob/$1/param.dat ] && cp $HARM_BASE_DIR/prob/$1/param.dat .
    [ -f $HARM_BASE_DIR/prob/$1/param_sane.dat ] && cp $HARM_BASE_DIR/prob/$1/param_sane.dat ./param.dat
  fi
}

# Usage: run_harm $OUT_DIR name 
run_harm() {
  if command -v ibrun >/dev/null 2>&1; then
    ibrun ./harm -p param.dat -o $1 > $1/out_$2.txt 2> $1/err_$2.txt
  else
    mpirun -n $HARM_NPROC ./harm -p param.dat -o $1 > $1/out_$2.txt 2> $1/err_$2.txt
  fi
}

verify() {
  PROB=$1

  cd results_$PROB

  if [ $PROB == "mhdmodes" ]
  then
    LAST_DUMP=dumps/dump_00000005.h5
  else
    LAST_DUMP=dumps/dump_00000001.h5
  fi

  cp ../../../analysis/*.py .
  python3 plot_diff.py $LAST_DUMP last_dump_gold.h5 differences_$PROB

  # Print verification to file
  exec > verification_$PROB.txt 2>&1
  set -x

  grep restart out_firsttime.txt
  grep restart out_secondtime.txt

  # Diff first dumps for a sanity check
  h5diff first_dump_gold.h5 dumps/dump_00000000.h5
  h5diff first_restart_gold.h5 restarts/restart_00000001.h5

  # Diff last dumps
  h5diff last_dump_gold.h5 $LAST_DUMP
  # These are useful for debugging
  #h5diff --delta=1e-12 last_dump_gold.h5 $LAST_DUMP
  #h5diff --delta=1e-8 last_dump_gold.h5 $LAST_DUMP
}
