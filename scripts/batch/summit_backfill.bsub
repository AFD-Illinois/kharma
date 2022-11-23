#!/bin/bash
#BSUB -P AST171
#BSUB -W 24:00
#BSUB -J KHARMA_killable
# debug or batch
#BSUB -q killable

# ALWAYS check the following 3 lines before submitting the script

#BSUB -nnodes 1
NNODES=1

KHARMA_DIR=~/kharma

#==============

# This ensures we load the modules we did for making kharma
ARGS=$(cat $KHARMA_DIR/make_args)
source $KHARMA_DIR/machines/incite.sh

# Make bsub behave like sbatch, moving to CWD
# But don't cd anywhere under $HOME, that would be bad
if [[ "$LS_SUBCWD" !=  *"home"* ]]; then
  cd $LS_SUBCWD
fi

# Stuff for posterity
date
echo "Job run on nodes:"
jsrun -n $NNODES -r 1 hostname
#module list

# Tell OpenMP explicitly what it's dealing with
# This also ensures that if we set e.g. NUM_THREADS=6,
# they will be spread 1/core
# Remember to change this if using 7 cores!
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=24

# Copy & correct final regular restart if present, otherwise start fresh
# h5ls will return 0 (false) if it CAN read the file, so negate that here
last_restart=$(ls -t . | grep torus.out1.0 | head -1)
h5ls $last_restart
if [[ $last_restart != "" && $? == 0 ]]
then
  echo "Restart file"
  cp $last_restart torus.out1.final.rhdf
  module load python
  sleep 1 
  python ~/kharma/scripts/fix_restart.py torus.out1.final.rhdf torus.out1.final_corrected.rhdf
  COMMAND="-r torus.out1.final_corrected.rhdf b_field/initial_cleanup=false"
else
  echo "No restart file"
  COMMAND="-i torus.par"
fi

# In order to run anything on the compute nodes, we must specify
# how we wish to slice them up.  This is done with arguments to
# 'jsrun', which takes the place of 'mpirun' on Summit
# -n # resource sets (== MPI tasks for us)
# -r # rs per node
# -a # MPI task per rs
# -g # GPU per rs
# -c # CPU cores (physical) per rs (total of 42=(22-1)*2, but 7 is such an awkward number)
# -b binding strat *within* a resource set

# The "smpiargs" argument is used to tell Spectrum MPI we're GPU-aware
jsrun --smpiargs="-gpu" -n $(($NNODES * 6)) -r 6 -a 1 -g 1 -c 6 -d packed -b packed:6 \
	$KHARMA_DIR/kharma.cuda $COMMAND
