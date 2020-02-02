#!/bin/bash

if ! [ -z "$1" ]; then
  export OMP_NUM_THREADS=$1
fi

# Alternately
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

if [ -f kharm.cuda ]; then
  ./kharm.cuda
elif [ -f kharm.host ]; then
  ./kharm.host
else
  echo "K/HARM executable not found!"
fi
