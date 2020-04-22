#!/bin/bash

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# Use GPU 0, 1 is flaky on cinnabar
export KOKKOS_DEVICE_ID=0

if [ -f kharma.cuda ]; then
  ./kharma.cuda --kokkos-device-id=0 "$@"
elif [ -f kharma.host ]; then
  ./kharma.host "$@"
else
  echo "KHARMA executable not found!"
fi
