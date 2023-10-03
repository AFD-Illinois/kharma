#!/bin/bash

for dir in */
do
  cd $dir
  if [ -f ./run.sh ]; then
    echo "Running $dir"
    ./run.sh
  fi
  cd -
done
