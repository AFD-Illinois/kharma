#!/bin/bash

for dir in */
do
  prob=${dir%?}
  cd $prob &>/dev/null
  if [ -f ./run.sh ]; then
    echo Running $prob
    exit_code=0
    ./run.sh >../log_${prob}.txt 2>&1 || exit_code=$?
    if [ $exit_code -ne 0 ]; then
      echo Test $prob FAIL
    else
      echo Test $prob PASS
    fi
  fi
  cd - &>/dev/null
done
