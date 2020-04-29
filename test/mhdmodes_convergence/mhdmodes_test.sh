#!/bin/bash

./run.sh -i test/mhdmodes_convergence/mhdc16.par &>test/mhdmodes_convergence/log.out
python scripts/l1norm.py mhdmodes.out1.00000.phdf mhdmodes.out1.00001.phdf
./run.sh -i test/mhdmodes_convergence/mhdc32.par &>>test/mhdmodes_convergence/log.out
python scripts/l1norm.py mhdmodes.out1.00000.phdf mhdmodes.out1.00001.phdf
./run.sh -i test/mhdmodes_convergence/mhdc64.par &>>test/mhdmodes_convergence/log.out
python scripts/l1norm.py mhdmodes.out1.00000.phdf mhdmodes.out1.00001.phdf
# ./run.sh -i test/mhdmodes_convergence/mhdc128.par &>^test/mhdmodes_convergence/log.out
# python scripts/l1norm.py mhdmodes.out1.00000.phdf mhdmodes.out1.00001.phdf