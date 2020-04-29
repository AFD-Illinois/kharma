#!/bin/bash

./run.sh -i test/bondi_convergence/bondic16.par &>test/bondi_convergence/log.out
python scripts/l1norm.py bondi.out1.00000.phdf bondi.out1.00001.phdf
./run.sh -i test/bondi_convergence/bondic32.par &>>test/bondi_convergence/log.out
python scripts/l1norm.py bondi.out1.00000.phdf bondi.out1.00001.phdf
./run.sh -i test/bondi_convergence/bondic64.par &>>test/bondi_convergence/log.out
python scripts/l1norm.py bondi.out1.00000.phdf bondi.out1.00001.phdf
#./run.sh -i test/bondi_convergence/bondic128.par &>>test/bondi_convergence/log.out
#python scripts/l1norm.py bondi.out1.00000.phdf bondi.out1.00001.phdf
