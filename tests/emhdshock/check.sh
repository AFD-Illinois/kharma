#!/bin/bash

# Run checks against analytic result for specified tests

# Very small amplitude by default, preserve double precision
pyharm convert --double *.phdf

RES1D="256,512,1024,2048"

fail=0

python3 check.py $RES1D "EMHD shock" emhd1d || fail=1

exit $fail
