#!/bin/bash

rm -rf build
mkdir build
cd build

cmake3 ..

make -j

cp kharma/kharma ../kharma.host
