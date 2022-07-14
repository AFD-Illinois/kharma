#!/bin/bash

# This one's a clear case.  Binary or bust
h5diff --exclude-path=/Info torus.out1.final.first.rhdf torus.out1.final.second.rhdf
