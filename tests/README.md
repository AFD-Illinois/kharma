# KHARMA Tests

Since all KHARMA parameters are determined at runtime, testing KHARMA is relatively easy,
and many different tests are defined changing different options

Tests are housed in folders, each containing a bash script `run.sh` to perform any runs for
the test, and another, `check.sh` to veryify the results.  `check.sh` usually calls a
python script `check.py` to produce any relevant plots and check that the output matches
expectations.  Note that while `run.sh` will exit on the first failed run, `check.sh` runs
all checks, accumulating a single return value `0` for success or `1` if any check fails.

While tests sometimes use many meshblocks, they do not by default use more than 1 MPI
process.  This may change if MPI-related issues crop up requiring KHARMA-specific tests
(e.g. of problem initialization requiring all-to-all operations).  All are designed to be
run on a single node in a reasonable amount of time (<1h).

Current and near-future planned tests are outlined below.

## (GR)MHD convergence tests

* Unmagnetized static Bondi accretion `bondi`
* MHD linear modes `mhdmodes`
* Komissarov shock tube tests `komissarov_shocks`

Tests outlined in many code papers, notably Gammie+ [2003](https://doi.org/10.1086/374594).

## Electron transport convergence tests

* Hubble flow with energy source term `hubble`
* Noh shock heating `noh_shocks`

Tests outlined in Ressler+ [2015](https://doi.org/10.1093/mnras/stv2084).

## Regression tests

* State at 1M after initialization vs restarting a problem `init_vs_restart`
* Stability stress test `bz_monopole` for polar boundary conditions, high-B operation
* Restart from mid-run of a MAD simulation `get_mad`

Note that the BZ monopole test has 2 parts: a stability test running through to 100M, a test
outputting state after a single step.  Currently both are imaged in the same way, with the
first two images showing initial condition and single-step state, and the rest showing the
full 100M run at normal dump cadence.  Plots for this test show the primitive radial velocity
U1 since this in particular shows erratic behavior near the polar bound.

## Performance tests

* torus_scaling.par input with single block and 8 blocks, cycle=100
* Same with orszag_tang, mhdmodes

## Testing wishlist

* Linear modes in cylindrical or spherical coordinates, to test polar boundary effects efficiently
* Driven turbulence in 2D, for testing electrons in more realistic scenarios e.g. with floors
* Test for unique random perturbations in all blocks
