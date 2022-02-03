# KHARMA Tests

Since all KHARMA parameters are determined at runtime, testing KHARMA is relatively easy,
and many different tests are defined changing different options

Tests are housed in folders, each containing a bash script `run.sh` to be run by CI, and a
python script `check.py` to produce any relevant plots and check that the output matches
expectations.

While tests sometimes use many meshblocks, they do not by default use more than 1 MPI
process.  This may change if MPI-related issues crop up requiring KHARMA-specific tests
(e.g. of problem initialization requiring all-to-all operations).  All are designed to be
run on a single node in a reasonable amount of time (<1h).

Current and near-future planned tests are outlined below.

## (GR)MHD convergence tests

* Unmagnetized static Bondi accretion `bondi`
* MHD linear modes `mhdmodes`
* Komissarov shock tube tests `komissarov_shocks`
* BZ monopole stability test `bz_monopole`

See pretty much any GRMHD code paper, but notably Gammie+ [2003](https://doi.org/10.1086/374594).
Several variants of each test are run, using different coordinate systems, reconstruction, etc
to catch regressions in particular features.

Note that the BZ monopole test has 2 parts: a stability test running through to 100M, a test
outputting state after a single step.  Currently both are imaged in the same way, with the
first two images showing initial condition and single-step state, and the rest showing the
full 100M run at normal dump cadence.  Plots for this test show the primitive radial velocity
U1 since this in particular shows erratic behavior near the polar bound.

## Identity regression tests

* Near-identical output of the same problem evolved with different block geometry
* MHD linear modes convergence using tiny (8x8x8) meshblocks
* State at 1M of a problem run from initialization, vs state at 1M of a problem initialized
  from its first restart file

These are basic regression tests in MPI operation, catching smaller differences which wouldn't
necessarily show up in conversion

## Testing wishlist

* Record `torus_scaling.par` stepwise performance at step=100, due to lower systematics
  than early step average
* Linear modes in cylindrical and spherical coordinates, to test polar boundary effects efficiently
