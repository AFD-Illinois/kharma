# KHARMA Tests

Since all KHARMA parameters are determined at runtime, testing KHARMA is relatively easy,
and many different tests are defined changing different options

Tests are housed in folders, each containing a bash script `run.sh` to be run by CI, and a
python script `check.py` to produce any relevant plots and check that the 

While tests sometimes use many meshblocks, they do not by default use more than 1 MPI
process.  This may change if MPI-related issues crop up requiring KHARMA-specific tests
(e.g. of problem initialization requiring all-to-all operations)

## (GR)MHD convergence tests

* Unmagnetized static Bondi accretion `bondi`
* MHD linear modes `mhdmodes`
* Komissarov shock tube tests `komissarov_shocks`

See pretty much any GRMHD code paper, but notably Gammie+ [2003](), 

## Electron transport convergence tests

* Hubble flow with energy source term `hubble`
* Noh shock heating `noh_shocks`

See Ressler+ [2015]()

## Regression tests

* MHD linear modes convergence using tiny (8x8x8) meshblocks
* State at 1M of a problem, vs state at 1M of a problem initialized from its first restart file

## Performance tests

* Coming once CI w/GPUs is set up
* torus_scaling.par input out to 100 steps, recording cycle=100 performance
  * Also record avg performance?

## Testing wishlist

* Linear modes in cylindrical or spherical coordinates, to test polar boundary effects efficiently
* Driven turbulence in 2D, for testing electrons in more realistic scenarios e.g. with floors
