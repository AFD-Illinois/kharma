# KHARMA
KHARMA is an implementation of the HARM GRMHD algorithm in C++ based on the Parthenon AMR infrastructure, using Kokkos for parallelism and GPU support.  It is implemented via extensible "packages," which in theory make it easy to add or swap components representing different physical processes.

The project is capable of most GRMHD functions found in e.g. [iharm3d](https://github.com/AFD-Illinois/iharm3d). Support for adaptive mesh refinement is planned but not yet imlemented.

# Building

## Prerequisites
First, be sure to check out all of KHARMA's dependencies by running
```bash
$ git submodule update --init --recursive
```
This will grab KHARMA's two direct dependencies (as well as some incidental things):
1. The [Parthenon](https://github.com/lanl/parthenon) AMR framework from LANL (accompanying [documentation](https://github.com/lanl/parthenon/tree/develop/docs)).  Note KHARMA actually uses a [fork](https://github.com/bprather/parthenon) of Parthenon, which exists mostly to change the default Parthenon coordiante system to KHARMA's `GRCoordinates`, with suitable include paths pointing back into the KHARMA repository, and makes minimal other changes to the Parthenon codebase.  This fork isn't planned to be permanent.

2. The [Kokkos](https://github.com/kokkos/kokkos) performance-portability library, originally from SNL.  Many pieces of how KHARMA and Parthenon work are best explained in the Kokkos [wiki](https://github.com/kokkos/kokkos/wiki) and [tutorials](https://github.com/kokkos/kokkos-tutorials).  Parthenon includes a list of their names for Kokkos functions in their [developer guide](https://github.com/lanl/parthenon/blob/develop/docs/development.md).

The dependencies KHARMA needs from the system are ~exactly the same as Parthenon and Kokkos:
1. A C++14 compliant compiler with OpenMP (tested on several of GCC >= 7, Intel >= 19, nvc++ (formerly PGI) >= 20.11)
2. MPI of some sort
3. Parallel HDF5 compiled against said MPI
4. (for GPU) CUDA >= 10.2 and a supported C++ compiler as a backend.  More GPU/HPC hardware backends will become available as they are added to Kokkos.

All of these should come packaged, come with installers, or be available as modules on larger systems -- except parallel HDF5.  Luckily it is quite easy to compile manually (instructions and compile script forthcoming), and the installation location can be specified with the `PREFIX_PATH` variable when building KHARMA, as described below.

## Compiling
Generally, on systems with a parallel HDF5 module, one can then run the following to compile for CPU with OpenMP:
```bash
./make.sh clean
```
And (possibly) the following to compile for GPU with CUDA:
```bash
./make.sh clean cuda
```

If (when) these fail, take a look at the make.sh source, which is mostly an interface to automatically set `cmake` parameters.  In many other cases, you should be able to get away with:
```bash
PREFIX_PATH=/absolute/path/to/phdf5 HOST_ARCH=CPUVER ./make.sh clean
```
or
```bash
PREFIX_PATH=/absolute/path/to/phdf5 HOST_ARCH=CPUVER DEVICE_ARCH=GPUVER ./make.sh clean cuda
```

Where `CPUVER` and `GPUVER` are the strings used by Kokkos to denote a particular architecture & set of compile flags, e.g. "SKX" for Skylake-X, "HSW" for Haswell, or "AMDAVX" for Ryzen/EPYC processors.  When in doubt, omitting the variable defaults to "HSW" on unrecognized Intel machines, and "AMDAVX" on AMD machines.  A list of many architecture strings is provided in make.sh, and a full (usually) up-to-date list is kept in the Kokkos [documentation](https://github.com/kokkos/kokkos/wiki/Compiling).  (Note `make.sh` needs only the portion of the flag *after* `Kokkos_ARCH_`).

If you need to specify multiple custom-installed dependencies (e.g. CUDA), you can set PREFIX_PATH="/path/to/one;/path/to/two".  PREFIX_PATH does not support spaces in paths, becuase shell escapes are hard.

# Running
Run a particular problem with e.g.
```bash
$ ./kharma.host -i pars/orszag_tang.par
```
KHARMA benefits from certain runtime environment variables and CPU pinning, which I've attempted to include in the wrapper `run.sh`. (Not to be confused with `run.sb`, a submit script for some SLURM batch systems.) YMMV.

KHARMA takes no compile-time options, so all the parameters for a simulation are provided by this input "deck."  Several sample inputs corresponding to standard tests and astrophysical/EHT-relevant systems are included in `pars/`. Note convention is to end a parameter file with `.par` (this is required for autodetection by `pyHARM` for example).

KHARMA will attempt to guess most parameters if they are not specified, e.g. boundary conditions and coordinate sizes for simulations in spherical polar coordinates, or interior boundary locations for black hole simulations based on keeping 5 zones inside the event horizon.  Most of the guessing is done in `kharma.cpp` and the defaults are mostly specified in `grmhd/grmhd.cpp`.

# Hacking
Much of KHARMA's design (function names, signatures, etc) reflects being a "package" and "driver" in the Parthenon framework.  The main KHARMA package is "GRMHD" (`grmhd/grmhd.cpp`), which implements HARM's primitive<->conserved calculations as well as calculating its HLLE fluxes.  The "HARMDriver" (`harm.cpp`) lists out the tasks needed to advance the fluid by a step, specifying when to call which function in GRMHD.

Eventually more packages (and potentially more drivers) will be added, as KHARMA gains capabilities (passives, tracer particles, radiation, etc) and algorithms (face-centered B, PCP methods).
## Header functions
KHARMA's implementation is split between header-defined inline functions and body-defined functions.  The general rule is that header functions operate on a single zone, whereas body functions operate on the whole fluid state.
Header functions are declared `KOKKOS_INLINE_FUNCTION`, which allows them to be compiled for the host and device and therefore called from inside a Kokkos loop, which may run on either.  They generally take the coordinates `k,j,i` of the zone in question, and either the pointer to a full Kokkos "View" (array) which they will index (called "global" functions in comments), or a small array corresponding to the particular values at that zone ("immediate" functions).  This leads to a lot of overloaded functions with similar implementations, but allows combining several operations into a loop without writing back the results until the end.
