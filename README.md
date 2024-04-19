# KHARMA
KHARMA is an implementation of the HARM scheme for gerneral relativistic magnetohydrodynamics (GRMHD) in C++.  It is based on the Parthenon AMR infrastructure, using Kokkos for parallelism and GPU support.  It is composed of modular "packages," which in theory make it easy to add or swap components representing different physical processes.

KHARMA is capable of closely matching other HARM implementations, e.g. [iharm3d](https://github.com/AFD-Illinois/iharm3d). However, it also extends the scheme with additional options for magnetic field transport, reconstruction, etc.  Notably, it implements a split face-centered CT scheme, allowing static and adaptive mesh refinement.

## Prerequisites
KHARMA requires that the system have a C++17-compliant compiler, MPI, and parallel HDF5.  All other dependencies are included as submodules, and can be checked out with `git` by running
```bash
$ git submodule update --init --recursive
```

When updating the KHARMA source code, you may also have to update the submodules with
```bash
$ git submodule update --recursive
```
Old submodules are a common cause of compile errors!

## Compiling
On directly supported systems, or systems with standard install locations, you may be able to run:
```bash
./make.sh clean [cuda hip sycl]
```
after a *successful* compile, subsequent invocations can omit `clean`.  If this command fails on supported machines (those with a file in `machines/`), please open an issue.  Broken builds aren't uncommon, as HPC machines change software all the time.

If running KHARMA on a new machine (or repairing the build on an old one), take a look at the [wiki page](https://github.com/AFD-Illinois/kharma/wiki/Building-KHARMA) describing the build system.

## Running
Run a particular problem with e.g.
```bash
$ ./kharma.host -i pars/tests/orszag_tang.par
```
note that *all* options are runtime.  The single KHARMA binary can run any of the parameter files in `pars/`, and indeed this is checked as a part of the regression tests.  Note you can still disable some sub-systems manually at compile time, and of course in that case the accompanying problems will crash.

KHARMA benefits from certain runtime environment variables and CPU pinning, included in a short wrapper script `run.sh`.  This script is provided mostly as an optional convenience, and an example of how to construct your own batch scripts for running KHARMA in production.  Other example batch scripts are in the `scripts/batch/` folder.

Further information can be found on the [wiki page](https://github.com/AFD-Illinois/kharma/wiki/Running-KHARMA).

## Hacking
KHARMA has some preliminary documentation for developers, hosted in its GitHub [wiki](https://github.com/AFD-Illinois/kharma/wiki).

## Licenses
KHARMA is made available under the BSD 3-clause license included in each file and in the file LICENSE at the root of this repository.

This repository also carries a substantial portion of the [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels), in the directory `external/kokkos-kernels`, which is provided under the license included in that directory.

Submodules of this repository are subject to their own licenses.
