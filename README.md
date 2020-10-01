# KHARMA
KHARMA is a version of HARM in C++ based on the Parthenon AMR infrastructure, using Kokkos for
parallelism/GPU support.

This project is in early stages!
Thus it is *not* guaranteed to be operable as work progresses!  See
[iharm3D](https://github.com/AFD-Illinois/iharm3d) for a more stable version
of the same algorithm.

# Building
KHARMA is built with CMake.  This can be done manually with the CMake CLI or GUI,
or by looking through and editing the file `make.sh`, which has a sample set of
flags for different compilation modes and architectures.

After editing `make.sh` to taste, run:

```bash
$ git submodule update --init --recursive
$ ./make.sh clean
```

# Hacking
KHARMA is split between header-defined functions and 

# Parthenon mirror
For the moment, this code uses a mirror of parthenon from [here](https://github.com/bprather/parthenon),
which changes the default coordinate system to KHARMA's GRCoordinates, and makes a couple changes geared
toward GPU support.
