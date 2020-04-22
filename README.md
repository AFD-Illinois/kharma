# k-harm
KHARMA: A version of HARM in C++ based on the Parthenon framework, using Kokkos for
parallelism/GPU support.

This project is in early stages of the Parthenon port!
Thus it is *not* guaranteed to be operable as work progresses!  See
[iharm3D](https://github.com/AFD-Illinois/iharm3d) for a more stable version
of the same algorithm.

# Building
KHARMA is built with CMake.  This can be done manually with the CMake CLI or GUI,
or by looking through and editing the file `make.sh`, which has a sample set of
flags for different compilation modes and architectures.

**Note** that you need to set the Parthenon option NUMBER_GHOST_CELLS to 4 (for WENO reconstruction),
and the option PAR_LOOP_LAYOUT to "MDRANGE_LOOP" for better performance.