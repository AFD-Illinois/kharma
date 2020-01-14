# k-harm
K/HARM: Implementing HARM in C++ with Kokkos

This project is in early stages of a large refactor from C to C++/Kokkos!
Thus it is *not* guaranteed to be operable as work progresses!  See
[iharm3D](https://github.com/AFD-Illinois/iharm3d) for a more stable version
of the same algorithm.

# Building
While just running `make` should work, the provided script make.sh builds in a
separate (git-ignored) directory and copies back the resulting executable,
for cleanliness.