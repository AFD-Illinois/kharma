# Machine files

## Writing a machine file

`make.sh` sources a series of machine-specific
definitions from the machines/ directory.

If the host isn't listed, the CPU & GPU arch will be guessed

Example Kokkos_ARCH options:
CPUs: BDW, SKX, KNL, AMDAVX, ZEN2, ZEN3, POWER9
ARM: ARMV80, ARMV81, ARMV8_THUNDERX2, A64FX
HOST_ARCH=

GPUs: VOLTA70, TURING75, AMPERE80, HOPPER90, VEGA90A, INTEL_GEN
DEVICE_ARCH=

Compilers to use.
C_NATIVE=
CXX_NATIVE=

Less common options:
PREFIX_PATH=

EXTRA_FLAGS

CXXFLAGS
CFLAGS
