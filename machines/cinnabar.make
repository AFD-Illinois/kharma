ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  KOKKOS_DEVICES=Cuda,OpenMP
  KOKKOS_ARCH=HSW,Kepler35
else
  KOKKOS_DEVICES=OpenMP
  KOKKOS_ARCH=HSW
endif
