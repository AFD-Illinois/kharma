ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  KOKKOS_DEVICES=Cuda,OpenMP
  KOKKOS_ARCH=BDW,Maxwell52
else
  KOKKOS_DEVICES=OpenMP
  KOKKOS_ARCH=BDW
endif