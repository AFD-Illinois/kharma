# This makefile is lifted wholesale from a Kokkos example.
# Most customizations/options are in a surrounding shell script,
# to avoid unintended consequences with Kokkos' kind of intense make infra

KHARM_PATH := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
KOKKOS_PATH ?= $(KHARM_PATH)/kokkos

SRC_DIR := $(KHARM_PATH)/kharm

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=%.o)

KOKKOS_CXX_STANDARD=c++14
CXXFLAGS = -O3 -I$(SRC_DIR)
LDFLAGS ?=

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  KOKKOS_DEVICES=Cuda,OpenMP
  KOKKOS_ARCH=BDW,Volta70
  CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper --expt-extended-lambda
  EXE = $(addsuffix .cuda, $(shell basename $(SRC_DIR)))
  INC += -I$(CUDA_PATH)
  LDFLAGS += -L$(CUDA_PATH)/lib64 -lcusparse
else
  KOKKOS_DEVICES=OpenMP
  KOKKOS_ARCH=HSW
  CXX = /usr/bin/h5c++
  EXE = $(addsuffix .host, $(shell basename $(SRC_DIR)))
endif

INC += -I$(KHARM_PATH)/HighFive/include

# Mpich/Boost MPI (TODO generalize/machinefile)
LDFLAGS += -L/usr/lib64/mpich/lib -lboost_mpi-mt
INC += -I/usr/include/mpich-x86_64/

# Machine-specific overrides
HOST := $(shell hostname)
ifneq (,$(findstring stampede2,$(HOST)))
	-include $(KHARM_PATH)/machines/stampede2.make
endif
-include $(MAKEFILE_PATH)/machines/$(HOST).make

# Link with compiler to avoid confusion
LINK ?= $(CXX)

# Define >=1 target before including Kokkos Makefile
default: $(EXE)

clean:
	@rm -f KokkosCore_config.* *.a *.o *.host *.cuda

include $(KOKKOS_PATH)/Makefile.kokkos

DEPFLAGS = -M

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

%.o:$(SRC_DIR)/%.cpp $(KOKKOS_CPP_DEPENDS) |$(BUILD_DIR)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(INC) -c $< -o $@


