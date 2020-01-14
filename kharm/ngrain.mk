KOKKOS_PATH ?= $(HOME)/libs/kokkos

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
SRC_DIR := $(dir $(MAKEFILE_PATH))/../src/

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=%.o)

default: build
	echo "Build Complete"

# Need KOKKOS_DEVICES="Cuda" to compile for Nvidia card
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
  EXE = ngrain.cuda
  KOKKOS_DEVICES = "Cuda,OpenMP"
  KOKKOS_ARCH = "HSW,Kepler35"
  KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
  CXX = g++
  EXE = ngrain.host
  KOKKOS_DEVICES = "OpenMP"
  KOKKOS_ARCH = "HSW"
endif

CXXFLAGS = -O3 -I$(SRC_DIR) -Wall -funroll-loops -use_fast_math
LINK ?= $(CXX)
LDFLAGS ?= -O3 -funroll-loops

KOKKOS_USE_TPLS = hwloc

include $(KOKKOS_PATH)/Makefile.kokkos

DEPFLAGS = -M

LIB = 

build: $(EXE)

# Todo Kokkos assumes /lib is a good place to link.  This is not always true.
$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) -L/lib64 $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: 
	rm -f *.a *.o *.cuda *.host

# Compilation rules

%.o:$(SRC_DIR)/%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

