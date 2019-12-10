S2TARGET = knl

CC = h5pcc

ifneq (,$(findstring icc,$(shell $(CC) --version)))

	ifneq (,$(findstring 18.0,$(shell $(CC) --version)))
		GSL_DIR = /opt/apps/intel18/gsl/2.3
	else
		GSL_DIR = /opt/apps/intel17/gsl/2.3
	endif

	ifeq ($(S2TARGET),skx)
		CFLAGS = -xCORE-AVX512
	endif
	ifeq ($(S2TARGET),knl)
		CFLAGS = -xMIC-AVX512
	endif

	CFLAGS += -std=gnu11 -O3 -funroll-loops -ipo -qopenmp -qopt-prefetch=5
	MATH_LIB =
endif

ifneq (,$(findstring gcc,$(shell $(CC) --version)))

	GSL_DIR = /opt/apps/gcc7_1/gsl/2.3/

	CC = h5pcc -shlib
	
	ifeq ($(S2TARGET),knl)
		CFLAGS = -march=knl -mtune=knl
	endif
	ifeq ($(S2TARGET),skx)
		CFLAGS = -march=skylake-avx512 -mtune=skylake-avx512
	endif

	CFLAGS += -std=gnu11 -O3 -flto -fopenmp -funroll-loops
endif

# Notes on the above:
# Additional arguments that have been tried
#-fargument-noalias -qopt-threads-per-core=4
#-vec-threshold0
#-qopt-zmm-usage=high
# Report vectorization
#-qopt-report-phase=vec -qopt-report-file=vec.txt

