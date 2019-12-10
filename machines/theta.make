THTARGET = knl

CC = cc

ifneq (,$(findstring icc,$(shell $(CC) --version)))
	CFLAGS = -xMIC-AVX512 -std=gnu11 -O3 -funroll-loops -ipo -qopenmp -qopt-prefetch=5
	MATH_LIB =
        GSL_DIR = /home/bprather/.local/
endif

ifneq (,$(findstring gcc,$(shell $(CC) --version)))
	GSL_DIR = ??
	CFLAGS = -march=knl -mtune=knl -std=gnu11 -O3 -flto -fopenmp -funroll-loops
endif

# Notes on the above:
# Additional arguments that have been tried
#-fargument-noalias -qopt-threads-per-core=4
#-vec-threshold0
#-qopt-zmm-usage=high
# Report vectorization
#-qopt-report-phase=vec -qopt-report-file=vec.txt

