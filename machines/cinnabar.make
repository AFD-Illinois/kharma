CC = h5pcc
#CC = /opt/phdf5-intel/h5pcc

ifneq (,$(findstring GCC,$(shell $(CC) --version)))
	CFLAGS = -std=gnu99 -O3 -march=core-avx2 -mtune=core-avx2 -flto -fopenmp -funroll-loops -mcmodel=large
endif

ifneq (,$(findstring icc,$(shell $(CC) --version)))
	CFLAGS = -std=gnu99 -xCORE-AVX2 -O3 -qopenmp -ipo
	GSL_DIR = /opt/gsl-intel/
	MATH_LIB =
endif
