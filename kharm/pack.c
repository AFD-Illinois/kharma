/*
 * pack.c
 *
 * HARM-specific output calls to hdf5_utils library
 */

#include "hdf5_utils.h"

#include "decs.h"

// Reverse and write a backwards-index N{3,2,1}-size array of doubles (GridDouble) to a file
void pack_write_scalar(double in[N3+2*NG][N2+2*NG][N1+2*NG], const char* name, hsize_t hdf5_type)
{
  void *out = calloc(N1*N2*N3, sizeof(hdf5_type));

  // Do some ungodly things with casts
  // Please keep this code away from children
  int ind = 0;
  if (hdf5_type == H5T_IEEE_F64LE) {
    ZLOOP_OUT {
      ((double*) out)[ind] = in[k][j][i];
      ind++;
    }
  } else if (hdf5_type == H5T_IEEE_F32LE) {
    ZLOOP_OUT {
      ((float*) out)[ind] = (float) in[k][j][i];
      ind++;
    }
  } else {
    fprintf(stderr, "Scalar type not supported!\n\n");
    exit(-1);
  }

  hsize_t fdims[] = {N1TOT, N2TOT, N3TOT};
  hsize_t fstart[] = {global_start[0], global_start[1], global_start[2]};
  hsize_t fcount[] = {N1, N2, N3}; // = mdims since this was packed above
  hsize_t mstart[] = {0, 0, 0};

  hdf5_write_array(out, name, 3, fdims, fstart, fcount, fcount, mstart, hdf5_type);

  free(out);
}

// Reverse and write a backwards-index N{3,2,1}-size array of ints (GridInt) to a file
void pack_write_int(int in[N3+2*NG][N2+2*NG][N1+2*NG], const char* name)
{
  int *out = calloc(N1*N2*N3, sizeof(int));

  int ind = 0;
  ZLOOP_OUT {
    out[ind] = in[k][j][i];
    ind++;
  }

  hsize_t fdims[] = {N1TOT, N2TOT, N3TOT};
  hsize_t fstart[] = {global_start[0], global_start[1], global_start[2]};
  hsize_t fcount[] = {N1, N2, N3}; // = mdims since this was packed above
  hsize_t mstart[] = {0, 0, 0};

  hdf5_write_array(out, name, 3, fdims, fstart, fcount, fcount, mstart, H5T_STD_I32LE);

  free(out);
}

// Reverse and write a backwards-index len,N{3,2,1}-size array of ints (GridVector or GridPrim) to a file
void pack_write_vector(double in[][N3+2*NG][N2+2*NG][N1+2*NG], int len, const char* name, hsize_t hdf5_type)
{
  void *out = calloc(N1*N2*N3*len, sizeof(hdf5_type));

  int ind = 0;
  if (hdf5_type == H5T_IEEE_F64LE) {
    ZLOOP_OUT {
      for (int mu=0; mu < len; mu++) {
        ((double*) out)[ind] = in[mu][k][j][i];
        ind++;
      }
    }
  } else if (hdf5_type == H5T_IEEE_F32LE) {
    ZLOOP_OUT {
      for (int mu=0; mu < len; mu++) {
        ((float*) out)[ind] = (float) in[mu][k][j][i];
        ind++;
      }
    }
  } else {
    fprintf(stderr, "Scalar type not supported!\n\n");
    exit(-1);
  }

  hsize_t fdims[] = {N1TOT, N2TOT, N3TOT, len};
  hsize_t fstart[] = {global_start[0], global_start[1], global_start[2], 0};
  hsize_t fcount[] = {N1, N2, N3, len};
  hsize_t mstart[] = {0, 0, 0, 0};

  hdf5_write_array(out, name, 4, fdims, fstart, fcount, fcount, mstart, hdf5_type);

  free(out);
}

// Reverse and write a backwards-index N{2,1}-size axisymmetric scalar (i.e. gdet or similar)
void pack_write_axiscalar(double in[N2+2*NG][N1+2*NG], const char* name, hsize_t hdf5_type)
{
  void *out = calloc(N1*N2*N3, sizeof(hdf5_type)); // Still write full phi for compatibility

  int ind = 0;
  if (hdf5_type == H5T_IEEE_F64LE) {
    ZLOOP_OUT {
      ((double*) out)[ind] = in[j][i];
      ind++;
    }
  } else if (hdf5_type == H5T_IEEE_F32LE) {
    ZLOOP_OUT {
      ((float*) out)[ind] = (float) in[j][i];
      ind++;
    }
  } else {
    fprintf(stderr, "Scalar type not supported!\n\n");
    exit(-1);
  }

  hsize_t fdims[] = {N1TOT, N2TOT, N3TOT};
  hsize_t fstart[] = {global_start[0], global_start[1], global_start[2]};
  hsize_t fcount[] = {N1, N2, N3}; // = mdims since this was packed above
  hsize_t mstart[] = {0, 0, 0};

  hdf5_write_array(out, name, 3, fdims, fstart, fcount, fcount, mstart, hdf5_type);

  free(out);
}

// Reverse and write an axisymmetric NDIMxNDIM tensor (i.e. Gcov/con)
void pack_write_Gtensor(double in[NDIM][NDIM][N2+2*NG][N1+2*NG], const char* name, hsize_t hdf5_type)
{
  void *out = calloc(N1*N2*N3*NDIM*NDIM, sizeof(hdf5_type));

  int ind = 0;
  if (hdf5_type == H5T_IEEE_F64LE) {
    ZLOOP_OUT {
      DLOOP2 {
        ((double*) out)[ind] = in[mu][nu][j][i];
        ind++;
      }
    }
  } else if (hdf5_type == H5T_IEEE_F32LE) {
    ZLOOP_OUT {
      DLOOP2 {
        ((float*) out)[ind] = (float) in[mu][nu][j][i];
        ind++;
      }
    }
  } else {
    fprintf(stderr, "Scalar type not supported!\n\n");
    exit(-1);
  }

  hsize_t fdims[] = {N1TOT, N2TOT, N3TOT, NDIM, NDIM};
  hsize_t fstart[] = {global_start[0], global_start[1], global_start[2], 0, 0};
  hsize_t fcount[] = {N1, N2, N3, NDIM, NDIM}; // = mdims since this was packed above
  hsize_t mstart[] = {0, 0, 0, 0, 0};

  hdf5_write_array(out, name, 5, fdims, fstart, fcount, fcount, mstart, hdf5_type);

  free(out);
}
