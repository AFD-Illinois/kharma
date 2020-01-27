/*
 * pack.c
 *
 * Guarantee orderings of Kokkos arrays for writing
 */

#include "grid.hpp"
#include "hdf5_utils.hpp"

#include "decs.hpp"

/**
 * Write a Grid-sized single variable (n1,n2,n3) to a file
 */
template <typename T>
void pack_write_scalar(H5F *outf, const Grid &G, const Kokkos::View<T ***, Kokkos::HostSpace> in, const char *name, const hsize_t hdf5_type)
{
    void *out = calloc(G.n1 * G.n2 * G.n3, sizeof(hdf5_type));

    // Do some ungodly things with casts
    // Please keep this code away from children
    int ind = 0;
    if (hdf5_type == H5T_IEEE_F64LE)
    {
        for (int i = G.ng; i < G.n1 + G.ng; ++i)
            for (int j = G.ng; j < G.n2 + G.ng; ++j)
                for (int k = G.ng; k < G.n3 + G.ng; ++k)
                {
                    ((double *)out)[ind] = (double)in(i, j, k);
                    ind++;
                }
    }
    else if (hdf5_type == H5T_IEEE_F32LE)
    {
        for (int i = G.ng; i < G.n1 + G.ng; ++i)
            for (int j = G.ng; j < G.n2 + G.ng; ++j)
                for (int k = G.ng; k < G.n3 + G.ng; ++k)
                {
                    ((float *)out)[ind] = (float)in(i, j, k);
                    ind++;
                }
    }
    else if (hdf5_type == H5T_STD_I32LE)
    {
        for (int i = G.ng; i < G.n1 + G.ng; ++i)
            for (int j = G.ng; j < G.n2 + G.ng; ++j)
                for (int k = G.ng; k < G.n3 + G.ng; ++k)
                {
                    ((int *)out)[ind] = in(i, j, k);
                    ind++;
                }
    }
    else
    {
        fprintf(stderr, "Scalar type not supported!\n\n");
        exit(-1);
    }

    hsize_t fdims[] = {G.n1tot, G.n2tot, G.n3tot};
    hsize_t fstart[] = {G.n1start, G.n2start, G.n3start};
    hsize_t fcount[] = {G.n1, G.n2, G.n3}; // = mdims since this was packed above
    hsize_t mstart[] = {0, 0, 0};

    outf->write_array(out, name, 3, fdims, fstart, fcount, fcount, mstart, hdf5_type);

    free(out);
}

/**
 * Write a grid-sized vector (n1,n2,n3,len) to a file
 */
template <typename T>
void pack_write_vector(H5F *outf, const Grid &G, const Kokkos::View<T ****, Kokkos::HostSpace> in, const int len, const char *name, const hsize_t hdf5_type)
{
    void *out = calloc(G.n1 * G.n2 * G.n3 * len, sizeof(hdf5_type));

    int ind = 0;
    if (hdf5_type == H5T_IEEE_F64LE)
    {
        for (int i = G.ng; i < G.n1 + G.ng; ++i)
            for (int j = G.ng; j < G.n2 + G.ng; ++j)
                for (int k = G.ng; k < G.n3 + G.ng; ++k)
                    for (int mu = 0; mu < len; mu++)
                    {
                        ((double *)out)[ind] = (double)in(i, j, k, mu);
                        ind++;
                    }
    }
    else if (hdf5_type == H5T_IEEE_F32LE)
    {
        for (int i = G.ng; i < G.n1 + G.ng; ++i)
            for (int j = G.ng; j < G.n2 + G.ng; ++j)
                for (int k = G.ng; k < G.n3 + G.ng; ++k)
                    for (int mu = 0; mu < len; mu++)
                    {
                        ((float *)out)[ind] = (float)in(i, j, k, mu);
                        ind++;
                    }
    }
    else if (hdf5_type == H5T_STD_I32LE)
    {
        for (int i = G.ng; i < G.n1 + G.ng; ++i)
            for (int j = G.ng; j < G.n2 + G.ng; ++j)
                for (int k = G.ng; k < G.n3 + G.ng; ++k)
                    for (int mu = 0; mu < len; mu++)
                    {
                        ((int *)out)[ind] = in(i, j, k, mu);
                        ind++;
                    }
    }
    else
    {
        fprintf(stderr, "Scalar type not supported!\n\n");
        exit(-1);
    }

    hsize_t fdims[] = {G.n1tot, G.n2tot, G.n3tot, len};
    hsize_t fstart[] = {G.n1start, G.n2start, G.n3start, 0};
    hsize_t fcount[] = {G.n1, G.n2, G.n3, len}; // = mdims since this was packed above
    hsize_t mstart[] = {0, 0, 0, 0};

    outf->write_array(out, name, 4, fdims, fstart, fcount, fcount, mstart, hdf5_type);

    free(out);
}
