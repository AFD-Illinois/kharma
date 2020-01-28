/*
 * pack.c
 *
 * Guarantee orderings of Kokkos arrays for writing
 */

#include "grid.hpp"
#include "hdf5_utils.hpp"

#include "decs.hpp"

using namespace std;

/**
 * Write a Grid-sized single variable (n1,n2,n3) to a file
 */
template <typename T>
void pack_write_scalar(H5F *outf, const Grid &G, const T in, const char *name, const hsize_t hdf5_type, const bool write_ghosts=false)
{
    // Decide automatically whether we've got a bulk-only or ghost grid
    void *out;
    auto range = G.h_bulk_0();
    int offset;

    if(in.extent(0) == G.n1 &&
        in.extent(1) == G.n2 &&
        in.extent(2) == G.n3) {

        range = G.h_bulk_0();
        out = calloc(G.n1 * G.n2 * G.n3, sizeof(hdf5_type));
        offset = 0;

    } else if (in.extent(0) == G.gn1 &&
                in.extent(1) == G.gn2 &&
                in.extent(2) == G.gn3) {
        if (write_ghosts) {
            range = G.h_all_0();
            out = calloc(G.gn1 * G.gn2 * G.gn3, sizeof(hdf5_type));
            offset = 0;
        } else {
            range = G.h_bulk_0();
            out = calloc(G.n1 * G.n2 * G.n3, sizeof(hdf5_type));
            offset = G.ng;
        }
    } else {
        throw std::invalid_argument("Input array is wrong size for grid");
    }

    // TODO can these be consolidated?
    if (hdf5_type == H5T_IEEE_F64LE)
    {
        Kokkos::parallel_for("pack_double_scalar", range,
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                ((double *)out)[G.i3(i,j,k,write_ghosts)] = (double) in(i+offset, j+offset, k+offset);
            }
        );
    }
    else if (hdf5_type == H5T_IEEE_F32LE)
    {
        Kokkos::parallel_for("pack_float_scalar", range,
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                ((float *)out)[G.i3(i,j,k,write_ghosts)] = (float) in(i+offset, j+offset, k+offset);
            }
        );
    }
    else if (hdf5_type == H5T_STD_I32LE)
    {
        Kokkos::parallel_for("pack_int_scalar", range,
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                ((int *)out)[G.i3(i,j,k,write_ghosts)] = (int) in(i+offset, j+offset, k+offset);
            }
        );
    }
    else
    {
        throw std::invalid_argument("Passed unsupported scalar type");
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
void pack_write_vector(H5F *outf, const Grid &G, const T in, const int len, const char *name, const hsize_t hdf5_type, const bool write_ghosts=false)
{
    // Decide automatically whether we've got a bulk-only or ghost grid
    void *out;
    auto range = G.h_bulk_0();
    int offset;

    if(in.extent(0) == G.n1 &&
        in.extent(1) == G.n2 &&
        in.extent(2) == G.n3) {

        range = G.h_bulk_0();
        out = calloc(G.n1 * G.n2 * G.n3 * len, sizeof(hdf5_type));
        offset = 0;

    } else if (in.extent(0) == G.gn1 &&
                in.extent(1) == G.gn2 &&
                in.extent(2) == G.gn3) {
        if (write_ghosts) {
            range = G.h_all_0();
            out = calloc(G.gn1 * G.gn2 * G.gn3 * len, sizeof(hdf5_type));
            offset = 0;
        } else {
            range = G.h_bulk_0();
            out = calloc(G.n1 * G.n2 * G.n3 * len, sizeof(hdf5_type));
            offset = G.ng;
        }
    } else {
        throw std::invalid_argument("Input array is wrong size for grid");
    }

    // TODO can these be consolidated?
    if (hdf5_type == H5T_IEEE_F64LE)
    {
        Kokkos::parallel_for("pack_double_vec", range,
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                for (int p = 0; p < len; ++p)
                    ((double *)out)[G.i4(i,j,k,p,write_ghosts)] = (double) in(i+offset, j+offset, k+offset, p);
            }
        );
    }
    else if (hdf5_type == H5T_IEEE_F32LE)
    {
        Kokkos::parallel_for("pack_float_vec", range,
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                for (int p = 0; p < len; ++p)
                    ((float *)out)[G.i4(i,j,k,p,write_ghosts)] = (float) in(i+offset, j+offset, k+offset, p);
            }
        );
    }
    else if (hdf5_type == H5T_STD_I32LE)
    {
        Kokkos::parallel_for("pack_int_vec", range,
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                for (int p = 0; p < len; ++p)
                    ((int *)out)[G.i4(i,j,k,p,write_ghosts)] = (int) in(i+offset, j+offset, k+offset, p);
            }
        );
    }
    else
    {
        throw std::invalid_argument("Passed unsupported scalar type");
    }

    hsize_t fdims[] = {G.n1tot, G.n2tot, G.n3tot, len};
    hsize_t fstart[] = {G.n1start, G.n2start, G.n3start, 0};
    hsize_t fcount[] = {G.n1, G.n2, G.n3, len}; // = mdims since this was packed above
    hsize_t mstart[] = {0, 0, 0, 0};

    outf->write_array(out, name, 4, fdims, fstart, fcount, fcount, mstart, hdf5_type);

    free(out);
}
