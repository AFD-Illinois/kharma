/*
 * io.hpp: 
 */

#pragma once

#include "decs.hpp"
#include "grid.hpp"

#include "pack.hpp"

#include "hdf5_utils.hpp"
#include "hdf5.h"

#define HDF_STR_LEN 20

/**
 * Dump state to a file.
 * Templated to accommodate array orderings
 */
template <typename T>
void dump(Grid &G, T P, Parameters params, std::string fname, bool write_double=false)
{
    std::cout << "DUMP " << fname << std::endl;

    H5F *outf = new H5F();

    outf->create(fname.c_str());
    //Can use H5T_VARIABLE for any-length strings. But not compatible with parallel IO
    hid_t string_type = outf->make_str_type(HDF_STR_LEN);

    // Write header
    outf->set_directory("/");
    outf->make_directory("header");
    outf->set_directory("/header/");

    outf->write_single_val(VERSION, "version", string_type);
    //   outf->write_single_val(&has_electrons, "has_electrons", H5T_STD_I32LE);

    // TODO adapt these to options
    outf->write_single_val("MINKOWSKI", "metric", string_type);
    outf->write_single_val("WENO", "reconstruction", string_type);

    outf->write_single_val(&(G.n1tot), "n1", H5T_STD_I32LE);
    outf->write_single_val(&(G.n2tot), "n2", H5T_STD_I32LE);
    outf->write_single_val(&(G.n3tot), "n3", H5T_STD_I32LE);

    outf->write_single_val(&(G.nvar), "n_prims", H5T_STD_I32LE);
    // TODO when grids get passives
    // TODO rename these in spec?
    int n_passive = 0;
    outf->write_single_val(&n_passive, "n_prims_passive", H5T_STD_I32LE);
    //   outf->write_str_list(varNames, "prim_names", HDF_STR_LEN, G.nvar);

    // Sometimes, C++ is very frustrating.  Structs are *stupid*.
    // for (std::pair<std::string, double> param : params) {
    //     outf->write_single_val(&(param.second), param.first.c_str(), H5T_IEEE_F64LE);
    // }
    //   outf->write_single_val(&gam, "gam", H5T_IEEE_F64LE);
    //   outf->write_single_val(&game, "gam_e", H5T_IEEE_F64LE);
    //   outf->write_single_val(&gamp, "gam_p", H5T_IEEE_F64LE);
    //   outf->write_single_val(&tptemin, "tptemin", H5T_IEEE_F64LE);
    //   outf->write_single_val(&tptemax, "tptemax", H5T_IEEE_F64LE);
    //   outf->write_single_val(&fel0, "fel0", H5T_IEEE_F64LE);
    //   outf->write_single_val(&cour, "cour", H5T_IEEE_F64LE);
    //   outf->write_single_val(&tf, "tf", H5T_IEEE_F64LE);
    //   outf->add_units("tf", "code");

    // Geometry
    outf->make_directory("geom");
    outf->set_directory("/header/geom/");

    outf->write_single_val(&(G.startx1), "startx1", H5T_IEEE_F64LE);
    outf->write_single_val(&(G.startx2), "startx2", H5T_IEEE_F64LE);
    outf->write_single_val(&(G.startx3), "startx3", H5T_IEEE_F64LE);
    outf->write_single_val(&(G.dx1), "dx1", H5T_IEEE_F64LE);
    outf->write_single_val(&(G.dx2), "dx2", H5T_IEEE_F64LE);
    outf->write_single_val(&(G.dx3), "dx3", H5T_IEEE_F64LE);
    int n_dim = NDIM;
    outf->write_single_val(&n_dim, "n_dim", H5T_STD_I32LE);
    // TODO NON-CART COORDS
    // #if METRIC == MKS
    // #if DEREFINE_POLES
    //   outf->make_directory("mmks");
    //   outf->set_directory("/header/geom/mmks/");
    //   outf->write_single_val(&poly_xt, "poly_xt", H5T_IEEE_F64LE);
    //   outf->write_single_val(&poly_alpha, "poly_alpha", H5T_IEEE_F64LE);
    //   outf->write_single_val(&mks_smooth, "mks_smooth", H5T_IEEE_F64LE);
    // #else
    //   outf->make_directory("mks");
    //   outf->set_directory("/header/geom/mks/");
    // #endif
    //   outf->write_single_val(&Rin, "r_in", H5T_IEEE_F64LE);
    //   outf->write_single_val(&Rout, "r_out", H5T_IEEE_F64LE);
    //   outf->write_single_val(&Rhor, "r_eh", H5T_IEEE_F64LE);
    //   // I don't need this in code but it's in the spec to output it
    //   double z1 = 1 + pow(1 - a*a,1./3.)*(pow(1+a,1./3.) + pow(1-a,1./3.));
    //   double z2 = sqrt(3*a*a + z1*z1);
    //   double Risco = 3 + z2 - sqrt((3-z1)*(3 + z1 + 2*z2));
    //   outf->write_single_val(&Risco, "r_isco", H5T_IEEE_F64LE);
    //   outf->write_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
    //   outf->write_single_val(&a, "a", H5T_IEEE_F64LE);
    // #endif

    outf->set_directory("/");

    int is_full_dump = 1;
    outf->write_single_val(&is_full_dump, "is_full_dump", H5T_STD_I32LE);

    // TODO NON-FLUID STATE
    //   outf->write_single_val(&t, "t", H5T_IEEE_F64LE);
    //   outf->add_units("t", "code");
    //   outf->write_single_val(&dt, "dt", H5T_IEEE_F64LE);
    //   outf->add_units("dt", "code");
    //   outf->write_single_val(&nstep, "n_step", H5T_STD_I32LE);
    //   outf->write_single_val(&dump_cnt, "n_dump", H5T_STD_I32LE);
    //   outf->write_single_val(&DTd, "dump_cadence", H5T_IEEE_F64LE);
    //   outf->write_single_val(&DTf, "full_dump_cadence", H5T_IEEE_F64LE);

    // Write primitive variables
    if (write_double) {
        pack_write_vector(outf, G, P, G.nvar, "prims", H5T_IEEE_F64LE);
    } else {
        pack_write_vector(outf, G, P, G.nvar, "prims", H5T_IEEE_F32LE);
    }
    outf->add_units("prims", "code");

    // TODO Preserve git commit or tag of the run -- see Makefile
#ifdef GIT_VERSION
    outf->write_single_val(QUOTE(GIT_VERSION), "git_version", string_type);
#endif

    outf->close();
    delete outf;
}
