/* 
 *  File: iharm_restart.cpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "iharm_restart.hpp"

#include "hdf5_utils.h"
#include "mpi.hpp"

#include <sys/stat.h>
#include <ctype.h>

// First boundary sync
void outflow_x1(const GRCoordinates& G, GridVars P, int n1, int n2, int n3);
void polar_x2(const GRCoordinates& G, GridVars P, int n1, int n2, int n3);
void periodic_x3(const GRCoordinates& G, GridVars P, int n1, int n2, int n3);

using namespace Kokkos;

// TODO
// At least check that Rin,Rout match
// Actually look at Rin,Rout,gamma and (re)build the Coordinates and mesh on them
// Re-gridding algorithm
// Start with multiple meshes i.e. find full file dimensions, where to start reading

void ReadIharmRestartHeader(std::string fname, std::unique_ptr<ParameterInput>& pin)
{
    // Read the restart file and set parameters that need to be specified at early loading
    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");

    // Get size
    int n1file, n2file, n3file;
    hdf5_read_single_val(&n1file, "n1", H5T_STD_I32LE);
    hdf5_read_single_val(&n2file, "n2", H5T_STD_I32LE);
    hdf5_read_single_val(&n3file, "n3", H5T_STD_I32LE);
    pin->SetInteger("parthenon/mesh", "nx1", n1file);
    pin->SetInteger("parthenon/mesh", "nx2", n2file);
    pin->SetInteger("parthenon/mesh", "nx3", n3file);

    double gam, cour, t, dt;
    hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);
    hdf5_read_single_val(&cour, "cour", H5T_IEEE_F64LE);
    hdf5_read_single_val(&t, "t", H5T_IEEE_F64LE);
    hdf5_read_single_val(&dt, "dt", H5T_IEEE_F64LE);

    pin->SetReal("GRMHD", "gamma", gam);
    //pin->SetReal("GRMHD", "cfl", cour);
    pin->SetReal("parthenon/time", "dt", dt);
    pin->SetReal("parthenon/time", "start_time", t);

    if (hdf5_exists("a")) {
        double a, hslope, Rout;
        hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
        hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
        hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
        pin->SetReal("coordinates", "a", a);
        pin->SetReal("coordinates", "hslope", hslope);
        pin->SetReal("coordinates", "r_out", Rout);

        // Sadly restarts did not record MKS vs FMKS
        // Guess if not specified in parameter file
        pin->SetString("coordinates", "base", "spherical_ks");
        if (!pin->DoesParameterExist("coordinates", "transform")) {
            pin->SetString("coordinates", "transform", "funky");
        }

        pin->SetReal("parthenon/mesh", "x2min", 0.0);
        pin->SetReal("parthenon/mesh", "x2max", 1.0);
        pin->SetReal("parthenon/mesh", "x3min", 0.0);
        pin->SetReal("parthenon/mesh", "x3max", 2*M_PI);

        // All MKS sims had the usual bcs
        pin->SetString("parthenon/mesh", "ix1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ox1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ix2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ox2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ix3_bc", "periodic");
        pin->SetString("parthenon/mesh", "ox3_bc", "periodic");
    } else if (hdf5_exists("x1Min")) {
        double x1min, x2min, x3min;
        double x1max, x2max, x3max;
        hdf5_read_single_val(&x1min, "x1Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x1max, "x1Max", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x2min, "x2Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x2max, "x2Max", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x3min, "x3Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x3max, "x3Max", H5T_IEEE_F64LE);
        pin->SetReal("parthenon/mesh", "x1min", x1min);
        pin->SetReal("parthenon/mesh", "x1max", x1max);
        pin->SetReal("parthenon/mesh", "x2min", x2min);
        pin->SetReal("parthenon/mesh", "x2max", x2max);
        pin->SetReal("parthenon/mesh", "x3min", x3min);
        pin->SetReal("parthenon/mesh", "x3max", x3max);

        // Sims like this were of course cartesian
        pin->SetString("coordinates", "base", "cartesian_minkowski");
        pin->SetString("coordinates", "transform", "null");

        // All cartesian sims were periodic
        pin->SetString("parthenon/mesh", "ix1_bc", "periodic");
        pin->SetString("parthenon/mesh", "ox1_bc", "periodic");
        pin->SetString("parthenon/mesh", "ix2_bc", "periodic");
        pin->SetString("parthenon/mesh", "ox2_bc", "periodic");
        pin->SetString("parthenon/mesh", "ix3_bc", "periodic");
        pin->SetString("parthenon/mesh", "ox3_bc", "periodic");
    } else {
        throw std::runtime_error("Unknown restart file format!");
    }

    // End HDF5 reads
    hdf5_close();
}

double ReadIharmRestart(MeshBlock *pmb, GRCoordinates G, GridVars P, std::string fname)
{
    IndexDomain domain = IndexDomain::interior;
    // Full mesh size
    hsize_t n1tot = pmb->pmy_mesh->mesh_size.nx1;
    hsize_t n2tot = pmb->pmy_mesh->mesh_size.nx2;
    hsize_t n3tot = pmb->pmy_mesh->mesh_size.nx3;
    // Our block size, start, and bounds for the GridVars
    hsize_t n1 = pmb->cellbounds.ncellsi(domain);
    hsize_t n2 = pmb->cellbounds.ncellsj(domain);
    hsize_t n3 = pmb->cellbounds.ncellsk(domain);
    // TODO starting location in the global?  Only accessible/calculable without refinement

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");

    hid_t string_type = hdf5_make_str_type(20);
    char version[20];
    hdf5_read_single_val(version, "version", string_type);
    if (MPIRank0())
    {
        cout << "Restarting from " << fname << ", file version " << version << endl << endl;
    }

    // Get tf here and not when reading the header, since using this
    // value *itself* depends on a parameter, "use_tf"
    Real tf;
    hdf5_read_single_val(&tf, "tf", H5T_IEEE_F64LE);

    hsize_t nfprim;
    if(hdf5_exists("game")) {
        nfprim = NPRIM + 2;
    } else {
        nfprim = NPRIM;
    }

    // Declare known sizes for outputting primitives
    static hsize_t fdims[] = {nfprim, n3tot, n2tot, n1tot};
    static hsize_t fcount[] = {nfprim, n3, n2, n1};

    // TODO figure out single restart -> multi mesh
    //hsize_t fstart[] = {0, global_start[2], global_start[1], global_start[0]};
    hsize_t fstart[] = {0, 0, 0, 0};

    // These are dimensions for memory,
    static hsize_t mdims[] = {nfprim, n3, n2, n1};
    static hsize_t mstart[] = {0, 0, 0, 0};

    Real *ptmp = new Real[nfprim*n3*n2*n1];
    hdf5_read_array(ptmp, "p", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);

    // End HDF5 reads
    hdf5_close();

    auto Phost = P.GetHostMirror();

    // Host-side copy into the mirror
    Kokkos::parallel_for("copy_restart_state", Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<4>>({0, ks, js, is}, {NPRIM, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_VARS {
            Phost(p, k, j, i) = ptmp[p*n3*n2*n1 + (k-ks)*n2*n1 + (j-js)*n1 + (i-is)];
        }
    );
    delete[] ptmp;

    // Deep copy to device
    P.DeepCopy(Phost);
    Kokkos::fence();

    // Every iharm3d sim we'd be restarting had these
    // TODO Switch to a KHARMA bounds sync
    outflow_x1(G, P, n1, n2, n3);
    polar_x2(G, P, n1, n2, n3);
    periodic_x3(G, P, n1, n2, n3);

    return tf;
}

// Boundary functions for the initial sync
// Many possible speed improvements but these are run once & shouldn't exist at all
void outflow_x1(const GRCoordinates& G, GridVars P, int n1, int n2, int n3)
{
    Kokkos::parallel_for("outflow_x1_l", MDRangePolicy<Rank<3>>({NGHOST, NGHOST, 0}, {n3+NGHOST, n2+NGHOST, NGHOST}),
        KOKKOS_LAMBDA_3D {
            int iz = NGHOST;

            PLOOP P(p, k, j, i) = P(p, k, j, iz);

            double rescale = G.gdet(Loci::center, j, iz) / G.gdet(Loci::center, j, i);
            P(prims::B1, k, j, i) *= rescale;
            P(prims::B2, k, j, i) *= rescale;
            P(prims::B3, k, j, i) *= rescale;
        }
    );
    Kokkos::parallel_for("outflow_x1_r", MDRangePolicy<Rank<3>>({NGHOST, NGHOST, n1+NGHOST}, {n3+NGHOST, n2+NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
            int iz = n1 + NGHOST - 1;

            PLOOP P(p, k, j, i) = P(p, k, j, iz);

            double rescale = G.gdet(Loci::center, j, iz) / G.gdet(Loci::center, j, i);
            P(prims::B1, k, j, i) *= rescale;
            P(prims::B2, k, j, i) *= rescale;
            P(prims::B3, k, j, i) *= rescale;
        }
    );
}

void polar_x2(const GRCoordinates& G, GridVars P, int n1, int n2, int n3)
{
    Kokkos::parallel_for("reflect_x2_l", MDRangePolicy<Rank<3>>({NGHOST, 0, 0}, {n3+NGHOST, NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
          // Reflect across NG.  The zone j is (NG-j) prior to reflection,
          // set it equal to the zone that far *beyond* NG
          int jrefl = NGHOST + (NGHOST - j) - 1;
          PLOOP P(p, k, j, i) = P(p, k, jrefl, i);

          P(prims::u2, k, j, i) *= -1.;
          P(prims::B2, k, j, i) *= -1.;
        }
    );
    Kokkos::parallel_for("reflect_x2_r", MDRangePolicy<Rank<3>>({NGHOST, n2+NGHOST, 0}, {n3+NGHOST, n2+2*NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
          // Reflect across (NG+N2).  The zone j is (j - (NG+N2)) after reflection,
          // set it equal to the zone that far *before* (NG+N2)
          int jrefl = (NGHOST + n2) - (j - (NGHOST + n2)) - 1;
          PLOOP P(p, k, j, i) = P(p, k, jrefl, i);

          P(prims::u2, k, j, i) *= -1.;
          P(prims::B2, k, j, i) *= -1.;
        }
    );
}

void periodic_x3(const GRCoordinates& G, GridVars P, int n1, int n2, int n3)
{
    Kokkos::parallel_for("periodic_x3_l", MDRangePolicy<Rank<3>>({0, 0, 0}, {NGHOST, n2+2*NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
            int kz = k + n3;

            PLOOP P(p, k, j, i) = P(p, kz, j, i);
        }
    );
    Kokkos::parallel_for("periodic_x3_r", MDRangePolicy<Rank<3>>({n3+NGHOST, 0, 0}, {n3+2*NGHOST, n2+2*NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
            int kz = k - n3;

            PLOOP P(p, k, j, i) = P(p, kz, j, i);
        }
    );
}
