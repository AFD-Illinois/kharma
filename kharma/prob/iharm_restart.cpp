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
#include "types.hpp"

#include <sys/stat.h>
#include <ctype.h>

// First boundary sync
void outflow_x1(const GRCoordinates& G, GridVars P, int nghost, int n1, int n2, int n3);
void polar_x2(const GRCoordinates& G, GridVars P, int nghost, int n1, int n2, int n3);
void periodic_x3(const GRCoordinates& G, GridVars P, int nghost, int n1, int n2, int n3);

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

TaskStatus ReadIharmRestart(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Restarting from iharm3d checkpoint file");

    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    auto& G = pmb->coords;

    auto fname = pin->GetString("iharm_restart", "fname"); // Require this, don't guess
    bool use_tf = pin->GetOrAddBoolean("iharm_restart", "use_tf", false);

    IndexDomain domain = IndexDomain::interior;
    // Full mesh size
    hsize_t n1tot = pmb->pmy_mesh->mesh_size.nx1;
    hsize_t n2tot = pmb->pmy_mesh->mesh_size.nx2;
    hsize_t n3tot = pmb->pmy_mesh->mesh_size.nx3;
    // Our block size, start, and bounds for the GridVars
    hsize_t n1 = pmb->cellbounds.ncellsi(domain);
    hsize_t n2 = pmb->cellbounds.ncellsj(domain);
    hsize_t n3 = pmb->cellbounds.ncellsk(domain);

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");

    hid_t string_type = hdf5_make_str_type(20);
    char version[20];
    hdf5_read_single_val(version, "version", string_type);
    if (MPIRank0()) {
        cout << "Restarting from " << fname << ", file version " << version << endl << endl;
    }

    // Get tf here and not when reading the header, since using this
    // value *itself* depends on a parameter, "use_tf"
    Real tf;
    hdf5_read_single_val(&tf, "tf", H5T_IEEE_F64LE);

    hsize_t nfprim;
    if(hdf5_exists("game")) {
        nfprim = 10;
    } else {
        nfprim = 8;
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

    auto rho_host = rho.GetHostMirror();
    auto u_host = u.GetHostMirror();
    auto uvec_host = uvec.GetHostMirror();
    auto B_host = B_P.GetHostMirror();

    // Host-side copy into the mirror.
    // TODO traditional OpenMP still works...
    Kokkos::parallel_for("copy_restart_state",
        Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<3>>({ks, js, is}, {ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_3D {
            rho_host(k, j, i) = ptmp[0*n3*n2*n1 + (k-ks)*n2*n1 + (j-js)*n1 + (i-is)];
            u_host(k, j, i) = ptmp[1*n3*n2*n1 + (k-ks)*n2*n1 + (j-js)*n1 + (i-is)];
            VLOOP uvec_host(v, k, j, i) = ptmp[(2+v)*n3*n2*n1 + (k-ks)*n2*n1 + (j-js)*n1 + (i-is)];
            VLOOP B_host(v, k, j, i) = ptmp[(5+v)*n3*n2*n1 + (k-ks)*n2*n1 + (j-js)*n1 + (i-is)];
        }
    );
    delete[] ptmp;

    // Deep copy to device
    rho.DeepCopy(rho_host);
    u.DeepCopy(u_host);
    uvec.DeepCopy(uvec_host);
    B_P.DeepCopy(B_host);
    Kokkos::fence();

    // Initialize the guesses for fluid prims in boundary zones
    // TODO Is this still necessary?
    // periodic_x3(G, P, Globals::nghost, n1, n2, n3);

    // Set the original simulation's end time, if we wanted that
    if (use_tf) {
        pin->SetReal("parthenon/time", "tlim", tf);
    }

    return TaskStatus::complete;
}

// void periodic_x3(const GRCoordinates& G, GridVars P, int nghost, int n1, int n2, int n3)
// {
//     Kokkos::parallel_for("periodic_x3_l", MDRangePolicy<Rank<3>>({0, 0, 0}, {nghost, n2+2*nghost, n1+2*nghost}),
//         KOKKOS_LAMBDA_3D {
//             int kz = k + n3;

//             PLOOP P(p, k, j, i) = P(p, kz, j, i);
//         }
//     );
//     Kokkos::parallel_for("periodic_x3_r", MDRangePolicy<Rank<3>>({n3+nghost, 0, 0}, {n3+2*nghost, n2+2*nghost, n1+2*nghost}),
//         KOKKOS_LAMBDA_3D {
//             int kz = k - n3;

//             PLOOP P(p, k, j, i) = P(p, kz, j, i);
//         }
//     );
// }
