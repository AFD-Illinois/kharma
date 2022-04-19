/* 
 *  File: resize_restart.cpp
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

#include "resize_restart.hpp"

#include "b_flux_ct.hpp"
#include "debug.hpp"
#include "hdf5_utils.h"
#include "mpi.hpp"
#include "resize.hpp"
#include "types.hpp"

#include <sys/stat.h>
#include <ctype.h>

using namespace Kokkos;

// TODO
// Record & read:
// 1. startx/stopx/dx
// 2. coordinate name FMKS/MKS/etc
// 3. all coordinate params in play
// 4. Electron MODEL bitflag param
// 5. nprim for sanity check?
// 6. Indication of EMHD vs MHD

// TODO this code is very specific to spherical systems/boundares or entirely periodic boxes.
// No other boundaries/geometries are really supported.

void ReadIharmRestartHeader(std::string fname, std::unique_ptr<ParameterInput>& pin)
{
    // Read the restart file and set parameters that need to be specified at early loading
    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");

    // Get the grid size
    int n1file, n2file, n3file;
    hdf5_read_single_val(&n1file, "n1", H5T_STD_I32LE);
    hdf5_read_single_val(&n2file, "n2", H5T_STD_I32LE);
    hdf5_read_single_val(&n3file, "n3", H5T_STD_I32LE);
    if (pin->GetOrAddBoolean("resize_restart", "use_restart_size", false)) {
        // This locks the mesh size to be zone-for-zone the same as the iharm3d dump file
        pin->SetInteger("parthenon/mesh", "nx1", n1file);
        pin->SetInteger("parthenon/mesh", "nx2", n2file);
        pin->SetInteger("parthenon/mesh", "nx3", n3file);
        pin->SetInteger("parthenon/meshblock", "nx1", n1file);
        pin->SetInteger("parthenon/meshblock", "nx2", n2file);
        pin->SetInteger("parthenon/meshblock", "nx3", n3file);
    }
    // Record the old values in any case
    pin->SetInteger("parthenon/mesh", "restart_nx1", n1file);
    pin->SetInteger("parthenon/mesh", "restart_nx2", n2file);
    pin->SetInteger("parthenon/mesh", "restart_nx3", n3file);

    double gam, cour, t;
    hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);
    hdf5_read_single_val(&cour, "cour", H5T_IEEE_F64LE);
    hdf5_read_single_val(&t, "t", H5T_IEEE_F64LE);

    pin->SetPrecise("GRMHD", "gamma", gam);
    pin->SetPrecise("parthenon/time", "start_time", t);
    // TODO NSTEP, next tdump/tlog, etc?

    if (hdf5_exists("a")) {
        double a, hslope, Rout;
        hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
        hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
        hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
        pin->SetPrecise("coordinates", "a", a);
        pin->SetPrecise("coordinates", "hslope", hslope);
        pin->SetPrecise("coordinates", "r_out", Rout);

        // Sadly restarts did not record MKS vs FMKS
        // Guess if not specified in parameter file
        pin->SetString("coordinates", "base", "spherical_ks");
        if (!pin->DoesParameterExist("coordinates", "transform")) {
            pin->SetString("coordinates", "transform", "funky");
        }

        pin->SetPrecise("parthenon/mesh", "x2min", 0.0);
        pin->SetPrecise("parthenon/mesh", "x2max", 1.0);
        pin->SetPrecise("parthenon/mesh", "x3min", 0.0);
        pin->SetPrecise("parthenon/mesh", "x3max", 2*M_PI);

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
        pin->SetPrecise("parthenon/mesh", "x1min", x1min);
        pin->SetPrecise("parthenon/mesh", "x1max", x1max);
        pin->SetPrecise("parthenon/mesh", "x2min", x2min);
        pin->SetPrecise("parthenon/mesh", "x2max", x2max);
        pin->SetPrecise("parthenon/mesh", "x3min", x3min);
        pin->SetPrecise("parthenon/mesh", "x3max", x3max);

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

    auto fname = pin->GetString("resize_restart", "fname"); // Require this, don't guess
    bool use_tf = pin->GetOrAddBoolean("resize_restart", "use_tf", false);
    bool use_dt = pin->GetOrAddBoolean("resize_restart", "use_dt", true);
    const bool is_spherical = pin->GetBoolean("coordinates", "spherical");

    // Size of the file mesh
    hsize_t n1tot = pin->GetInteger("parthenon/mesh", "restart_nx1");
    hsize_t n2tot = pin->GetInteger("parthenon/mesh", "restart_nx2");
    hsize_t n3tot = pin->GetInteger("parthenon/mesh", "restart_nx3");

    // Size/domain of the MeshBlock we're reading to
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");
    // Print version
    hid_t string_type = hdf5_make_str_type(20);
    char version[20];
    hdf5_read_single_val(version, "version", string_type);
    if (MPIRank0()) {
        cout << "Restarting from " << fname << ", file version " << version << endl << endl;
    }

    // Get tf/dt here and not when reading the header, since whether we use them
    // depends on another parameter, "use_tf" & "use_dt" which need to be initialized
    double tf, dt;
    hdf5_read_single_val(&tf, "tf", H5T_IEEE_F64LE);
    hdf5_read_single_val(&dt, "dt", H5T_IEEE_F64LE);

    // TODO do this better by recording/counting flags in MODEL
    hsize_t nfprim;
    if(hdf5_exists("game")) {
        nfprim = 10;
    } else {
        nfprim = 8;
    }

    // Declare known sizes for inputting/outputting primitives
    // We'll only ever read the full block, so this is the size we want
    static hsize_t fdims[] = {nfprim, n3tot, n2tot, n1tot};
    hsize_t fstart[] = {0, 0, 0, 0};

    // TODO don't repeat this read for every block!
    // Likely requires read once in e.g. InitUserMeshData
    // -> pass in (pointer) -> delete[] in PostInit or something
    Real *ptmp = new double[nfprim*n3tot*n2tot*n1tot]; // These will include B & thus be double or upconverted to it
    hdf5_read_array(ptmp, "p", 4, fdims, fstart, fdims, fdims, fstart, H5T_IEEE_F64LE);

    // End HDF5 reads
    hdf5_close();

    auto rho_host = rho.GetHostMirror();
    auto u_host = u.GetHostMirror();
    auto uvec_host = uvec.GetHostMirror();
    auto B_host = B_P.GetHostMirror();

    // These are set to probably mirror the restart file,
    // but ideally should be read straight from it.
    const GReal startx[GR_DIM] = {0,
        pin->GetReal("parthenon/mesh", "x1min"),
        pin->GetReal("parthenon/mesh", "x2min"),
        pin->GetReal("parthenon/mesh", "x3min")};
    const GReal stopx[GR_DIM] = {0,
        pin->GetReal("parthenon/mesh", "x1max"),
        pin->GetReal("parthenon/mesh", "x2max"),
        pin->GetReal("parthenon/mesh", "x3max")};
    // Same here
    const GReal dx[GR_DIM] = {0., (stopx[1] - startx[1])/n1tot,
                                  (stopx[2] - startx[2])/n2tot,
                                  (stopx[3] - startx[3])/n3tot};

    const int block_sz = n3tot*n2tot*n1tot;

    // Host-side interpolate & copy into the mirror array
    // TODO Support restart native coordinates != new native coordinates
    // NOTE: KOKKOS USES < not <=!! Therefore the RangePolicy below will seem like it is too big
    Kokkos::parallel_for("copy_restart_state",
        Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<3>>({ks, js, is}, {ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_3D {
            // Get the zone center location
            GReal X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);
            // Interpolate the value at this location from the global grid
            rho_host(k, j, i) = interp_scalar(G, X, startx, stopx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[0*block_sz]));
            u_host(k, j, i) = interp_scalar(G, X, startx, stopx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[1*block_sz]));
            VLOOP uvec_host(v, k, j, i) = interp_scalar(G, X, startx, stopx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[(2+v)*block_sz]));
            VLOOP B_host(v, k, j, i) = interp_scalar(G, X, startx, stopx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[(5+v)*block_sz]));
        }
    );
    delete[] ptmp;

    // Deep copy to device
    rho.DeepCopy(rho_host);
    u.DeepCopy(u_host);
    uvec.DeepCopy(uvec_host);
    B_P.DeepCopy(B_host);
    Kokkos::fence();

    // Set the original simulation's end time, if we wanted that
    // Used pretty much only for MHDModes restart test
    if (use_tf) {
        pin->SetPrecise("parthenon/time", "tlim", tf);
    }
    if (use_dt) {
        // Setting dt here is actually for KHARMA,
        // which returns this from EstimateTimestep in step 0
        pin->SetPrecise("parthenon/time", "dt", dt);
    }

    return TaskStatus::complete;
}
