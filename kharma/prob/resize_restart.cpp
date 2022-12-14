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
#include "kharma_utils.hpp"
#include "mpi.hpp"
#include "interpolation.hpp"
#include "types.hpp"

#include <sys/stat.h>
#include <ctype.h>

// This is gross, but everything else is grosser
// What's a little leaked host mem between friends?
static Real *ptmp = NULL;
static int blocks_initialized = 0;

// TODO: The iharm3d restart format fails to record several things we must guess:
// 1. Sometimes, even precise domain boundaries in native coordinates
// 2. Which coordinate system was used
// 3. Any coordinate system parameters
// Better to either:
// a. read KHARMA restart files so we can re-grid
// b. use the IL dump format, but in double
// Either are useful capabilities.

void ReadIharmRestartHeader(std::string fname, std::unique_ptr<ParameterInput>& pin)
{
    // Read the restart file and set parameters that need to be specified at early loading
    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");
    // Print version
    hid_t string_type = hdf5_make_str_type(20);
    char version[20];
    hdf5_read_single_val(version, "version", string_type);
    if (MPIRank0()) {
        std::cout << "Initialized from " << fname << ", file version " << version << std::endl << std::endl;
    }


    // Read what we need from the file, regardless of where we're putting it
    int n1file, n2file, n3file;
    hdf5_read_single_val(&n1file, "n1", H5T_STD_I32LE);
    hdf5_read_single_val(&n2file, "n2", H5T_STD_I32LE);
    hdf5_read_single_val(&n3file, "n3", H5T_STD_I32LE);

    double x1min, x2min, x3min;
    double x1max, x2max, x3max;
    bool use_native_bounds = false;
    // Spherical for guessing
    double Rin, Rout;
    if (hdf5_exists("x1Min")) {
        // If available, read domain boundaries exactly.  This is mostly
        // for re-gridded KHARMA dumps.
        hdf5_read_single_val(&x1min, "x1Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x1max, "x1Max", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x2min, "x2Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x2max, "x2Max", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x3min, "x3Min", H5T_IEEE_F64LE);
        hdf5_read_single_val(&x3max, "x3Max", H5T_IEEE_F64LE);
        use_native_bounds = true;
    } else if (hdf5_exists("a")) {
        // Only read these if the better versions aren't available
        hdf5_read_single_val(&Rin, "Rin", H5T_IEEE_F64LE);
        hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
    } else {
        throw std::runtime_error("Unknown restart file type!");
    }

    // Anything always necessary for spherical coordinates
    double a, hslope;
    bool file_in_spherical = false;
    if (hdf5_exists("a")) {
        hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
        hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
        file_in_spherical = true;
    }

    // Anything else
    double gam, t, dt, tf;
    hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);
    hdf5_read_single_val(&t, "t", H5T_IEEE_F64LE);
    hdf5_read_single_val(&dt, "dt", H5T_IEEE_F64LE);
    hdf5_read_single_val(&tf, "tf", H5T_IEEE_F64LE);

    // Set the number of primitive vars
    // TODO do this better by recording/counting flags in MODEL
    if(hdf5_exists("game")) {
        pin->SetInteger("resize_restart", "nfprim", 10);
    } else {
        pin->SetInteger("resize_restart", "nfprim", 8);
    }

    // End HDF5 reads
    hdf5_close();

    // Record the parameters of the file grid
    // Note the iharm3d-style naming as mnemonic
    pin->SetInteger("resize_restart", "n1tot", n1file);
    pin->SetInteger("resize_restart", "n2tot", n2file);
    pin->SetInteger("resize_restart", "n3tot", n3file);
    if (use_native_bounds) {
        // If available, set the domain boundaries exactly
        pin->SetReal("resize_restart", "startx1", x1min);
        pin->SetReal("resize_restart", "stopx1", x1max);
        pin->SetReal("resize_restart", "startx2", x2min);
        pin->SetReal("resize_restart", "stopx2", x2max);
        pin->SetReal("resize_restart", "startx3", x3min);
        pin->SetReal("resize_restart", "stopx3", x3max);
    } else {
        // Otherwise, guess them
        pin->SetReal("resize_restart", "startx1", m::log(Rin));
        pin->SetReal("resize_restart", "stopx1", m::log(Rout));
        pin->SetReal("resize_restart", "startx2", 0.0);
        pin->SetReal("resize_restart", "stopx2", 1.0);
        pin->SetReal("resize_restart", "startx3", 0.0);
        pin->SetReal("resize_restart", "stopx3", 2*M_PI);
    }

    // If specified, set *our* grid to exactly match the *file's* grid
    if (pin->GetOrAddBoolean("resize_restart", "regrid_only", false)) {
        // This locks the Parthenon mesh size to be zone-for-zone the same as the iharm3d dump file...
        pin->SetInteger("parthenon/mesh", "nx1", n1file);
        pin->SetInteger("parthenon/mesh", "nx2", n2file);
        pin->SetInteger("parthenon/mesh", "nx3", n3file);
        if (pin->GetOrAddBoolean("resize_restart", "one_meshblock", false)) {
            // We can re-split the mesh now by using nearest-neighbor
            // The option is provided to restart without necessarily knowing
            // the mesh size beforehand to build a compatible meshblock
            pin->SetInteger("parthenon/meshblock", "nx1", n1file);
            pin->SetInteger("parthenon/meshblock", "nx2", n2file);
            pin->SetInteger("parthenon/meshblock", "nx3", n3file);
        }
        // ...which of course also means setting our geometry to exactly match.
        if (use_native_bounds) {
            // If available, set the boundaries exactly as read from native coords
            pin->SetReal("parthenon/mesh", "x1min", x1min);
            pin->SetReal("parthenon/mesh", "x1max", x1max);
            pin->SetReal("parthenon/mesh", "x2min", x2min);
            pin->SetReal("parthenon/mesh", "x2max", x2max);
            pin->SetReal("parthenon/mesh", "x3min", x3min);
            pin->SetReal("parthenon/mesh", "x3max", x3max);

            // Going from domain->coords values is better to match everything
            if (file_in_spherical) {
                pin->SetReal("coordinates", "r_in", exp(x1min));
                pin->SetReal("coordinates", "r_out", exp(x1max));
            }
        } else {
            std::cout << "Guessing geometry when restarting! This is potentially very bad to do!" << std::endl;
            // NOTE: the reason guessing is bad here has to do with old KHARMA versions:
            // input parameters (even geometry) were only stored to 6-digit precision in old KHARMA.
            // This means that r_in and x3max especially were cut off, and only correct to 6 digits.
            // Restarting with more accurate parameters can mess with the B field divergence,
            // so we warn about it.
            pin->SetReal("coordinates", "r_in", Rin);
            pin->SetReal("coordinates", "r_out", Rout);
            // xNmin/max will then be set by kharma.cpp.
        }
    }

    // Set the coordinate system to match the restart file *even if we're resizing*
    // Mapping to new spins/systems is theoretically fine, but we do not want to
    // do this in any applications yet
    if (file_in_spherical) {
        pin->SetReal("coordinates", "a", a);
        pin->SetReal("coordinates", "hslope", hslope);

        // Sadly restarts did not record MKS vs FMKS
        // Guess FMKS if not specified in parameter file
        pin->SetString("coordinates", "base", "spherical_ks");
        if (!pin->DoesParameterExist("coordinates", "transform")) {
            pin->SetString("coordinates", "transform", "funky");
        }
    } else {
        std::cout << "Guessing the restart file is in Cartesian coordinates!" << std::endl;
        // Anything without a BH spin was pretty likely Cartesian
        pin->SetString("coordinates", "base", "cartesian_minkowski");
        pin->SetString("coordinates", "transform", "null");
    }
    // Boundary conditions will be set based on coordinate system in kharma.cpp

    // We should always use the restart's fluid gamma & current time
    pin->SetReal("GRMHD", "gamma", gam);
    pin->SetReal("parthenon/time", "start_time", t);
    // Set the original simulation's end time, if we wanted that
    // Maybe for MHDModes restart test, or keeping to lower res end time
    if (pin->GetOrAddBoolean("resize_restart", "use_tf", false)) {
        pin->SetReal("parthenon/time", "tlim", tf);
    }
    if (pin->GetOrAddBoolean("resize_restart", "use_dt", true)) {
        // Setting dt here is actually for KHARMA,
        // which returns this from EstimateTimestep in step 0
        pin->SetReal("parthenon/time", "dt", dt);
    }
}

TaskStatus ReadIharmRestart(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Restarting from iharm3d checkpoint file");

    // TODO pack?  Probably not worth it
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    const auto fname = pin->GetString("resize_restart", "fname"); // Require this, don't guess
    const bool regrid_only = pin->GetOrAddBoolean("resize_restart", "regrid_only", false);
    const bool is_spherical = pin->GetBoolean("coordinates", "spherical");

    // Size/domain of the file we're reading *from*.
    const hsize_t nfprim = pin->GetInteger("resize_restart", "nfprim");
    const hsize_t n1tot = pin->GetInteger("resize_restart", "n1tot");
    const hsize_t n2tot = pin->GetInteger("resize_restart", "n2tot");
    const hsize_t n3tot = pin->GetInteger("resize_restart", "n3tot");
    const GReal startx[GR_DIM] = {0,
        pin->GetReal("resize_restart", "startx1"),
        pin->GetReal("resize_restart", "startx2"),
        pin->GetReal("resize_restart", "startx3")};
    const GReal stopx[GR_DIM] = {0,
        pin->GetReal("resize_restart", "stopx1"),
        pin->GetReal("resize_restart", "stopx2"),
        pin->GetReal("resize_restart", "stopx3")};
    const GReal dx[GR_DIM] = {0., (stopx[1] - startx[1])/n1tot,
                                  (stopx[2] - startx[2])/n2tot,
                                  (stopx[3] - startx[3])/n3tot};

    // Sanity checks.  Unlikely to fire but nice to have
    if (regrid_only) {
        // Check the mesh we're declaring matches
        if (pin->GetInteger("parthenon/mesh", "nx1") != n1tot ||
            pin->GetInteger("parthenon/mesh", "nx2") != n2tot ||
            pin->GetInteger("parthenon/mesh", "nx3") != n3tot) {
            printf("Mesh size does not match!\n");
            printf("[%d %d %d] vs [%d %d %d]",
                pin->GetInteger("parthenon/mesh", "nx1"),
                pin->GetInteger("parthenon/mesh", "nx2"),
                pin->GetInteger("parthenon/mesh", "nx3"),
                n1tot, n2tot, n3tot);
        }
        
        if (!close_to(pin->GetReal("parthenon/mesh", "x1min"),
                      m::log(pin->GetReal("coordinates", "r_in"))) ||
            !close_to(pin->GetReal("parthenon/mesh", "x1max"),
                      m::log(pin->GetReal("coordinates", "r_out")))) {
            printf("Mesh shape does not match!");
            printf("Rin %g vs %g, Rout %g vs %g",
                m::exp(pin->GetReal("parthenon/mesh", "x1min")),
                pin->GetReal("coordinates", "r_in"),
                m::exp(pin->GetReal("parthenon/mesh", "x1max")),
                pin->GetReal("coordinates", "r_out"));
        }

        if (!close_to(pin->GetReal("parthenon/mesh", "x1min"), startx[1]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x1max"), stopx[1]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x2min"), startx[2]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x2max"), stopx[2]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x3min"), startx[3]) ||
            !close_to(pin->GetReal("parthenon/mesh", "x3max"), stopx[3])) {
            printf("Mesh shape does not match!\n");
            printf("X1 %g vs %g & %g vs %g\nX2 %g vs %g & %g vs %g\nX3 %g vs %g & %g vs %g",
                pin->GetReal("parthenon/mesh", "x1min"), startx[1],
                pin->GetReal("parthenon/mesh", "x1max"), stopx[1],
                pin->GetReal("parthenon/mesh", "x2min"), startx[2],
                pin->GetReal("parthenon/mesh", "x2max"), stopx[2],
                pin->GetReal("parthenon/mesh", "x3min"), startx[3],
                pin->GetReal("parthenon/mesh", "x3max"), stopx[3]);
        }
    }

    // TODO there must be a better way to cache this.  InitUserData and make it a big variable or something?
    if (ptmp == NULL) {
        std::cout << "Reading mesh from file to cache..." << std::endl;

        // Declare known sizes for inputting/outputting primitives
        // We'll only ever read the full block, so this is the size we want
        hsize_t fdims[] = {nfprim, n3tot, n2tot, n1tot};
        hsize_t fstart[] = {0, 0, 0, 0};
        ptmp = new double[nfprim*n3tot*n2tot*n1tot]; // These will include B & thus be double or upconverted to it

        hdf5_open(fname.c_str());
        hdf5_set_directory("/");
        hdf5_read_array(ptmp, "p", 4, fdims, fstart, fdims, fdims, fstart, H5T_IEEE_F64LE);
        hdf5_close();

        std::cout << "Read!" << std::endl;
    }
    // If we are going to keep a static pointer, keep count so the last guy can kill it
    blocks_initialized += 1;

    auto rho_host = rho.GetHostMirror();
    auto u_host = u.GetHostMirror();
    auto uvec_host = uvec.GetHostMirror();
    auto B_host = B_P.GetHostMirror();

    // Size/domain of the MeshBlock we're reading *to*.
    // Note that we only read physical zones. 
    IndexDomain domain = IndexDomain::interior;
    const IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    const IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    const IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    auto& G = pmb->coords;

    Flag("Reordering meshblock...");
    // Host-side interpolate & copy into the mirror array
    // TODO Support restart native coordinates != new native coordinates
    // NOTE: KOKKOS USES < not <=!! Therefore the RangePolicy below will seem like it is too big
    if (regrid_only) {
        // Kokkos::parallel_for("copy_restart_state",
        //     Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<3>>({kb.s, jb.s, ib.s}, {kb.e+1, jb.e+1, ib.e+1}),
        //         KOKKOS_LAMBDA_3D {
        for (int k=kb.s; k <= kb.e; ++k) for (int j=jb.s; j <= jb.e; ++j) for (int i=ib.s; i <= ib.e; ++i) {
                GReal X[GR_DIM];
                G.coord(k, j, i, Loci::center, X); double tmp[GR_DIM];
                int gk,gj,gi; Xtoijk(X, startx, dx, gi, gj, gk, tmp, true);
                // Fill block cells with global equivalents
                rho_host(k, j, i) = ptmp[0*n3tot*n2tot*n1tot + gk*n2tot*n1tot + gj*n1tot + gi];
                u_host(k, j, i)   = ptmp[1*n3tot*n2tot*n1tot + gk*n2tot*n1tot + gj*n1tot + gi];
                VLOOP uvec_host(v, k, j, i) = ptmp[(2+v)*n3tot*n2tot*n1tot + gk*n2tot*n1tot + gj*n1tot + gi];
                VLOOP B_host(v, k, j, i) = ptmp[(5+v)*n3tot*n2tot*n1tot + gk*n2tot*n1tot + gj*n1tot + gi];
            }
        // );
    } else {
        // Kokkos::parallel_for("interp_restart_state",
        //     Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<3>>({kb.s, jb.s, ib.s}, {kb.e+1, jb.e+1, ib.e+1}),
        //     KOKKOS_LAMBDA_3D {
        for (int k=kb.s; k <= kb.e; ++k) for (int j=jb.s; j <= jb.e; ++j) for (int i=ib.s; i <= ib.e; ++i) {
                // Get the zone center location
                GReal X[GR_DIM];
                G.coord(k, j, i, Loci::center, X);
                // Interpolate the value at this location from the global grid
                rho_host(k, j, i) = linear_interp(G, X, startx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[0*n3tot*n2tot*n1tot]));
                u_host(k, j, i) = linear_interp(G, X, startx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[1*n3tot*n2tot*n1tot]));
                VLOOP uvec_host(v, k, j, i) = linear_interp(G, X, startx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[(2+v)*n3tot*n2tot*n1tot]));
                VLOOP B_host(v, k, j, i) = linear_interp(G, X, startx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(ptmp[(5+v)*n3tot*n2tot*n1tot]));
            }
        // );
    }

    // Deep copy to device
    Flag("Copying meshblock to device...");
    rho.DeepCopy(rho_host);
    u.DeepCopy(u_host);
    uvec.DeepCopy(uvec_host);
    B_P.DeepCopy(B_host);
    Kokkos::fence();

    // Close the door on our way out
    if (blocks_initialized == pmb->pmy_mesh->GetNumMeshBlocksThisRank()) {
        std::cout << "Deleting cached mesh" << std::endl;
        delete[] ptmp;
    }

    return TaskStatus::complete;
}
