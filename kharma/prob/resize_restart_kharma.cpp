/* 
 *  File: resize_restart_kharma.cpp
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

#include "resize_restart_kharma.hpp"

#include "hdf5_utils.h"
#include "types.hpp"

#include <sys/stat.h>
#include <ctype.h>

//using namespace Kokkos; // Hyerin: 10/07/22 comment this out, use par_for instead


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
//
// Reads in KHARMA restart file but at a different simulation size


void ReadKharmaRestartHeader(std::string fname, std::unique_ptr<ParameterInput>& pin)
{
    bool use_dt = pin->GetOrAddBoolean("resize_restart", "use_dt", true);
    bool use_tf = pin->GetOrAddBoolean("resize_restart", "use_tf", false);

    // Read input from restart file 
    // (from external/parthenon/src/parthenon_manager.cpp)
    std::unique_ptr<RestartReader> restartReader;
    restartReader = std::make_unique<RestartReader>(fname.c_str());

    // Load input stream
    std::unique_ptr<ParameterInput> fpinput;
    fpinput = std::make_unique<ParameterInput>();
    auto inputString = restartReader->GetAttr<std::string>("Input", "File");
    std::istringstream is(inputString);
    fpinput->LoadFromStream(is);

    int fnx1, fnx2, fnx3, fmbnx1, fmbnx2, fmbnx3;
    fnx1 = fpinput->GetInteger("parthenon/mesh", "nx1");
    fnx2 = fpinput->GetInteger("parthenon/mesh", "nx2");
    fnx3 = fpinput->GetInteger("parthenon/mesh", "nx3");
    fmbnx1 = fpinput->GetInteger("parthenon/meshblock", "nx1");
    fmbnx2 = fpinput->GetInteger("parthenon/meshblock", "nx2");
    fmbnx3 = fpinput->GetInteger("parthenon/meshblock", "nx3");
    Real fx1min = fpinput->GetReal("parthenon/mesh", "x1min");
    Real fx1max = fpinput->GetReal("parthenon/mesh", "x1max");
    bool fghostzones = fpinput->GetBoolean("parthenon/output1", "ghost_zones");
    int fnghost = fpinput->GetInteger("parthenon/mesh", "nghost");
    auto fBfield = fpinput->GetOrAddString("b_field", "type", "none");
    if (pin->GetOrAddBoolean("resize_restart", "use_restart_size", false)) {
        // This locks the mesh size to be zone-for-zone the same as the iharm3d dump file
        pin->SetInteger("parthenon/mesh", "nx1", fnx1);
        pin->SetInteger("parthenon/mesh", "nx2", fnx2);
        pin->SetInteger("parthenon/mesh", "nx3", fnx3);
        pin->SetInteger("parthenon/meshblock", "nx1", fmbnx1);
        pin->SetInteger("parthenon/meshblock", "nx2", fmbnx2);
        pin->SetInteger("parthenon/meshblock", "nx3", fmbnx3);
    }
    // Record the old values in any case
    pin->SetInteger("parthenon/mesh", "restart_nx1", fnx1);
    pin->SetInteger("parthenon/mesh", "restart_nx2", fnx2);
    pin->SetInteger("parthenon/mesh", "restart_nx3", fnx3);
    pin->SetInteger("parthenon/meshblock", "restart_nx1", fmbnx1);
    pin->SetInteger("parthenon/meshblock", "restart_nx2", fmbnx2);
    pin->SetInteger("parthenon/meshblock", "restart_nx3", fmbnx3);
    pin->SetReal("parthenon/mesh", "restart_x1min", fx1min);
    pin->SetReal("parthenon/mesh", "restart_x1max", fx1max);
    pin->SetInteger("parthenon/mesh", "restart_nghost", fnghost);
    pin->SetBoolean("parthenon/mesh", "restart_ghostzones", fghostzones);
    pin->SetString("b_field", "type", fBfield); // (12/07/22) Hyerin need to test

    Real gam, tNow, dt, tf;
    gam = fpinput->GetReal("GRMHD", "gamma");
    tNow = restartReader->GetAttr<Real>("Info", "Time");
    dt = restartReader->GetAttr<Real>("Info", "dt");
    tf = fpinput->GetReal("parthenon/time", "tlim");
    int ncycle = restartReader->GetAttr<int>("Info", "NCycle");

    pin->SetReal("GRMHD", "gamma", gam);
    pin->SetReal("parthenon/time", "start_time", tNow);
    if (use_dt) {
        pin->SetReal("parthenon/time", "dt", dt);
    }
    if (use_tf) {
        pin->SetReal("parthenon/time", "tlim", tf);
    }
    pin->SetInteger("parthenon/time", "ncycle", ncycle);
    // TODO NSTEP, next tdump/tlog, etc?

    Real  a, hslope;//, Rout;
    a = fpinput->GetReal("coordinates", "a");
    pin->SetReal("coordinates", "a", a);
    hslope = fpinput->GetReal("coordinates", "hslope");
    pin->SetReal("coordinates", "hslope", hslope);

    // close hdf5 file to prevent HDF5 hangs and corrupted files
    restartReader = nullptr;
}

TaskStatus ReadKharmaRestart(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Restarting from KHARMA checkpoint file");

    auto pmb = rc->GetBlockPointer();

    const int n1tot = pin->GetInteger("parthenon/mesh", "restart_nx1");
    const int n2tot = pin->GetInteger("parthenon/mesh", "restart_nx2");
    const int n3tot = pin->GetInteger("parthenon/mesh", "restart_nx3");
    const int n1mb = pin->GetInteger("parthenon/meshblock", "restart_nx1");
    const int n2mb = pin->GetInteger("parthenon/meshblock", "restart_nx2");
    const int n3mb = pin->GetInteger("parthenon/meshblock", "restart_nx3");
    auto fname = pin->GetString("resize_restart", "fname"); // Require this, don't guess
    auto fname_fill = pin->GetOrAddString("resize_restart", "fname_fill", "none");
    const bool is_spherical = pin->GetBoolean("coordinates", "spherical");
    const Real fx1min = pin->GetReal("parthenon/mesh", "restart_x1min");
    const Real fx1max = pin->GetReal("parthenon/mesh", "restart_x1max");
    const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
    const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
    const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
    const int nghost = pin->GetReal("parthenon/mesh", "restart_nghost");
    const bool ghost_zones = pin->GetBoolean("parthenon/mesh", "restart_ghostzones");
    auto fBfield = pin->GetOrAddString("b_field", "type", "none");

    // Add these to package properties, since they continue to be needed on boundaries
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rnx1")))
        pmb->packages.Get("GRMHD")->AddParam<int>("rnx1", n1tot);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rnx2")))
        pmb->packages.Get("GRMHD")->AddParam<int>("rnx2", n2tot);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rnx3")))
        pmb->packages.Get("GRMHD")->AddParam<int>("rnx3", n3tot);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rmbnx1")))
        pmb->packages.Get("GRMHD")->AddParam<int>("rmbnx1", n1mb);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rmbnx2")))
        pmb->packages.Get("GRMHD")->AddParam<int>("rmbnx2", n2mb);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rmbnx3")))
        pmb->packages.Get("GRMHD")->AddParam<int>("rmbnx3", n3mb);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("fname")))
        pmb->packages.Get("GRMHD")->AddParam<std::string>("fname", fname);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("fname_fill")))
        pmb->packages.Get("GRMHD")->AddParam<std::string>("fname_fill", fname_fill);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("spherical")))
        pmb->packages.Get("GRMHD")->AddParam<bool>("spherical", is_spherical);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rx1min")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rx1min", fx1min);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rx1max")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rx1max", fx1max);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("mdot")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("mdot", mdot);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rs")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rs", rs);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("x1min")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("x1min", x1min);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rnghost")))
        pmb->packages.Get("GRMHD")->AddParam<int>("rnghost", nghost);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rghostzones")))
        pmb->packages.Get("GRMHD")->AddParam<bool>("rghostzones", ghost_zones);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("b_field_type")))
        pmb->packages.Get("GRMHD")->AddParam<std::string>("b_field_type", fBfield);

    // Set the whole domain
    SetKharmaRestart(rc);

   return TaskStatus::complete;
}

TaskStatus SetKharmaRestart(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Setting KHARMA restart zones");
    auto pmb = rc->GetBlockPointer();
    auto b_field_type = pmb->packages.Get("GRMHD")->Param<std::string>("b_field_type");
    const bool include_B = (b_field_type != "none");
    // A placeholder to save the B fields for SeedBField
    GridVector B_Save;
    if (include_B) B_Save = rc->Get("B_Save").data;

    auto& G = pmb->coords;
    
    // Size/domain of the MeshBlock we're reading to
    int is, ie;
    if (domain == IndexDomain::outer_x1) {// copying from bondi
        is = pmb->cellbounds.GetBoundsI(IndexDomain::interior).e+1;
        ie = pmb->cellbounds.GetBoundsI(IndexDomain::entire).e;
    } else if (domain == IndexDomain::inner_x1) {
        is = pmb->cellbounds.GetBoundsI(IndexDomain::entire).s;
        ie = pmb->cellbounds.GetBoundsI(IndexDomain::interior).s-1;
    } else {
        is = pmb->cellbounds.is(domain);
        ie = pmb->cellbounds.ie(domain);
    }
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    //IndexRange block = IndexRange{0, nb - 1};
    
    const int n1tot = pmb->packages.Get("GRMHD")->Param<int>("rnx1");
    const int n2tot = pmb->packages.Get("GRMHD")->Param<int>("rnx2");
    const int n3tot = pmb->packages.Get("GRMHD")->Param<int>("rnx3");
    hsize_t n1mb = pmb->packages.Get("GRMHD")->Param<int>("rmbnx1");
    hsize_t n2mb = pmb->packages.Get("GRMHD")->Param<int>("rmbnx2");
    hsize_t n3mb = pmb->packages.Get("GRMHD")->Param<int>("rmbnx3");
    hsize_t nBlocks = (int) (n1tot*n2tot*n3tot)/(n1mb*n2mb*n3mb);
    auto fname = pmb->packages.Get("GRMHD")->Param<std::string>("fname");
    auto fname_fill = pmb->packages.Get("GRMHD")->Param<std::string>("fname_fill");
    const bool should_fill = !(fname_fill == "none");
    const Real fx1min = pmb->packages.Get("GRMHD")->Param<Real>("rx1min");
    const Real fx1max = pmb->packages.Get("GRMHD")->Param<Real>("rx1max");
    const Real dx1 = (fx1max - fx1min) / n1tot;
    const bool fghostzones = pmb->packages.Get("GRMHD")->Param<bool>("rghostzones");
    int fnghost = pmb->packages.Get("GRMHD")->Param<int>("rnghost");
    const Real fx1min_ghost = fx1min - fnghost*dx1;
    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    
    if ((domain != IndexDomain::outer_x1) && (domain != IndexDomain::inner_x1)) { 
        // read from a restart file and save it to static GridScalar
        //cout << "Hyerin: reading files" << endl;


        if (! fghostzones) fnghost=0; // reset to 0
        int x3factor=1;
        if (n3tot <= 1) x3factor=0; // if less than 3D, do not add ghosts in x3
        hsize_t length[GR_DIM] = {nBlocks,
                                    n1mb+2*fnghost,
                                    n2mb+2*fnghost,
                                    n3mb+2*fnghost*x3factor}; 
        const int block_sz = length[0]*length[1]*length[2]*length[3];
        //std::cout << "lengths " << length[0]  << " " << length[1] <<" " <<  length[2]<<" " << length[3] << std::endl;
        //printf("lengths %i %i %i %i \n", length[0], length[1], length[2], length[3]);
        
        
        // read from file and stored in device Hyerin (10/18/2022)
        GridScalar x1_f_device("x1_f_device", length[0], length[1]); 
        GridScalar x2_f_device("x2_f_device", length[0], length[2]); 
        GridScalar x3_f_device("x3_f_device", length[0], length[3]); 
        GridScalar rho_f_device("rho_f_device", length[0], length[3], length[2], length[1]); 
        GridScalar u_f_device("u_f_device", length[0], length[3], length[2], length[1]); 
        GridVector uvec_f_device("uvec_f_device", NVEC, length[0], length[3], length[2], length[1]); 
        GridVector B_f_device("B_f_device", NVEC, length[0], length[3], length[2], length[1]);
        auto x1_f_host = x1_f_device.GetHostMirror();
        auto x2_f_host = x2_f_device.GetHostMirror();
        auto x3_f_host = x3_f_device.GetHostMirror();
        auto rho_f_host = rho_f_device.GetHostMirror();
        auto u_f_host = u_f_device.GetHostMirror();
        auto uvec_f_host = uvec_f_device.GetHostMirror();
        auto B_f_host = B_f_device.GetHostMirror();
        // Hyerin (09/19/2022) : new attempt to read the file 
        hdf5_open(fname.c_str());
        hdf5_set_directory("/");
        Real *rho_file = new double[block_sz];
        Real *u_file = new double[block_sz];
        Real *uvec_file = new double[block_sz*3];
        Real *B_file = new double[block_sz*3];
        Real *x1_file = new double[length[0]*length[1]];
        Real *x2_file = new double[length[0]*length[2]];
        Real *x3_file = new double[length[0]*length[3]];
        //static hsize_t fdims[] = {length[0], length[3], length[2], length[1],1}; //outdated
        static hsize_t fdims[] = {length[0], 1, length[3], length[2], length[1]};
        //static hsize_t fdims_vec[] = {length[0], length[3], length[2], length[1],3}; //outdated
        static hsize_t fdims_vec[] = {length[0], 3, length[3], length[2], length[1]};
        static hsize_t fdims_x1[] = {length[0], length[1]};
        static hsize_t fdims_x2[] = {length[0], length[2]};
        static hsize_t fdims_x3[] = {length[0], length[3]};
        hsize_t fstart[] = {0, 0, 0, 0, 0};
        hsize_t fstart_x[] = {0, 0};
        hdf5_read_array(rho_file, "prims.rho", 5, fdims, fstart,fdims,fdims,fstart,H5T_IEEE_F64LE);
        hdf5_read_array(u_file, "prims.u", 5, fdims, fstart,fdims,fdims,fstart,H5T_IEEE_F64LE);
        hdf5_read_array(uvec_file, "prims.uvec", 5, fdims_vec, fstart,fdims_vec,fdims_vec,fstart,H5T_IEEE_F64LE);
        //if (include_B) hdf5_read_array(B_file, "prims.B", 5, fdims_vec, fstart,fdims_vec,fdims_vec,fstart,H5T_IEEE_F64LE);
        if (include_B) hdf5_read_array(B_file, "cons.B", 5, fdims_vec, fstart,fdims_vec,fdims_vec,fstart,H5T_IEEE_F64LE);
        hdf5_read_array(x1_file, "VolumeLocations/x", 2, fdims_x1, fstart_x,fdims_x1,fdims_x1,fstart_x,H5T_IEEE_F64LE);
        hdf5_read_array(x2_file, "VolumeLocations/y", 2, fdims_x2, fstart_x,fdims_x2,fdims_x2,fstart_x,H5T_IEEE_F64LE);
        hdf5_read_array(x3_file, "VolumeLocations/z", 2, fdims_x3, fstart_x,fdims_x3,fdims_x3,fstart_x,H5T_IEEE_F64LE);
        hdf5_close();
        
        GridScalar x1_fill_device("x1_fill_device", length[0], length[1]); 
        GridScalar x2_fill_device("x2_fill_device", length[0], length[2]); 
        GridScalar x3_fill_device("x2_fill_device", length[0], length[3]); 
        GridScalar rho_fill_device("rho_fill_device", length[0], length[3], length[2], length[1]); 
        GridScalar u_fill_device("u_fill_device", length[0], length[3], length[2], length[1]); 
        GridVector uvec_fill_device("uvec_fill_device", NVEC, length[0], length[3], length[2], length[1]); 
        GridVector B_fill_device("B_fill_device", NVEC, length[0], length[3], length[2], length[1]); 
        auto x1_fill_host = x1_fill_device.GetHostMirror();
        auto x2_fill_host = x2_fill_device.GetHostMirror();
        auto x3_fill_host = x3_fill_device.GetHostMirror();
        auto rho_fill_host = rho_fill_device.GetHostMirror();
        auto u_fill_host = u_fill_device.GetHostMirror();
        auto uvec_fill_host = uvec_fill_device.GetHostMirror();
        auto B_fill_host = B_fill_device.GetHostMirror();
        Real *rho_filefill = new double[block_sz];
        Real *u_filefill = new double[block_sz];
        Real *uvec_filefill = new double[block_sz*3];
        Real *B_filefill = new double[block_sz*3];
        Real *x1_filefill = new double[length[0]*length[1]];
        Real *x2_filefill = new double[length[0]*length[2]];
        Real *x3_filefill = new double[length[0]*length[3]];
        if (fname_fill != "none") { // TODO: here I'm assuming fname and fname_fill has same dimensions, which is not always the case.
            hdf5_open(fname_fill.c_str());
            hdf5_set_directory("/");
            hdf5_read_array(rho_filefill, "prims.rho", 5, fdims, fstart,fdims,fdims,fstart,H5T_IEEE_F64LE);
            hdf5_read_array(u_filefill, "prims.u", 5, fdims, fstart,fdims,fdims,fstart,H5T_IEEE_F64LE);
            hdf5_read_array(uvec_filefill, "prims.uvec", 5, fdims_vec, fstart,fdims_vec,fdims_vec,fstart,H5T_IEEE_F64LE);
            //if (include_B) hdf5_read_array(B_filefill, "prims.B", 5, fdims_vec, fstart,fdims_vec,fdims_vec,fstart,H5T_IEEE_F64LE);
            if (include_B) hdf5_read_array(B_filefill, "cons.B", 5, fdims_vec, fstart,fdims_vec,fdims_vec,fstart,H5T_IEEE_F64LE);
            hdf5_read_array(x1_filefill, "VolumeLocations/x", 2, fdims_x1, fstart_x,fdims_x1,fdims_x1,fstart_x,H5T_IEEE_F64LE);
            hdf5_read_array(x2_filefill, "VolumeLocations/y", 2, fdims_x2, fstart_x,fdims_x2,fdims_x2,fstart_x,H5T_IEEE_F64LE);
            hdf5_read_array(x3_filefill, "VolumeLocations/z", 2, fdims_x3, fstart_x,fdims_x3,fdims_x3,fstart_x,H5T_IEEE_F64LE);
            hdf5_close();
        }

        // save the grid coordinate values to host array
        for (int iblocktemp = 0; iblocktemp < length[0]; iblocktemp++) {
            for (int itemp = 0; itemp < length[1]; itemp++) {
                x1_f_host(iblocktemp,itemp) = x1_file[length[1]*iblocktemp+itemp];
                if (fname_fill != "none") x1_fill_host(iblocktemp,itemp) = x1_filefill[length[1]*iblocktemp+itemp];
            } for (int jtemp = 0; jtemp < length[2]; jtemp++) {
                x2_f_host(iblocktemp,jtemp) = x2_file[length[2]*iblocktemp+jtemp];
                if (fname_fill != "none") x2_fill_host(iblocktemp,jtemp) = x2_filefill[length[2]*iblocktemp+jtemp];
            } for (int ktemp = 0; ktemp < length[3]; ktemp++) {
                x3_f_host(iblocktemp,ktemp) = x3_file[length[3]*iblocktemp+ktemp];
                if (fname_fill != "none") x3_fill_host(iblocktemp,ktemp) = x3_filefill[length[3]*iblocktemp+ktemp];
            }
        }
        // re-arrange uvec such that it can be read in the VLOOP
        int vector_file_index, scalar_file_index;
        for (int iblocktemp = 0; iblocktemp < length[0]; iblocktemp++) {
            for (int itemp = 0; itemp < length[1]; itemp++) {
                for (int jtemp = 0; jtemp < length[2]; jtemp++) {
                    for (int ktemp = 0; ktemp < length[3]; ktemp++) {
                        scalar_file_index = length[1]*(length[2]*(length[3]*iblocktemp+ktemp)+jtemp)+itemp;

                        rho_f_host(iblocktemp,ktemp,jtemp,itemp) = rho_file[scalar_file_index];
                        u_f_host(iblocktemp,ktemp,jtemp,itemp) = u_file[scalar_file_index];
                        if (fname_fill != "none") {
                            rho_fill_host(iblocktemp,ktemp,jtemp,itemp) = rho_filefill[scalar_file_index];
                            u_fill_host(iblocktemp,ktemp,jtemp,itemp) = u_filefill[scalar_file_index];
                        }
                        for (int ltemp = 0; ltemp < 3; ltemp++) {
                            //vector_file_index = 3*(scalar_file_index)+ltemp; // outdated parthenon phdf5 saving order
                            vector_file_index = length[1]*(length[2]*(length[3]*(3*iblocktemp+ltemp)+ktemp)+jtemp)+itemp;
                            
                            uvec_f_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = uvec_file[vector_file_index];
                            if (include_B) B_f_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = B_file[vector_file_index];
                            if (fname_fill != "none") {
                                uvec_fill_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = uvec_filefill[vector_file_index];
                                if (include_B) B_fill_host(ltemp,iblocktemp,ktemp,jtemp,itemp) = B_filefill[vector_file_index];
                            }
                        }
                    }
                }
            }
        }
        //std::cout << "Hyerin: first five Bs" << B_file[0] << " " << B_file[1] << " " << B_file[2] << " " << B_file[3] << " " << B_file[4] << std::endl; 
        //std::cout << "Hyerin: 6,7,8,9,10 B_f " << B_f_host(0,0,0,0,6) << " " << B_f_host(0,0,0,0,7) << " " << B_f_host(0,0,0,0,8) << " " << B_f_host(0,0,0,0,9) << " " << B_f_host(0,0,0,0,10) << std::endl; 
        const bool is_spherical = pmb->packages.Get("GRMHD")->Param<bool>("spherical");
        const Real mdot = pmb->packages.Get("GRMHD")->Param<Real>("mdot");
        const Real rs = pmb->packages.Get("GRMHD")->Param<Real>("rs");
        const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

        SphKSCoords kscoord = mpark::get<SphKSCoords>(G.coords.base);
        SphBLCoords blcoord = SphBLCoords(kscoord.a); //, kscoord.ext_g); // modified (11/15/22)
        CoordinateEmbedding coords = G.coords;

      
        // Deep copy to device
        x1_f_device.DeepCopy(x1_f_host);
        x2_f_device.DeepCopy(x2_f_host);
        x3_f_device.DeepCopy(x3_f_host);
        rho_f_device.DeepCopy(rho_f_host);
        u_f_device.DeepCopy(u_f_host);
        uvec_f_device.DeepCopy(uvec_f_host);
        if (include_B) B_f_device.DeepCopy(B_f_host);
        if (fname_fill != "none") {
            x1_fill_device.DeepCopy(x1_fill_host);
            x2_fill_device.DeepCopy(x2_fill_host);
            x3_fill_device.DeepCopy(x3_fill_host);
            rho_fill_device.DeepCopy(rho_fill_host);
            u_fill_device.DeepCopy(u_fill_host);
            uvec_fill_device.DeepCopy(uvec_fill_host);
            if (include_B) B_fill_device.DeepCopy(B_fill_host);
        }
        //if (pin->GetOrAddString("b_field", "type", "none") != "none") {
        //    B_P.DeepCopy(B_host);
        //}
        Kokkos::fence();

        // Host-side interpolate & copy into the mirror array
        pmb->par_for("copy_restart_state_kharma", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                get_prim_restart_kharma(G, coords, P, m_p, blcoord,  kscoord, 
                    fx1min, fx1max, fnghost, should_fill, is_spherical, include_B, gam, rs, mdot, length,
                    x1_f_device, x2_f_device, x3_f_device, rho_f_device, u_f_device, uvec_f_device, B_f_device,
                    x1_fill_device, x2_fill_device, x3_fill_device, rho_fill_device, u_fill_device, uvec_fill_device, B_fill_device,
                    k, j, i);
                //GRMHD::p_to_u(G,P,m_p,gam,k,j,i,U,m_u);  //TODO: is this needed? I don't see it in resize_restart.cpp
                //if (pin->GetOrAddString("b_field", "type", "none") != "none") {
                //    VLOOP B_host(v, k, j, i) = interp_scalar(G, X, startx, stopx, dx, is_spherical, false, n3tot, n2tot, n1tot, &(B_file[v*block_sz]));
                //}
                if (include_B)
                    get_B_restart_kharma(G, coords, P, m_p, blcoord,  kscoord, 
                        fx1min, fx1max, should_fill, length,
                        x1_f_device, x2_f_device, x3_f_device, B_f_device,
                        x1_fill_device, x2_fill_device, x3_fill_device, B_fill_device, B_Save,
                        k, j, i);
            }
        );
        //if (include_B) B_FluxCT::PtoU(rc,domain); // added for B fields
    }

   return TaskStatus::complete;
}
