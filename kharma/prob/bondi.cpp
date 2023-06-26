/* 
 *  File: bondi.cpp
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

#include "bondi.hpp"

/**
 * Initialization of a Bondi problem with specified sonic point, BH mdot, and horizon radius
 * TODO mdot and rs are redundant and should be merged into one parameter. Uh, no.
 */
TaskStatus InitializeBondi(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Initializing Bondi problem");
    auto pmb = rc->GetBlockPointer();

    const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
    const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
    // r_shell : the radius of the shell where inside this radius is filled with vacuum. If 0, the simulation is initialized to Bondi everywhere
    const Real r_shell = pin->GetOrAddReal("bondi", "r_shell", 0.); 
    const bool use_gizmo = pin->GetOrAddBoolean("bondi", "use_gizmo", false);
    auto datfn = pin->GetOrAddString("gizmo_shell", "datfn", "none");
    const Real ur_frac = pin->GetOrAddReal("bondi", "ur_frac", 1.); 
    const Real uphi = pin->GetOrAddReal("bondi", "uphi", 0.); 

    // Add these to package properties, since they continue to be needed on boundaries
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("mdot")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("mdot", mdot);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rs")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rs", rs);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("r_shell")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("r_shell", r_shell);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("use_gizmo")))
        pmb->packages.Get("GRMHD")->AddParam<bool>("use_gizmo", use_gizmo);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("gizmo_dat")))
        pmb->packages.Get("GRMHD")->AddParam<std::string>("gizmo_dat", datfn);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("ur_frac")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("ur_frac", ur_frac);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("uphi")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("uphi", uphi);

    // Set the whole domain to the analytic solution to begin
    SetBondi(rc);

    Flag(rc, "Initialized");
    return TaskStatus::complete;
}

TaskStatus SetBondi(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Setting Bondi zones");
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const Real mdot = pmb->packages.Get("GRMHD")->Param<Real>("mdot");
    const Real rs = pmb->packages.Get("GRMHD")->Param<Real>("rs");
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real r_shell = pmb->packages.Get("GRMHD")->Param<Real>("r_shell");
    const bool use_gizmo = pmb->packages.Get("GRMHD")->Param<bool>("use_gizmo");
    auto datfn = pmb->packages.Get("GRMHD")->Param<std::string>("gizmo_dat");
    const Real ur_frac = pmb->packages.Get("GRMHD")->Param<Real>("ur_frac");
    const Real uphi = pmb->packages.Get("GRMHD")->Param<Real>("uphi");

    // Just the X1 right boundary
    GRCoordinates G = pmb->coords;
    SphKSCoords ks = mpark::get<SphKSCoords>(G.coords.base);
    SphBLCoords bl = SphBLCoords(ks.a, ks.ext_g); // modified
    CoordinateEmbedding cs = G.coords;

    // This function currently only handles "outer X1" and "entire" grid domains,
    // but is the special-casing here necessary?
    // Can we define outer_x1 w/priority more flexibly?
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int ibs, ibe;
    if (domain == IndexDomain::outer_x1) {
        ibs = bounds.GetBoundsI(IndexDomain::interior).e+1;
        ibe = bounds.GetBoundsI(IndexDomain::entire).e;
    } else if (domain == IndexDomain::inner_x1) {
        ibs = bounds.GetBoundsI(IndexDomain::entire).s;
        ibe = bounds.GetBoundsI(IndexDomain::interior).s-1;
    } else {
        ibs = bounds.GetBoundsI(domain).s;
        ibe = bounds.GetBoundsI(domain).e;
    }
    IndexRange jb_e = bounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb_e = bounds.GetBoundsK(IndexDomain::entire);
    IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);
    
    // GIZMO shell, doesn't do anything for radial bdry
    if (use_gizmo && (domain != IndexDomain::outer_x1) && (domain != IndexDomain::inner_x1)) { 
        // Read the gizmo data file
        // TODO: Hyerin: maybe put this into some other function?
        std::string fnstr(datfn.c_str());
        std::string dat_type=fnstr.substr(fnstr.find_last_of(".") + 1);
        const bool use_3d = (dat_type != "txt");

        // txt file is 1D data
        if (! use_3d) { 
            std::cout << "GIZMO dat file is txt" << std::endl;
            FILE *fptr;
            fptr = fopen(datfn.c_str(),"r");
            const int datlen=100000;
            Real *rarr = new double[datlen];
            Real *rhoarr = new double[datlen]; 
            Real *Tarr = new double[datlen]; 
            Real *vrarr = new double[datlen]; 
            Real *Mencarr = new double[datlen]; 
            int length=0, itemp=0;
            while (fscanf(fptr,"%lf %lf %lf %lf %lf\n", &(rarr[itemp]), &(rhoarr[itemp]), &(Tarr[itemp]), &(vrarr[itemp]), &(Mencarr[itemp])) == 5) { // assign the read value to variable, and enter it in array
                    itemp++;
            }
            fclose(fptr);
            length=itemp;

            GridVector r_device("r_device", length); 
            GridVector rho_device("rho_device", length); 
            GridVector T_device("T_device", length); 
            GridVector vr_device("vr_device", length); 
            auto r_host = r_device.GetHostMirror();
            auto rho_host = rho_device.GetHostMirror();
            auto T_host = T_device.GetHostMirror();
            auto vr_host = vr_device.GetHostMirror();
            for (itemp = 0; itemp < length; itemp++) {
                r_host(itemp) = rarr[itemp];
                rho_host(itemp) = rhoarr[itemp];
                T_host(itemp) = Tarr[itemp];
                vr_host(itemp) = vrarr[itemp];
            }
            r_device.DeepCopy(r_host);
            rho_device.DeepCopy(rho_host);
            T_device.DeepCopy(T_host);
            vr_device.DeepCopy(vr_host);
                
            Kokkos::fence();
            
            pmb->par_for("gizmo_shell", kb_e.s, kb_e.e, jb_e.s, jb_e.e, ibs, ibe,
                KOKKOS_LAMBDA_3D {
                    // same vacuum conditions at r_shell
                    GReal Xshell[GR_DIM] = {0, r_shell, 0, 0};
                    int i_sh;
                    GReal del_sh;
                    Real vacuum_rho, vacuum_u_over_rho, vacuum_logrho, vacuum_log_u_over_rho;
                    XtoindexGizmo(Xshell, r_device, length, i_sh, del_sh);
                    vacuum_rho = rho_device(i_sh)*(1.-del_sh)+rho_device(i_sh+1)*del_sh;
                    vacuum_u_over_rho = (T_device(i_sh)*(1.-del_sh)+T_device(i_sh+1)*del_sh)/(gam-1.);
                    vacuum_logrho = log10(vacuum_rho);
                    vacuum_log_u_over_rho = log10(vacuum_u_over_rho);
                
                    get_prim_gizmo_shell(G, cs, P, m_p, gam, bl, ks, r_shell, rs, vacuum_logrho, vacuum_log_u_over_rho, 
                                          r_device, rho_device, T_device, vr_device, length, k, j, i);
                    // TODO all flux
                    GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
                }
            );
        } else { 
            // hdf5 files are 3D data
            std::cout << "GIZMO dat file is hdf5" << std::endl;

            // read from hdf5 file
            hdf5_open(datfn.c_str());
            hdf5_set_directory("/");
            hsize_t length=7372800; // TODO: should be able to be retrieved from the data file
            Real *coordarr = new double[length*3];
            Real *rhoarr = new double[length];
            Real *Tarr = new double[length];
            Real *varr = new double[length*3];
            static hsize_t fdims_vec[] = {length, 3};
            static hsize_t fdims_scl[] = {length};
            hsize_t fstart_vec[] = {0, 0};
            hsize_t fstart_scl[] = {0};
            hdf5_read_array(coordarr, "PartType0_dimless/Coordinates", 2, fdims_vec, fstart_vec,fdims_vec,fdims_vec,fstart_vec,H5T_IEEE_F64LE);
            hdf5_read_array(rhoarr, "PartType0_dimless/Density", 1, fdims_scl, fstart_scl,fdims_scl,fdims_scl,fstart_scl,H5T_IEEE_F64LE);
            hdf5_read_array(Tarr, "PartType0_dimless/Temperature", 1, fdims_scl, fstart_scl,fdims_scl,fdims_scl,fstart_scl,H5T_IEEE_F64LE);
            hdf5_read_array(varr, "PartType0_dimless/Velocities", 2, fdims_vec, fstart_vec,fdims_vec,fdims_vec,fstart_vec,H5T_IEEE_F64LE);
            hdf5_close();

            // save in a device arrays
            GridVector coord_device("coord_device", length, 3); 
            GridScalar rho_device("rho_device", length); 
            GridScalar T_device("rho_device", length); 
            GridVector v_device("v_device", length, 3); 
            auto coord_host = coord_device.GetHostMirror();
            auto rho_host = rho_device.GetHostMirror();
            auto T_host = T_device.GetHostMirror();
            auto v_host = v_device.GetHostMirror();
            int vector_file_index;
            for (int itemp = 0; itemp < length; itemp++) {
                for (int ltemp = 0; ltemp < 3; ltemp++) {
                    vector_file_index = 3*itemp+ltemp;
                    coord_host(itemp,ltemp) = coordarr[vector_file_index];
                    v_host(itemp,ltemp) = varr[vector_file_index];
                }
                rho_host(itemp) = rhoarr[itemp];
                T_host(itemp) = Tarr[itemp];
            }
            coord_device.DeepCopy(coord_host);
            rho_device.DeepCopy(rho_host);
            T_device.DeepCopy(T_host);
            v_device.DeepCopy(v_host);
                
            Kokkos::fence();

            //int i_sh;
            // same vacuum conditions at r_shell // (05/01/23) a better way to do this?
            //pmb->par_for("gizmo_r_shell", 0, 0,
            //    KOKKOS_LAMBDA_1D {
            //        GReal Xshell[GR_DIM] = {0, r_shell, 0, 0};
            //        GReal del_sh;
            //        int i_sh_dev;
            //        XtoindexGizmo3D(Xshell, coord_device, length, i_sh_dev, del_sh);
            //    }
            //);
            //i_sh=i_sh_dev;
            int i_sh=0; // TODO! Ask Ben

            pmb->par_for("gizmo_shell", kb.s, kb.e, jb.s, jb.e, ibs, ibe,
                KOKKOS_LAMBDA_3D {
                    Real vacuum_rho, vacuum_u_over_rho, vacuum_logrho, vacuum_log_u_over_rho;
                    
                    vacuum_rho = rho_device(i_sh);
                    vacuum_u_over_rho = T_device(i_sh)/(gam-1.);
                    vacuum_logrho = log10(vacuum_rho);
                    vacuum_log_u_over_rho = log10(vacuum_u_over_rho);

                    get_prim_gizmo_shell_3d(G, cs, P, m_p, gam, bl, ks, r_shell, rs, vacuum_logrho, vacuum_log_u_over_rho, 
                                          coord_device, rho_device, T_device, v_device, length, k, j, i);
                    // TODO all flux
                    GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
                }
            );
        }
    }
    // Bondi
    else if (! (use_gizmo)) {
        pmb->par_for("bondi_boundary", kb_e.s, kb_e.e, jb_e.s, jb_e.e, ibs, ibe,
            KOKKOS_LAMBDA_3D {
                get_prim_bondi(G, cs, P, m_p, gam, bl, ks, mdot, rs, r_shell, ur_frac, uphi, k, j, i);
                // TODO all flux
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
            }
        );
    }

    Flag(rc, "Set");
    return TaskStatus::complete;
}
