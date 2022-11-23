/* 
 *  File: conducting_atmosphere.cpp
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

#include "emhd/conducting_atmosphere.hpp"

#ifdef KOKKOS_ENABLE_CUDA

TaskStatus InitializeAtmosphere(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    throw std::runtime_error("Conducting Atmosphere problem is not implemented for GPUs!!");
}

TaskStatus dirichlet_bc(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    throw std::runtime_error("Dirichlet BCs are not implemented for GPUs!!");
}

#else

using namespace std;
using namespace parthenon;

#define STRLEN 2048

/*
 * Initialization of the hydrostatic conducting atmosphere test
 * 
 * The ODE solution (kharma/prob/emhd/conducting_atmosphere_${RES}_default) is the input to the code.
 * Since the ODE solution is a steady-state solution of the EMHD equations,
 * the code should maintain the solution.
 * 
 */

ParArrayND<double> p_bound;

TaskStatus InitializeAtmosphere(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    
    auto pmb = rc->GetBlockPointer();
    PackIndexMap prims_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);

    const bool use_emhd     = pmb->packages.AllPackages().count("EMHD");
    bool higher_order_terms = false;
    EMHD::EMHD_parameters emhd_params_tmp;
    
    if (use_emhd)
    {
        Flag(rc, "Initializing hydrostatic conducting atmosphere problem");
        
        const auto& emhd_pars = pmb->packages.Get("EMHD")->AllParams();
        emhd_params_tmp       = emhd_pars.Get<EMHD::EMHD_parameters>("emhd_params");
        higher_order_terms    = emhd_params_tmp.higher_order_terms;

        MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
        P = rc->PackVariables({isPrimitive}, prims_map);
    }
    else
        Flag(rc, "Initializing hydrostatic atmosphere problem");

    const EMHD::EMHD_parameters& emhd_params = emhd_params_tmp;

    // Obtain GRMHD params
    const auto& grmhd_pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real& gam        = grmhd_pars.Get<Real>("gamma");

    int nvar = P.GetDim(4);

    const auto& G = pmb->coords;

    // Type of input to the problem
    const std::string input = pin->GetOrAddString("conducting_atmosphere", "input", "ODE");

    // Bounds of the domain
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

    // Load file names into strings
    char fode_rCoords[STRLEN], fode_rho[STRLEN], fode_u[STRLEN], fode_q[STRLEN];
    sprintf(fode_rCoords, "atmosphere_soln_rCoords.txt");
    sprintf(fode_rho,     "atmosphere_soln_rho.txt");
    sprintf(fode_u,       "atmosphere_soln_u.txt");
    sprintf(fode_q,       "atmosphere_soln_phi.txt");

    // Assign file pointers
    FILE *fp_r, *fp_rho, *fp_u, *fp_q;;
    fp_r   = fopen(fode_rCoords, "r");
    fp_rho = fopen(fode_rho, "r");
    fp_u   = fopen(fode_u,   "r");
    if (use_emhd)
        fp_q = fopen(fode_q, "r");

    
    GridScalar rho  = rc->Get("prims.rho").data; 
    GridScalar u    = rc->Get("prims.u").data; 
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P  = rc->Get("prims.B").data;
    GridScalar q;
    GridScalar dP;
    if (use_emhd)
    {
        q  = rc->Get("prims.q").data;
        dP = rc->Get("prims.dP").data;
    }

    // Host side mirror of primitives
    // TODO Better way to create mirrors using the primitive pack P instead of rc?
    auto rho_host   = rho.GetHostMirror();
    auto u_host     = u.GetHostMirror();
    auto uvec_host  = uvec.GetHostMirror();
    auto B_host     = B_P.GetHostMirror();
    auto q_host     = rho.GetHostMirror(); // TODO Temporary initialization necessary for auto type
    auto dP_host    = rho.GetHostMirror();
    if (use_emhd)
    {
        q_host  = q.GetHostMirror();
        dP_host = dP.GetHostMirror();
    }

    // Define ParArrays to store radial boundary values
    IndexRange ib_in = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb_in = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb_in = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    const int n1 = pmb->cellbounds.ncellsi(IndexDomain::interior);
    const int ng = (int)(ib.e - ib_in.e);

    p_bound = ParArrayND<double>("Dirichlet boundary values", nvar, n1 + 2*ng);
    auto p_bound_host = p_bound.GetHostMirror();

    // Load coordinates 'r' and compare against grid values
    double rCoords[n1 + 2*ng] = {0}, error = 0.;
    for (int i = ib.s; i <= ib.e; i++) {
        
        fscanf(fp_r, "%lf", &(rCoords[i]));
        GReal Xnative[GR_DIM], Xembed[GR_DIM]; 
        G.coord(0, ng, i, Loci::center, Xnative); // j and k don't matter since we need to compare only the radial coordinate
        G.coord_embed(0, ng, i, Loci::center, Xembed);
        error = fabs(Xembed[1] - rCoords[i]);
        if (error > 1.e-10) {
            fprintf(stdout, "Error at radial zone i = %d, Error = %8.5e KHARMA: %8.7e, sage nb: %8.7e\n", i, error, Xembed[1], rCoords[i]);
        }
    }
    if (error > 1.e-10) exit(-1);

    // Initialize primitives
    double rho_temp, u_temp, q_temp;

    for (int i = ib.s; i <= ib.e; i++) {

        fscanf(fp_rho, "%lf", &(rho_temp));
        fscanf(fp_u,   "%lf", &(u_temp));
        if (use_emhd)
            fscanf(fp_q, "%lf", &(q_temp));

        for (int j = jb_in.s; j <= jb_in.e; j++) {
            for (int k = kb_in.s; k <= kb_in.e; k++) {

                GReal Xnative[GR_DIM], Xembed[GR_DIM]; 
                G.coord(k, j, i, Loci::center, Xnative);
                G.coord_embed(k, j, i, Loci::center, Xembed);

                // First initialize primitives that are read from .txt files
                rho_host(k, j, i)   = rho_temp;
                u_host(k, j, i)     = u_temp;
                if (use_emhd)
                    q_host(k, j, i) = q_temp;

                // Now the remaining primitives
                uvec_host(V1, k, j, i) = 0.;
                uvec_host(V2, k, j, i) = 0.;
                uvec_host(V3, k, j, i) = 0.;
                B_host(V1, k, j, i)    = 1./pow(Xembed[1], 3.);
                B_host(V2, k, j, i)    = 0.;
                B_host(V3, k, j, i)    = 0.;
                if (use_emhd)
                    dP_host(k, j, i)   = 0.;

                // Note that the  velocity primitives defined up there isn't quite right.
                // For a fluid at rest wrt. the normal observer, ucon = {-1/g_tt,0,0,0}. 
                // We need to use this info to obtain the correct values for U1, U2 and U3

                Real ucon[GR_DIM]         = {0};
                Real gcov[GR_DIM][GR_DIM] = {0};
                Real gcon[GR_DIM][GR_DIM] = {0};
                G.gcov(Loci::center, j, i, gcov);
                G.gcon(Loci::center, j, i, gcon);

                ucon[0] = 1./sqrt(-gcov[0][0]);
                ucon[1] = 0.;
                ucon[2] = 0.;
                ucon[3] = 0.;

                double alpha, beta[GR_DIM], gamma;

                // Solve for primitive velocities (utilde)
                alpha = 1/sqrt(-gcon[0][0]);
                gamma = ucon[0] * alpha;

                beta[0] = 0.;
                beta[1] = alpha*alpha*gcon[0][1];
                beta[2] = alpha*alpha*gcon[0][2];
                beta[3] = alpha*alpha*gcon[0][3];

                uvec_host(V1, k, j, i) = ucon[1] + beta[1]*gamma/alpha;
                uvec_host(V2, k, j, i) = ucon[2] + beta[2]*gamma/alpha;
                uvec_host(V3, k, j, i) = ucon[3] + beta[3]*gamma/alpha;

                if (use_emhd)
                    if (higher_order_terms){

                        const Real Theta = (gam - 1.) * u_temp / rho_temp;

                        // Set EMHD parameters
                        Real tau, chi_e, nu_e;
                        EMHD::set_parameters(G, rho_temp, u_temp, emhd_params, gam, k, j, i, tau, chi_e, nu_e);

                        // Update q_host (and dP_host, which is zero in this problem). These are now q_tilde and dP_tilde
                        Real q_tilde  = q_host(k, j, i);
                        Real dP_tilde = dP_host(k, j, i);
                        if (emhd_params.higher_order_terms) {
                            q_tilde  *= (chi_e != 0) ? sqrt(tau / (chi_e * rho_temp * pow(Theta, 2.))) : 0.;
                            dP_tilde *= (nu_e  != 0) ? sqrt(tau / (nu_e * rho_temp * Theta)) : 0.;
                        }
                        q_host(k, j, i)   = q_tilde;
                        dP_host(k, j, i)  = dP_tilde;
                    }
            }
        }

        // Save boundary values for Dirichlet boundary conditions
        if (i < ng) {
            p_bound_host(0, i) = rho_host(0, ng, i);
            p_bound_host(1, i) = u_host(0, ng, i);
            p_bound_host(2, i) = uvec_host(V1, 0, ng, i);
            p_bound_host(3, i) = uvec_host(V2, 0, ng, i);
            p_bound_host(4, i) = uvec_host(V3, 0, ng, i);
            p_bound_host(5, i) = B_host(V1, 0, ng, i);
            p_bound_host(6, i) = B_host(V2, 0, ng, i);
            p_bound_host(7, i) = B_host(V3, 0, ng, i);
            if (use_emhd)
                p_bound_host(8, i) = q_host(0, ng, i);
                p_bound_host(9, i) = dP_host(0, ng, i);
        }
        if (i > n1 + ng - 1) {
            p_bound_host(0, i-n1) = rho_host(0, ng, i);
            p_bound_host(1, i-n1) = u_host(0, ng, i);
            p_bound_host(2, i-n1) = uvec_host(V1, 0, ng, i);
            p_bound_host(3, i-n1) = uvec_host(V2, 0, ng, i);
            p_bound_host(4, i-n1) = uvec_host(V3, 0, ng, i);
            p_bound_host(5, i-n1) = B_host(V1, 0, ng, i);
            p_bound_host(6, i-n1) = B_host(V2, 0, ng, i);
            p_bound_host(7, i-n1) = B_host(V3, 0, ng, i);
            if (use_emhd)
                p_bound_host(8, i-n1) = q_host(0, ng, i);
                p_bound_host(9, i-n1) = dP_host(0, ng, i);
        }
    }

    // disassociate file pointer
    fclose(fp_rho);
    fclose(fp_u);
    if (use_emhd)
        fclose(fp_q);

    // Deep copy to device
    rho.DeepCopy(rho_host);
    u.DeepCopy(u_host);
    uvec.DeepCopy(uvec_host);
    B_P.DeepCopy(B_host);
    if (use_emhd)
    {
        q.DeepCopy(q_host);
        dP.DeepCopy(dP_host);
    }
    p_bound.DeepCopy(p_bound_host);
    Kokkos::fence();

    return TaskStatus::complete;

}

TaskStatus dirichlet_bc(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{

    Flag(rc, "Applying Dirichlet boundary conditions along radial direction");

    auto pmb = rc->GetBlockPointer();
    const bool use_emhd = pmb->packages.AllPackages().count("EMHD");

    GridScalar rho  = rc->Get("prims.rho").data;
    GridScalar u    = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P  = rc->Get("prims.B").data;
    GridScalar q;
    GridScalar dP;
    if (use_emhd)
    {
        q  = rc->Get("prims.q").data;
        dP = rc->Get("prims.dP").data;
    }

    const auto& G = pmb->coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    // Need number of physical zones to access outer boundary elements of p_bound
    const int n1 = pmb->cellbounds.ncellsi(IndexDomain::interior);

    pmb->par_for("dirichlet_boundary", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            if (domain == IndexDomain::inner_x1) {
                rho(k, j, i)      = p_bound(0, i);
                u(k, j, i)        = p_bound(1, i);
                uvec(V1, k, j, i) = p_bound(2, i);
                uvec(V2, k, j, i) = p_bound(3, i);
                uvec(V3, k, j, i) = p_bound(4, i);
                B_P(V1, k, j, i)  = p_bound(5, i);
                B_P(V2, k, j, i)  = p_bound(6, i);
                B_P(V3, k, j, i)  = p_bound(7, i);
                if (use_emhd)
                {
                    q(k, j, i)  = p_bound(8, i);
                    dP(k, j, i) = p_bound(9, i);
                }
            }
            else {
                rho(k, j, i)      = p_bound(0, i - n1);
                u(k, j, i)        = p_bound(1, i - n1);
                uvec(V1, k, j, i) = p_bound(2, i - n1);
                uvec(V2, k, j, i) = p_bound(3, i - n1);
                uvec(V3, k, j, i) = p_bound(4, i - n1);
                B_P(V1, k, j, i)  = p_bound(5, i - n1);
                B_P(V2, k, j, i)  = p_bound(6, i - n1);
                B_P(V3, k, j, i)  = p_bound(7, i - n1);
                if (use_emhd)
                {
                    q(k, j, i)  = p_bound(8, i - n1);
                    dP(k, j, i) = p_bound(9, i - n1);
                }
            }
        }
    );

    return TaskStatus::complete;
}

#endif
