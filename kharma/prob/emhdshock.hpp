/* 
 *  File: emhdshock.hpp
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
#pragma once

#include "decs.hpp"
#include "emhd.hpp"

using namespace std::literals::complex_literals;
using namespace std;
using namespace parthenon;

#define STRLEN 2048

/**
 * Initialization of the EMHD shock test in magnetized plasma w/viscosity and heat conduction
 * 
 * The BVP solution (EMHD_shock_test.ipynb) is the input to the code.
 * Since the BVP solution is a steady-state, time-independent solution of the EMHD equations,
 * the code should maintain the solution.
 * 
 * An alternate option is to initialize with the ideal MHD Rankine-Hugoniot jump condition.
 * If higher order terms have been implemented correctly, the primitives should relax to the
 * steady state solution. However, they may differ by a translation to the BVP solution.
 * 
 * Therefore, to quantitatively check the EMHD implementation, we prefer the BVP solution as the input
 */

TaskStatus InitializeEMHDShock(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Initializing EMHD shock problem");
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    // It is well and good this problem should cry if B/EMHD are disabled.
    GridVector B_P = rc->Get("prims.B").data;
    GridVector q = rc->Get("prims.q").data;
    GridVector dP = rc->Get("prims.dP").data;

    // Need P for EMHD::set_parameters
    PackIndexMap prims_map;
    const MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    auto P = rc->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map);
    const VarMap m_p(prims_map, false);

    // Need fluid adiabatic index to compute pgas and sound speed (for higher order terms)
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const auto& G = pmb->coords;

    const std::string input = pin->GetOrAddString("emhd", "input" "BVP");

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    // Need the EMHD package if higher order terms are considered
    const bool use_emhd = pkgs.count("EMHD");
    EMHD::EMHD_parameters emhd_params_tmp;
    if (use_emhd) {
        const auto& emhd_pars = pmb0->packages.Get("EMHD")->AllParams();
        emhd_params_tmp       = emhd_pars.Get<EMHD::EMHD_parameters>("emhd_params");
    }
    const EMHD::EMHD_parameters& emhd_params = emhd_params_tmp;

    if (input == "BVP"){

        // Load file names into strings
        char fbvp_rho[STRLEN], fbvp_u[STRLEN], fbvp_u1[STRLEN], fbvp_q[STRLEN], fbvp_dP[STRLEN];
        sprintf(fbvp_rho, "shock_soln_rho.txt");
        sprintf(fbvp_u,   "shock_soln_u.txt");
        sprintf(fbvp_u1,  "shock_soln_u1.txt");
        sprintf(fbvp_q,   "shock_soln_q.txt");
        sprintf(fbvp_dP,  "shock_soln_dP.txt");

        // Assign file pointers
        FILE *fp_rho, *fp_u, *fp_u1, *fp_q, *fp_dP;
        fp_rho = fopen(fname_rho, "r");
        fp_u   = fopen(fname_u,   "r");
        fp_u1  = fopen(fname_u1,  "r");
        fp_q   = fopen(fname_q,   "r");
        fp_dP  = fopen(fname_dP,  "r");

        pmb->par_for("emhdshock_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_3D {
                Real X[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, X);

                // First initialize primitives that are read from .txt files
                fscanf(fp_rho, "%lf", &(rho(k, j, i)));
                fscanf(fp_u,   "%lf", &(u(k, j, i)));
                fscanf(fp_u1,  "%lf", &(uvec(0, k, j, i)));
                fscanf(fp_q,   "%lf", &(q(k, j, i)));
                fscanf(fp_dP,  "%lf", &(dP(k, j, i)));

                // Now the remaining primitives
                uvec(1, k, j, i) = 0.;
                uvec(2, k, j, i) = 0.;
                B_P(0, k, j, i)  = 0.;
                B_P(1, k, j, i)  = 0.;
                B_P(2, k, j, i)  = 0.;

                if (emhd_params.higher_order_terms) {

                    // Initialize local variables (for improved readability)
                    const Real& rho   = rho(k, j, i);
                    const Real& u     = u(k, j, i);
                    const Real& Theta = (gam - 1.) * u / rho;

                    // Set EMHD parameters
                    Real tau, chi_e, nu_e;
                    EMHD::set_parameters(G, P, m_p, emhd_params, gam, k, j, i, tau, chi_e, nu_e);

                    // Update q and dP (which now are q_tilde and dP_tilde)
                    q(k, j, i)  *= sqrt(tau / (chi_e * rho * pow(Theta, 2)));
                    dP(k, j, i) *= sqrt(tau / (nu_e * rho * Theta));
                }

            }
        );

        // disassociate file pointer
        fclose(fp_rho);
        fclose(fp_u);
        fclose(fp_u1);
        fclose(fp_q);
        fclose(fp_dP);
    }

    else {

        // Need the limits of the problem size to determine center
        const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
        const Real x1max = pin->GetReal("parthenon/mesh", "x1max");

        // Left and right states
        double rhoL = 1.,     rhoR = 3.08312999;
        double uL   = 1.,     uR   = 4.94577705;
        double u1L  = 1.,     u1R  = 0.32434571;
        double u2L  = 0.,     u2R  = 0.;
        double u3L  = 0.,     u3R  = 0.;
        double B1L  = 1.e-5,  B1R  = 1.e-5;
        double B2L  = 0,      B2R  = 0.;
        double B3L  = 0.,     B3R  = 0.;

        pmb->par_for("emhdshock_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_3D {

                Real X[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, X);
                const Real x1_center = (x1min + x1max) / 2.;

                bool lhs = X[1] < x1_center;

                // Initialize primitives
                rho(k, j, i)     = (lhs) ? rhoL : rhoR;
                u(k, j, i)       = (lhs) ? uL : uR;
                uvec(0, k, j, i) = (lhs) ? u1L : u1R;
                uvec(1, k, j, i) = (lhs) ? u2L : u2R;
                uvec(2, k, j, i) = (lhs) ? u3L : u3R;
                B_P(0, k, j, i)  = (lhs) ? B1L : B1R;
                B_P(1, k, j, i)  = (lhs) ? B2L : B2R;
                B_P(3, k, j, i)  = (lhs) ? B3L : B3R;
                q(k ,j, i)       = 0.;   
                dP(k ,j, i)      = 0.;   

            }

        );

    }

    return TaskStatus::complete;

}