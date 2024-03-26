/* 
 *  File: fofc.cpp
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

#include "flux.hpp"

#include "domain.hpp"
#include "inverter.hpp"

using namespace parthenon;

// Very bad definitions. TODO get rid of them eventually
#define NPRIM_MAX 12
#define PLOOP for(int ip=0; ip < nvar; ++ip)

TaskStatus Flux::MarkFOFC(MeshData<Real> *guess)
{
    auto pmb0 = guess->GetBlockData(0)->GetBlockPointer();

    // flags of the guess indicate where we lower
    // (not that it matters, the flags are OneCopy)
    auto fflag = guess->PackVariables(std::vector<std::string>{"fflag"});
    auto pflag = guess->PackVariables(std::vector<std::string>{"pflag"});
    auto fofcflag = guess->PackVariables(std::vector<std::string>{"fofcflag"});

    // Parameters
    const bool spherical = pmb0->coords.coords.is_spherical();
    const GReal r_eh = pmb0->coords.coords.get_horizon();

    // Pre-mark cells which will need fluxes reduced.
    // This avoids a race condition marking them multiple times when iterating faces,
    // and isolates the potentially slow/weird integer conversion stuff so we can measure the kernel time
    const IndexRange3 b = KDomain::GetRange(guess, IndexDomain::entire);
    const IndexRange block = IndexRange{0, fofcflag.GetDim(5) - 1};
    pmb0->par_for("fofc_mark", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
            const auto& G = fofcflag.GetCoords(b);
            // if cell failed to invert or would call floors...
            // TODO preserve cause in the fofcflag
            if (static_cast<int>(fflag(b, 0, k, j, i)) || //Inverter::failed(pflag(b, 0, k, j, i)) ||
                (spherical && G.r(k, j, i) < r_eh + 0.1)) {
                fofcflag(b, 0, k, j, i) = 1;
            } else {
                fofcflag(b, 0, k, j, i) = 0;
            }
        }
    );
    return TaskStatus::complete;
}

TaskStatus Flux::FOFC(MeshData<Real> *md, MeshData<Real> *guess)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto& packages = pmb0->packages;
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // Pick up flag. Optionally synced
    auto fofcflag = guess->PackVariables(std::vector<std::string>{"fofcflag"});

    // But we're modifying the live temporaries, and eventually fluxes, here
    const auto& Pl_all = md->PackVariables(std::vector<std::string>{"Flux.Pl"});
    const auto& Pr_all = md->PackVariables(std::vector<std::string>{"Flux.Pr"});
    const auto& Ul_all = md->PackVariables(std::vector<std::string>{"Flux.Ul"});
    const auto& Ur_all = md->PackVariables(std::vector<std::string>{"Flux.Ur"});
    const auto& Fl_all = md->PackVariables(std::vector<std::string>{"Flux.Fl"});
    const auto& Fr_all = md->PackVariables(std::vector<std::string>{"Flux.Fr"});
    // I assume we should update cmax/cmin. Else we should use the old ones, so
    const auto& cmax  = md->PackVariables(std::vector<std::string>{"Flux.cmax"});
    const auto& cmin  = md->PackVariables(std::vector<std::string>{"Flux.cmin"});

    // TODO this does NOT necessarily leave Flux.xyz vars in a matching state with GetFlux
    // It will be filled according to m_u/cons_map, which does not contain B
    PackIndexMap cons_map, prims_map;
    std::vector<MetadataFlag> prims_flags = {Metadata::GetUserFlag("Primitive"), Metadata::Cell};
    std::vector<MetadataFlag> cons_flags = {Metadata::Conserved, Metadata::Cell};
    const auto& P_all = md->PackVariables(prims_flags, prims_map);
    const auto& U_all = md->PackVariablesAndFluxes(cons_flags, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const int nvar = U_all.GetDim(4);

    // Parameters
    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");
    const bool use_global = pmb0->packages.Get("Flux")->Param<bool>("fofc_use_glf");
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(packages);
    // With B_CT this will load the preference, without it will set face_b false and never access (empty) Bf
    // Weird but it works
    const bool face_b = (packages.AllPackages().count("B_CT") &&
                        packages.Get("B_CT")->Param<bool>("consistent_face_b"));
    const auto& Bf = md->PackVariables(std::vector<std::string>{"cons.fB"});

    for (int dir=1; dir <= ndim; dir++) { // TODO if(trivial_direction) etc
        const TE el = FaceOf(dir);
        const Loci loc = loc_of(dir);
        const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior, el, -1, 1);
        const IndexRange block = IndexRange{0, P_all.GetDim(5) - 1};
        pmb0->par_for("fofc_replacement", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
                const auto& G = P_all.GetCoords(b);

                // Face i,j,k borders cell with same index and 1 left with index:
                int kk = (dir == 3) ? k - 1 : k;
                int jj = (dir == 2) ? j - 1 : j;
                int ii = (dir == 1) ? i - 1 : i;
                // If either bordering cell is marked, and always inside the EH
                if (static_cast<int>(fofcflag(b, 0, k, j, i)) ||
                    static_cast<int>(fofcflag(b, 0, kk, jj, ii))) { // TODO allow customizing

                    // "Reconstruct" left & right of this face: left is left cell, right is shared-index
                    PLOOP Pl_all(b, ip, k, j, i) = P_all(b, ip, kk, jj, ii);
                    PLOOP Pr_all(b, ip, k, j, i) = P_all(b, ip, k, j, i);
                    // Preserve the existing field at the face
                    if (face_b) {
                        Pl_all(b, m_p.B1+dir-1, k, j, i) = Bf(b, el, 0, k, j, i) / G.gdet(loc, j, i);
                        Pr_all(b, m_p.B1+dir-1, k, j, i) = Bf(b, el, 0, k, j, i) / G.gdet(loc, j, i);
                    }

                    FourVectors Dtmp;
                    // Left
                    GRMHD::calc_4vecs(G, Pl_all(b), m_p, k, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pl_all(b), m_p, Dtmp, emhd_params, gam, k, j, i, 0, Ul_all(b), m_u, loc);
                    Flux::prim_to_flux(G, Pl_all(b), m_p, Dtmp, emhd_params, gam, k, j, i, dir, Fl_all(b), m_u, loc);
                    // Magnetosonic speeds
                    Real cmaxL, cminL;
                    Flux::vchar_global(G, Pl_all(b), m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxL, cminL);
                    // Record speeds
                    cmax(b, dir-1, k, j, i) = m::max(0., cmaxL);
                    cmin(b, dir-1, k, j, i) = m::min(0., cminL);

                    // Right
                    GRMHD::calc_4vecs(G, Pr_all(b), m_p, k, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pr_all(b), m_p, Dtmp, emhd_params, gam, k, j, i, 0, Ur_all(b), m_u, loc);
                    Flux::prim_to_flux(G, Pr_all(b), m_p, Dtmp, emhd_params, gam, k, j, i, dir, Fr_all(b), m_u, loc);
                    // Magnetosonic speeds
                    Real cmaxR, cminR;
                    Flux::vchar_global(G, Pr_all(b), m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxR, cminR);
                    // Calculate cmax/min based on comparison with cached values
                    if (!use_global) {
                        cmax(b, dir-1, k, j, i) =  m::max(cmax(b, dir-1, k, j, i), cmaxR);
                        cmin(b, dir-1, k, j, i) = -m::min(cmin(b, dir-1, k, j, i), cminR);
                    } else {
                        // This conveniently also reduces the timestep if necessary
                        // Though, you should almost certainly set use_dt_light w/this
                        cmax(b, dir-1, k, j, i) = 1.;
                        cmin(b, dir-1, k, j, i) = 1.;
                    }

                    // Use LLF flux. Note we replace fluxes of all variables (including B!)
                    // This is for a consistent scheme, i.e. all cells FOFC == using DC+LLF
                    PLOOP
                        U_all(b).flux(dir, ip, k, j, i) = llf(Fl_all(b, ip, k, j, i), Fr_all(b, ip, k, j, i),
                                                            cmax(b, dir-1, k, j, i), cmin(b, dir-1, k, j, i),
                                                            Ul_all(b, ip, k, j, i), Ur_all(b, ip, k, j, i));
                }
            }
        );
    }

    return TaskStatus::complete;

}
