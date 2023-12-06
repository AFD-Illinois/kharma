/* 
 *  File: seed_B.cpp
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
#include "seed_B.hpp"

#include "b_ct.hpp"
#include "b_flux_ct.hpp"

#include "boundaries.hpp"
#include "coordinate_utils.hpp"
#include "domain.hpp"
#include "fm_torus.hpp"
#include "chakrabarti_torus.hpp"
#include "grmhd_functions.hpp"

using namespace parthenon;

/**
 * Perform a Parthenon MPI reduction.
 * Should only be used in initialization code, as the
 * reducer object & MPI comm are created on entry &
 * cleaned on exit
 */
template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
    parthenon::AllReduce<T> reduction;
    reduction.val = f;
    reduction.StartReduce(O);
    // Wait on results
    while (reduction.CheckReduce() == parthenon::TaskStatus::incomplete);
    // TODO catch errors?
    return reduction.val;
}

// Shorter names for the reductions we use here
Real MaxBsq(MeshData<Real> *md)
{
    return Reductions::DomainReduction<Reductions::Var::bsq, Real>(md, UserHistoryOperation::max);
}
Real MaxPressure(MeshData<Real> *md)
{
    return Reductions::DomainReduction<Reductions::Var::gas_pressure, Real>(md, UserHistoryOperation::max);
}
Real MinBeta(MeshData<Real> *md)
{
    return Reductions::DomainReduction<Reductions::Var::beta, Real>(md, UserHistoryOperation::min);
}


template <BSeedType Seed>
TaskStatus SeedBFieldType(MeshBlockData<Real> *rc, ParameterInput *pin, IndexDomain domain = IndexDomain::entire)
{
    auto pmb = rc->GetBlockPointer();
    auto pkgs = pmb->packages.AllPackages();

    // Fields
    GridScalar rho = rc->Get("prims.rho").data;
    const auto &G = pmb->coords;

    // Parameters
    std::string b_field_type = pin->GetString("b_field", "type");
    auto prob = pin->GetString("parthenon/job", "problem_id");
    bool is_torus = (prob == "torus");
    // What kind?
    std::string torus_type;
    if (is_torus) {
        torus_type = pin->GetString("parthenon/job", "torus_type");
    }
    bool is_fm, is_chakrabarti = false;
    if (torus_type == "fishbone_moncrief") {
        is_fm = true;
    } else if (torus_type == "chakrabarti") {
        is_chakrabarti = true;
    }

    // Indices
    // TODO handle filling faces with domain < entire more gracefully
    IndexRange3 b = KDomain::GetRange(rc, domain);
    int ndim = pmb->pmy_mesh->ndim;

    // Shortcut to field values for easy fields
    if constexpr (Seed == BSeedType::constant ||
                  Seed == BSeedType::monopole ||
                  Seed == BSeedType::monopole_cube ||
                  Seed == BSeedType::orszag_tang ||
                  Seed == BSeedType::wave || 
                  Seed == BSeedType::shock_tube)
    {
        // All custom B fields should set what they need of these.
        // We take the same names, but they may mean different things to the
        // particular init function, check seed_B.hpp
        const Real B10 = pin->GetOrAddReal("b_field", "B10", 0.);
        const Real B20 = pin->GetOrAddReal("b_field", "B20", 0.);
        const Real B30 = pin->GetOrAddReal("b_field", "B30", 0.);
        const Real k1 = pin->GetOrAddReal("b_field", "k1", 0.);
        const Real k2 = pin->GetOrAddReal("b_field", "k2", 0.);
        const Real k3 = pin->GetOrAddReal("b_field", "k3", 0.);
        const Real phase = pin->GetOrAddReal("b_field", "phase", 0.);
        const Real amp_B1 = pin->GetOrAddReal("b_field", "amp_B1", 0.);
        const Real amp_B2 = pin->GetOrAddReal("b_field", "amp_B2", 0.);
        const Real amp_B3 = pin->GetOrAddReal("b_field", "amp_B3", 0.);
        const Real amp2_B1 = pin->GetOrAddReal("b_field", "amp2_B1", 0.);
        const Real amp2_B2 = pin->GetOrAddReal("b_field", "amp2_B2", 0.);
        const Real amp2_B3 = pin->GetOrAddReal("b_field", "amp2_B3", 0.);

        if (pkgs.count("B_CT")) {
            auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
            // Fill at 3 different locations
            pmb->par_for(
                "B_field_B", b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    GReal Xembed[GR_DIM];
                    double null1, null2;
                    double B_Pf1, B_Pf2, B_Pf3;
                    // TODO handle calling Seed() mid-run and adding field
                    G.coord_embed(k, j, i, Loci::face1, Xembed);
                    GReal gdet = G.gdet(Loci::face1, j, i);
                    B_Pf1 = B10;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase,
                                 amp_B1, amp_B2, amp_B3,
                                 amp2_B1, amp2_B2, amp2_B3,
                                 B_Pf1, null1, null2);
                    B_Uf(F1, 0, k, j, i) = B_Pf1 * gdet;

                    G.coord_embed(k, j, i, Loci::face2, Xembed);
                    gdet = G.gdet(Loci::face2, j, i);
                    B_Pf2 = B20;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase,
                                 amp_B1, amp_B2, amp_B3,
                                 amp2_B1, amp2_B2, amp2_B3,
                                 null1, B_Pf2, null2);
                    B_Uf(F2, 0, k, j, i) = B_Pf2 * gdet;

                    G.coord_embed(k, j, i, Loci::face3, Xembed);
                    gdet = G.gdet(Loci::face3, j, i);
                    B_Pf3 = B30;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase,
                                 amp_B1, amp_B2, amp_B3,
                                 amp2_B1, amp2_B2, amp2_B3,
                                 null1, null2, B_Pf3);
                    B_Uf(F3, 0, k, j, i) = B_Pf3 * gdet;
                }
            );
            // Update primitive variables
            B_CT::BlockUtoP(rc, domain);
        } else if (pkgs.count("B_FluxCT")) {
            GridVector B_P = rc->Get("prims.B").data;
            pmb->par_for(
                "B_field_B", b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    GReal Xembed[GR_DIM];
                    G.coord_embed(k, j, i, Loci::center, Xembed);
                    const GReal gdet = G.gdet(Loci::center, j, i);
                    B_P(V1, k, j, i) = B10;
                    B_P(V2, k, j, i) = B20;
                    B_P(V3, k, j, i) = B30;
                    seed_b<Seed>(Xembed, gdet, k1, k2, k3, phase,
                                 amp_B1, amp_B2, amp_B3,
                                 amp2_B1, amp2_B2, amp2_B3,
                                 B_P(V1, k, j, i),
                                 B_P(V2, k, j, i),
                                 B_P(V3, k, j, i));
                }
            );
            // We still need to update conserved flux values, but then we're done
            B_FluxCT::BlockPtoU(rc, domain);
        } // TODO B_CD!!
        return TaskStatus::complete;
    } else { // Seed with vector potential A otherwise
        // Require and load what we need if necessary
        Real A0 = pin->GetOrAddReal("b_field", "A0", 0.);
        Real min_A = pin->GetOrAddReal("b_field", "min_A", 0.2);
        // Init-specific loads
        Real a, rin, rmax, gam, kappa, rho_norm, arg1;
        Real tilt = 0; // Needs to be initialized
        // Chakrabarti-specific variables
        const Real rho_max = pin->GetOrAddReal("torus", "rho_max", 1.0);
        Real gm1, lnh_in, lnh_peak, pgas_over_rho_peak, rho_peak;
        GReal cc, nn; 
        Real potential_rho_pow, potential_falloff, potential_r_pow;
        switch (Seed) {
        case BSeedType::sane:
        case BSeedType::mad:
        case BSeedType::mad_quadrupole:
        case BSeedType::r3s3:
        case BSeedType::r5s5:
        case BSeedType::gaussian:
            // Torus parameters
            rin   = pin->GetReal("torus", "rin");
            rmax  = pin->GetReal("torus", "rmax");
            kappa = pin->GetReal("torus", "kappa");
            tilt  = pin->GetReal("torus", "tilt") / 180. * M_PI;
            // Other things we need only for torus evaluation
            gam      = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
            rho_norm = pmb->packages.Get("GRMHD")->Param<Real>("rho_norm");
            a        = G.coords.get_a();
            break;
        case BSeedType::vertical_chakrabarti:
            // A separate case for the vertical field initialized for the Chakrabarti torus
            // Fluid parameters
            gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
            gm1 = gam - 1.;
            // Field init parameters
            potential_rho_pow = pin->GetOrAddReal("b_field", "potential_rho_pow", 1.0);
            potential_falloff = pin->GetOrAddReal("b_field", "potential_falloff", 0.0);
            potential_r_pow   = pin->GetOrAddReal("b_field", "potential_r_pow", 0.0);
            // Torus parameters
            rin  = pin->GetReal("torus", "rin");
            rmax = pin->GetReal("torus", "rmax");
            tilt = pin->GetReal("torus", "tilt") / 180. * M_PI;
            // Spacetime geometry parameters
            a = G.coords.get_a();
            // Now compute relevant parameters
            cn_calc(a, rin, rmax, &cc, &nn);
            lnh_in             = lnh_calc(a, rin, rin, 1.0, cc, nn);
            lnh_peak           = lnh_calc(a, rin, rmax, 1.0, cc, nn) - lnh_in;
            pgas_over_rho_peak = gm1/gam * (m::exp(lnh_peak) - 1.0);
            rho_peak           = m::pow(pgas_over_rho_peak, 1.0 / gm1) / rho_max;
            break;
        case BSeedType::orszag_tang_a:
            A0 = pin->GetReal("orszag_tang", "tscale");
            arg1 = pin->GetReal("orszag_tang", "phase");
            break;
        default:
            break;
        }

        // For all other fields...
        // Find the magnetic vector potential.  In X3 symmetry only A_phi is non-zero,
        // But for tilted conditions we must keep track of all components
        IndexSize3 sz = KDomain::GetBlockSize(rc);
        ParArrayND<double> A("A", NVEC, sz.n3, sz.n2, sz.n1);
        pmb->par_for(
            "B_field_A", b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                GReal Xnative[GR_DIM];
                GReal Xembed[GR_DIM], Xmidplane[GR_DIM];
                G.coord(k, j, i, Loci::corner, Xnative);
                G.coord_embed(k, j, i, Loci::corner, Xembed);
                // What are our corresponding "midplane" values for evaluating the function?
                rotate_polar(Xembed, tilt, Xmidplane);
                const GReal r = Xmidplane[1], th = Xmidplane[2];

                // Trigonometric values
                const GReal sth = sin(th);
                const GReal cth = cos(th);

                // In case we need zone sizes
                const GReal dxc[GR_DIM] = {0., G.Dxc<1>(i), G.Dxc<2>(j), G.Dxc<3>(k)};

                // This is written under the assumption re-computed rho is more accurate than a bunch
                // of averaging in a meaningful way.  Just use the average if not.
                Real rho_av;
                bool in_torus = false;
                if (is_torus) {
                    if (is_fm) {
                        // Find rho at corner directly for torii
                        rho_av = fm_torus_rho(a, rin, rmax, gam, kappa, r, th) / rho_norm;
                    }
                    else if (is_chakrabarti){
                        // Find rho
                        const Real lnh = lnh_calc(a, rin, r, sth, cc, nn);
                        if (lnh >= 0.0) {
                            in_torus = true;
                            Real pg_over_rho = gm1 / gam * (m::exp(lnh) - 1.0);
                            rho_av = m::pow(pg_over_rho, 1. / gm1) / rho_peak;
                        }
                    }
                } else {
                    // Use averages for anything else
                    // This loop runs over every corner. Centers do not exist before the first
                    // or after the last, so use the last (ghost) zones available.
                    const int ii = clip((uint)i, b.is + 1, b.ie);
                    const int jj = clip((uint)j, b.js + 1, b.je);
                    const int kk = clip((uint)k, b.ks + 1, b.ke);
                    if (ndim > 2)
                    {
                        rho_av = (rho(kk, jj, ii) + rho(kk, jj, ii - 1) +
                                rho(kk, jj - 1, ii) + rho(kk, jj - 1, ii - 1) +
                                rho(kk - 1, jj, ii) + rho(kk - 1, jj, ii - 1) +
                                rho(kk - 1, jj - 1, ii) + rho(kk - 1, jj - 1, ii - 1)) /
                                8;
                    }
                    else
                    {
                        rho_av = (rho(kk, jj, ii) + rho(kk, jj, ii - 1) +
                                rho(kk, jj - 1, ii) + rho(kk, jj - 1, ii - 1)) /
                                4;
                    }
                }

                Real Aphi = seed_a<Seed>(Xmidplane, dxc, rho_av, rin, min_A, A0, arg1, in_torus, rho_max,\
                                        potential_rho_pow, potential_falloff, potential_r_pow);

                if (tilt != 0.0) {
                    // This is *covariant* A_mu of an untilted disk
                    const double A_untilt_lower[GR_DIM] = {0., 0., 0., Aphi};
                    // Raise to contravariant vector, since rotate_polar_vec will need that.
                    // Note we have to do this in the midplane!
                    // The coord_to_native calculation involves an iterative solve for MKS/FMKS
                    GReal Xnative_midplane[GR_DIM] = {0}, gcon_midplane[GR_DIM][GR_DIM] = {0};
                    G.coords.coord_to_native(Xmidplane, Xnative_midplane);
                    G.coords.gcon_native(Xnative_midplane, gcon_midplane);
                    double A_untilt[GR_DIM] = {0};
                    DLOOP2 A_untilt[mu] += gcon_midplane[mu][nu] * A_untilt_lower[nu];

                    // Then rotate
                    double A_tilt[GR_DIM] = {0};
                    double A_untilt_embed[GR_DIM] = {0}, A_tilt_embed[GR_DIM] = {0};
                    G.coords.con_vec_to_embed(Xnative_midplane, A_untilt, A_untilt_embed);
                    rotate_polar_vec(Xmidplane, A_untilt_embed, -tilt, Xembed, A_tilt_embed);
                    G.coords.con_vec_to_native(Xnative, A_tilt_embed, A_tilt);

                    // Lower the result as we need curl(A_mu).  Done at local zone.
                    double A_tilt_lower[GR_DIM] = {0}, gcov[GR_DIM][GR_DIM] = {0};
                    G.coords.gcov_native(Xnative, gcov);
                    DLOOP2 A_tilt_lower[mu] += gcov[mu][nu] * A_tilt[nu];
                    VLOOP A(v, k, j, i) = A_tilt_lower[1 + v];
                } else {
                    // Some problems rely on a very accurate A->B, which the rotation lacks.
                    // So, we preserve exact values in the no-tilt case.
                    A(V3, k, j, i) = Aphi;
                }
            });

        if (pkgs.count("B_CT")) {
            auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
            // This fills a couple zones outside the exact interior with bad data
            // Careful of that w/e.g. Dirichlet bounds.
            IndexRange3 bB = KDomain::GetRange(rc, domain, 0, -1);
            if (ndim > 2) {
                pmb->par_for(
                    "ot_B", bB.ks, bB.ke, bB.js, bB.je, bB.is, bB.ie,
                    KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                        B_CT::curl_3D(G, A, B_Uf, k, j, i);
                    });
            } else if (ndim > 1) {
                pmb->par_for(
                    "ot_B", bB.ks, bB.ke, bB.js, bB.je, bB.is, bB.ie,
                    KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                        B_CT::curl_2D(G, A, B_Uf, k, j, i);
                    });
            } else {
                throw std::runtime_error("Must initialize 1D field directly!");
            }
            B_CT::BlockUtoP(rc, domain);
            //std::cout << "Block divB: " << B_CT::BlockMaxDivB(rc) << std::endl;
        } else if (pkgs.count("B_FluxCT")) {
            // Calculate B-field
            GridVector B_U = rc->Get("cons.B").data;
            IndexRange3 bl = KDomain::GetRange(rc, domain, 0, -1); // TODO will need changes if domain < entire
            if (ndim > 2) {
                pmb->par_for(
                    "B_field_B_3D", bl.ks, bl.ke, bl.js, bl.je, bl.is, bl.ie,
                    KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                        B_FluxCT::averaged_curl_3D(G, A, B_U, k, j, i);
                    });
            } else if (ndim > 1) {
                pmb->par_for(
                    "B_field_B_2D", bl.ks, bl.ke, bl.js, bl.je, bl.is, bl.ie,
                    KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                        B_FluxCT::averaged_curl_2D(G, A, B_U, k, j, i);
                    });
            } else {
                throw std::runtime_error("Must initialize 1D field directly!");
            }
            // Finally, make sure we initialize the primitive field too
            B_FluxCT::BlockUtoP(rc, domain);
        }

        return TaskStatus::complete;
    }
}

TaskStatus SeedBField(MeshData<Real> *md, ParameterInput *pin)
{
    Flag("SeedBField");
    std::string b_field_type = pin->GetString("b_field", "type");
    auto pmesh = md->GetMeshPointer();
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    if (MPIRank0() && verbose) {
        std::cout << "Seeding B field with type " << b_field_type << std::endl;
    }

    TaskStatus status = TaskStatus::incomplete;
    for (int i=0; i < md->NumBlocks(); i++) {
        auto *rc = md->GetBlockData(i).get();

        // I could make this a map or something,
        // but this is the only place I decode it.
        // TODO could also save it to a package...
        // TODO accumulate TaskStatus properly?
        if (b_field_type == "constant") {
            status = SeedBFieldType<BSeedType::constant>(rc, pin);
        } else if (b_field_type == "monopole") {
            status = SeedBFieldType<BSeedType::monopole>(rc, pin);
        } else if (b_field_type == "monopole_cube") {
            status = SeedBFieldType<BSeedType::monopole_cube>(rc, pin);
        } else if (b_field_type == "sane") {
            status = SeedBFieldType<BSeedType::sane>(rc, pin);
        } else if (b_field_type == "mad") {
            status = SeedBFieldType<BSeedType::mad>(rc, pin);
        } else if (b_field_type == "mad_quadrupole") {
            status = SeedBFieldType<BSeedType::mad_quadrupole>(rc, pin);
        } else if (b_field_type == "r3s3") {
            status = SeedBFieldType<BSeedType::r3s3>(rc, pin);
        } else if (b_field_type == "steep" || b_field_type == "r5s5") {
            status = SeedBFieldType<BSeedType::r5s5>(rc, pin);
        } else if (b_field_type == "gaussian") {
            status = SeedBFieldType<BSeedType::gaussian>(rc, pin);
        } else if (b_field_type == "bz_monopole") {
            status = SeedBFieldType<BSeedType::bz_monopole>(rc, pin);
        } else if (b_field_type == "vertical") {
            status = SeedBFieldType<BSeedType::vertical>(rc, pin);
        } else if (b_field_type == "vertical_chakrabarti") {
            status = SeedBFieldType<BSeedType::vertical_chakrabarti>(rc, pin);
        } else if (b_field_type == "orszag_tang") {
            status = SeedBFieldType<BSeedType::orszag_tang>(rc, pin);
        } else if (b_field_type == "orszag_tang_a") {
            status = SeedBFieldType<BSeedType::orszag_tang_a>(rc, pin);
        } else if (b_field_type == "wave") {
            status = SeedBFieldType<BSeedType::wave>(rc, pin);
        } else if (b_field_type == "shock_tube") {
            status = SeedBFieldType<BSeedType::shock_tube>(rc, pin);
        } else {
            throw std::invalid_argument("Magnetic field seed type not supported: " + b_field_type);
        }
    }

    EndFlag();
    return status;
}

TaskStatus NormalizeBField(MeshData<Real> *md, ParameterInput *pin)
{
    Flag("NormBField");
    // Check which solver we'll be using
    auto pmesh = md->GetMeshPointer();
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    // Default to the general literature beta_min of 100.
    // As noted above, by default this uses the definition max(P)/max(P_B)!
    Real desired_beta_min = pin->GetOrAddReal("b_field", "beta_min", 100.);

    // "Legacy" is the much more common normalization:
    // It's the ratio of max values over the domain i.e. max(P) / max(P_B),
    // not necessarily a local min(beta)
    Real beta_calc_legacy = pin->GetOrAddBoolean("b_field", "legacy_norm", true);

    // Calculate current beta_min value
    Real bsq_max, p_max, beta_min;
    if (beta_calc_legacy) {
        bsq_max = MPIReduce_once(MaxBsq(md), MPI_MAX);
        p_max = MPIReduce_once(MaxPressure(md), MPI_MAX);
        beta_min = p_max / (0.5 * bsq_max);
    } else {
        beta_min = MPIReduce_once(MinBeta(md), MPI_MIN);
    }

    if (MPIRank0() && verbose > 0) {
        if (beta_calc_legacy) {
            std::cout << "B^2 max pre-norm: " << bsq_max << std::endl;
            std::cout << "Pressure max pre-norm: " << p_max << std::endl;
        }
        std::cout << "Beta min pre-norm: " << beta_min << std::endl;
    }

    // Then normalize B by sqrt(beta/beta_min)
    if (beta_min > 0) {
        Real norm = m::sqrt(beta_min/desired_beta_min);
        for (auto &pmb : pmesh->block_list) {
            auto& rc = pmb->meshblock_data.Get();
            KHARMADriver::Scale(std::vector<std::string>{"prims.B"}, rc.get(), norm);
        }
    } // else yell?

    // Measure again to check
    if (verbose > 0) {
        Real bsq_max, p_max, beta_min;
        if (beta_calc_legacy) {
            bsq_max = MPIReduce_once(MaxBsq(md), MPI_MAX);
            p_max = MPIReduce_once(MaxPressure(md), MPI_MAX);
            beta_min = p_max / (0.5 * bsq_max);
        } else {
            beta_min = MPIReduce_once(MinBeta(md), MPI_MIN);
        }
        if (MPIRank0()) {
            if (beta_calc_legacy) {
                std::cout << "B^2 max post-norm: " << bsq_max << std::endl;
                std::cout << "Pressure max post-norm: " << p_max << std::endl;
            }
            std::cout << "Beta min post-norm: " << beta_min << std::endl;
        }
    }

    // We've been initializing/manipulating P
    Flux::MeshPtoU(md, IndexDomain::entire);

    EndFlag(); //NormBField
    return TaskStatus::complete;
}