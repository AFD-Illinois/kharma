/*
 *  File: radM1_solvers.hpp
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

#include "radM1.hpp"

#include "entropy.hpp"
#include "inverter.hpp"

#define RAD_LARGE (0.1 * std::numeric_limits<Real>::max())
#define RAD_SMALL (10.0 * std::numeric_limits<Real>::min())
#define RAD_EPS (10.0 * std::numeric_limits<Real>::epsilon())

namespace RadM1
{

KOKKOS_INLINE_FUNCTION Real compute_y_max(const Real GAMMAMAX)
{
    const Real target = GAMMAMAX * GAMMAMAX;
    Real y_old = 0.9998;
    Real E_old =
        target - (2.0 - y_old + m::sqrt(4.0 - 3.0 * y_old)) / (4.0 - 4.0 * y_old);
    for (int n = 0; n < 40; ++n) {
        Real dEdy = (0.375 * y_old - 0.25 * m::sqrt(4.0 - 3.0 * y_old) - 0.625) /
                    (m::sqrt(4.0 - 3.0 * y_old) * (1.0 - y_old) * (1.0 - y_old));
        Real y_new = m::min(y_old - E_old / dEdy, 0.99999999999999);
        Real E_new =
            target - (2.0 - y_new + m::sqrt(4.0 - 3.0 * y_new)) / (4.0 - 4.0 * y_new);
        y_old = y_new;
        if (m::abs(E_new) / target <= 1.e-9) break;
        E_old = E_new;
    }
    return y_old;
}

KOKKOS_INLINE_FUNCTION StatusRadiationInversion u_to_p_rad(const GRCoordinates& G,
    const Real U_rad[4], Real P_rad[4], const int k, const int j, const int i,
    bool* used_normal_out = nullptr, RadLimiterType type = RadLimiterType::TYPE2)
{
    Real gdet = G.gdet(Loci::center, j, i);

    if (U_rad[0] == 0.0 && U_rad[1] == 0.0 && U_rad[2] == 0.0 && U_rad[3] == 0.0) {
        P_rad[0] = 0.0;
        P_rad[1] = 0.0;
        P_rad[2] = 0.0;
        P_rad[3] = 0.0;
        if (used_normal_out != nullptr) *used_normal_out = true;
        return StatusRadiationInversion::success;
    }

    Real gcon_tt = G.gcon(Loci::center, j, i, 0, 0);
    Real alpha = m::sqrt(-1.0 / gcon_tt);

    Real Ucov_zamo[4];
    for (int mu = 0; mu < 4; ++mu) Ucov_zamo[mu] = alpha * U_rad[mu] / gdet;

    if (!m::isfinite(Ucov_zamo[0]) || !m::isfinite(Ucov_zamo[1]) ||
        !m::isfinite(Ucov_zamo[2]) || !m::isfinite(Ucov_zamo[3])) {
        P_rad[0] = 1.e-300;
        P_rad[1] = 0.0;
        P_rad[2] = 0.0;
        P_rad[3] = 0.0;
        if (used_normal_out != nullptr) *used_normal_out = false;
        return StatusRadiationInversion::not_finite;
    }

    Real Ucon_zamo[4];
    G.raise(Ucov_zamo, Ucon_zamo, k, j, i, Loci::center);

    Real eta_cov0 = -alpha;
    Real eta_con[4];
    for (int mu = 0; mu < 4; ++mu)
        eta_con[mu] = G.gcon(Loci::center, j, i, 0, mu) * eta_cov0;

    Real U_dot_eta = Ucon_zamo[0] * eta_cov0;

    Real Utilde_con[4] = {0.0, 0.0, 0.0, 0.0};
    for (int mu = 1; mu < 4; ++mu)
        Utilde_con[mu] = Ucon_zamo[mu] + eta_con[mu] * U_dot_eta;

    Real Utilde_sq = 0.0;
    for (int mu = 1; mu < 4; ++mu)
        for (int nu = 1; nu < 4; ++nu)
            Utilde_sq +=
                G.gcov(Loci::center, j, i, mu, nu) * Utilde_con[mu] * Utilde_con[nu];

    if (Utilde_sq < 0.0) {
        Utilde_sq = 0.0;
        Utilde_con[1] = 0.0;
        Utilde_con[2] = 0.0;
        Utilde_con[3] = 0.0;
    }

    const Real GAMMAMAX = 50.0;
    const Real y_max = compute_y_max(GAMMAMAX);

    Real y = Utilde_sq / (U_dot_eta * U_dot_eta + 1.e-150);
    Real gamma_rad_sq = (2.0 - y + m::sqrt(m::max(0.0, 4.0 - 3.0 * y))) / (4.0 - 4.0 * y);

    Real p_rad = -U_dot_eta / (4.0 * gamma_rad_sq - 1.0);
    Real Erf = p_rad * 3.0;

    Real uvec_rad[4] = {0.0, 0.0, 0.0, 0.0};
    for (int mu = 1; mu < 4; ++mu)
        uvec_rad[mu] =
            m::sqrt(gamma_rad_sq) * Utilde_con[mu] / (4.0 * p_rad * gamma_rad_sq);

    bool failed = (y > y_max) || (y < 0.0) || !m::isfinite(U_dot_eta) ||
                  (U_dot_eta > 0.0) || !m::isfinite(uvec_rad[1]) ||
                  !m::isfinite(uvec_rad[2]) || !m::isfinite(uvec_rad[3]);

    bool used_normal = !failed;
    if (used_normal_out != nullptr) *used_normal_out = used_normal;

    if (failed) {
        if (type == RadLimiterType::BASIC) {
            if (U_dot_eta > 0.0) {
                // Erad <= 0
                P_rad[0] = 1.e-300;
                P_rad[1] = 0.0;
                P_rad[2] = 0.0;
                P_rad[3] = 0.0;
                return StatusRadiationInversion::urad_negative;
            } else if (y < 0.0 && y > -RAD_EPS) {
                // 0 > y > -m: zero the velocity only, leave Erf as already computed above
                if (!m::isfinite(Erf)) Erf = 1.e-300;
                P_rad[0] = Erf;
                P_rad[1] = 0.0;
                P_rad[2] = 0.0;
                P_rad[3] = 0.0;
                return StatusRadiationInversion::negative_gamma;
            } else {
                // y > y_max: rescale directly to y = y_max with no intermediate
                // Uabs/GAMMAMAX step
                Real U_dot_eta_basic = -(1.e-30 + m::sqrt(m::abs(Utilde_sq) / y_max));
                Real gamma_rad_sq_basic =
                    (2.0 - y_max + m::sqrt(4.0 - 3.0 * y_max)) / (4.0 - 4.0 * y_max);
                Real p_rad_basic = -U_dot_eta_basic / (4.0 * gamma_rad_sq_basic - 1.0);
                Erf = p_rad_basic * 3.0;
                for (int mu = 1; mu < 4; ++mu)
                    uvec_rad[mu] = m::sqrt(gamma_rad_sq_basic) * Utilde_con[mu] /
                                   (4.0 * p_rad_basic * gamma_rad_sq_basic);

                if (!m::isfinite(Erf)) Erf = 1.e-300;
                if (!m::isfinite(uvec_rad[1])) uvec_rad[1] = 0.0;
                if (!m::isfinite(uvec_rad[2])) uvec_rad[2] = 0.0;
                if (!m::isfinite(uvec_rad[3])) uvec_rad[3] = 0.0;

                P_rad[0] = Erf;
                P_rad[1] = uvec_rad[1];
                P_rad[2] = uvec_rad[2];
                P_rad[3] = uvec_rad[3];
                return StatusRadiationInversion::maximum_gamma;
            }
        } else {
            // Do TYPE2 fallback
            Real Uabs = 0.5 * (m::abs(U_dot_eta) + m::sqrt(m::abs(Utilde_sq)) + 1.e-150);
            for (int mu = 1; mu < 4; ++mu)
                uvec_rad[mu] = GAMMAMAX * Utilde_con[mu] / Uabs;

            Real qsq =
                G.gcov(Loci::center, j, i, 1, 1) * uvec_rad[1] * uvec_rad[1] +
                G.gcov(Loci::center, j, i, 2, 2) * uvec_rad[2] * uvec_rad[2] +
                G.gcov(Loci::center, j, i, 3, 3) * uvec_rad[3] * uvec_rad[3] +
                2.0 * G.gcov(Loci::center, j, i, 1, 2) * uvec_rad[1] * uvec_rad[2] +
                2.0 * G.gcov(Loci::center, j, i, 1, 3) * uvec_rad[1] * uvec_rad[3] +
                2.0 * G.gcov(Loci::center, j, i, 2, 3) * uvec_rad[2] * uvec_rad[3];
            if (qsq < 0.0 || m::abs(qsq) < 1.e-10) qsq = 1.e-10;
            Real gamma_rad_sq_fb = 1.0 + qsq;

            Real f = m::sqrt((GAMMAMAX * GAMMAMAX - 1.0) / (gamma_rad_sq_fb - 1.0));
            uvec_rad[1] *= f;
            uvec_rad[2] *= f;
            uvec_rad[3] *= f;

            Real U_dot_eta_tp2 = -(1.e-30 + m::sqrt(m::abs(Utilde_sq) / y_max));
            Real gamma_rad_sq_tp2 =
                (2.0 - y_max + m::sqrt(4.0 - 3.0 * y_max)) / (4.0 - 4.0 * y_max);
            Real p_rad_tp2 = -U_dot_eta_tp2 / (4.0 * gamma_rad_sq_tp2 - 1.0);
            Erf = p_rad_tp2 * 3.0;
            for (int mu = 1; mu < 4; ++mu)
                uvec_rad[mu] = m::sqrt(gamma_rad_sq_tp2) * Utilde_con[mu] /
                               (4.0 * p_rad_tp2 * gamma_rad_sq_tp2);
            if (!m::isfinite(Erf)) Erf = 1.e-300;
            if (!m::isfinite(uvec_rad[1])) uvec_rad[1] = 0.0;
            if (!m::isfinite(uvec_rad[2])) uvec_rad[2] = 0.0;
            if (!m::isfinite(uvec_rad[3])) uvec_rad[3] = 0.0;
            P_rad[0] = Erf;
            P_rad[1] = uvec_rad[1];
            P_rad[2] = uvec_rad[2];
            P_rad[3] = uvec_rad[3];
            return StatusRadiationInversion::type2;
        }
    }

    if (!m::isfinite(Erf)) Erf = 1.e-300;
    if (!m::isfinite(uvec_rad[1])) uvec_rad[1] = 0.0;
    if (!m::isfinite(uvec_rad[2])) uvec_rad[2] = 0.0;
    if (!m::isfinite(uvec_rad[3])) uvec_rad[3] = 0.0;

    P_rad[0] = Erf;
    P_rad[1] = uvec_rad[1];
    P_rad[2] = uvec_rad[2];
    P_rad[3] = uvec_rad[3];

    return StatusRadiationInversion::success;
}

KOKKOS_INLINE_FUNCTION void compute_covariant_fourforce(const GRCoordinates& G,
    const Real P_mhd[4], const Real P_rad[4], const Real rho, const Real B_P[NVEC],
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const int k, const int j,
    const int i, Real dS[4], Real* kappa_tot_out = nullptr, Real* coupling_out = nullptr)
{

    if (static_cast<RadM1::OpacityType>(rad_opac.opacity_type) ==
        RadM1::OpacityType::Transparent) {
        dS[0] = 0.0;
        dS[1] = 0.0;
        dS[2] = 0.0;
        dS[3] = 0.0;
        return;
    }

    Real uvec_mhd[3] = {P_mhd[1], P_mhd[2], P_mhd[3]};
    Real ucon_mhd[4], ucov_mhd[4];
    GRMHD::calc_ucon(G, uvec_mhd, k, j, i, Loci::center, ucon_mhd);
    G.lower(ucon_mhd, ucov_mhd, k, j, i, Loci::center);

    Real Erf = P_rad[0];
    Real ucon_rad[4], ucov_rad[4];
    RadM1::calc_ucon_rad(G, P_rad, j, i, ucon_rad);
    G.lower(ucon_rad, ucov_rad, k, j, i, Loci::center);

    Real gamma_rel = -(ucon_rad[0] * ucov_mhd[0] + ucon_rad[1] * ucov_mhd[1] +
                       ucon_rad[2] * ucov_mhd[2] + ucon_rad[3] * ucov_mhd[3]);
    gamma_rel = m::max(gamma_rel, 1.0);

    Real E_hat = Erf * ((4.0 / 3.0) * gamma_rel * gamma_rel - (1.0 / 3.0));
    Real F_hat_cov[4];
    for (int mu = 0; mu < 4; ++mu) {
        F_hat_cov[mu] = (4.0 / 3.0) * Erf * gamma_rel * ucov_rad[mu] -
                        ((1.0 / 3.0) * Erf + E_hat) * ucov_mhd[mu];
    }

    Real Tg = eos.TemperatureFromDensityInternalEnergy(rho, P_mhd[0] / rho);
    Real bcon[4] = {0., 0., 0., 0.};
    Real bcov[4] = {0., 0., 0., 0.};
    VLOOP
        bcon[0] += B_P[v] * ucov_mhd[v + 1];
    VLOOP
        bcon[v + 1] = (B_P[v] + bcon[0] * ucon_mhd[v + 1]) / ucon_mhd[0];

    G.lower(bcon, bcov, k, j, i, Loci::center);
    Real bsq = dot(bcov, bcon);
    Real Trad = rad_opac.Trad(E_hat, rho, bsq);

    Real kappa_abs_rad = RadM1::calc_kabs(rho, Tg, Trad, bsq, rad_opac); // phi uses Trad
    Real kappa_emit_gas =
        RadM1::calc_kabs(rho, Tg, Tg, bsq, rad_opac); // phi uses Tg (Kirchhoff)
    Real kappa_sc = RadM1::calc_kscattering(rho, Tg, bsq, rad_opac);
    Real JBB_val = rad_opac.JBB(Tg, rho, bsq);

    Real kappa_tot = kappa_abs_rad + kappa_sc;

    Real coupling_term = kappa_emit_gas * JBB_val - kappa_abs_rad * E_hat;
    dS[0] = coupling_term * ucov_mhd[0] - kappa_tot * F_hat_cov[0];
    dS[1] = coupling_term * ucov_mhd[1] - kappa_tot * F_hat_cov[1];
    dS[2] = coupling_term * ucov_mhd[2] - kappa_tot * F_hat_cov[2];
    dS[3] = coupling_term * ucov_mhd[3] - kappa_tot * F_hat_cov[3];
    if (kappa_tot_out != nullptr)
        *kappa_tot_out = kappa_sc + m::max(kappa_abs_rad, kappa_emit_gas);
    if (coupling_out != nullptr) *coupling_out = coupling_term;
}

KOKKOS_INLINE_FUNCTION Real calculate_energy_residual(const GRCoordinates& G,
    const Real u_trial, const Real uvec_frozen[NVEC], const Real B_P[NVEC],
    const Real U_mhd_0[4], const Real U_rad_0[4], const Real rho,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const Real dt,
    const Real gdet, const int k, const int j, const int i, Real U_mhd_trial_out[4],
    Real U_rad_trial_out[4], Real P_rad_trial_out[4], Real dS_trial_out[4],
    bool& rad_recovery_ok)
{
    Real uvec[NVEC] = {uvec_frozen[0], uvec_frozen[1], uvec_frozen[2]};
    Real rho_new;
    GRMHD::p_to_u_mhd(
        G, rho, u_trial, uvec, B_P, eos, k, j, i, rho_new, U_mhd_trial_out, Loci::center);

    for (int n = 0; n < 4; n++) {
        U_rad_trial_out[n] = U_rad_0[n] - (U_mhd_trial_out[n] - U_mhd_0[n]);
    }

    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
    rho_new = rho_new / (gdet * Dtmp.ucon[0]);

    auto status = u_to_p_rad(G, U_rad_trial_out, P_rad_trial_out, k, j, i);
    rad_recovery_ok = (status == StatusRadiationInversion::success);

    Real P_mhd_trial[4] = {u_trial, uvec_frozen[0], uvec_frozen[1], uvec_frozen[2]};
    compute_covariant_fourforce(G, P_mhd_trial, P_rad_trial_out, rho_new, B_P, eos,
        rad_opac, k, j, i, dS_trial_out);
    for (int n = 0; n < 4; n++) dS_trial_out[n] = gdet * dS_trial_out[n];

    Real resid = (U_mhd_trial_out[0] - U_mhd_0[0]) + dt * dS_trial_out[0];
    Real scale = m::max(RAD_SMALL,
        m::abs(U_mhd_trial_out[0]) + m::abs(U_mhd_0[0]) + m::abs(dt * dS_trial_out[0]));
    return resid / scale;
}

KOKKOS_INLINE_FUNCTION StatusImplicitStep solve_radiation_1d(const GRCoordinates& G,
    const VariablePack<Real> P_init, const VarMap m_p, const VarMap m_u,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const int k, const int j,
    const int i, const Real dt, const double tol, const int maxiter,
    const VariablePack<Real> pflag, const VariablePack<Real> rinvflag,
    const Real U_entry[8], Real dS_final[5])
{

    const Real gdet = G.gdet(Loci::center, j, i);
    const Real uvec_frozen[NVEC] = {
        P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0) {
        B_P[0] = P_init(m_p.B1, k, j, i);
        B_P[1] = P_init(m_p.B2, k, j, i);
        B_P[2] = P_init(m_p.B3, k, j, i);
    }
    const Real U_mhd_0[4] = {U_entry[0], U_entry[1], U_entry[2], U_entry[3]};
    const Real U_rad_0[4] = {U_entry[4], U_entry[5], U_entry[6], U_entry[7]};
    const Real rho_init = P_init(m_p.RHO, k, j, i);
    const Real u_init = P_init(m_p.UU, k, j, i);

    Real U_mhd_trial[4], U_rad_trial[4], P_rad_trial[4], dS_trial[4];
    bool rad_ok;

    Real u_lo = 1.e-2 * u_init;
    Real u_hi = 1.e2 * u_init;
    Real f_lo = calculate_energy_residual(G, u_lo, uvec_frozen, B_P, U_mhd_0, U_rad_0,
        rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial,
        dS_trial, rad_ok);
    Real f_hi = calculate_energy_residual(G, u_hi, uvec_frozen, B_P, U_mhd_0, U_rad_0,
        rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial,
        dS_trial, rad_ok);

    bool bracketed = (f_lo * f_hi < 0.0);
    Real rebracket_fac = 10.0;
    int n_rebracket = 0;
    const int MAX_REBRACKET = 4;
    while (!bracketed && n_rebracket < MAX_REBRACKET) {
        u_lo = (1.e-1 / rebracket_fac) * u_init;
        u_hi = (1.e1 * rebracket_fac) * u_init;
        f_lo = calculate_energy_residual(G, u_lo, uvec_frozen, B_P, U_mhd_0, U_rad_0,
            rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial,
            P_rad_trial, dS_trial, rad_ok);
        f_hi = calculate_energy_residual(G, u_hi, uvec_frozen, B_P, U_mhd_0, U_rad_0,
            rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial,
            P_rad_trial, dS_trial, rad_ok);
        bracketed = (f_lo * f_hi < 0.0);
        rebracket_fac *= 10.0;
        n_rebracket++;
    }

    if (!bracketed) {
        return StatusImplicitStep::failure;
    }

    Real u_root = 0.5 * (u_lo + u_hi);
    bool converged = false;
    int stuck_lo = 0, stuck_hi = 0;
    for (int iter = 0; iter < maxiter; iter++) {
        u_root = (u_lo * f_hi - u_hi * f_lo) / (f_hi - f_lo);
        Real f_root = calculate_energy_residual(G, u_root, uvec_frozen, B_P, U_mhd_0,
            U_rad_0, rho_init, eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial,
            P_rad_trial, dS_trial, rad_ok);

        if (!rad_ok) {
            if (f_root * f_lo > 0.0) {
                u_lo = u_root;
                f_lo = f_root;
                stuck_hi++;
                stuck_lo = 0;
            } else {
                u_hi = u_root;
                f_hi = f_root;
                stuck_lo++;
                stuck_hi = 0;
            }
            continue;
        }

        if (m::abs(f_root) < tol || m::abs(u_hi - u_lo) < tol * u_init) {
            converged = true;
            break;
        }

        if (f_root * f_lo > 0.0) {
            u_lo = u_root;
            f_lo = f_root;
            stuck_hi++;
            stuck_lo = 0;
            if (stuck_hi >= 2) {
                f_hi *= 0.5;
                stuck_hi = 0;
            }
        } else {
            u_hi = u_root;
            f_hi = f_root;
            stuck_lo++;
            stuck_hi = 0;
            if (stuck_lo >= 2) {
                f_lo *= 0.5;
                stuck_lo = 0;
            }
        }
    }

    if (!converged) {
        return StatusImplicitStep::onedfallback_failure;
    }

    calculate_energy_residual(G, u_root, uvec_frozen, B_P, U_mhd_0, U_rad_0, rho_init,
        eos, rad_opac, dt, gdet, k, j, i, U_mhd_trial, U_rad_trial, P_rad_trial, dS_trial,
        rad_ok);
    if (!rad_ok) {
        return StatusImplicitStep::onedfallback_failure;
    }

    dS_final[0] = dS_trial[0];
    dS_final[1] = dS_trial[1];
    dS_final[2] = dS_trial[2];
    dS_final[3] = dS_trial[3];

    {
        Real ucon_final[4];
        GRMHD::calc_ucon(G, uvec_frozen, k, j, i, Loci::center, ucon_final);
        Real Gdotu = dS_trial[0] * ucon_final[0] + dS_trial[1] * ucon_final[1] +
                     dS_trial[2] * ucon_final[2] + dS_trial[3] * ucon_final[3];
        dS_final[4] = Gdotu;
    }

    return StatusImplicitStep::success;
}

KOKKOS_INLINE_FUNCTION Real calculate_error(const Real resid[4],
    const Real U_mhd_guess[4], const Real U_mhd_0[4], const Real dtdS[4])
{
    constexpr Real KORAL_SMALL = 1.e-80;

    Real err[4];

    // Energy component (koral's err[0], MHD-lab-frame energy case):
    // err[0] = |f[0]| / (|uu[UU]| + |uu0[UU]| + |dt*gdetu*Gi[0]|)
    if (m::abs(resid[0]) > KORAL_SMALL) {
        err[0] = m::abs(resid[0]) /
                 (m::abs(U_mhd_guess[0]) + m::abs(U_mhd_0[0]) + m::abs(dtdS[0]));
    } else {
        err[0] = 0.0;
    }

    // Momentum components each normalized only by its
    // own history, plus a small floor tied to the energy component's own
    // magnitude (1e-20*uu[UU])
    for (int n = 1; n < 4; n++) {
        if (m::abs(resid[n]) > KORAL_SMALL) {
            err[n] = m::abs(resid[n]) / (m::abs(U_mhd_guess[0]) + m::abs(U_mhd_guess[n]) +
                                            m::abs(U_mhd_0[n]) + m::abs(dtdS[n]));
        } else {
            err[n] = 0.0;
        }
    }

    Real err_max = 0.0;
    for (int n = 0; n < 4; n++) {
        err_max = m::max(err_max, err[n]);
    }
    return err_max;
}

KOKKOS_INLINE_FUNCTION Real rho_from_D(const GRCoordinates& G, const Real D,
    const Real uvec[NVEC], const int k, const int j, const int i)
{
    Real uv[NVEC] = {uvec[0], uvec[1], uvec[2]};
    Real ucon[GR_DIM];
    GRMHD::calc_ucon(G, uv, k, j, i, Loci::center, ucon);
    return D / ucon[0];
}

KOKKOS_INLINE_FUNCTION int solve_4d_pmhd(const GRCoordinates& G,
    const VariablePack<Real> P_init, const VarMap m_p, const VarMap m_u, const int k,
    const int j, const int i, const Real dt, const Microphysics::EOS::EOS& eos,
    const double src_rootfind_eps, const double src_rootfind_tol,
    const int src_rootfind_maxiter, const RadOpac& rad_opac,
    const VariablePack<Real> pflag, const VariablePack<Real> rinvflag,
    const Real U_entry[8], Real dS_final[5])
{
    const Real rho_init = P_init(m_p.RHO, k, j, i);

    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0) {
        B_P[V1] = P_init(m_p.B1, k, j, i);
        B_P[V2] = P_init(m_p.B2, k, j, i);
        B_P[V3] = P_init(m_p.B3, k, j, i);
    }
    Real P_mhd_guess[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
        P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};

    Real U_rad_0[4] = {U_entry[4], U_entry[5], U_entry[6], U_entry[7]};

    Real resid[4];

    Real U_mhd_0[4];

    Real uvec[NVEC] = {
        P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
    const Real D_rho = rho_init / rho_from_D(G, 1.0, uvec, k, j, i);

    Real U_mhd_guess[4];
    Real U_rad_guess[4];
    Real P_rad_guess[4];
    Real dS_guess[4];

    U_mhd_0[0] = U_entry[0];
    U_mhd_0[1] = U_entry[1];
    U_mhd_0[2] = U_entry[2];
    U_mhd_0[3] = U_entry[3];

    // Iteration 0
    U_mhd_guess[0] = U_mhd_0[0];
    U_mhd_guess[1] = U_mhd_0[1];
    U_mhd_guess[2] = U_mhd_0[2];
    U_mhd_guess[3] = U_mhd_0[3];

    // Conservation law: Delta U_rad = - Delta U_mhd
    DLOOP1
        U_rad_guess[mu] = U_rad_0[mu];
    Real gdet = G.gdet(Loci::center, j, i);

    bool used_normal_guess = true;
    u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i, &used_normal_guess);
    Real kappa_tot_entry = 0.0;
    compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, rho_init, B_P, eos, rad_opac,
        k, j, i, dS_guess, &kappa_tot_entry);
    const Real tol_eff = m::max(static_cast<Real>(src_rootfind_tol),
        10.0 * std::numeric_limits<Real>::epsilon() * kappa_tot_entry * dt);

    for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

    DLOOP1 {
        resid[mu] = U_mhd_guess[mu] - U_mhd_0[mu] + dt * dS_guess[mu];
    }

    // Compute err here and do a convergence check
    Real dtdS_0[4];
    for (int n = 0; n < 4; n++) dtdS_0[n] = dt * dS_guess[n];

    Real err = calculate_error(resid, U_mhd_guess, U_mhd_0, dtdS_0);
    int niter = 0;
    bool bad_guess = false;

    Real rho_iter = rho_init;

    do {
        if (err <= tol_eff) {
            break;
        }

        Real P_rad_m[4];
        Real P_rad_p[4];
        Real U_mhd_m[4];
        Real U_mhd_p[4];
        Real U_rad_m[4];
        Real U_rad_p[4];
        Real dS_m[4];
        Real dS_p[4];

        Real jac[4][4] = {0};

        // Find minimum non-zero magnitude from P_mhd_guess to scale FD step safely
        Real P_mhd_mag_min = RAD_LARGE;
        for (int m = 0; m < 4; m++) {
            if (m::abs(P_mhd_guess[m]) > 0.) {
                P_mhd_mag_min = m::min(P_mhd_mag_min, m::abs(P_mhd_guess[m]));
            }
        }

        bool bad_guess_m = false;
        bool bad_guess_p = false;

        // Loop over the 4 fluid variables to perturb each one
        for (int m = 0; m < 4; m++) {
            Real P_mhd_m[4] = {
                P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
            Real P_mhd_p[4] = {
                P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};

            const Real fd_step = m::max(src_rootfind_eps * P_mhd_mag_min,
                src_rootfind_eps * m::abs(P_mhd_guess[m]));
            P_mhd_m[m] -= fd_step;
            P_mhd_p[m] += fd_step;

            // Evaluate minus perturbation
            Real rho_m;
            uvec[0] = P_mhd_m[1];
            uvec[1] = P_mhd_m[2];
            uvec[2] = P_mhd_m[3];

            GRMHD::p_to_u_mhd(G, rho_from_D(G, D_rho, uvec, k, j, i), P_mhd_m[0], uvec,
                B_P, eos, k, j, i, rho_m, U_mhd_m, Loci::center);

            // The rho output from p_to_u is rho_ut_gdet;
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
            rho_m = rho_m / (gdet * Dtmp.ucon[0]);
            // Conservation law: Delta U_rad = - Delta U_mhd

            for (int n = 0; n < 4; n++) {
                U_rad_m[n] = U_rad_0[n] - (U_mhd_m[n] - U_mhd_0[n]);
            }

            // Recover rad primitives
            bool used_normal_m;
            auto status_m = u_to_p_rad(G, U_rad_m, P_rad_m, k, j, i, &used_normal_m);
            if (status_m != StatusRadiationInversion::success) {
                bad_guess_m = true;
            }

            if (used_normal_m != used_normal_guess) {
                bad_guess_m = true;
            }

            // If a bad guess has already been found before, we can't really skip the
            // whole Jacobian evaluation. We need to keep evaluating the other blocks to
            // figure out if, during the next m's, the other side (plus/minus) will also
            // go bad, trigerring a bad_guess_m == true && bad_guess_p == true
            if (!bad_guess_m && !bad_guess_p) {
                compute_covariant_fourforce(
                    G, P_mhd_m, P_rad_m, rho_m, B_P, eos, rad_opac, k, j, i, dS_m);
                for (int n = 0; n < 4; n++) dS_m[n] = gdet * dS_m[n];
            }
            // Evaluate plus perturbation
            Real rho_p;
            uvec[0] = P_mhd_p[1];
            uvec[1] = P_mhd_p[2];
            uvec[2] = P_mhd_p[3];

            GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);

            GRMHD::p_to_u_mhd(G, rho_from_D(G, D_rho, uvec, k, j, i), P_mhd_p[0], uvec,
                B_P, eos, k, j, i, rho_p, U_mhd_p, Loci::center);

            // The rho output from p_to_u is rho_ut_gdet;
            rho_p = rho_p / (gdet * Dtmp.ucon[0]);

            for (int n = 0; n < 4; n++) {
                U_rad_p[n] = U_rad_0[n] - (U_mhd_p[n] - U_mhd_0[n]);
            }
            // TODO (PNM): Change name of KOKKOS kernel to lower snake case
            bool used_normal_p;
            auto status_p = u_to_p_rad(G, U_rad_p, P_rad_p, k, j, i, &used_normal_p);
            if (status_p != StatusRadiationInversion::success) {
                bad_guess_p = true;
            }
            if (used_normal_p != used_normal_guess) {
                bad_guess_p = true;
            }

            // If a bad guess has already been found before, we can't really skip the
            // whole Jacobian evaluation. We need to keep evaluating the other blocks to
            // figure out if, during the next m's, the other side (plus/minus) will also
            // go bad, trigerring a bad_guess_m == true && bad_guess_p == true
            if (!bad_guess_m && !bad_guess_p) {
                compute_covariant_fourforce(
                    G, P_mhd_p, P_rad_p, rho_p, B_P, eos, rad_opac, k, j, i, dS_p);
                for (int n = 0; n < 4; n++) dS_p[n] = gdet * dS_p[n];

                // Populate Jacobian
                for (int n = 0; n < 4; n++) {
                    Real fp = U_mhd_p[n] - U_mhd_0[n] + dt * dS_p[n];
                    Real fm = U_mhd_m[n] - U_mhd_0[n] + dt * dS_m[n];
                    // Jacobian here is dU_rad/dP_mhd
                    // Since div R^mu_nu = G_nu
                    // and div T^mu_nu = -G_nu
                    jac[n][m] = (fp - fm) / (P_mhd_p[m] - P_mhd_m[m]);
                }
            }
        }
        // TODO (PNM): Separate these if/elses into different kernels.
        if (bad_guess_m == true && bad_guess_p == true) {
            bad_guess = true;
            break; // Exit the iteration loop if both perturbations yield bad guesses
        } else if (bad_guess_m == true) {
            // If only - finite difference support point is bad, do one-sided
            // difference with + support point
            bool onesided_mismatch = false;

            for (int m = 0; m < 4; m++) {
                Real P_mhd_p[4] = {
                    P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
                P_mhd_p[m] += std::max(src_rootfind_eps * P_mhd_mag_min,
                    src_rootfind_eps * m::abs(P_mhd_p[m]));

                Real rho_p;
                uvec[0] = P_mhd_p[1];
                uvec[1] = P_mhd_p[2];
                uvec[2] = P_mhd_p[3];
                // TODO (PNM): This is stupid since we have already computed the plus
                // perturbation above. PNM: Actually, I don't know if that's actually
                // stupid, we would need to save a lot of 4x4 matrices to get this
                // working. it's a trade-off between register pressure and doing a few
                // more calculations, which I think would be more efficient.
                GRMHD::p_to_u_mhd(G, rho_from_D(G, D_rho, uvec, k, j, i), P_mhd_p[0],
                    uvec, B_P, eos, k, j, i, rho_p, U_mhd_p, Loci::center);
                FourVectors Dtmp;
                GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);

                rho_p = rho_p / (gdet * Dtmp.ucon[0]);

                for (int n = 0; n < 4; n++) {
                    U_rad_p[n] = U_rad_0[n] - (U_mhd_p[n] - U_mhd_0[n]);
                }
                bool used_normal_p;
                auto status_p = u_to_p_rad(G, U_rad_p, P_rad_p, k, j, i, &used_normal_p);
                compute_covariant_fourforce(
                    G, P_mhd_p, P_rad_p, rho_p, B_P, eos, rad_opac, k, j, i, dS_p);
                for (int n = 0; n < 4; n++) dS_p[n] = gdet * dS_p[n];

                if (used_normal_p != used_normal_guess) {
                    onesided_mismatch = true;
                    break;
                }

                for (int n = 0; n < 4; n++) {
                    Real fp = U_mhd_p[n] - U_mhd_0[n] + dt * dS_p[n];
                    Real fguess = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
                    // Jacobian here is dU_rad/dP_mhd
                    // Since div R^mu_nu = G_nu
                    // and div T^mu_nu = -G_nu
                    jac[n][m] = (fp - fguess) / (P_mhd_p[m] - P_mhd_guess[m]);
                }
            }
            if (onesided_mismatch) {
                bad_guess = true;
                break;
            }
        } else if (bad_guess_p == true) {
            // If only + finite difference support point is bad, do one-sided
            // difference with - support point
            bool onesided_mismatch = false;

            for (int m = 0; m < 4; m++) {
                Real P_mhd_m[4] = {
                    P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
                P_mhd_m[m] -= std::max(src_rootfind_eps * P_mhd_mag_min,
                    src_rootfind_eps * m::abs(P_mhd_m[m]));

                Real rho_m;
                uvec[0] = P_mhd_m[1];
                uvec[1] = P_mhd_m[2];
                uvec[2] = P_mhd_m[3];
                GRMHD::p_to_u_mhd(G, rho_from_D(G, D_rho, uvec, k, j, i), P_mhd_m[0],
                    uvec, B_P, eos, k, j, i, rho_m, U_mhd_m, Loci::center);
                FourVectors Dtmp;
                GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
                rho_m = rho_m / (gdet * Dtmp.ucon[0]);

                for (int n = 0; n < 4; n++) {
                    U_rad_m[n] = U_rad_0[n] - (U_mhd_m[n] - U_mhd_0[n]);
                }
                bool used_normal_m;
                auto status_m = u_to_p_rad(G, U_rad_m, P_rad_m, k, j, i, &used_normal_m);
                compute_covariant_fourforce(
                    G, P_mhd_m, P_rad_m, rho_m, B_P, eos, rad_opac, k, j, i, dS_m);
                for (int n = 0; n < 4; n++) dS_m[n] = gdet * dS_m[n];

                // See the mirror-image comment in the bad_guess_m branch above.
                if (used_normal_m != used_normal_guess) {
                    onesided_mismatch = true;
                    break;
                }

                for (int n = 0; n < 4; n++) {
                    Real fm = U_mhd_m[n] - U_mhd_0[n] + dt * dS_m[n];
                    Real fguess = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
                    jac[n][m] = (fguess - fm) / (P_mhd_guess[m] - P_mhd_m[m]);
                }
            }
            if (onesided_mismatch) {
                bad_guess = true;
                break;
            }
        }

        Real jacinv[4][4];
        // Inverting the 4x4 matrix;
        invert(&jac[0][0], &jacinv[0][0]);

        // Update guess via a damped Newton step, e.g, we take the full step, and if
        // it violates any of the checks below, shrink the step and retry.
        // Currently we just shrink it by 3 with a max of 15 iterations or if
        // scailing_factor < 1.e-8;
        Real P_mhd_pre[4] = {
            P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        Real scaling_factor = 1.0;
        int track = 0;
        bool step_ok = false;
        Real rho_iter_next;
        do {
            for (int m = 0; m < 4; m++) {
                P_mhd_guess[m] = P_mhd_pre[m];
                for (int n = 0; n < 4; n++) {
                    P_mhd_guess[m] -= scaling_factor * jacinv[m][n] * resid[n];
                }
            }

            if (P_mhd_guess[0] <= 0.0) {
                scaling_factor /= 3.0;
                track++;
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }

            // Re-evaluate residual with updated guess
            uvec[0] = P_mhd_guess[1];
            uvec[1] = P_mhd_guess[2];
            uvec[2] = P_mhd_guess[3];
            GRMHD::p_to_u_mhd(G, rho_from_D(G, D_rho, uvec, k, j, i), P_mhd_guess[0],
                uvec, B_P, eos, k, j, i, rho_iter_next, U_mhd_guess, Loci::center);

            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
            rho_iter_next = rho_iter_next / (gdet * Dtmp.ucon[0]);

            for (int n = 0; n < 4; n++) {
                U_rad_guess[n] = U_rad_0[n] - (U_mhd_guess[n] - U_mhd_0[n]);
            }

            auto status =
                u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i, &used_normal_guess);
            compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, rho_iter_next, B_P,
                eos, rad_opac, k, j, i, dS_guess);

            for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

            if (status != StatusRadiationInversion::success) {
                scaling_factor /= 3.0;
                track++;
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }

            step_ok = true;
            rho_iter = rho_iter_next;
        } while (!step_ok && track <= 15);

        if (track > 15 || scaling_factor < 1.e-8) {
            bad_guess = true;
            break;
        }

        // Update residuals
        for (int n = 0; n < 4; n++) {
            resid[n] = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];

            if (!m::isfinite(resid[n])) {
                bad_guess = true;
                break;
            }
        }

        // This is needed since the previous bad_guess = true would only break out of the
        // for loop
        if (bad_guess) {
            break;
        }

        // Calculate error now
        Real dtdS_iter[4];
        for (int n = 0; n < 4; n++) dtdS_iter[n] = dt * dS_guess[n];

        err = calculate_error(resid, U_mhd_guess, U_mhd_0, dtdS_iter);

        niter++;
    } while (err > tol_eff && niter < src_rootfind_maxiter);

    if (err > tol_eff || !m::isfinite(U_rad_guess[0]) || !m::isfinite(U_rad_guess[1]) ||
        !m::isfinite(U_rad_guess[2]) || !m::isfinite(U_rad_guess[3]) || bad_guess) {

        return static_cast<int>(StatusImplicitStep::failure);
    }

    {
        Real uvec_g[NVEC] = {P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        FourVectors D_g;
        GRMHD::calc_4vecs(G, uvec_g, B_P, k, j, i, Loci::center, D_g);
        const Real w_gas =
            rho_iter + P_mhd_guess[0] +
            eos.PressureFromDensityInternalEnergy(rho_iter, P_mhd_guess[0] / rho_iter) +
            dot(D_g.bcon, D_g.bcov);
        const bool e_rad = P_rad_guess[0] > P_mhd_guess[0];
        const bool m_rad = (4.0 / 3.0) * P_rad_guess[0] > w_gas;
        for (int n = 0; n < 4; n++) {
            const bool solved_state = (n == 0) ? e_rad : m_rad;
            dS_final[n] = solved_state ? (U_rad_guess[n] - U_rad_0[n]) / dt : dS_guess[n];
        }
    }

    {
        Real uvec_final[NVEC] = {P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        Real ucon_final[4];
        GRMHD::calc_ucon(G, uvec_final, k, j, i, Loci::center, ucon_final);
        Real Gdotu = dS_guess[0] * ucon_final[0] + dS_guess[1] * ucon_final[1] +
                     dS_guess[2] * ucon_final[2] + dS_guess[3] * ucon_final[3];
        dS_final[4] = Gdotu;
    }

    return static_cast<int>(StatusImplicitStep::success);
}

KOKKOS_INLINE_FUNCTION int prad_comoving_energy_residual(const GRCoordinates& G,
    const VariablePack<Real> U_init, const VariablePack<Real> P_init, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const Real B_P[NVEC],
    const Real dt, const Real gdet, const Real U_mhd_0[4], const Real U_rad_0[4],
    Real P_rad[4], Real& f, Real& scale)
{
    for (int it = 0; it < 40; it++) {
        Real U_rad[4];
        RadM1::calc_tensor(G, P_rad, 0, j, i, U_rad);
        for (int n = 0; n < 4; n++) U_rad[n] *= gdet;
        U_init(m_u.UU, k, j, i) = U_mhd_0[0] - (U_rad[0] - U_rad_0[0]);
        U_init(m_u.U1, k, j, i) = U_mhd_0[1] - (U_rad[1] - U_rad_0[1]);
        U_init(m_u.U2, k, j, i) = U_mhd_0[2] - (U_rad[2] - U_rad_0[2]);
        U_init(m_u.U3, k, j, i) = U_mhd_0[3] - (U_rad[3] - U_rad_0[3]);
        const int status = Inverter::u_to_p<Inverter::Type::kastaun>(
            G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
        if (status != static_cast<int>(Inverter::Status::success)) return status;
        const Real rho_g = P_init(m_p.RHO, k, j, i);
        Real P_mhd[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
            P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
        Real uvec_g[NVEC] = {P_mhd[1], P_mhd[2], P_mhd[3]};
        FourVectors D_g;
        GRMHD::calc_4vecs(G, uvec_g, B_P, k, j, i, Loci::center, D_g);
        const Real w_g = rho_g + P_mhd[0] +
                         eos.PressureFromDensityInternalEnergy(rho_g, P_mhd[0] / rho_g) +
                         dot(D_g.bcon, D_g.bcov);
        const Real lw_g = w_g * GRMHD::lorentz_calc(G, uvec_g, k, j, i, Loci::center);
        const Real lw_r =
            (4.0 / 3.0) * P_rad[0] * RadM1::lorentz_calc_rad(G, P_rad, j, i);
        Real uvec_new[NVEC];
        Real dv = 0.0, vn = 1.0;
        for (int v = 0; v < NVEC; v++) {
            uvec_new[v] = (lw_g * uvec_g[v] + lw_r * P_rad[v + 1]) / (lw_g + lw_r);
            dv = m::max(dv, m::abs(uvec_new[v] - P_rad[v + 1]));
            vn = m::max(vn, m::abs(uvec_new[v]));
        }
        if (dv <= 1.e-13 * vn) {
            Real dS[4], coupling;
            compute_covariant_fourforce(G, P_mhd, P_rad, rho_g, B_P, eos, rad_opac, k, j,
                i, dS, nullptr, &coupling);
            Real ucon_g[GR_DIM];
            GRMHD::calc_ucon(G, uvec_g, k, j, i, Loci::center, ucon_g);
            const Real src = dt * gdet * coupling;
            f = src;
            scale = m::abs(src);
            for (int mu = 0; mu < GR_DIM; mu++) {
                f += ucon_g[mu] * (U_rad[mu] - U_rad_0[mu]);
                scale +=
                    m::abs(ucon_g[mu] * U_rad[mu]) + m::abs(ucon_g[mu] * U_rad_0[mu]);
            }
            return status;
        }
        for (int v = 0; v < NVEC; v++) P_rad[v + 1] = uvec_new[v];
    }
    return static_cast<int>(Inverter::Status::max_iter);
}

KOKKOS_INLINE_FUNCTION int prad_residual_at(const GRCoordinates& G,
    const VariablePack<Real> U_init, const VariablePack<Real> P_init, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i,
    const Microphysics::EOS::EOS& eos, const RadOpac& rad_opac, const Real B_P[NVEC],
    const Real dt, const Real gdet, const Real U_mhd_0[4], const Real U_rad_0[4],
    const Real P_rad[4], Real resid[4], Real U_rad[4], Real dtdS[4], Real P_mhd[4],
    Real dS[4])
{
    RadM1::calc_tensor(G, P_rad, 0, j, i, U_rad);
    for (int n = 0; n < 4; n++) U_rad[n] *= gdet;
    U_init(m_u.UU, k, j, i) = U_mhd_0[0] - (U_rad[0] - U_rad_0[0]);
    U_init(m_u.U1, k, j, i) = U_mhd_0[1] - (U_rad[1] - U_rad_0[1]);
    U_init(m_u.U2, k, j, i) = U_mhd_0[2] - (U_rad[2] - U_rad_0[2]);
    U_init(m_u.U3, k, j, i) = U_mhd_0[3] - (U_rad[3] - U_rad_0[3]);
    const int status = Inverter::u_to_p<Inverter::Type::kastaun>(
        G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
    if (status != static_cast<int>(Inverter::Status::success)) return status;
    P_mhd[0] = P_init(m_p.UU, k, j, i);
    P_mhd[1] = P_init(m_p.U1, k, j, i);
    P_mhd[2] = P_init(m_p.U2, k, j, i);
    P_mhd[3] = P_init(m_p.U3, k, j, i);
    compute_covariant_fourforce(
        G, P_mhd, P_rad, P_init(m_p.RHO, k, j, i), B_P, eos, rad_opac, k, j, i, dS);
    for (int n = 0; n < 4; n++) {
        dS[n] *= gdet;
        dtdS[n] = dt * dS[n];
        resid[n] = U_rad[n] - U_rad_0[n] - dtdS[n];
    }
    return status;
}

KOKKOS_INLINE_FUNCTION int solve_4d_prad(const GRCoordinates& G,
    const VariablePack<Real> U_init, const VariablePack<Real> P_init, const VarMap m_p,
    const VarMap m_u, const int k, const int j, const int i, const Real dt,
    const Microphysics::EOS::EOS& eos, const double src_rootfind_eps,
    const double src_rootfind_tol, const int src_rootfind_maxiter,
    const RadOpac& rad_opac, const VariablePack<Real> pflag,
    const VariablePack<Real> rinvflag, const Real U_entry[8], Real dS_final[5])
{
    const Real rho_init = P_init(m_p.RHO, k, j, i);

    Real B_P[NVEC] = {0.};
    if (m_p.B1 >= 0) {
        B_P[V1] = P_init(m_p.B1, k, j, i);
        B_P[V2] = P_init(m_p.B2, k, j, i);
        B_P[V3] = P_init(m_p.B3, k, j, i);
    }
    Real P_mhd_guess[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
        P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};

    Real U_rad_0[4] = {U_entry[4], U_entry[5], U_entry[6], U_entry[7]};

    Real resid[4];

    Real U_mhd_0[4] = {U_entry[0], U_entry[1], U_entry[2], U_entry[3]};

    Real U_mhd_guess[4];
    Real U_rad_guess[4];
    Real P_rad_guess[4];
    Real dS_guess[4];

    // Iteration 0
    U_mhd_guess[0] = U_mhd_0[0];
    U_mhd_guess[1] = U_mhd_0[1];
    U_mhd_guess[2] = U_mhd_0[2];
    U_mhd_guess[3] = U_mhd_0[3];

    // Conservation law: Delta U_rad = - Delta U_mhd
    DLOOP1
        U_rad_guess[mu] = U_rad_0[mu];
    Real gdet = G.gdet(Loci::center, j, i);

    // Convert the newly guessed U_rad to P_rad
    // This will determine which closure branch the inversion took, so the Jacobian FD
    // loop below can detect when a perturbed sample has crossed onto a
    // different (maybe discontinuous) branch relative to the guess itself.
    bool used_normal_guess = true;
    const auto rad_status_entry =
        u_to_p_rad(G, U_rad_guess, P_rad_guess, k, j, i, &used_normal_guess);
    Real kappa_tot_entry = 0.0;
    compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, rho_init, B_P, eos, rad_opac,
        k, j, i, dS_guess, &kappa_tot_entry);
    const Real tol_eff = m::max(static_cast<Real>(src_rootfind_tol),
        10.0 * std::numeric_limits<Real>::epsilon() * kappa_tot_entry * dt);

    bool comoving_guess = false;
    if (kappa_tot_entry * dt > 1.0 ||
        rad_status_entry != StatusRadiationInversion::success) {
        Real uvec_gas[NVEC] = {P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        Real ucon_gas[GR_DIM], ucov_gas[GR_DIM];
        GRMHD::calc_ucon(G, uvec_gas, k, j, i, Loci::center, ucon_gas);
        G.lower(ucon_gas, ucov_gas, k, j, i, Loci::center);
        Real ucon_rad_in[GR_DIM];
        RadM1::calc_ucon_rad(G, P_rad_guess, j, i, ucon_rad_in);
        FourVectors D_gas;
        GRMHD::calc_4vecs(G, uvec_gas, B_P, k, j, i, Loci::center, D_gas);
        const Real w_gas =
            rho_init + P_mhd_guess[0] +
            eos.PressureFromDensityInternalEnergy(rho_init, P_mhd_guess[0] / rho_init) +
            dot(D_gas.bcon, D_gas.bcov);
        const Real lw_gas =
            w_gas * GRMHD::lorentz_calc(G, uvec_gas, k, j, i, Loci::center);
        const Real lw_rad =
            (4.0 / 3.0) * P_rad_guess[0] * RadM1::lorentz_calc_rad(G, P_rad_guess, j, i);
        Real uvec_com[NVEC];
        for (int v = 0; v < NVEC; v++)
            uvec_com[v] =
                (lw_gas * uvec_gas[v] + lw_rad * P_rad_guess[v + 1]) / (lw_gas + lw_rad);
        Real ucon_com[GR_DIM], ucov_com[GR_DIM];
        GRMHD::calc_ucon(G, uvec_com, k, j, i, Loci::center, ucon_com);
        G.lower(ucon_com, ucov_com, k, j, i, Loci::center);
        Real gamma_rel_com = 0.0;
        for (int mu = 0; mu < GR_DIM; mu++)
            gamma_rel_com -= ucon_rad_in[mu] * ucov_com[mu];
        const Real Erf_com =
            P_rad_guess[0] * ((4.0 / 3.0) * gamma_rel_com * gamma_rel_com - 1.0 / 3.0);
        const Real Erf_gas =
            U_rad_0[0] / gdet / ((4.0 / 3.0) * ucon_gas[0] * ucov_gas[0] + 1.0 / 3.0);
        const Real candidates[3][4] = {{Erf_com, uvec_com[0], uvec_com[1], uvec_com[2]},
            {Erf_gas, uvec_gas[0], uvec_gas[1], uvec_gas[2]},
            {P_init(m_p.UU_RAD, k, j, i), P_init(m_p.U1_RAD, k, j, i),
                P_init(m_p.U2_RAD, k, j, i), P_init(m_p.U3_RAD, k, j, i)}};
        const Real P_save[5] = {P_init(m_p.RHO, k, j, i), P_init(m_p.UU, k, j, i),
            P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
        const Real U_save[4] = {U_init(m_u.UU, k, j, i), U_init(m_u.U1, k, j, i),
            U_init(m_u.U2, k, j, i), U_init(m_u.U3, k, j, i)};
        for (int c = 0; c < 3; c++) {
            if (!(candidates[c][0] > 0.0) || !m::isfinite(candidates[c][0])) continue;
            Real P_rad_c[4] = {
                candidates[c][0], candidates[c][1], candidates[c][2], candidates[c][3]};
            Real U_rad_c[4], U_mhd_c[4];
            RadM1::calc_tensor(G, P_rad_c, 0, j, i, U_rad_c);
            for (int n = 0; n < 4; n++) {
                U_rad_c[n] *= gdet;
                U_mhd_c[n] = U_mhd_0[n] - (U_rad_c[n] - U_rad_0[n]);
            }
            U_init(m_u.UU, k, j, i) = U_mhd_c[0];
            U_init(m_u.U1, k, j, i) = U_mhd_c[1];
            U_init(m_u.U2, k, j, i) = U_mhd_c[2];
            U_init(m_u.U3, k, j, i) = U_mhd_c[3];
            auto status_c = Inverter::u_to_p<Inverter::Type::kastaun>(
                G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
            if (status_c == static_cast<int>(Inverter::Status::success)) {
                for (int n = 0; n < 4; n++) {
                    P_rad_guess[n] = P_rad_c[n];
                    U_rad_guess[n] = U_rad_c[n];
                    U_mhd_guess[n] = U_mhd_c[n];
                }
                P_mhd_guess[0] = P_init(m_p.UU, k, j, i);
                P_mhd_guess[1] = P_init(m_p.U1, k, j, i);
                P_mhd_guess[2] = P_init(m_p.U2, k, j, i);
                P_mhd_guess[3] = P_init(m_p.U3, k, j, i);
                used_normal_guess = true;
                compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess,
                    P_init(m_p.RHO, k, j, i), B_P, eos, rad_opac, k, j, i, dS_guess);
                comoving_guess = true;
                break;
            }
            P_init(m_p.RHO, k, j, i) = P_save[0];
            P_init(m_p.UU, k, j, i) = P_save[1];
            P_init(m_p.U1, k, j, i) = P_save[2];
            P_init(m_p.U2, k, j, i) = P_save[3];
            P_init(m_p.U3, k, j, i) = P_save[4];
            U_init(m_u.UU, k, j, i) = U_save[0];
            U_init(m_u.U1, k, j, i) = U_save[1];
            U_init(m_u.U2, k, j, i) = U_save[2];
            U_init(m_u.U3, k, j, i) = U_save[3];
        }
    }
    for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

    DLOOP1 {
        resid[mu] = U_rad_guess[mu] - U_rad_0[mu] - dt * dS_guess[mu];
    }

    // Compute err here and do a convergence check
    Real dtdS_0[4];
    for (int n = 0; n < 4; n++) dtdS_0[n] = dt * dS_guess[n];

    Real err = calculate_error(resid, U_rad_guess, U_rad_0, dtdS_0);

    // Here, if the comoving guess was used (in case of very optically thick gas) and err
    // is still large, we can make a better gas by trying to improve the initial guess,
    // this is done by checking the sign of the energy transfer, and increase/decrease
    // radiation and gas energy's initial guess. The search works in s = ln (E). We do a
    // bracket method.
    if (comoving_guess && err > 1.e-3) {

        // We save the gas state because this will overwrite P_init/U_init while testing.
        const Real P_keep[5] = {P_init(m_p.RHO, k, j, i), P_init(m_p.UU, k, j, i),
            P_init(m_p.U1, k, j, i), P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
        const Real U_keep[4] = {U_init(m_u.UU, k, j, i), U_init(m_u.U1, k, j, i),
            U_init(m_u.U2, k, j, i), U_init(m_u.U3, k, j, i)};

        constexpr Real LN10 = 2.302585092994046;
        const int ok = static_cast<int>(Inverter::Status::success);
        Real Pr[4] = {P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
        Real v_last[3] = {Pr[1], Pr[2], Pr[3]};
        Real f_a, sc;
        Real s_a = m::log(P_rad_guess[0]);
        bool have_bracket = false;
        Real s_lo = 0., s_hi = 0., f_lo = 0., f_hi = 0.;

        // Bracket search branch. the sign of f at the current E says whether E should go
        // up or down. Step E by x10 (or /10) until f changes sign. If the gas inversion
        // fails, decrease the step. v_last keeps the last converged velocity.
        if (prad_comoving_energy_residual(G, U_init, P_init, m_p, m_u, k, j, i, eos,
                rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr, f_a, sc) == ok) {
            for (int v = 0; v < 3; v++) v_last[v] = Pr[v + 1];
            const Real dir = (f_a > 0.0) ? 1.0 : -1.0;
            Real step = LN10;
            for (int b = 0; b < 60 && !have_bracket; b++) {
                const Real s_b = s_a + dir * step;
                Pr[0] = m::exp(s_b);
                for (int v = 0; v < 3; v++) Pr[v + 1] = v_last[v];
                Real f_b;
                const int st_b =
                    prad_comoving_energy_residual(G, U_init, P_init, m_p, m_u, k, j, i,
                        eos, rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr, f_b, sc);
                if (st_b != ok) {
                    step *= 0.5;
                    if (step < 1.e-6) break;
                    continue;
                }
                for (int v = 0; v < 3; v++) v_last[v] = Pr[v + 1];
                if (f_b * f_a <= 0.0) {
                    have_bracket = true;
                    if (s_a < s_b) {
                        s_lo = s_a;
                        f_lo = f_a;
                        s_hi = s_b;
                        f_hi = f_b;
                    } else {
                        s_lo = s_b;
                        f_lo = f_b;
                        s_hi = s_a;
                        f_hi = f_a;
                    }
                } else {
                    s_a = s_b;
                    f_a = f_b;
                }
            }
        }

        // Once we find the bracket, we can just approximate them by a line, with the
        // slope and point values, we can approximate which point it will cross E = 0.
        if (have_bracket) {
            Real s_c = s_lo;
            int side = 0;
            for (int it = 0; it < 60; it++) {
                s_c = (s_lo * f_hi - s_hi * f_lo) / (f_hi - f_lo);
                if (!(s_c > s_lo && s_c < s_hi)) s_c = 0.5 * (s_lo + s_hi);
                Pr[0] = m::exp(s_c);
                for (int v = 0; v < 3; v++) Pr[v + 1] = v_last[v];
                Real f_c;
                const int st_c =
                    prad_comoving_energy_residual(G, U_init, P_init, m_p, m_u, k, j, i,
                        eos, rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr, f_c, sc);
                if (st_c != ok) {
                    s_hi = s_c;
                    f_hi = -m::abs(f_lo);
                    continue;
                }
                for (int v = 0; v < 3; v++) v_last[v] = Pr[v + 1];
                if (m::abs(f_c) <= 1.e-3 * tol_eff * sc || s_hi - s_lo < 1.e-13) break;
                if (f_c * f_lo > 0.0) {
                    s_lo = s_c;
                    f_lo = f_c;
                    if (side == -1) f_hi *= 0.5;
                    side = -1;
                } else {
                    s_hi = s_c;
                    f_hi = f_c;
                    if (side == 1) f_lo *= 0.5;
                    side = 1;
                }
            }
            // Check the result in the full 4D problem, use it as Newton's starting
            // point only if the error is smaller than before.
            Real f_c;
            Pr[0] = m::exp(s_c);
            for (int v = 0; v < 3; v++) Pr[v + 1] = v_last[v];
            const int st_f = prad_comoving_energy_residual(G, U_init, P_init, m_p, m_u, k,
                j, i, eos, rad_opac, B_P, dt, gdet, U_mhd_0, U_rad_0, Pr, f_c, sc);
            if (st_f == ok) {
                Real U_rad_c[4], U_mhd_c[4], dS_c[4], res_c[4], dtdS_c[4];
                RadM1::calc_tensor(G, Pr, 0, j, i, U_rad_c);
                for (int n = 0; n < 4; n++) {
                    U_rad_c[n] *= gdet;
                    U_mhd_c[n] = U_mhd_0[n] - (U_rad_c[n] - U_rad_0[n]);
                }
                U_init(m_u.UU, k, j, i) = U_mhd_c[0];
                U_init(m_u.U1, k, j, i) = U_mhd_c[1];
                U_init(m_u.U2, k, j, i) = U_mhd_c[2];
                U_init(m_u.U3, k, j, i) = U_mhd_c[3];
                const int st_c = Inverter::u_to_p<Inverter::Type::kastaun>(
                    G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
                if (st_c == ok) {
                    Real P_mhd_c[4] = {P_init(m_p.UU, k, j, i), P_init(m_p.U1, k, j, i),
                        P_init(m_p.U2, k, j, i), P_init(m_p.U3, k, j, i)};
                    compute_covariant_fourforce(G, P_mhd_c, Pr, P_init(m_p.RHO, k, j, i),
                        B_P, eos, rad_opac, k, j, i, dS_c);
                    for (int n = 0; n < 4; n++) {
                        dS_c[n] *= gdet;
                        dtdS_c[n] = dt * dS_c[n];
                        res_c[n] = U_rad_c[n] - U_rad_0[n] - dtdS_c[n];
                    }
                    const Real err_c = calculate_error(res_c, U_rad_c, U_rad_0, dtdS_c);
                    if (err_c < err) {
                        for (int n = 0; n < 4; n++) {
                            P_rad_guess[n] = Pr[n];
                            U_rad_guess[n] = U_rad_c[n];
                            U_mhd_guess[n] = U_mhd_c[n];
                            dS_guess[n] = dS_c[n];
                            dtdS_0[n] = dtdS_c[n];
                            resid[n] = res_c[n];
                            P_mhd_guess[n] = P_mhd_c[n];
                        }
                        err = err_c;
                    }
                }
            }
        }
        // If the improved guess was not kept, restore the saved gas state
        if (P_init(m_p.UU, k, j, i) != P_mhd_guess[0] ||
            P_init(m_p.U1, k, j, i) != P_mhd_guess[1]) {
            P_init(m_p.RHO, k, j, i) = P_keep[0];
            P_init(m_p.UU, k, j, i) = P_keep[1];
            P_init(m_p.U1, k, j, i) = P_keep[2];
            P_init(m_p.U2, k, j, i) = P_keep[3];
            P_init(m_p.U3, k, j, i) = P_keep[4];
            U_init(m_u.UU, k, j, i) = U_keep[0];
            U_init(m_u.U1, k, j, i) = U_keep[1];
            U_init(m_u.U2, k, j, i) = U_keep[2];
            U_init(m_u.U3, k, j, i) = U_keep[3];
        }
    }

    Real best_err = err;
    Real best_P[4] = {P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
    int niter = 0;
    bool bad_guess = false;

    do {
        if (err <= tol_eff) {
            break;
        }

        Real P_mhd_m[4];
        Real P_mhd_p[4];
        Real U_mhd_m[4];
        Real U_mhd_p[4];
        Real U_rad_m[4];
        Real U_rad_p[4];
        Real dS_m[4];
        Real dS_p[4];
        Real rho_iter_next;

        Real jac[4][4] = {0};

        // Find minimum non-zero magnitude from P_mhd_guess to scale FD step safely
        Real P_rad_mag_min = RAD_LARGE;
        for (int m = 0; m < 4; m++) {
            if (m::abs(P_rad_guess[m]) > 0.) {
                P_rad_mag_min = m::min(P_rad_mag_min, m::abs(P_rad_guess[m]));
            }
        }

        bool bad_guess_m = false;
        bool bad_guess_p = false;

        // Loop over the 4 fluid variables to perturb each one
        for (int m = 0; m < 4; m++) {
            Real P_rad_m[4] = {
                P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
            Real P_rad_p[4] = {
                P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};

            Real fd_step = m::max(src_rootfind_eps * P_rad_mag_min,
                src_rootfind_eps * m::abs(P_rad_guess[m]));
            if (m == 0)
                fd_step = m::max(fd_step, src_rootfind_eps * m::abs(U_mhd_0[0]) / gdet);
            {
                const Real gas_energy_margin = 0.1 * m::abs(P_mhd_guess[0]);
                fd_step = m::min(fd_step, m::max(gas_energy_margin, RAD_SMALL));
            }
            P_rad_m[m] -= fd_step;
            P_rad_p[m] += fd_step;

            // Evaluate minus perturbation
            RadM1::calc_tensor(G, P_rad_m, 0, j, i, U_rad_m);
            for (int n = 0; n < 4; n++) U_rad_m[n] *= gdet;

            // set U_mhd from U_rad
            for (int n = 0; n < 4; n++) {
                U_mhd_m[n] = U_mhd_0[n] - (U_rad_m[n] - U_rad_0[n]);
            }

            // Invert to get fluid primitives
            U_init(m_u.UU, k, j, i) = U_mhd_m[0];
            U_init(m_u.U1, k, j, i) = U_mhd_m[1];
            U_init(m_u.U2, k, j, i) = U_mhd_m[2];
            U_init(m_u.U3, k, j, i) = U_mhd_m[3];
            auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
                G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);

            P_mhd_m[0] = P_init(m_p.UU, k, j, i);
            P_mhd_m[1] = P_init(m_p.U1, k, j, i);
            P_mhd_m[2] = P_init(m_p.U2, k, j, i);
            P_mhd_m[3] = P_init(m_p.U3, k, j, i);

            if (mhd_inverter_status != static_cast<int>(Inverter::Status::success)) {
                bad_guess_m = true;
            }

            // If a bad guess has already been found before, we can't really skip the
            // whole Jacobian evaluation. We need to keep evaluating the other blocks to
            // figure out if, during the next m's, the other side (plus/minus) will also
            // go bad, trigerring a bad_guess_m == true && bad_guess_p == true
            if (!bad_guess_m && !bad_guess_p) {
                Real rho_m = P_init(m_p.RHO, k, j, i);
                compute_covariant_fourforce(
                    G, P_mhd_m, P_rad_m, rho_m, B_P, eos, rad_opac, k, j, i, dS_m);
                for (int n = 0; n < 4; n++) dS_m[n] = gdet * dS_m[n];
            }

            // Evaluate plus perturbation
            RadM1::calc_tensor(G, P_rad_p, 0, j, i, U_rad_p);
            for (int n = 0; n < 4; n++) U_rad_p[n] *= gdet;
            // set U_mhd from U_rad
            for (int n = 0; n < 4; n++) {
                U_mhd_p[n] = U_mhd_0[n] - (U_rad_p[n] - U_rad_0[n]);
            }

            // Invert to get fluid primitives
            U_init(m_u.UU, k, j, i) = U_mhd_p[0];
            U_init(m_u.U1, k, j, i) = U_mhd_p[1];
            U_init(m_u.U2, k, j, i) = U_mhd_p[2];
            U_init(m_u.U3, k, j, i) = U_mhd_p[3];
            auto mhd_inverter_status_p = Inverter::u_to_p<Inverter::Type::kastaun>(
                G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);

            P_mhd_p[0] = P_init(m_p.UU, k, j, i);
            P_mhd_p[1] = P_init(m_p.U1, k, j, i);
            P_mhd_p[2] = P_init(m_p.U2, k, j, i);
            P_mhd_p[3] = P_init(m_p.U3, k, j, i);

            if (mhd_inverter_status_p != static_cast<int>(Inverter::Status::success)) {
                bad_guess_p = true;
            }

            // If a bad guess has already been found before, we can't really skip the
            // whole Jacobian evaluation. We need to keep evaluating the other blocks to
            // figure out if, during the next m's, the other side (plus/minus) will also
            // go bad, trigerring a bad_guess_m == true && bad_guess_p == true
            if (!bad_guess_m && !bad_guess_p) {
                Real rho_p = P_init(m_p.RHO, k, j, i);
                compute_covariant_fourforce(
                    G, P_mhd_p, P_rad_p, rho_p, B_P, eos, rad_opac, k, j, i, dS_p);
                for (int n = 0; n < 4; n++) dS_p[n] = gdet * dS_p[n];

                // Populate Jacobian
                for (int n = 0; n < 4; n++) {
                    Real fp = U_rad_p[n] - U_rad_0[n] - dt * dS_p[n];
                    Real fm = U_rad_m[n] - U_rad_0[n] - dt * dS_m[n];
                    // Jacobian here is dU_mhd/dP_rad
                    // Since div R^mu_nu = G_nu
                    // and div T^mu_nu = -G_nu
                    jac[n][m] = (fp - fm) / (P_rad_p[m] - P_rad_m[m]);
                }
            }
        }
        // TODO (PNM): Separate these if/elses into different kernels.
        if (bad_guess_m == true && bad_guess_p == true) {
            bad_guess = true;
            break; // Exit the iteration loop if both perturbations yield bad guesses
        } else if (bad_guess_m == true) {
            // If only - finite difference support point is bad, do one-sided
            // difference with + support point

            for (int m = 0; m < 4; m++) {
                Real P_rad_p[4] = {
                    P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
                Real fd_step_p = std::max(src_rootfind_eps * P_rad_mag_min,
                    src_rootfind_eps * m::abs(P_rad_p[m]));
                if (m == 0)
                    fd_step_p =
                        m::max(fd_step_p, src_rootfind_eps * m::abs(U_mhd_0[0]) / gdet);
                {
                    const Real gas_energy_margin = 0.1 * m::abs(P_mhd_guess[0]);
                    fd_step_p = m::min(fd_step_p, m::max(gas_energy_margin, RAD_SMALL));
                }
                P_rad_p[m] += fd_step_p;

                RadM1::calc_tensor(G, P_rad_p, 0, j, i, U_rad_p);
                for (int n = 0; n < 4; n++) U_rad_p[n] *= gdet;

                for (int n = 0; n < 4; n++) {
                    U_mhd_p[n] = U_mhd_0[n] - (U_rad_p[n] - U_rad_0[n]);
                }

                // invert
                U_init(m_u.UU, k, j, i) = U_mhd_p[0];
                U_init(m_u.U1, k, j, i) = U_mhd_p[1];
                U_init(m_u.U2, k, j, i) = U_mhd_p[2];
                U_init(m_u.U3, k, j, i) = U_mhd_p[3];
                auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
                    G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);

                P_mhd_p[0] = P_init(m_p.UU, k, j, i);
                P_mhd_p[1] = P_init(m_p.U1, k, j, i);
                P_mhd_p[2] = P_init(m_p.U2, k, j, i);
                P_mhd_p[3] = P_init(m_p.U3, k, j, i);

                Real rho_p = P_init(m_p.RHO, k, j, i);
                Real B_P[NVEC] = {0.};
                if (m_p.B1 >= 0) {
                    B_P[V1] = P_init(m_p.B1, k, j, i);
                    B_P[V2] = P_init(m_p.B2, k, j, i);
                    B_P[V3] = P_init(m_p.B3, k, j, i);
                }
                compute_covariant_fourforce(
                    G, P_mhd_p, P_rad_p, rho_p, B_P, eos, rad_opac, k, j, i, dS_p);
                for (int n = 0; n < 4; n++) dS_p[n] = gdet * dS_p[n];

                PARTHENON_REQUIRE(
                    mhd_inverter_status == static_cast<int>(Inverter::Status::success),
                    "This inversion should have already worked!");

                for (int n = 0; n < 4; n++) {
                    Real fp = U_rad_p[n] - U_rad_0[n] - dt * dS_p[n];
                    Real fguess = U_rad_guess[n] - U_rad_0[n] - dt * dS_guess[n];
                    // Jacobian here is dU_rad/dP_mhd
                    // Since div R^mu_nu = G_nu
                    // and div T^mu_nu = -G_nu
                    jac[n][m] = (fp - fguess) / (P_rad_p[m] - P_rad_guess[m]);
                }
            }
        } else if (bad_guess_p == true) {
            // If only + finite difference support point is bad, do one-sided
            // difference with - support point

            for (int m = 0; m < 4; m++) {
                Real P_rad_m[4] = {
                    P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
                Real fd_step_m = std::max(src_rootfind_eps * P_rad_mag_min,
                    src_rootfind_eps * m::abs(P_rad_m[m]));
                if (m == 0)
                    fd_step_m =
                        m::max(fd_step_m, src_rootfind_eps * m::abs(U_mhd_0[0]) / gdet);
                {
                    const Real gas_energy_margin = 0.1 * m::abs(P_mhd_guess[0]);
                    fd_step_m = m::min(fd_step_m, m::max(gas_energy_margin, RAD_SMALL));
                }
                P_rad_m[m] -= fd_step_m;

                RadM1::calc_tensor(G, P_rad_m, 0, j, i, U_rad_m);
                for (int n = 0; n < 4; n++) U_rad_m[n] *= gdet;

                for (int n = 0; n < 4; n++) {
                    U_mhd_m[n] = U_mhd_0[n] - (U_rad_m[n] - U_rad_0[n]);
                }

                U_init(m_u.UU, k, j, i) = U_mhd_m[0];
                U_init(m_u.U1, k, j, i) = U_mhd_m[1];
                U_init(m_u.U2, k, j, i) = U_mhd_m[2];
                U_init(m_u.U3, k, j, i) = U_mhd_m[3];
                auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
                    G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);
                P_mhd_m[0] = P_init(m_p.UU, k, j, i);
                P_mhd_m[1] = P_init(m_p.U1, k, j, i);
                P_mhd_m[2] = P_init(m_p.U2, k, j, i);
                P_mhd_m[3] = P_init(m_p.U3, k, j, i);
                Real rho_m = P_init(m_p.RHO, k, j, i);
                Real B_P[NVEC] = {0.};
                if (m_p.B1 >= 0) {
                    B_P[V1] = P_init(m_p.B1, k, j, i);
                    B_P[V2] = P_init(m_p.B2, k, j, i);
                    B_P[V3] = P_init(m_p.B3, k, j, i);
                }

                compute_covariant_fourforce(
                    G, P_mhd_m, P_rad_m, rho_m, B_P, eos, rad_opac, k, j, i, dS_m);
                for (int n = 0; n < 4; n++) dS_m[n] = gdet * dS_m[n];

                PARTHENON_REQUIRE(
                    mhd_inverter_status == static_cast<int>(Inverter::Status::success),
                    "This inversion should have already worked!");

                for (int n = 0; n < 4; n++) {
                    Real fm = U_rad_m[n] - U_rad_0[n] - dt * dS_m[n];
                    Real fguess = U_rad_guess[n] - U_rad_0[n] - dt * dS_guess[n];
                    jac[n][m] = (fguess - fm) / (P_rad_guess[m] - P_rad_m[m]);
                }
            }
        }

        Real jacinv[4][4];
        // Inverting the 4x4 matrix;
        invert(&jac[0][0], &jacinv[0][0]);

        // Update guess via a damped/backtracking Newton step: take the full step, and if
        // it violates either check below, shrink the step (Koral's fixed factor of 3)
        // and retry, up to 15 shrinks.
        Real P_rad_pre[4] = {
            P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]};
        Real scaling_factor = 1.0;
        int track = 0;
        bool step_ok = false;
        do {
            for (int m = 0; m < 4; m++) {
                P_rad_guess[m] = P_rad_pre[m];
                for (int n = 0; n < 4; n++) {
                    P_rad_guess[m] -= scaling_factor * jacinv[m][n] * resid[n];
                }
            }

            if (P_rad_guess[0] <= 0.0) {
                scaling_factor /= 3.0;
                track++;
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }

            RadM1::calc_tensor(G, P_rad_guess, 0, j, i, U_rad_guess);
            for (int n = 0; n < 4; n++) U_rad_guess[n] *= gdet;

            for (int n = 0; n < 4; n++) {
                U_mhd_guess[n] = U_mhd_0[n] - (U_rad_guess[n] - U_rad_0[n]);
            }

            U_init(m_u.UU, k, j, i) = U_mhd_guess[0];
            U_init(m_u.U1, k, j, i) = U_mhd_guess[1];
            U_init(m_u.U2, k, j, i) = U_mhd_guess[2];
            U_init(m_u.U3, k, j, i) = U_mhd_guess[3];

            auto mhd_inverter_status = Inverter::u_to_p<Inverter::Type::kastaun>(
                G, U_init, m_u, eos, k, j, i, P_init, m_p, Loci::center, 25, 1e-12);

            if (mhd_inverter_status != static_cast<int>(Inverter::Status::success)) {
                scaling_factor /= 3.0;
                track++;
                if (track > 15 || scaling_factor < 1.e-8) break;
                continue;
            }

            step_ok = true;
        } while (!step_ok && track <= 15);

        if (track > 15 || scaling_factor < 1.e-8) {
            bad_guess = true;
            break;
        }

        P_mhd_guess[0] = P_init(m_p.UU, k, j, i);
        P_mhd_guess[1] = P_init(m_p.U1, k, j, i);
        P_mhd_guess[2] = P_init(m_p.U2, k, j, i);
        P_mhd_guess[3] = P_init(m_p.U3, k, j, i);
        rho_iter_next = P_init(m_p.RHO, k, j, i);

        compute_covariant_fourforce(G, P_mhd_guess, P_rad_guess, rho_iter_next, B_P, eos,
            rad_opac, k, j, i, dS_guess);

        for (int n = 0; n < 4; n++) dS_guess[n] = gdet * dS_guess[n];

        // Update residuals
        for (int n = 0; n < 4; n++) {
            resid[n] = U_rad_guess[n] - U_rad_0[n] - dt * dS_guess[n];

            if (!m::isfinite(resid[n])) {
                bad_guess = true;
                break;
            }
        }

        // This is needed since the previous bad_guess = true would only break out of the
        // for loop
        if (bad_guess) {
            break;
        }

        // Calculate error now
        Real dtdS_iter[4];
        for (int n = 0; n < 4; n++) dtdS_iter[n] = dt * dS_guess[n];

        err = calculate_error(resid, U_rad_guess, U_rad_0, dtdS_iter);

        niter++;
        if (err < best_err) {
            best_err = err;
            for (int n = 0; n < 4; n++) best_P[n] = P_rad_guess[n];
        }
    } while (err > tol_eff && niter < src_rootfind_maxiter);

    bool failed = err > tol_eff || !m::isfinite(U_mhd_guess[0]) ||
                  !m::isfinite(U_mhd_guess[1]) || !m::isfinite(U_mhd_guess[2]) ||
                  !m::isfinite(U_mhd_guess[3]) || bad_guess;

    // Newton failed, but its best attempt may already be as accurate as double
    // precision allows. Accept it if its error is no bigger than the round-off noise
    if (failed && best_err <= 1.e-4) {
        Real r0[4], ur0[4], dd0[4], pm0[4], ds0[4];
        if (prad_residual_at(G, U_init, P_init, m_p, m_u, k, j, i, eos, rad_opac, B_P, dt,
                gdet, U_mhd_0, U_rad_0, best_P, r0, ur0, dd0, pm0,
                ds0) == static_cast<int>(Inverter::Status::success)) {
            const Real err0 = calculate_error(r0, ur0, U_rad_0, dd0);

            // Measure the round-off noise, nudge the unknowns by 4 ULPs (one at a time,
            // all down, then alternating signs) and record how much the error changes
            const Real ee = 4.0 * std::numeric_limits<Real>::epsilon();
            Real noise = 0.0;
            for (int q = 0; q < 6 && err0 <= 1.e-4 && err0 > tol_eff; q++) {
                Real Pq[4] = {best_P[0], best_P[1], best_P[2], best_P[3]};
                for (int n = 0; n < 4; n++) {
                    if (q < 4)
                        Pq[n] *= (n == q) ? (1.0 + ee) : 1.0;
                    else
                        Pq[n] *= (q == 4) ? (1.0 - ee) : (1.0 + ((n % 2) ? ee : -ee));
                }
                Real r1[4], ur1[4], dd1[4], pm1[4], ds1[4], dr[4];
                if (prad_residual_at(G, U_init, P_init, m_p, m_u, k, j, i, eos, rad_opac,
                        B_P, dt, gdet, U_mhd_0, U_rad_0, Pq, r1, ur1, dd1, pm1,
                        ds1) != static_cast<int>(Inverter::Status::success))
                    continue;
                for (int n = 0; n < 4; n++) dr[n] = r1[n] - r0[n];
                noise = m::max(noise, calculate_error(dr, ur0, U_rad_0, dd0));
            }

            // Accept if the error is within tolerance or within 10x the noise. Evaluate
            // once more at best_P so U_init/P_init match the accepted state.
            if (err0 <= 1.e-4 && (err0 <= tol_eff || err0 <= 10.0 * noise) &&
                prad_residual_at(G, U_init, P_init, m_p, m_u, k, j, i, eos, rad_opac, B_P,
                    dt, gdet, U_mhd_0, U_rad_0, best_P, r0, ur0, dd0, pm0,
                    ds0) == static_cast<int>(Inverter::Status::success)) {
                // Use best attempt as the solution.
                for (int n = 0; n < 4; n++) {
                    P_rad_guess[n] = best_P[n];
                    U_rad_guess[n] = ur0[n];
                    U_mhd_guess[n] = U_mhd_0[n] - (ur0[n] - U_rad_0[n]);
                    P_mhd_guess[n] = pm0[n];
                    dS_guess[n] = ds0[n];
                    resid[n] = r0[n];
                }
                err = err0;
                failed = false;
            }
        }
    }

    if (failed) {

        return static_cast<int>(StatusImplicitStep::failure);
    }

    for (int n = 0; n < 4; n++) dS_final[n] = (U_rad_guess[n] - U_rad_0[n]) / dt;

    {
        Real uvec_final[NVEC] = {P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
        Real ucon_final[4];
        GRMHD::calc_ucon(G, uvec_final, k, j, i, Loci::center, ucon_final);
        Real Gdotu = dS_guess[0] * ucon_final[0] + dS_guess[1] * ucon_final[1] +
                     dS_guess[2] * ucon_final[2] + dS_guess[3] * ucon_final[3];
        dS_final[4] = Gdotu;
    }

    return static_cast<int>(StatusImplicitStep::success);
}

} // namespace RadM1
