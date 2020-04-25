/**
 * Variable inversion to recover primitive variables from conserved quantities
 * See Mignone & McKinney 2007
 */
#pragma once

#include "decs.hpp"

#include "utils.hpp"

#define ERRTOL 1.e-8
#define ITERMAX 8

KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS* eos, const Real Bsq, const Real D, const Real Ep, const Real QdB,
                                    const Real Qtsq, const Real Wp, InversionStatus &eflag);
KOKKOS_INLINE_FUNCTION Real gamma_func(const Real Bsq, const Real D, const Real QdB,
                                        const Real Qtsq, const Real Wp, InversionStatus &eflag);
KOKKOS_INLINE_FUNCTION Real Wp_func(const Grid &G, const GridVars P, const EOS* eos,
                                    const int& k, const int& j, const int& i, const Loci loc, InversionStatus &eflag);

/**
 * Recover primitive variables
 */
KOKKOS_INLINE_FUNCTION InversionStatus U_to_P(const Grid &G, const GridVars U, const EOS* eos,
                                  const int& k, const int& j, const int& i, const Loci loc, GridVars P)
{
    InversionStatus eflag = InversionStatus::success;

    Real gdet = G.gdet(loc, j, i);

    // Update the primitive B-fields
    P(prims::B1, k, j, i) = U(prims::B1, k, j, i) / gdet;
    P(prims::B2, k, j, i) = U(prims::B2, k, j, i) / gdet;
    P(prims::B3, k, j, i) = U(prims::B3, k, j, i) / gdet;

#if DEBUG
    // Catch negative energy or density
    if (U(prims::rho, k, j, i) <= 0. || U(prims::u, k, j, i) <= 0.)
    {
        return InversionStatus::neg_input;
    }
#endif

    // Convert from conserved variables to four-vectors
    // TODO try a version of this with cached gcov/con/det
    Real a_over_g = 1./sqrt(-G.gcon(loc, j, i, 0, 0)) / G.gdet(loc, j, i);
    Real D = U(prims::rho, k, j, i) * a_over_g;

    Real Bcon[NDIM];
    Bcon[0] = 0.;
    Bcon[1] = U(prims::B1, k, j, i) * a_over_g;
    Bcon[2] = U(prims::B2, k, j, i) * a_over_g;
    Bcon[3] = U(prims::B3, k, j, i) * a_over_g;

    Real Qcov[NDIM];
    Qcov[0] = (U(prims::u, k, j, i) - U(prims::rho, k, j, i)) * a_over_g;
    Qcov[1] = U(prims::u1, k, j, i) * a_over_g;
    Qcov[2] = U(prims::u2, k, j, i) * a_over_g;
    Qcov[3] = U(prims::u3, k, j, i) * a_over_g;

    Real ncov[NDIM] = {(Real) -1./sqrt(-G.gcon(loc, j, i, 0, 0)), 0, 0, 0};

    Real Bcov[NDIM], Qcon[NDIM], ncon[NDIM];
    G.lower(Bcon, Bcov, k, j, i, loc);
    G.raise(Qcov, Qcon, k, j, i, loc);
    G.raise(ncov, ncon, k, j, i, loc);

    Real Bsq = dot(Bcon, Bcov);
    Real QdB = dot(Bcon, Qcov);
    Real Qdotn = dot(Qcon, ncov);

    Real Qtcon[NDIM];
    DLOOP1 Qtcon[mu] = Qcon[mu] + ncon[mu] * Qdotn;
    Real Qtsq = dot(Qcon, Qcov) + pow(Qdotn, 2);

    // Set up eqtn for W'; this is the energy density
    Real Ep = -Qdotn - D;

    // Numerical rootfinding
    // Take guesses from primitives.
    Real Wp = Wp_func(G, P, eos, k, j, i, loc, eflag);
    if (eflag)
        return eflag;

    // Step around the guess & evaluate errors
    Real Wpm = (1. - DELTA) * Wp; //heuristic
    Real h = Wp - Wpm;
    Real Wpp = Wp + h;
    Real errp = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wpp, eflag);
    Real err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);
    Real errm = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wpm, eflag);

    // Attempt a Halley/Muller/Bailey/Press step
    Real dedW = (errp - errm) / (Wpp - Wpm);
    Real dedW2 = (errp - 2. * err + errm) / pow(h,2);
    Real f = clip(0.5 * err * dedW2 / pow(dedW,2), -0.3, 0.3); // TODO take a hard look at this clip

    Real dW = clip(-err / dedW / (1. - f), -0.5*Wp, 2.0*Wp);

    // Wp, dW, err
    Real Wp1 = Wp;
    Real err1 = err;

    Wp += dW;
    err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

    // Not good enough?  apply secant method
    int iter = 0;
    for (iter = 0; iter < ITERMAX; iter++)
    {
        dW = clip((Wp1 - Wp) * err / (err - err1), (Real) -0.5*Wp, (Real) 2.0*Wp);

        Wp1 = Wp;
        err1 = err;

        Wp += dW;

        if (fabs(dW / Wp) < ERRTOL) break;

        err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

        if (fabs(err / Wp) < ERRTOL) break;
    }
    // Failure to converge; do not set primitives other than B
    if (iter == ITERMAX)
    {
        return InversionStatus::max_iter;
    }

    // Find utsq, gamma, rho0 from Wp
    Real gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, eflag);

    // Find the scalars
    Real rho0 = D / gamma;
    Real W = Wp + D;
    Real w = W / pow(gamma, 2);
    Real p = eos->p_w(rho0, w);
    Real u = w - (rho0 + p);

    // Return without updating non-B primitives
    if (rho0 < 0 && u < 0)
    {
        return InversionStatus::neg_rhou;
    }
    else if (rho0 < 0)
    {
        return InversionStatus::neg_rho;
    }
    else if (u < 0)
    {
        return InversionStatus::neg_u;
    }

    // Set primitives
    P(prims::rho, k, j, i) = rho0;
    P(prims::u, k, j, i) = u;

    // Find u(tilde); Eqn. 31 of Noble et al.
    P(prims::u1, k, j, i) = (gamma / (W + Bsq)) * (Qtcon[1] + QdB * Bcon[1] / W);
    P(prims::u2, k, j, i) = (gamma / (W + Bsq)) * (Qtcon[2] + QdB * Bcon[2] / W);
    P(prims::u3, k, j, i) = (gamma / (W + Bsq)) * (Qtcon[3] + QdB * Bcon[3] / W);

    return InversionStatus::success;
}

// TODO lighten up on checks in these functions in favor of sanity at the end
// Document these
KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS* eos, const Real Bsq, const Real D, const Real Ep, const Real QdB,
                                    const Real Qtsq, const Real Wp, InversionStatus &eflag)
{
    Real W = Wp + D;
    Real gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, eflag);
    Real w = W / pow(gamma,2);
    Real rho0 = D / gamma;
    Real p = eos->p_w(rho0, w);

    return -Ep + Wp - p + 0.5 * Bsq + 0.5 * (Bsq * Qtsq - QdB * QdB) / pow((Bsq + W), 2);

}

/**
 * Fluid relativistic factor gamma in terms of inversion state variables
 */
KOKKOS_INLINE_FUNCTION Real gamma_func(const Real Bsq, const Real D, const Real QdB,
                                        const Real Qtsq, const Real Wp, InversionStatus &eflag)
{
    Real QdBsq, W, utsq, gamma, W2, WB;

    QdBsq = QdB * QdB;
    W = Wp + D;
    W2 = W * W;
    WB = W + Bsq;

    // This is basically inversion of eq. A7 of Mignone & McKinney
    utsq = -((W + WB) * QdBsq + W2 * Qtsq) / (QdBsq * (W + WB) + W2 * (Qtsq - WB * WB));
    gamma = sqrt(1. + fabs(utsq));

#if DEBUG
    // Catch utsq < 0
    if (utsq < 0. || utsq > 1.e3 * GAMMAMAX * GAMMAMAX)
    {
        eflag = InversionStatus::bad_ut;
    }
    if (gamma < 1.)
    {
        eflag = InversionStatus::bad_gamma;
    }
#endif

    return gamma;
}

/**
 * See Mignone & McKinney
 * TODO make local?  Index stuff seems weird
 */
KOKKOS_INLINE_FUNCTION Real Wp_func(const Grid &G, const GridVars P, const EOS* eos,
                                    const int& k, const int& j, const int& i, const Loci loc, InversionStatus &eflag)
{
    Real rho0 = P(prims::rho, k, j, i);
    Real u = P(prims::u, k, j, i);

    Real gamma = mhd_gamma_calc(G, P, k, j, i, loc);
#if DEBUG
    if (gamma < 1.)
    {
        eflag = InversionStatus::bad_gamma;
    }
#endif

    return (rho0 + u + eos->p(rho0, u)) * gamma * gamma - rho0 * gamma;
}
