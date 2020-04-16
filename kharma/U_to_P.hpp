/**
 * Variable inversion to recover primitive variables from conserved quantities
 * See Mignone & McKinney 2007
 */
#pragma once

#include "mesh/mesh.hpp"

#include "decs.hpp"
#include "eos.hpp"
#include "utils.hpp"

#define ERRTOL 1.e-8
#define ITERMAX 8
#define GAMMAMAX 200

using namespace parthenon;

KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS* eos, const Real Bsq, const Real D, const Real Ep, const Real QdB,
                                    const Real Qtsq, const Real Wp, int &eflag);
KOKKOS_INLINE_FUNCTION Real gamma_func(const Real Bsq, const Real D, const Real QdB,
                                        const Real Qtsq, const Real Wp, int &eflag);
KOKKOS_INLINE_FUNCTION Real Wp_func(const CoordinateEmbedding &C, const CellVariable<Real> P, const EOS* eos,
                                    const int i, const int j, const int k, const Loci loc, InversionStatus &eflag);

/**
 * Recover primitive variables
 * 
 * TODO does U_to_P ever need to be called from non-center, even with face-centered B fields?
 */
KOKKOS_INLINE_FUNCTION InversionStatus U_to_P(MeshBlock *pmb, const CoordinateEmbedding &C, const EOS* eos,
                                  const int i, const int j, const int k, const Loci loc)
{
    InversionStatus eflag = InversionStatus::success;

    Container<Real>& rc = pmb->real_containers.Get();
    CellVariable<Real> U = rc.Get("c.c.bulk.cons");
    CellVariable<Real> P = rc.Get("c.c.bulk.prims");

    // TODO amend if ever not loci::cent
    GReal x1 = C.coord

    Real gdet = C.gdet(loc, i, j);

    // Update the primitive B-fields
    P(i, j, k, prims::B1) = U(i, j, k, prims::B1) / gdet;
    P(i, j, k, prims::B2) = U(i, j, k, prims::B2) / gdet;
    P(i, j, k, prims::B3) = U(i, j, k, prims::B3) / gdet;

#if DEBUG
    // Catch negative energy or density
    if (U(i, j, k, prims::rho) <= 0. || U(i, j, k, prims::u) <= 0.)
    {
        // TODO print if OpenMP or something?
        return InversionStatus::neg_input;
    }
#endif

    // Convert from conserved variables to four-vectors
    Real a_over_g = 1./sqrt(-G.gcon(loc, i, j, 0, 0)) / G.gdet(loc, i, j);
    Real D = U(i, j, k, prims::rho) * a_over_g;

    Real Bcon[NDIM];
    Bcon[0] = 0.;
    Bcon[1] = U(i, j, k, prims::B1) * a_over_g;
    Bcon[2] = U(i, j, k, prims::B2) * a_over_g;
    Bcon[3] = U(i, j, k, prims::B3) * a_over_g;

    Real Qcov[NDIM];
    Qcov[0] = (U(i, j, k, prims::u) - U(i, j, k, prims::rho)) * a_over_g;
    Qcov[1] = U(i, j, k, prims::u1) * a_over_g;
    Qcov[2] = U(i, j, k, prims::u2) * a_over_g;
    Qcov[3] = U(i, j, k, prims::u3) * a_over_g;

    Real ncov[NDIM] = {(Real) -1./sqrt(-G.gcon(loc, i, j, 0, 0)), 0, 0, 0};

    Real Bcov[NDIM], Qcon[NDIM], ncon[NDIM];
    // Interlaced upper/lower operation
    // TODO are separate ops faster?
    for (int mu = 0; mu < NDIM; mu++)
    {
        Bcov[mu] = 0.;
        Qcon[mu] = 0.;
        ncon[mu] = 0.;
        for (int nu = 0; nu < NDIM; nu++)
        {
            Bcov[mu] += G.gcov(Loci::center, i, j, mu, nu) * Bcon[nu];
            Qcon[mu] += G.gcon(Loci::center, i, j, mu, nu) * Qcov[nu];
            ncon[mu] += G.gcon(Loci::center, i, j, mu, nu) * ncov[nu];
        }
    }

    Real Qtcon[NDIM];
    DLOOP1 Qtcon[mu] = Qcon[mu] + ncon[mu] * Qdotn;
    Real Qtsq = dot(Qcon, Qcov) + pow(Qdotn, 2);

    // Set up eqtn for W'; this is the energy density
    Real Ep = -Qdotn - D;

    // Numerical rootfinding
    // Take guesses from primitives.
    Real Wp = Wp_func(G, P, eos, i, j, k, loc, eflag);
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

        if (fabs(dW / Wp) < ERRTOL)
        {
            break;
        }

        err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

        if (fabs(err / Wp) < ERRTOL)
        {
            break;
        }
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
    P(i, j, k, prims::rho) = rho0;
    P(i, j, k, prims::u) = u;

    // Find u(tilde); Eqn. 31 of Noble et al.
    P(i, j, k, prims::u1) = (gamma / (W + Bsq)) * (Qtcon[1] + QdB * Bcon[1] / W);
    P(i, j, k, prims::u2) = (gamma / (W + Bsq)) * (Qtcon[2] + QdB * Bcon[2] / W);
    P(i, j, k, prims::u3) = (gamma / (W + Bsq)) * (Qtcon[3] + QdB * Bcon[3] / W);

    return InversionStatus::success;
}

// TODO lighten up on checks in these functions in favor of sanity at the end
// Document these
KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS* eos, const Real Bsq, const Real D, const Real Ep, const Real QdB,
                                    const Real Qtsq, const Real Wp, int &eflag)
{

    Real W = Wp + D;
    Real gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, eflag);
    Real w = W / pow(gamma,2);
    Real rho0 = D / gamma;
    Real p = eos->p_w(rho0, w);

    return -Ep + Wp - p + 0.5 * Bsq + 0.5 * (Bsq * Qtsq - QdB * QdB) / pow((Bsq + W),2);

}

/**
 * Fluid relativistic factor gamma in terms of inversion state variables
 */
KOKKOS_INLINE_FUNCTION Real gamma_func(const Real Bsq, const Real D, const Real QdB,
                                        const Real Qtsq, const Real Wp, int &eflag)
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
        eflag = ERR_UTSQ;
    }
    if (gamma < 1.)
    {
        eflag = ERR_GAMMA;
    }
#endif

    return gamma;
}

/**
 * See Mignone & McKinney
 * TODO make local?  Index stuff seems weird
 */
KOKKOS_INLINE_FUNCTION Real Wp_func(const Grid &G, const GridVars P, const EOS* eos,
                                    const int i, const int j, const int k, const Loci loc, int &eflag)
{
    Real rho0, u, gamma;

    rho0 = P(i, j, k, prims::rho);
    u = P(i, j, k, prims::u);

    gamma = mhd_gamma_calc(G, P, i, j, k, loc);

    return (rho0 + u + eos->p(rho0, u)) * gamma * gamma - rho0 * gamma;
}
