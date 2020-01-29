/**
 * Variable inversion to recover primitive variables from conserved quantities
 * See Mignone & McKinney 2007
 */

#include "decs.hpp"

#define ERRTOL 1.e-8
#define ITERMAX 8
#define GAMMAMAX 200

#define ERR_NEG_INPUT -100
#define ERR_MAX_ITER 1
#define ERR_UTSQ 2
#define ERR_GAMMA 3
#define ERR_RHO_NEGATIVE 6
#define ERR_U_NEGATIVE 7
#define ERR_BOTH_NEGATIVE 8

KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS eos, const Real Bsq, const Real D, const Real Ep, const Real QdB,
                                    const Real Qtsq, const Real Wp, int &eflag);
KOKKOS_INLINE_FUNCTION Real gamma_func(const Real Bsq, const Real D, const Real QdB,
                                        const Real Qtsq, const Real Wp, int &eflag);
KOKKOS_INLINE_FUNCTION Real Wp_func(const Grid &G, const GridVars P, const EOS eos,
                                    const int i, const int j, const int k, const Loci loc, int &eflag);

/**
 * Recover primitive variables
 */
KOKKOS_INLINE_FUNCTION int U_to_P(const Grid &G, const GridVars U, const EOS eos,
                                  const int i, const int j, const int k, const Loci loc, GridVars P)
{
    int eflag = 0;

    Real gdet = G.gdet(loc, i, j);
    Real lapse = 1./sqrt(-G.gcon(loc, i, j, 0, 0));

    // Update the primitive B-fields
    P(i, j, k, prims::B1) = U(i, j, k, prims::B1) / gdet;
    P(i, j, k, prims::B2) = U(i, j, k, prims::B2) / gdet;
    P(i, j, k, prims::B3) = U(i, j, k, prims::B3) / gdet;

    // Catch negative density
    if (U(i, j, k, prims::rho) <= 0.)
    {
        return ERR_NEG_INPUT;
    }

    // Convert from conserved variables to four-vectors
    Real D = U(i, j, k, prims::rho) * lapse / gdet;

    Real Bcon[NDIM];
    Bcon[0] = 0.;
    Bcon[1] = U(i, j, k, prims::B1) * lapse / gdet;
    Bcon[2] = U(i, j, k, prims::B2) * lapse / gdet;
    Bcon[3] = U(i, j, k, prims::B3) * lapse / gdet;

    Real Qcov[NDIM];
    Qcov[0] = (U(i, j, k, prims::u) - U(i, j, k, prims::rho)) * lapse / gdet;
    Qcov[1] = U(i, j, k, prims::u1) * lapse / gdet;
    Qcov[2] = U(i, j, k, prims::u2) * lapse / gdet;
    Qcov[3] = U(i, j, k, prims::u3) * lapse / gdet;

    Real ncov[NDIM];
    ncov[0] = -lapse;
    ncov[1] = 0.;
    ncov[2] = 0.;
    ncov[3] = 0.;

    // Interlaced upper/lower operation
    Real Bcov[NDIM], Qcon[NDIM], ncon[NDIM];
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
    //raise_grid(ncov, ncon, G, i, j, k, loc);
    Real Bsq = dot(Bcon, Bcov);
    Real QdB = dot(Bcon, Qcov);
    Real Qdotn = dot(Qcon, ncov);
    Real Qsq = dot(Qcon, Qcov);

    Real Qtcon[NDIM];
    DLOOP1 Qtcon[mu] = Qcon[mu] + ncon[mu] * Qdotn;
    Real Qtsq = Qsq + Qdotn * Qdotn;

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
    Real dedW2 = (errp - 2. * err + errm) / (h * h);
    Real f = 0.5 * err * dedW2 / (dedW * dedW);
    // Limit size of 2nd derivative correction
    if (f < -0.3)
        f = -0.3;
    if (f > 0.3)
        f = 0.3;

    Real dW = -err / dedW / (1. - f);
    Real Wp1 = Wp;
    Real err1 = err;
    // Limit size of step
    if (dW < -0.5 * Wp)
        dW = -0.5 * Wp;
    if (dW > 2.0 * Wp)
        dW = 2.0 * Wp;

    Wp += dW;
    err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

    // Not good enough?  apply secant method
    int iter = 0;
    for (iter = 0; iter < ITERMAX; iter++)
    {
        dW = (Wp1 - Wp) * err / (err - err1);

        // TODO should this have limit applied?
        Wp1 = Wp;
        err1 = err;

        // Normal secant increment is dW. Also limit guess to between 0.5 and 2
        // times the current value
        if (dW < -0.5 * Wp)
            dW = -0.5 * Wp;
        if (dW > 2.0 * Wp)
            dW = 2.0 * Wp;

        Wp += dW;

        if (fabs(dW / Wp) < ERRTOL)
        {
            //fprintf(stderr, "Breaking!\n");
            break;
        }

        err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);
        //fprintf(stderr, "%.15f ", err);

        if (fabs(err / Wp) < ERRTOL)
        {
            //fprintf(stderr, "Breaking!\n");
            break;
        }
    }
    // Failure to converge; do not set primitives other than B
    if (iter == ITERMAX)
    {
        return (1);
    }

    // Find utsq, gamma, rho0 from Wp
    Real gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, eflag);

    // Find the scalars
    Real rho0 = D / gamma;
    Real W = Wp + D;
    Real w = W / (gamma * gamma);
    Real p = eos.p_w(rho0, w);
    Real u = w - (rho0 + p);

    // Return without updating non-B primitives
    if (rho0 < 0 && u < 0)
    {
        return ERR_BOTH_NEGATIVE;
    }
    else if (rho0 < 0)
    {
        return ERR_RHO_NEGATIVE;
    }
    else if (u < 0)
    {
        return ERR_U_NEGATIVE;
    }

    // Set primitives
    P(i, j, k, prims::rho) = rho0;
    P(i, j, k, prims::u) = u;

    // Find u(tilde); Eqn. 31 of Noble et al.
    P(i, j, k, prims::u1) = (gamma / (W + Bsq)) * (Qtcon[1] + QdB * Bcon[1] / W);
    P(i, j, k, prims::u2) = (gamma / (W + Bsq)) * (Qtcon[2] + QdB * Bcon[2] / W);
    P(i, j, k, prims::u3) = (gamma / (W + Bsq)) * (Qtcon[3] + QdB * Bcon[3] / W);

    return 0;
}

// TODO lighten up on checks in these functions in favor of sanity at the end
// Document these
KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS eos, const Real Bsq, const Real D, const Real Ep, const Real QdB,
                                    const Real Qtsq, const Real Wp, int &eflag)
{

    Real W = Wp + D;
    Real gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, eflag);
    Real w = W / (gamma * gamma);
    Real rho0 = D / gamma;
    Real p = eos.p_w(rho0, w);

    Real err = -Ep + Wp - p + 0.5 * Bsq + 0.5 * (Bsq * Qtsq - QdB * QdB) / ((Bsq + W) * (Bsq + W));

    return err;
}

/**
 * Fluid relativistic factor gamma in terms of inversion state variables
 */
KOKKOS_INLINE_FUNCTION Real gamma_func(const Real Bsq, const Real D, const Real QdB,
                                        const Real Qtsq, const Real Wp, int &eflag)
{
    Real QdBsq, W, utsq, gamma, W2, WB;

    QdBsq = QdB * QdB;
    W = D + Wp;
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
KOKKOS_INLINE_FUNCTION Real Wp_func(const Grid &G, const GridVars P, const EOS eos,
                                    const int i, const int j, const int k, const Loci loc, int &eflag)
{
    Real rho0, u, utsq, gamma;
    Real utcon[NDIM] = {0}, utcov[NDIM] = {0};

    rho0 = P(i, j, k, prims::rho);
    u = P(i, j, k, prims::u);

    utcon[0] = 0.;
    utcon[1] = P(i, j, k, prims::u1);
    utcon[2] = P(i, j, k, prims::u2);
    utcon[3] = P(i, j, k, prims::u3);

    // TODO can this be covered by the fluid_gamma in phys.c????
    G.lower(utcon, utcov, i, j, k, loc);
    for (int mu = 0; mu < NDIM; mu++)
    {
        utcov[mu] = 0.;
        for (int nu = 0; nu < NDIM; nu++)
        {
            utcov[mu] += G.gcov(Loci::center, i, j, mu, nu) * utcon[nu];
        }
    }
    utsq = dot(utcon, utcov);

    // Catch utsq < 0
    if ((utsq < 0.) && (fabs(utsq) < 1.e-13))
    {
        utsq = fabs(utsq);
    }
    if (utsq < 0. || utsq > 1.e3 * GAMMAMAX * GAMMAMAX)
    {
        eflag = ERR_UTSQ;
        return rho0 + u; // Not sure what to do here...
    }

    gamma = sqrt(1. + fabs(utsq));

    return (rho0 + u + eos.p(rho0, u)) * gamma * gamma - rho0 * gamma;
}
