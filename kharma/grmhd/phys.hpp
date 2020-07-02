/**
 * Physics functions
 * Purely device-side utility functions operating on local data or local parts of matrices.
 */
#pragma once

#include "decs.hpp"
#include "gr_coordinates.hpp"
#include "eos.hpp"
#include "utils.hpp"

/*
 * These functions mostly have several overloads, related to local vs global variables.
 *
 * One version usually takes a local cache e.g. P[GR_DIM] of a variable, in this case primitives in a single zone
 * The other version(s) take e.g. P, the pointer to all primitives, which will be indexed P(i,j,k) to get the zone's values
 *
 * This allows easy fusing/splitting of loops, while avoiding unnecessary global writes of temporary variables
 */

KOKKOS_INLINE_FUNCTION Real bsq_calc(const FourVectors& D)
{
    return D.bcon[0] * D.bcov[0] +
           D.bcon[1] * D.bcov[1] +
           D.bcon[2] * D.bcov[2] +
           D.bcon[3] * D.bcov[3];
}

/**
 * Get a row of the MHD stress-energy tensor with first index up, second index down.
 * A factor of sqrt(4 pi) is absorbed into the definition of b.
 * See Gammie & McKinney '04
 */
KOKKOS_INLINE_FUNCTION void mhd_calc(const GridVars P, const FourVectors& D, const EOS* eos,
                                     const int& k, const int& j, const int& i, const int dir,
                                     Real mhd[GR_DIM])
{
    Real rho, u, pgas, w, bsq, eta, ptot;

    rho = P(prims::rho, k, j, i);
    u =   P(prims::u, k, j, i);
    pgas = eos->p(rho, u);
    w = pgas + rho + u;
    bsq = bsq_calc(D);
    eta = w + bsq;
    ptot = pgas + 0.5 * bsq;

    DLOOP1
    {
        mhd[mu] = eta * D.ucon[dir] * D.ucov[mu] +
                  ptot * (dir == mu) -
                  D.bcon[dir] * D.bcov[mu];
    }
}
KOKKOS_INLINE_FUNCTION void mhd_calc(const Real P[NPRIM], const FourVectors& D, const EOS* eos,
                                     const int& k, const int& j, const int& i, const int dir,
                                     Real mhd[GR_DIM])
{
    Real rho, u, pgas, w, bsq, eta, ptot;

    rho = P[prims::rho];
    u = P[prims::u];
    pgas = eos->p(rho, u);
    w = pgas + rho + u;
    bsq = bsq_calc(D);
    eta = w + bsq;
    ptot = pgas + 0.5 * bsq;

    DLOOP1
    {
        mhd[mu] = eta * D.ucon[dir] * D.ucov[mu] +
                  ptot * (dir == mu) -
                  D.bcon[dir] * D.bcov[mu];
    }
}

/**
 *  Calculate magnetic field four-vector, see Gammie et al '03
 */
KOKKOS_INLINE_FUNCTION void bcon_calc(const GridVars P, FourVectors& D,
                                      const int& k, const int& j, const int& i,
                                      Real bcon[GR_DIM])
{
    bcon[0] = P(prims::B1, k, j, i) * D.ucov[1] +
              P(prims::B2, k, j, i) * D.ucov[2] +
              P(prims::B3, k, j, i) * D.ucov[3];
    for (int mu = 1; mu < GR_DIM; ++mu)
    {
        bcon[mu] = (P(prims::B1 - 1 + mu, k, j, i) +
                             bcon[0] * D.ucon[mu]) /
                            D.ucon[0];
    }
}
KOKKOS_INLINE_FUNCTION void bcon_calc(const Real P[NPRIM], FourVectors& D,
                                      const int& k, const int& j, const int& i,
                                      Real bcon[GR_DIM])
{
    bcon[0] = P[prims::B1] * D.ucov[1] +
                P[prims::B2] * D.ucov[2] +
                P[prims::B3] * D.ucov[3];
    for (int mu = 1; mu < GR_DIM; ++mu)
    {
        bcon[mu] = (P[prims::B1 - 1 + mu] +
                             bcon[0] * D.ucon[mu]) /
                            D.ucon[0];
    }
}

/**
 *  Find gamma-factor of the fluid w.r.t. normal observer
 *
 * TODO Error or print/clip if outside min or max value
 */
KOKKOS_INLINE_FUNCTION Real mhd_gamma_calc(const GRCoordinates &G, const GridVars P,
                                             const int& k, const int& j, const int& i,
                                             const Loci loc)
{

    Real qsq = G.gcov(loc, j, i, 1, 1) * P(prims::u1, k, j, i) * P(prims::u1, k, j, i) +
    G.gcov(loc, j, i, 2, 2) * P(prims::u2, k, j, i) * P(prims::u2, k, j, i) +
    G.gcov(loc, j, i, 3, 3) * P(prims::u3, k, j, i) * P(prims::u3, k, j, i) +
    2. * (G.gcov(loc, j, i, 1, 2) * P(prims::u1, k, j, i) * P(prims::u2, k, j, i) +
            G.gcov(loc, j, i, 1, 3) * P(prims::u1, k, j, i) * P(prims::u3, k, j, i) +
            G.gcov(loc, j, i, 2, 3) * P(prims::u2, k, j, i) * P(prims::u3, k, j, i));

    return sqrt(1. + qsq);
}
KOKKOS_INLINE_FUNCTION Real mhd_gamma_calc(const GRCoordinates &G, const Real P[NPRIM],
                                             const int& k, const int& j, const int& i,
                                             const Loci loc)
{
    Real qsq = G.gcov(loc, j, i, 1, 1) * P[prims::u1] * P[prims::u1] +
    G.gcov(loc, j, i, 2, 2) * P[prims::u2] * P[prims::u2] +
    G.gcov(loc, j, i, 3, 3) * P[prims::u3] * P[prims::u3] +
    2. * (G.gcov(loc, j, i, 1, 2) * P[prims::u1] * P[prims::u2] +
            G.gcov(loc, j, i, 1, 3) * P[prims::u1] * P[prims::u3] +
            G.gcov(loc, j, i, 2, 3) * P[prims::u2] * P[prims::u3]);

    return sqrt(1. + qsq);
}

/**
 *  Find contravariant four-velocity from the primitive 3-velocity
 */
KOKKOS_INLINE_FUNCTION void ucon_calc(const GRCoordinates &G, const GridVars P,
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      Real ucon[GR_DIM])
{
    Real gamma = mhd_gamma_calc(G, P, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));
    ucon[0] = gamma / alpha;

    for (int mu = 1; mu < GR_DIM; ++mu)
    {
        ucon[mu] = P(prims::u1 + mu - 1, k, j, i) -
                            gamma * alpha * G.gcon(loc, j, i, 0, mu);
    }
}
KOKKOS_INLINE_FUNCTION void ucon_calc(const GRCoordinates &G, const Real P[NPRIM],
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      Real ucon[GR_DIM])
{
    Real gamma = mhd_gamma_calc(G, P, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));
    ucon[0] = gamma / alpha;

    for (int mu = 1; mu < GR_DIM; ++mu)
    {
        ucon[mu] = P[prims::u1 + mu - 1] -
                            gamma * alpha * G.gcon(loc, j, i, 0, mu);
    }
}

/**
 * Calculate ucon, ucov, bcon, bcov from primitive variables
 * Note each member of D must be allocated first
 */
KOKKOS_INLINE_FUNCTION void get_state(const GRCoordinates& G, const GridVars P,
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      FourVectors& D)
{
    // Half-local version: immediate derived vars
    ucon_calc(G, P, k, j, i, loc, D.ucon);
    G.lower(D.ucon, D.ucov, k, j, i, loc);
    bcon_calc(P, D, k, j, i, D.bcon);
    G.lower(D.bcon, D.bcov, k, j, i, loc);
}
KOKKOS_INLINE_FUNCTION void get_state(const GRCoordinates& G, const Real P[NPRIM],
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      FourVectors& D)
{
    // All-local version: immediate prims and derived
    ucon_calc(G, P, k, j, i, loc, D.ucon);
    G.lower(D.ucon, D.ucov, k, j, i, loc);
    bcon_calc(P, D, k, j, i, D.bcon);
    G.lower(D.bcon, D.bcov, k, j, i, loc);
}

/**
 * Turn the primitive variables at a location into the local conserved variables, or fluxes at a face
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const GridVars P, const FourVectors D, const EOS* eos,
                                         const int& k, const int& j, const int& i, const Loci loc, const int dir,
                                         GridVars flux)
{
    Real mhd[GR_DIM];

    // Particle number flux
    flux(prims::rho, k, j, i) = P(prims::rho, k, j, i) * D.ucon[dir];

    mhd_calc(P, D, eos, k, j, i, dir, mhd);

    // MHD stress-energy tensor w/ first index up, second index down
    flux(prims::u, k, j, i) = mhd[0] + flux(prims::rho, k, j, i);
    flux(prims::u1, k, j, i) = mhd[1];
    flux(prims::u2, k, j, i) = mhd[2];
    flux(prims::u3, k, j, i) = mhd[3];

    // Dual of Maxwell tensor
    flux(prims::B1, k, j, i) = D.bcon[1] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[1];
    flux(prims::B2, k, j, i) = D.bcon[2] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[2];
    flux(prims::B3, k, j, i) = D.bcon[3] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[3];

    PLOOP flux(p, k, j, i) *= G.gdet(loc, j, i);
}
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates &G, const GridVars P, const FourVectors D, const EOS* eos,
                                         const int& k, const int& j, const int& i, const Loci loc, const int dir,
                                         Real flux[NPRIM])
{
    Real mhd[GR_DIM];

    // Particle number flux
    flux[prims::rho] = P(prims::rho, k, j, i) * D.ucon[dir];

    mhd_calc(P, D, eos, k, j, i, dir, mhd);

    // MHD stress-energy tensor w/ first index up, second index down
    flux[prims::u] = mhd[0] + flux[prims::rho];
    flux[prims::u1] = mhd[1];
    flux[prims::u2] = mhd[2];
    flux[prims::u3] = mhd[3];

    // Dual of Maxwell tensor
    flux[prims::B1] = D.bcon[1] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[1];
    flux[prims::B2] = D.bcon[2] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[2];
    flux[prims::B3] = D.bcon[3] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[3];

    PLOOP flux[p] *= G.gdet(loc, j, i);
}
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates &G, const Real P[NPRIM], const FourVectors D, const EOS* eos,
                                         const int& k, const int& j, const int& i, const Loci loc, const int dir,
                                         Real flux[NPRIM])
{
    Real mhd[GR_DIM];

    // Particle number flux
    flux[prims::rho] = P[prims::rho] * D.ucon[dir];

    mhd_calc(P, D, eos, k, j, i, dir, mhd);

    // MHD stress-energy tensor w/ first index up, second index down
    flux[prims::u] = mhd[0] + flux[prims::rho];
    flux[prims::u1] = mhd[1];
    flux[prims::u2] = mhd[2];
    flux[prims::u3] = mhd[3];

    // Dual of Maxwell tensor
    flux[prims::B1] = D.bcon[1] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[1];
    flux[prims::B2] = D.bcon[2] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[2];
    flux[prims::B3] = D.bcon[3] * D.ucon[dir] -
                               D.bcon[dir] * D.ucon[3];

    PLOOP flux[p] *= G.gdet(loc, j, i);
}


/**
 *  Calculate components of magnetosonic velocity from primitive variables
 */
KOKKOS_INLINE_FUNCTION void mhd_vchar(const GRCoordinates &G, const GridVars P, const FourVectors D, const EOS* eos,
                                      const int& k, const int& j, const int& i, const Loci loc, const int dir,
                                      Real& cmax, Real& cmin)
{
    Real discr, vp, vm, bsq, ee, ef, va2, cs2, cms2, u;
    Real Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
    Real Acov[GR_DIM] = {0}, Bcov[GR_DIM] = {0};
    Real Acon[GR_DIM] = {0}, Bcon[GR_DIM] = {0};

    Acov[dir] = 1.;
    Bcov[0] = 1.;

    DLOOP2 // TODO use lower()
    {
        Acon[mu] += G.gcon(loc, j, i, mu, nu) * Acov[nu];
        Bcon[mu] += G.gcon(loc, j, i, mu, nu) * Bcov[nu];
    }

    // Find fast magnetosonic speed
    bsq = bsq_calc(D);
    u =  P(prims::u, k, j, i);
    ef = P(prims::rho, k, j, i) + eos->gam * u;
    ee = bsq + ef;
    va2 = bsq / ee;
    cs2 = eos->gam * eos->p(0, u) / ef;

    cms2 = cs2 + va2 - cs2 * va2;

    clip(cms2, 0., 1.);

    // Require that speed of wave measured by observer q.ucon is cms2
    Asq = dot(Acon, Acov);
    Bsq = dot(Bcon, Bcov);
    Au = dot(Acov, D.ucon);
    Bu = dot(Bcov, D.ucon);
    AB = dot(Acon, Bcov);
    Au2 = Au * Au;
    Bu2 = Bu * Bu;
    AuBu = Au * Bu;

    A = Bu2 - (Bsq + Bu2) * cms2;
    B = 2. * (AuBu - (AB + AuBu) * cms2);
    C = Au2 - (Asq + Au2) * cms2;

    discr = B * B - 4. * A * C;
    discr = (discr < 0.) ? 0. : discr;
    discr = sqrt(discr);

    vp = -(-B + discr) / (2. * A);
    vm = -(-B - discr) / (2. * A);

    cmax = (vp > vm) ? vp : vm;
    cmin = (vp > vm) ? vm : vp;
}