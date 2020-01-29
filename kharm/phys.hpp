/**
 * Physics functions
 * 
 */

#include "decs.hpp"

#include "eos.hpp"

KOKKOS_INLINE_FUNCTION double bsq_calc(const GridDerived D, const int i, const int j, const int k)
{
    return D.bcon(i, j, k, 0) * D.bcov(i, j, k, 0) +
           D.bcon(i, j, k, 1) * D.bcov(i, j, k, 1) +
           D.bcon(i, j, k, 2) * D.bcov(i, j, k, 2) +
           D.bcon(i, j, k, 3) * D.bcov(i, j, k, 3);
}

/**
 * MHD stress-energy tensor with first index up, second index down. A factor of
 * sqrt(4 pi) is absorbed into the definition of b.
 * See Gammie & McKinney '04
 */
KOKKOS_INLINE_FUNCTION void mhd_calc(const GridVars P, const GridDerived D, const EOS eos,
                                     const int i, const int j, const int k, const int dir,
                                     Real *mhd)
{
    Real u, pgas, w, bsq, eta, ptot;

    u = P(i, j, k, prims::u);
    pgas = eos.p(0, u);
    w = pgas + P(i, j, k, prims::rho) + u;
    bsq = bsq_calc(D, i, j, k);
    eta = w + bsq;
    ptot = pgas + 0.5 * bsq;

    DLOOP1
    {
        mhd[mu] = eta * D.ucon(i, j, k, dir) * D.ucov(i, j, k, mu) +
                  ptot * (dir == mu) -
                  D.bcon(i, j, k, dir) * D.bcov(i, j, k, mu);
    }
}

/**
 * Turn the primitive variables at a location into 
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const Grid &G, const GridVars P, const GridDerived D, const EOS eos,
                                         const int i, const int j, const int k, const Loci loc, const int dir,
                                         GridVars flux)
{
    Real mhd[NDIM];

    // Particle number flux
    flux(i, j, k, prims::rho) = P(i, j, k, prims::rho) * D.ucon(i, j, k, dir);

    mhd_calc(P, D, eos, i, j, k, dir, mhd);

    // MHD stress-energy tensor w/ first index up, second index down
    flux(i, j, k, prims::u) = mhd[0] + flux(i, j, k, prims::rho);
    flux(i, j, k, prims::u1) = mhd[1];
    flux(i, j, k, prims::u2) = mhd[2];
    flux(i, j, k, prims::u3) = mhd[3];

    // Dual of Maxwell tensor
    flux(i, j, k, prims::B1) = D.bcon(i, j, k, 1) * D.ucon(i, j, k, dir) -
                               D.bcon(i, j, k, dir) * D.ucon(i, j, k, 1);
    flux(i, j, k, prims::B2) = D.bcon(i, j, k, 2) * D.ucon(i, j, k, dir) -
                               D.bcon(i, j, k, dir) * D.ucon(i, j, k, 2);
    flux(i, j, k, prims::B3) = D.bcon(i, j, k, 3) * D.ucon(i, j, k, dir) -
                               D.bcon(i, j, k, dir) * D.ucon(i, j, k, 3);

    // Note for later all passives go here
    //   flux(i, j, k, KEL) = flux(i, j, k, prims::rho) * P(i, j, k, KEL);
    //   etc

    for (int p = 0; p < G.nvar; ++p)
        flux(i, j, k, p) *= G.gdet(loc, i, j);
}

// calculate magnetic field four-vector
KOKKOS_INLINE_FUNCTION void bcon_calc(const GridVars P, const GridDerived D,
                                      const int i, const int j, const int k,
                                      GridVector bcon)
{
    bcon(i, j, k, 0) = P(i, j, k, prims::B1) * D.ucov(i, j, k, 1) +
                       P(i, j, k, prims::B2) * D.ucov(i, j, k, 2) +
                       P(i, j, k, prims::B3) * D.ucov(i, j, k, 3);
    for (int mu = 1; mu < NDIM; ++mu)
    {
        bcon(i, j, k, mu) = (P(i, j, k, prims::B1 - 1 + mu) +
                             bcon(i, j, k, 0) * D.ucon(i, j, k, mu)) /
                            D.ucon(i, j, k, 0);
    }
}

// Find gamma-factor wrt normal observer
KOKKOS_INLINE_FUNCTION double mhd_gamma_calc(const Grid &G, const GridVars P,
                                             const int i, const int j, const int k,
                                             const Loci loc)
{
    Real qsq = G.gcov(loc, i, j, 1, 1) * P(i, j, k, prims::u1) * P(i, j, k, prims::u1) +
    G.gcov(loc, i, j, 2, 2) * P(i, j, k, prims::u2) * P(i, j, k, prims::u2) +
    G.gcov(loc, i, j, 3, 3) * P(i, j, k, prims::u3) * P(i, j, k, prims::u3) +
    2. * (G.gcov(loc, i, j, 1, 2) * P(i, j, k, prims::u1) * P(i, j, k, prims::u2) +
            G.gcov(loc, i, j, 1, 3) * P(i, j, k, prims::u1) * P(i, j, k, prims::u3) +
            G.gcov(loc, i, j, 2, 3) * P(i, j, k, prims::u2) * P(i, j, k, prims::u3));

#if DEBUG
    if (qsq < 0.)
    {
        if (fabs(qsq) > 1.E-10)
        { // Then assume not just machine precision
            fprintf(stderr,
                    "gamma_calc():  failed: [%i %i %i] qsq = %28.18e \n",
                    i, j, k, qsq);
            fprintf(stderr,
                    "v[1-3] = %28.18e %28.18e %28.18e  \n",
                    P(i, j, k, prims::u1), P(i, j, k, prims::u2), P(i, j, k, prims::u3));
            return 1.0;
        }
        else
        {
            qsq = 1.E-10; // Set floor
        }
    }
#endif

    return sqrt(1. + qsq);
}

// Find contravariant four-velocity
KOKKOS_INLINE_FUNCTION void ucon_calc(const Grid &G, const GridVars P,
                                      const int i, const int j, const int k, const Loci loc,
                                      GridVector ucon)
{
    double gamma = mhd_gamma_calc(G, P, i, j, k, loc);
    double alpha = 1. / sqrt(-G.gcon(loc, i, j, 0, 0));
    ucon(i, j, k, 0) = gamma / alpha;

    for (int mu = 1; mu < NDIM; mu++)
    {
        ucon(i, j, k, mu) = P(i, j, k, prims::u1 + mu - 1) -
                            gamma * alpha * G.gcon(loc, i, j, 0, mu);
    }
}

/**
 * Calculate ucon, ucov, bcon, bcov from primitive variables
 * Note each member of D must be allocated first
 */
KOKKOS_INLINE_FUNCTION void get_state(const Grid &G, const GridVars P,
                                      const int i, const int j, const int k, const Loci loc,
                                      GridDerived D)
{
    ucon_calc(G, P, i, j, k, loc, D.ucon);
    G.lower_grid(D.ucon, D.ucov, i, j, k, loc);
    bcon_calc(P, D, i, j, k, D.bcon);
    G.lower_grid(D.bcon, D.bcov, i, j, k, loc);
}

// Calculate components of magnetosonic velocity from primitive variables
// TODO this is a primary candidate for splitting/vectorizing
KOKKOS_INLINE_FUNCTION void mhd_vchar(const Grid &G, const GridVars P, const GridDerived D, const EOS eos,
                                      const int i, const int j, const int k, const Loci loc, const int dir,
                                      GridScalar cmax, GridScalar cmin)
{
    Real discr, vp, vm, bsq, ee, ef, va2, cs2, cms2, u;
    Real Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
    Real Acov[NDIM] = {0}, Bcov[NDIM] = {0};
    Real Acon[NDIM] = {0}, Bcon[NDIM] = {0};

    Acov[dir] = 1.;
    Bcov[0] = 1.;

    DLOOP2
    {
        Acon[mu] += G.gcon(loc, i, j, mu, nu) * Acov[nu];
        Bcon[mu] += G.gcon(loc, i, j, mu, nu) * Bcov[nu];
    }

    // Find fast magnetosonic speed
    bsq = bsq_calc(D, i, j, k);
    u = P(i, j, k, prims::u);
    ef = P(i, j, k, prims::rho) + eos.gam * u;
    ee = bsq + ef;
    va2 = bsq / ee;
    cs2 = eos.gam * eos.p(0, u) / ef;

    cms2 = cs2 + va2 - cs2 * va2;

    cms2 = (cms2 < 0) ? SMALL : cms2;
    cms2 = (cms2 > 1) ? 1 : cms2;

    // Require that speed of wave measured by observer q.ucon is cms2
    Asq = dot(Acon, Acov);
    Bsq = dot(Bcon, Bcov);
    Au = Bu = 0.;
    DLOOP1
    {
        Au += Acov[mu] * D.ucon(i, j, k, mu);
        Bu += Bcov[mu] * D.ucon(i, j, k, mu);
    }
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

    cmax(i, j, k) = (vp > vm) ? vp : vm;
    cmin(i, j, k) = (vp > vm) ? vm : vp;
}

// Source terms for equations of motion
// Note this is a single function for the state, so it is *not* INLINE_ETC
void get_fluid_source(const Grid &G, const GridVars P, const GridDerived D,
                      const EOS eos, GridVars dU, bool wind=false)
{
    GridVars dP("dP", G.gn1, G.gn2, G.gn3, G.nvar);
    GridDerived dD;
    dD.ucon = GridVector("dD_ucon", G.gn1, G.gn2, G.gn3);
    dD.ucov = GridVector("dD_ucov", G.gn1, G.gn2, G.gn3);
    dD.bcon = GridVector("dD_bcon", G.gn1, G.gn2, G.gn3);
    dD.bcov = GridVector("dD_bcov", G.gn1, G.gn2, G.gn3);
    GridVars dUw("dUw", G.gn1, G.gn2, G.gn3, G.nvar);

    Kokkos::parallel_for("fluid_source", G.bulk_ng(),
                         KOKKOS_LAMBDA(const int i, const int j, const int k) {
                             Real mhd[NDIM][NDIM]; // Too much local memory?

                             DLOOP1 mhd_calc(P, D, eos, i, j, k, mu, mhd[mu]);

                             // Contract mhd stress tensor with connection
                             for (int p = 0; p < G.nvar; ++p)
                                 dU(i, j, k, p) = 0.;
                             DLOOP2
                             {
                                 // Put all 4 values into U(UU,U1,U2,U3)
                                 for (int lam = 0; lam < NDIM; lam++)
                                     dU(i, j, k, prims::u + lam) += mhd[mu][nu] * G.conn(i, j, nu, lam, mu);
                             }

                             for (int p = 0; p < G.nvar; ++p)
                                 dU(i, j, k, p) *= G.gdet(Loci::center, i, j);
                         });

    if (wind) {
        // Add a small "wind" source term in RHO,UU
        // Stolen shamelessly from iharm2d_v3
        Kokkos::parallel_for("fluid_source", G.bulk_ng(),
                         KOKKOS_LAMBDA(const int i, const int j, const int k) {
                             // TODO make these local rather than writing back?

                             // Need coordinates to evaluate particle addtn rate
                             // Note that makes the wind spherical-only
                             GReal r, th;
                             G.ks_coord(i, j, k, Loci::center, r, th);

                             // Particle addition rate: concentrate at poles & center
                             Real drhopdt = 2.e-4 * pow(cos(th), 4) / pow(1. + r * r, 2);

                             dP(i, j, k, prims::u) = drhopdt;

                             Real Tp = 10.; // New fluid's temperature in units of c^2
                             dP(i, j, k, prims::u) = drhopdt * Tp * 3.;

                             // Leave everything else: we're inserting only fluid in normal observer frame

                             // Add plasma to the T^t_a component of the stress-energy tensor
                             // Notice that U already contains a factor of sqrt{-g}
                             get_state(G, dP, i, j, k, Loci::center, dD);
                             prim_to_flux(G, dP, dD, eos, i, j, k, Loci::center, 0, dUw);

                             for (int p = 0; p < G.nvar; ++p)
                                 dU(i, j, k, p) += dUw(i, j, k, p);
                         });
    }
}
