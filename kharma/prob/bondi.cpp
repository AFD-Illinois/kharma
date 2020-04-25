
// BONDI PROBLEM

#include "decs.hpp"

#include "grid.hpp"
#include "eos.hpp"
#include "phys.hpp"

using namespace parthenon;
using namespace std;

void get_prim_bondi(const Grid& G, GridVars P, const EOS* eos, const Real mdot, const Real rs,
                    const int& k, const int& j, const int& i);

/**
 * Initialization of a Bondi problem with specified sonic point, BH mdot, and horizon radius
 * TODO this can/should be just mdot (and the grid ofc), if this problem is to be used as anything more than a test
 */
void InitializeBondi(MeshBlock *pmb, const Grid& G, GridVars P,
                     const EOS* eos, const Real mdot, const Real rs)
{
    FLAG("Initializing Bondi problem");

    pmb->par_for( "init_bondi", 0, pmb->ncells3-1, 0, pmb->ncells2-1, 0, pmb->ncells1-1,
        KOKKOS_LAMBDA_3D {
            get_prim_bondi(G, P, eos, mdot, rs, k, j, i);
        }
    );
}

void ApplyBondiBoundary(Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    GridVars U = rc.Get("c.c.bulk.cons").data;
    GridVars P = rc.Get("c.c.bulk.prims").data;

    Grid G(pmb);

    FLAG("Applying Bondi X1R boundary");

    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    Real mdot = pmb->packages["GRMHD"]->Param<Real>("mdot");
    Real rs = pmb->packages["GRMHD"]->Param<Real>("rs");
    EOS* eos = new GammaLaw(gamma);

    // Just the X1 right boundary
    pmb->par_for("bondi_boundary", 0, pmb->ncells3-1, 0, pmb->ncells2-1, pmb->ncells1 - NGHOST, pmb->ncells1-1,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;
            get_prim_bondi(G, P, eos, mdot, rs, k, j, i);
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
        }
    );
}
// Adapted from M. Chandra
KOKKOS_INLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
    return pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + pow(C1 / pow(r,2) / pow(T, n), 2.)) - C2;
}

KOKKOS_INLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tmin = 0.6 * (sqrt(C2) - 1.) / (n + 1);
    Real Tmax = pow(C1 * sqrt(2. / pow(r,3)), 1. / n);

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r, C1, C2, n);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r, C1, C2, n);
    if (f0 * f1 > 0) return -1;

    Th = (f1 * T0 - f0 * T1) / (f1 - f0);
    fh = get_Tfunc(Th, r, C1, C2, n);
    Real epsT = rtol * (Tmin + Tmax);
    while (fabs(Th - T0) > epsT && fabs(Th - T1) > epsT && fabs(fh) > ftol)
    {
        if (fh * f0 < 0.)
        {
            T0 = Th;
            f0 = fh;
        }
        else
        {
            T1 = Th;
            f1 = fh;
        }

        Th = (f1 * T0 - f0 * T1) / (f1 - f0);
        fh = get_Tfunc(Th, r, C1, C2, n);
    }

    return Th;
}

/**
 * Make primitive velocities out of 4-velocity.  See Gammie '04
 * Returns in the given
 */
KOKKOS_INLINE_FUNCTION void fourvel_to_prim(const Real gcon[NDIM][NDIM], const Real ucon[NDIM], Real u_prim[NDIM])
{
    Real alpha2 = -1.0 / gcon[0][0];
    // Note gamma/alpha is ucon[0]
    u_prim[1] = ucon[1] + ucon[0] * alpha2 * gcon[0][1];
    u_prim[2] = ucon[2] + ucon[0] * alpha2 * gcon[0][2];
    u_prim[3] = ucon[3] + ucon[0] * alpha2 * gcon[0][3];
}
// KOKKOS_INLINE_FUNCTION void fourvel_to_prim(const Real gcon[NDIM][NDIM], Real ucon[NDIM], Real u_prim[NDIM])
// {
//     Real beta[NDIM];
//     Real alpha = 1.0/sqrt(-gcon[0][0]);
//     beta[1] = alpha*alpha*gcon[0][1];
//     beta[2] = alpha*alpha*gcon[0][2];
//     beta[3] = alpha*alpha*gcon[0][3];
//     Real gamma = ucon[0]*alpha;

//     u_prim[0] = 0;
//     u_prim[1] = ucon[1] + beta[1]*gamma/alpha;
//     u_prim[2] = ucon[2] + beta[2]*gamma/alpha;
//     u_prim[3] = ucon[3] + beta[3]*gamma/alpha;
// }

/**
 * Set time component for consistency given a 3-velocity
 */
KOKKOS_INLINE_FUNCTION void set_ut(const Real gcov[NDIM][NDIM], Real ucon[NDIM])
{
    Real AA, BB, CC;

    AA = gcov[0][0];
    BB = 2. * (gcov[0][1] * ucon[1] +
               gcov[0][2] * ucon[2] +
               gcov[0][3] * ucon[3]);
    CC = 1. + gcov[1][1] * ucon[1] * ucon[1] +
         gcov[2][2] * ucon[2] * ucon[2] +
         gcov[3][3] * ucon[3] * ucon[3] +
         2. * (gcov[1][2] * ucon[1] * ucon[2] +
               gcov[1][3] * ucon[1] * ucon[3] +
               gcov[2][3] * ucon[2] * ucon[3]);

    Real discr = BB * BB - 4. * AA * CC;
    ucon[0] = (-BB - sqrt(discr)) / (2. * AA);
}

/**
 * Get the Bondi solution at a particular zone.  Templated to be host- or device-side.
 * Note this assumes that there are ghost zones!
 */
void get_prim_bondi(const Grid& G, GridVars P, const EOS* eos, const Real mdot, const Real rs,
                    const int& k, const int& j, const int& i)
{
    // Solution constants
    // TODO cache these?  State is awful
    Real n = 1. / (eos->gam - 1.);
    Real uc = sqrt(mdot / (2. * rs));
    Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
    Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
    Real C1 = uc * pow(rs, 2) * pow(Tc, n);
    Real C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

    // TODO this seems awkward.  Is this really the way to use my new tooling?
    // Note some awkwardness is the fact we need both native coords and embedding r
    SphKSCoords ks = mpark::get<SphKSCoords>(G.coords->base);
    SphBLCoords bl = SphBLCoords(ks.a);

    GReal X[NDIM], Xembed[NDIM];
    G.coord(k, j, i, Loci::center, X);
    G.coord_embed(k, j, i, Loci::center, Xembed);
    Real Rhor = ks.rhor();
    // Any zone inside the horizon gets the horizon's values
    int ii = i;
    while (Xembed[1] < Rhor)
    {
        ++ii;
        G.coord(k, j, ii, Loci::center, X);
        G.coord_embed(k, j, ii, Loci::center, Xembed);
    }
    GReal r = Xembed[1];

    Real T = get_T(r, C1, C2, n);
    //if (T < 0) T = 0; // If you can't error, NaN
    Real ur = -C1 / (pow(T, n) * pow(r, 2));
    Real rho = pow(T, n);
    Real u = rho * T * n;

    // Set u^t to make u^r a 4-vector
    Real ucon_bl[NDIM] = {0, ur, 0, 0};
    Real gcov_bl[NDIM][NDIM];
    bl.gcov_embed(Xembed, gcov_bl);
    set_ut(gcov_bl, ucon_bl);

    // Then transform that 4-vector to KS, then to native
    Real ucon_ks[NDIM], ucon_mks[NDIM], u_prim[NDIM];
    ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
    G.coords->con_vec_to_native(X, ucon_ks, ucon_mks);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[NDIM][NDIM];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_mks, u_prim);

#if DEBUG
    static int fails = 0;
    if (T < 0 || rho < 0 || u < 0) {
        Real ucov_mks[NDIM];
        G.lower(ucon_mks, ucov_mks, k, j, i, Loci::center);
        Real uu_mks = dot(ucon_mks, ucov_mks);

        cerr << "Bondi Bad in zone " << i << " " << j << " " << k << std::endl;
        cerr << "Native X:  " << X[1] << " " << X[2] << " " << X[3] << std::endl;
        cerr << "Embed X:  " << Xembed[1] << " " << Xembed[2] << " " << Xembed[3] << std::endl;
        cerr << "gam, C1, C2, n, T: " << eos->gam << " " << C1 << " " << C2 << " " << n << " " << T << std::endl;
        cerr << "Tmin, Tmax: " << 0.6 * (sqrt(C2) - 1.) / (n + 1) << " " << pow(C1 * sqrt(2. / pow(r,3)), 1. / n) << std::endl;
        cerr << "rho, u:  " << rho << " " << u << std::endl;
        cerr << "u [BL]:  " << ucon_bl[0] << " " << ucon_bl[1] << " " << ucon_bl[2] << " " << ucon_bl[3] << std::endl;
        cerr << "u [KS]:  " << ucon_ks[0] << " " << ucon_ks[1] << " " << ucon_ks[2] << " " << ucon_ks[3] << std::endl;
        cerr << "u [native]:  " << ucon_mks[0] << " " << ucon_mks[1] << " " << ucon_mks[2] << " " << ucon_mks[3] << std::endl;
        cerr << "u.u [native]:  " << uu_mks << std::endl;
        cerr << "u [prim]:  " << u_prim[1] << " " << u_prim[2] << " " << u_prim[3] << std::endl;
        fails++;
    }
    if(fails > 100) {
        exit(-1);
    }
#endif

    P(prims::rho, k, j, i) = rho;
    P(prims::u, k, j, i) = u;
    P(prims::u1, k, j, i) = u_prim[1];
    P(prims::u2, k, j, i) = u_prim[2];
    P(prims::u3, k, j, i) = u_prim[3];
    P(prims::B1, k, j, i) = 0.;
    P(prims::B2, k, j, i) = 0.;
    P(prims::B3, k, j, i) = 0.;
}