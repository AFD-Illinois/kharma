/*
 * coordinates.hpp:  Coordinate systems as objects
 */
#pragma once

#include "decs.hpp"
#include "matrix.hpp"

/**
 * Abstract class defining any coordinate system.
 * All functions should be callable device-side.
 */
class CoordinateSystem {
    public:
        KOKKOS_INLINE_FUNCTION virtual void ks_coord(GReal X[NDIM], Real &r, Real &th) {};
        KOKKOS_INLINE_FUNCTION virtual void sph_coord(GReal X[NDIM], Real &r, Real &th, Real &phi) {};
        KOKKOS_INLINE_FUNCTION virtual void cart_coord(GReal X[NDIM], Real &x, Real &y, Real &z) {};

        KOKKOS_INLINE_FUNCTION virtual void gcov_native(GReal X[NDIM], Real gcov[NDIM][NDIM]) {};
        KOKKOS_INLINE_FUNCTION virtual void gcon_native(GReal X[NDIM], Real gcon[NDIM][NDIM]) {
            Real gcov[NDIM][NDIM];
            gcov_native(X, gcov);
            gcon_native(gcov, gcon);
        }
        KOKKOS_INLINE_FUNCTION Real gcon_native(Real gcov[NDIM][NDIM], Real gcon[NDIM][NDIM]){
            Real gdet = invert(&gcov[0][0],&gcon[0][0]);
            return sqrt(fabs(gdet));
        };
        KOKKOS_INLINE_FUNCTION void conn_func(GReal X[NDIM], Real conn[NDIM][NDIM][NDIM]) {
            Real tmp[NDIM][NDIM][NDIM];
            Real gcon[NDIM][NDIM];
            Real Xh[NDIM], Xl[NDIM];
            Real gh[NDIM][NDIM];
            Real gl[NDIM][NDIM];

            for (int nu = 0; nu < NDIM; nu++) {
                DLOOP1 Xh[mu] = Xl[mu] = X[mu];
                Xh[nu] += DELTA;
                Xl[nu] -= DELTA;
                gcov_native(Xh, gh);
                gcov_native(Xl, gl);

                for (int lam = 0; lam < NDIM; lam++) {
                    for (int kap = 0; kap < NDIM; kap++) {
                        conn[lam][kap][nu] = (gh[lam][kap] - gl[lam][kap])/
                                                        (Xh[nu] - Xl[nu]);
                    }
                }
            }

            // Rearrange to find \Gamma_{lam nu mu}
            for (int lam = 0; lam < NDIM; lam++) {
                for (int nu = 0; nu < NDIM; nu++) {
                    for (int mu = 0; mu < NDIM; mu++) {
                        tmp[lam][nu][mu] = 0.5 * (conn[nu][lam][mu] + 
                                                  conn[mu][lam][nu] - 
                                                  conn[mu][nu][lam]);
                    }
                }
            }

            // now mu nu kap

            // Need gcon for next bit
            gcon_native(X, gcon);

            // Raise index to get \Gamma^lam_{nu mu}
            for (int lam = 0; lam < NDIM; lam++) {
                for (int nu = 0; nu < NDIM; nu++) {
                    for (int mu = 0; mu < NDIM; mu++) {
                        conn[lam][nu][mu] = 0.;

                        for (int kap = 0; kap < NDIM; kap++)
                            conn[lam][nu][mu] += gcon[lam][kap] * tmp[kap][nu][mu];
                    }
                }
            }
        }
    protected:
        KOKKOS_INLINE_FUNCTION virtual void dxdX_to_native(Real X[NDIM], Real dxdX[NDIM][NDIM]) {};
        KOKKOS_INLINE_FUNCTION virtual void dxdX_to_embed(Real X[NDIM], Real dxdX[NDIM][NDIM]) {};
};

/**
 * Class defining properties mapping a native coordinate system to minkowski space
 */
class Minkowski : public CoordinateSystem {
    public:
        KOKKOS_INLINE_FUNCTION void cart_coord(GReal X[NDIM], Real &x, Real &y, Real &z) {x = X[1]; y = X[2]; z = X[3];};
        KOKKOS_INLINE_FUNCTION void gcov_native(GReal X[NDIM], Real gcov[NDIM][NDIM]) {DLOOP1 gcov[mu][mu] = -1 + 2*(mu != 0);};
        KOKKOS_INLINE_FUNCTION void gcon_native(GReal X[NDIM], Real gcon[NDIM][NDIM]) {DLOOP1 gcon[mu][mu] = -1 + 2*(mu != 0);};

};

// class KS : public CoordinateSystem {
//     public:
//         KS(double spin): a(spin) {};

//         virtual void ks_coord(GReal X[NDIM], Real &r, Real &th) {
//             r = r_of_X(X);
//             th = th_of_X(X);

//   // Avoid singularity at polar axis
// #if COORDSINGFIX
//             if (fabs(th) < SINGSMALL) {
//                 if ((th) >= 0)
//                     th = SINGSMALL;
//                 if ((*th) < 0)
//                     th = -SINGSMALL;
//             }
//             if (fabs(M_PI - th) < SINGSMALL) {
//                 if ((*th) >= M_PI)
//                     th = M_PI + SINGSMALL;
//                 if ((*th) < M_PI)
//                     th = M_PI - SINGSMALL;
//             }
// #endif
//         }

//         virtual void sph_coord(GReal X[NDIM], Real &r, Real &th, Real &phi) {
//             ks_coord(X, r, th);
//             phi = phi_of_X(X);
//         }

//         virtual void gcov_native(GReal X[NDIM], Real gcov[NDIM][NDIM]) {
//             double sth, cth, s2, rho2;
//             double r, th;

//             ks_coord(X, r, th);

//             cth = cos(th);
//             sth = sin(th);

//             s2 = sth*sth;
//             rho2 = r*r + a*a*cth*cth;

//             double gcov_ks[NDIM][NDIM];
//             gcov_ks[0][0] = -1. + 2.*r/rho2;
//             gcov_ks[0][1] = 2.*r/rho2;
//             gcov_ks[0][3] = -2.*a*r*s2/rho2;

//             gcov_ks[1][0] = gcov[0][1];
//             gcov_ks[1][1] = 1. + 2.*r/rho2;
//             gcov_ks[1][3] = -a*s2*(1. + 2.*r/rho2);

//             gcov_ks[2][2] = rho2;

//             gcov_ks[3][0] = gcov[0][3];
//             gcov_ks[3][1] = gcov[1][3];
//             gcov_ks[3][3] = s2*(rho2 + a*a*s2*(1. + 2.*r/rho2));

//             // Apply coordinate transformation to code coordinates X
//             double dxdX[NDIM][NDIM];
//             dxdX_to_native(X, dxdX);

//             for (int mu = 0; mu < NDIM; mu++) {
//                 for (int nu = 0; nu < NDIM; nu++) {
//                     for (int lam = 0; lam < NDIM; lam++) {
//                         for (int kap = 0; kap < NDIM; kap++) {
//                             gcov[mu][nu] += gcov_ks[lam][kap]*dxdX[lam][mu]*dxdX[kap][nu];
//                         }
//                     }
//                 }
//             }
//         }

//     protected:
//         // BH Spin is a property of KS
//         double a;

//         virtual double r_of_X(Real X[NDIM]) {
//             return X[1];
//         }
//         virtual double th_of_X(Real X[NDIM]) {
//             return X[2];
//         }
//         virtual double phi_of_X(Real X[NDIM]) {
//             return X[3];
//         }
//         virtual void dxdX_to_native(Real X[NDIM], Real dxdX[NDIM][NDIM]) {
//             DLOOP1 dxdX[mu][mu] = 1;
//         }
//         virtual void dxdX_to_embed(Real X[NDIM], Real dxdX[NDIM][NDIM]) {
//             DLOOP1 dxdX[mu][mu] = 1;
//         }
// };