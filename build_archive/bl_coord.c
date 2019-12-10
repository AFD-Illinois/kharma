/*
 * Utility functions for Boyer-Lindquist coordinates
 * Provided for problem setups but not otherwise used in core functions
 */

// TODO cleanup/minimize this file, it duplicates some of coord.c

#include "bl_coord.h"

void blgset(int i, int j, struct of_geom *geom)
{
  double r, th, X[NDIM];

  coord(i, j, 0, CENT, X);
  bl_coord(X, &r, &th);

  if (th < 0)
    th *= -1.;
  if (th > M_PI)
    th = 2. * M_PI - th;

  geom->g = bl_gdet_func(r, th);
  bl_gcov_func(r, th, geom->gcov);
  bl_gcon_func(r, th, geom->gcon);
}

double bl_gdet_func(double r, double th)
{
  double a2, r2;

  a2 = a * a;
  r2 = r * r;
  return (r * r * fabs(sin(th)) *
    (1. + 0.5 * (a2 / r2) * (1. + cos(2. * th))));
}

void bl_gcov_func(double r, double th, double gcov[NDIM][NDIM])
{
  DLOOP2 gcov[mu][nu] = 0.;

  double sth, cth, s2, a2, r2, DD, mu;
  sth = fabs(sin(th));
  s2 = sth*sth;
  cth = cos(th);
  a2 = a*a;
  r2 = r*r;
  DD = 1. - 2./r + a2/r2;
  mu = 1. + a2*cth*cth/r2;

  gcov[0][0] = -(1. - 2./(r*mu));
  gcov[0][3]  = -2.*a*s2/(r*mu);
  gcov[3][0]  = gcov[0][3];
  gcov[1][1]   = mu/DD;
  gcov[2][2]   = r2*mu;
  gcov[3][3]   = r2*sth*sth*(1. + a2/r2 + 2.*a2*s2/(r2*r*mu));

}

void bl_gcon_func(double r, double th, double gcon[NDIM][NDIM])
{
  double sth, cth, a2, r2, r3, DD, mu;

  DLOOP2 gcon[mu][nu] = 0.;

  sth = sin(th);
  cth = cos(th);

#if(COORDSINGFIX)
  if (fabs(sth) < SINGSMALL) {
    if (sth >= 0)
      sth = SINGSMALL;
    if (sth < 0)
      sth = -SINGSMALL;
  }
#endif

  a2 = a*a;
  r2 = r*r;
  r3 = r2*r;
  DD = 1. - 2./r + a2/r2;
  mu = 1. + a2*cth*cth/r2;

  gcon[0][0] = -1. - 2.*(1. + a2/r2)/(r*DD*mu);
  gcon[0][3] = -2.*a/(r3*DD*mu);
  gcon[3][0] = gcon[0][3];
  gcon[1][1] = DD/mu;
  gcon[2][2] = 1./(r2*mu);
  gcon[3][3] = (1. - 2./(r*mu))/(r2*sth*sth*DD);
}

void bl_to_ks(double X[NDIM], double ucon_bl[NDIM], double ucon_ks[NDIM])
{
  double r, th;
  bl_coord(X, &r, &th);

  double trans[NDIM][NDIM];
  DLOOP2 trans[mu][nu] = 0.;
  DLOOP1 trans[mu][mu] = 1.;
  trans[0][1] = 2.*r/(r*r - 2.*r + a*a);
  trans[3][1] = a/(r*r - 2.*r + a*a);

  DLOOP1 ucon_ks[mu] = 0.;
  DLOOP2 ucon_ks[mu] += trans[mu][nu]*ucon_bl[nu];
}

// Convert Boyer-Lindquist four-velocity to MKS 3-velocity
void coord_transform(struct GridGeom *G, struct FluidState *S, int i, int j,
  int k)
{
  double X[NDIM], r, th, ucon[NDIM], trans[NDIM][NDIM], tmp[NDIM];
  double AA, BB, CC, discr;
  double alpha, gamma, beta[NDIM];
  struct blgeom;
  struct of_geom blgeom;

  coord(i, j, k, CENT, X);
  bl_coord(X, &r, &th);
  blgset(i, j, &blgeom);

  ucon[1] = S->P[U1][k][j][i];
  ucon[2] = S->P[U2][k][j][i];
  ucon[3] = S->P[U3][k][j][i];

  AA = blgeom.gcov[0][0];
  BB = 2.*(blgeom.gcov[0][1]*ucon[1] +
           blgeom.gcov[0][2]*ucon[2] +
           blgeom.gcov[0][3]*ucon[3]);
  CC = 1. +
      blgeom.gcov[1][1]*ucon[1]*ucon[1] +
      blgeom.gcov[2][2]*ucon[2]*ucon[2] +
      blgeom.gcov[3][3]*ucon[3]*ucon[3] +
      2.*(blgeom.gcov[1][2]*ucon[1]*ucon[2] +
          blgeom.gcov[1][3]*ucon[1]*ucon[3] +
          blgeom.gcov[2][3]*ucon[2]*ucon[3]);

  discr = BB*BB - 4.*AA*CC;
  ucon[0] = (-BB - sqrt(discr))/(2.*AA);
  // This is ucon in BL coords

  // transform to Kerr-Schild
  // Make transform matrix
  memset(trans, 0, 16*sizeof(double));
  for (int mu = 0; mu < NDIM; mu++) {
    trans[mu][mu] = 1.;
  }
  trans[0][1] = 2.*r/(r*r - 2.*r + a*a);
  trans[3][1] = a/(r*r - 2.*r + a*a);

  // Transform ucon
  for (int mu = 0; mu < NDIM; mu++) {
    tmp[mu] = 0.;
  }
  for (int mu = 0; mu < NDIM; mu++) {
    for (int nu = 0; nu < NDIM; nu++) {
      tmp[mu] += trans[mu][nu]*ucon[nu];
    }
  }
  for (int mu = 0; mu < NDIM; mu++) {
    ucon[mu] = tmp[mu];
  }

  // This is ucon in KS coords

  // Transform to MKS or MMKS
  double invtrans[NDIM][NDIM];
  set_dxdX(X, invtrans);
  invert(&invtrans[0][0], &trans[0][0]);

  DLOOP1 tmp[mu] = 0.;
  DLOOP2 {
     tmp[mu] += trans[mu][nu]*ucon[nu];
  }
  DLOOP1 ucon[mu] = tmp[mu];

  // Solve for v. Use same u^t, unchanged under KS -> KS'
  alpha = G->lapse[CENT][j][i];
  gamma = ucon[0]*alpha;

  beta[1] = alpha*alpha*G->gcon[CENT][0][1][j][i];
  beta[2] = alpha*alpha*G->gcon[CENT][0][2][j][i];
  beta[3] = alpha*alpha*G->gcon[CENT][0][3][j][i];

  S->P[U1][k][j][i] = ucon[1] + beta[1]*gamma/alpha;
  S->P[U2][k][j][i] = ucon[2] + beta[2]*gamma/alpha;
  S->P[U3][k][j][i] = ucon[3] + beta[3]*gamma/alpha;
}




