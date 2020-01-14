/******************************************************************************
 *                                                                            *
 * RECONSTRUCTION.C                                                           *
 *                                                                            *
 * RECONSTRUCTION ALGORITHMS                                                  *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

// Play some pre-processor games
#if RECONSTRUCTION == LINEAR
// MC is the only limiter used
#define RECON_ALGO linear_mc
#elif RECONSTRUCTION == PPM
#error "PPM currently broken!"
#elif RECONSTRUCTION == WENO
#define RECON_ALGO weno
#elif RECONSTRUCTION == MP5
#define RECON_ALGO mp5
#else
#error "Reconstruction not specified!"
#endif

  // Sanity checks
#if (RECONSTRUCTION == PPM || RECONSTRUCTION == WENO || RECONSTRUCTION == MP5) && NG < 3
#error "not enough ghost zones! PPM/WENO/MP5 + NG < 3\n"
#endif

void linear_mc(double unused1, double x1, double x2, double x3, double unused2, double *lout, double *rout);
void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);
void weno(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);

double median(double a, double b, double c);
double mp5_subcalc(double Fjm2, double Fjm1, double Fj, double Fjp1, double Fjp2);
void mp5(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);

inline void linear_mc(double unused1, double x1, double x2, double x3, double unused2, double *lout, double *rout)
{
  double Dqm,Dqp,Dqc,s;

  Dqm = 2. * (x2 - x1);
  Dqp = 2. * (x3 - x2);
  Dqc = 0.5 * (x3 - x1);

  s = Dqm * Dqp;

  if (s <= 0.)
    s = 0.;
  else {
    if (fabs(Dqm) < fabs(Dqp) && fabs(Dqm) < fabs(Dqc))
      s = Dqm;
    else if (fabs(Dqp) < fabs(Dqc))
      s = Dqp;
    else
      s = Dqc;
  }

  // Reconstruct left, right
  *lout = x2 - 0.5*s;
  *rout = x2 + 0.5*s;
}

// Parabolic interpolation (see Colella & Woodward 1984; CW)
// Implemented by Xiaoyue Guan
inline void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout)
{
  double y[5], dq[5];
  double Dqm, Dqc, Dqp, aDqm,aDqp,aDqc,s,l,r,qa, qd, qe;

  y[0] = x1;
  y[1] = x2;
  y[2] = x3;
  y[3] = x4;
  y[4] = x5;

  // CW 1.7
  for(int i = 1; i < 4; i++) {
    Dqm = 2. *(y[i  ] - y[i-1]);
    Dqp = 2. *(y[i+1] - y[i  ]);
    Dqc = 0.5*(y[i+1] - y[i-1]);
    aDqm = fabs(Dqm) ;
    aDqp = fabs(Dqp) ;
    aDqc = fabs(Dqc) ;
    s = Dqm*Dqp;

    // CW 1.8
    if (s <= 0.) {
      dq[i] = 0.;
    } else {
      dq[i] = MY_MIN(aDqc,MY_MIN(aDqm,aDqp))*MY_SIGN(Dqc);
    }
  }

  // CW 1.6
  l = 0.5*(y[2] + y[1]) - (dq[2] - dq[1])/6.0;
  r = 0.5*(y[3] + y[2]) - (dq[3] - dq[2])/6.0;

  qa = (r - y[2])*(y[2] - l);
  qd = (r - l);
  qe = 6.0*(y[2] - 0.5*(l + r));

  if (qa <= 0.) {
    l = y[2];
    r = y[2];
  }

  if (qd*(qd - qe) < 0.0) {
    l = 3.0*y[2] - 2.0*r;
  } else if (qd*(qd + qe) < 0.0) {
    r = 3.0*y[2] - 2.0*l;
  }

  *lout = l;
  *rout = r;
}

// WENO interpolation. See Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)
// Implemented by Monika Moscibrodzka
inline void weno(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout)
{
  // S11 1, 2, 3
  double vr[3], vl[3];
  vr[0] =  (3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3;
  vr[1] = (-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4;
  vr[2] =  (3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5;

  vl[0] =  (3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3;
  vl[1] = (-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2;
  vl[2] =  (3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1;

  // Smoothness indicators, T07 A18 or S11 8
  double beta[3];
  beta[0] = (13./12.)*pow(x1 - 2.*x2 + x3, 2) +
            (1./4.)*pow(x1 - 4.*x2 + 3.*x3, 2);
  beta[1] = (13./12.)*pow(x2 - 2.*x3 + x4, 2) +
            (1./4.)*pow(x4 - x2, 2);
  beta[2] = (13./12.)*pow(x3 - 2.*x4 + x5, 2) +
            (1./4.)*pow(x5 - 4.*x4 + 3.*x3, 2);

  // Nonlinear weights S11 9
  double den, wtr[3], Wr, wr[3], wtl[3], Wl, wl[3], eps;
  eps=1.e-26;

  den = eps + beta[0]; den *= den; wtr[0] = (1./16.)/den;
  den = eps + beta[1]; den *= den; wtr[1] = (5./8. )/den;
  den = eps + beta[2]; den *= den; wtr[2] = (5./16.)/den;
  Wr = wtr[0] + wtr[1] + wtr[2];
  wr[0] = wtr[0]/Wr ;
  wr[1] = wtr[1]/Wr ;
  wr[2] = wtr[2]/Wr ;

  den = eps + beta[2]; den *= den; wtl[0] = (1./16.)/den;
  den = eps + beta[1]; den *= den; wtl[1] = (5./8. )/den;
  den = eps + beta[0]; den *= den; wtl[2] = (5./16.)/den;
  Wl = wtl[0] + wtl[1] + wtl[2];
  wl[0] = wtl[0]/Wl;
  wl[1] = wtl[1]/Wl;
  wl[2] = wtl[2]/Wl;

  *lout = vl[0]*wl[0] + vl[1]*wl[1] + vl[2]*wl[2];
  *rout = vr[0]*wr[0] + vr[1]*wr[1] + vr[2]*wr[2];
}

// MP5 reconstruction from PLUTO
// Imported by Mani Chandra
#define MINMOD(a, b) ((a)*(b) > 0.0 ? (fabs(a) < fabs(b) ? (a):(b)):0.0)
inline double median(double a, double b, double c)
{
  return (a + MINMOD(b - a, c - a));
}
#define ALPHA (4.0)
#define EPSM (1.e-12)
inline double mp5_subcalc(double Fjm2, double Fjm1, double Fj, double Fjp1, double Fjp2)
{
  double f, d2, d2p, d2m;
  double dMMm, dMMp;
  double scrh1,scrh2, Fmin, Fmax;
  double fAV, fMD, fLC, fUL, fMP;

  f  = 2.0*Fjm2 - 13.0*Fjm1 + 47.0*Fj + 27.0*Fjp1 - 3.0*Fjp2;
  f /= 60.0;

  fMP = Fj + MINMOD(Fjp1 - Fj, ALPHA*(Fj - Fjm1));

  if ((f - Fj)*(f - fMP) <= EPSM)
    return f;

  d2m = Fjm2 + Fj   - 2.0*Fjm1;              // Eqn. 2.19
  d2  = Fjm1 + Fjp1 - 2.0*Fj;
  d2p = Fj   + Fjp2 - 2.0*Fjp1;              // Eqn. 2.19

  scrh1 = MINMOD(4.0*d2 - d2p, 4.0*d2p - d2);
  scrh2 = MINMOD(d2, d2p);
  dMMp  = MINMOD(scrh1,scrh2);               // Eqn. 2.27

  scrh1 = MINMOD(4.0*d2m - d2, 4.0*d2 - d2m);
  scrh2 = MINMOD(d2, d2m);
  dMMm  = MINMOD(scrh1,scrh2);               // Eqn. 2.27

  fUL = Fj + ALPHA*(Fj - Fjm1);              // Eqn. 2.8
  fAV = 0.5*(Fj + Fjp1);                     // Eqn. 2.16
  fMD = fAV - 0.5*dMMp;                      // Eqn. 2.28
  fLC = 0.5*(3.0*Fj - Fjm1) + 4.0/3.0*dMMm;  // Eqn. 2.29

  scrh1 = fmin(Fj, Fjp1); scrh1 = fmin(scrh1, fMD);
  scrh2 = fmin(Fj, fUL);    scrh2 = fmin(scrh2, fLC);
  Fmin  = fmax(scrh1, scrh2);                // Eqn. (2.24a)

  scrh1 = fmax(Fj, Fjp1); scrh1 = fmax(scrh1, fMD);
  scrh2 = fmax(Fj, fUL);    scrh2 = fmax(scrh2, fLC);
  Fmax  = fmin(scrh1, scrh2);                // Eqn. 2.24b

  f = median(f, Fmin, Fmax);                 // Eqn. 2.26
  return f;
}

inline void mp5(double x1, double x2, double x3, double x4, double x5, double *lout,
  double *rout)
{
  *rout = mp5_subcalc(x1, x2, x3, x4, x5);
  *lout = mp5_subcalc(x5, x4, x3, x2, x1);
}
#undef MINMOD

// Use the pre-processor for poor man's multiple dispatch
void reconstruct(struct FluidState *S, GridPrim Pl, GridPrim Pr, int dir)
{
  timer_start(TIMER_RECON);
  if (dir == 1) {
#pragma omp parallel for collapse(3)
    PLOOP {
      KSLOOP(-1, N3) {
        JSLOOP(-1, N2) {
          ISLOOP(-1, N1) {
            RECON_ALGO(S->P[ip][k][j][i-2], S->P[ip][k][j][i-1], S->P[ip][k][j][i],
                 S->P[ip][k][j][i+1], S->P[ip][k][j][i+2], &(Pl[ip][k][j][i]),
                 &(Pr[ip][k][j][i]));
          }
        }
      }
    }
  } else if (dir == 2) {
#pragma omp parallel for collapse(3)
    PLOOP {
      KSLOOP(-1, N3) {
        JSLOOP(-1, N2) {
          ISLOOP(-1, N1) {
            RECON_ALGO(S->P[ip][k][j-2][i], S->P[ip][k][j-1][i], S->P[ip][k][j][i],
                 S->P[ip][k][j+1][i], S->P[ip][k][j+2][i], &(Pl[ip][k][j][i]),
                 &(Pr[ip][k][j][i]));
          }
        }
      }
    }
  } else if (dir == 3) {
#pragma omp parallel for collapse(3)
    PLOOP {
      KSLOOP(-1, N3) {
        JSLOOP(-1, N2) {
          ISLOOP(-1, N1) {
            RECON_ALGO(S->P[ip][k-2][j][i], S->P[ip][k-1][j][i], S->P[ip][k][j][i],
                 S->P[ip][k+1][j][i], S->P[ip][k+2][j][i], &(Pl[ip][k][j][i]),
                 &(Pr[ip][k][j][i]));
          }
        }
      }
    }
  }
  timer_stop(TIMER_RECON);
}

