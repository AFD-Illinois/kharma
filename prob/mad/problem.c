/******************************************************************************
 *                                                                            *
 * PROBLEM.C                                                                  *
 *                                                                            *
 * INITIAL CONDITIONS FOR FISHBONE-MONCRIEF TORUS                             *
 *                                                                            *
 ******************************************************************************/

#include "bl_coord.h"
#include "decs.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

gsl_rng *rng;

// Local declarations
double lfish_calc(double rmax);

static int MAD, NORM_WITH_MAXR = 0; //TODO clean up these parameters
static double BHflux, beta;
static double rin, rmax;
static double rBstart, rBend;
void set_problem_params() {
  set_param("rin", &rin);
  set_param("rmax", &rmax);

  set_param("MAD", &MAD);
  set_param("BHflux", &BHflux);
  set_param("beta", &beta);

  set_param("rBstart", &rBstart);
  set_param("rBend", &rBend);
}

void init(struct GridGeom *G, struct FluidState *S)
{
  // Magnetic field
  double (*A)[N2 + 2*NG] = malloc(sizeof(*A) * (N1 + 2*NG));

  // Initialize RNG
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, mpi_myrank());

  // Fishbone-Moncrief parameters
  double l = lfish_calc(rmax);
  double kappa = 1.e-3;

  // Grid parameters
  R0 = 0.0;
  Rhor = (1. + sqrt(1. - a*a));
  //double z1 = 1 + pow(1 - a*a,1./3.)*(pow(1+a,1./3.) + pow(1-a,1./3.));
  //double z2 = sqrt(3*a*a + z1*z1);
  //Risco = 3 + z2 - sqrt((3-z1)*(3 + z1 + 2*z2));

  set_grid(G);
  LOG("Grid set");

  double rhomax = 0.;
  double umax = 0.;
  ZSLOOP(-1, N3, -1, N2, -1, N1) {
    double X[NDIM];
    coord(i, j, k, CENT, X);
    double r, th;
    bl_coord(X, &r, &th);

    double sth = sin(th);
    double cth = cos(th);

    // Calculate lnh
    double DD = r * r - 2. * r + a * a;
    double AA = (r * r + a * a) * (r * r + a * a) -
             DD * a * a * sth * sth;
    double SS = r * r + a * a * cth * cth;

    double thin = M_PI / 2.;
    double sthin = sin(thin);
    double cthin = cos(thin);

    double DDin = rin * rin - 2. * rin + a * a;
    double AAin = (rin * rin + a * a) * (rin * rin + a * a)
             - DDin * a * a * sthin * sthin;
    double SSin = rin * rin + a * a * cthin * cthin;

    double lnh;
    if (r >= rin) {
      lnh =
          0.5 *
          log((1. +
         sqrt(1. +
              4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth)))
        / (SS * DD / AA))
          - 0.5 * sqrt(1. +
           4. * (l * l * SS * SS) * DD /
           (AA * AA * sth * sth))
          - 2. * a * r * l / AA -
          (0.5 *
           log((1. +
          sqrt(1. +
               4. * (l * l * SSin * SSin) * DDin /
               (AAin * AAin * sthin * sthin))) /
         (SSin * DDin / AAin))
           - 0.5 * sqrt(1. +
            4. * (l * l * SSin * SSin) * DDin / (AAin * AAin * sthin * sthin))
           - 2. * a * rin * l / AAin);
    } else {
      lnh = 1.;
    }

    // regions outside torus
    if (lnh < 0. || r < rin) {
      // Nominal values; real value set by fixup

      S->P[RHO][k][j][i] = 1.e-7 * RHOMIN;
      S->P[UU][k][j][i] = 1.e-7 * UUMIN;
      S->P[U1][k][j][i] = 0.;
      S->P[U2][k][j][i] = 0.;
      S->P[U3][k][j][i] = 0.;
    }
    /* region inside magnetized torus; u^i is calculated in
     * Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
     * so it needs to be transformed at the end */
    else {
      double hm1 = exp(lnh) - 1.;
      double rho = pow(hm1 * (gam - 1.) / (kappa * gam),
               1. / (gam - 1.));
      double u = kappa * pow(rho, gam) / (gam - 1.);

      // Calculate u^phi
      double expm2chi = SS * SS * DD / (AA * AA * sth * sth);
      double up1 =
          sqrt((-1. +
          sqrt(1. + 4. * l * l * expm2chi)) / 2.);
      double up = 2. * a * r * sqrt(1. +
                 up1 * up1) / sqrt(AA * SS *
                 DD) +
          sqrt(SS / AA) * up1 / sth;


      S->P[RHO][k][j][i] = rho;
      if (rho > rhomax) rhomax = rho;
      u *= (1. + 4.e-2 * (gsl_rng_uniform(rng) - 0.5));
      S->P[UU][k][j][i] = u;
      if (u > umax && r > rin) umax = u;
      S->P[U1][k][j][i] = 0.;
      S->P[U2][k][j][i] = 0.;
      S->P[U3][k][j][i] = up;

      // Convert from 4-velocity to 3-velocity
      coord_transform(G, S, i, j, k);
    }

    S->P[B1][k][j][i] = 0.;
    S->P[B2][k][j][i] = 0.;
    S->P[B3][k][j][i] = 0.;
  } // ZSLOOP

  // Find the zone in which rBend of Narayan condition resides
  // This just uses the farthest process in R
  // For /very/ large N1CPU it might fail
  int iend_global = 0;
  if (global_stop[0] == N1TOT && global_start[1] == 0 && global_start[2] == 0) {
    int iend = NG;
    double r_iend = 0.0;
    while (r_iend < rBend) {
      iend++;
      double Xend[NDIM];
      coord(iend,N2/2+NG,NG,CORN,Xend);
      double thend;
      bl_coord(Xend,&r_iend,&thend);
    }
    iend--;
    iend_global = global_start[0] + iend - NG; //Translate to coordinates for uu_mid below
    if(DEBUG) printf("[MPI %d] Furthest torus zone is %d (locally %d), at r = %f\n", mpi_myrank(), iend_global, iend, r_iend);
  }
  iend_global = mpi_reduce_int(iend_global); //TODO This should be a broadcast I know.

  // Normalize the densities so that max(rho) = 1
  umax = mpi_max(umax);
  rhomax = mpi_max(rhomax);

  //ZSLOOP(-1, N3, -1, N2, -1, N1) {
  ZLOOPALL {
    S->P[RHO][k][j][i] /= rhomax;
    S->P[UU][k][j][i] /= rhomax;
  }
  umax /= rhomax;
  rhomax = 1.;
  fixup(G, S);
  set_bounds(G, S);

  // Calculate UU along midplane, propagate to all processes
  double *uu_plane_send = calloc(N1TOT,sizeof(double));

  // This relies on an even N2TOT /and/ N2CPU
  if ((global_start[1] == N2TOT/2 || N2CPU == 1) && global_start[2] == 0) {
    int j_mid = N2TOT/2 - global_start[1] + NG;
    int k = NG; // Axisymmetric
    ILOOP {
      int i_global = global_start[0] + i - NG;
      uu_plane_send[i_global] = 0.25*(S->P[UU][k][j_mid][i] + S->P[UU][k][j_mid][i-1] +
                          S->P[UU][k][j_mid-1][i] + S->P[UU][k][j_mid-1][i-1]);
    }
  }

  double *uu_plane = calloc(N1TOT,sizeof(double));
  mpi_reduce_vector(uu_plane_send, uu_plane, N1TOT);
  free(uu_plane_send);
  //printf ("UU in plane is "); for (int i =0; i < N1TOT; i++) printf("%.10e ", uu_plane[i]);

  // first find corner-centered vector potential
  ZSLOOP(0, 0, -NG, N2+NG-1, -NG, N1+NG-1) A[i][j] = 0.;
  ZSLOOP(0, 0, -NG+1, N2+NG-1, -NG+1, N1+NG-1) {
    double X[NDIM];
    coord(i,j,k,CORN,X);
    double r, th;
    bl_coord(X,&r,&th);

    double q;

    // Field in disk
    double rho_av = 0.25*(S->P[RHO][k][j][i] + S->P[RHO][k][j][i-1] +
                    S->P[RHO][k][j-1][i] + S->P[RHO][k][j-1][i-1]);
    double uu_av = 0.25*(S->P[UU][k][j][i] + S->P[UU][k][j][i-1] +
                         S->P[UU][k][j-1][i] + S->P[UU][k][j-1][i-1]);

    int i_global = global_start[0] + i - NG;
    double uu_plane_av = uu_plane[i_global];
    double uu_end = uu_plane[iend_global];

    double b_buffer = 0.2; //Minimum rho at which there will be B field
    if (N3 > 1) {
      if (MAD == 0) { // SANE
        q = rho_av/rhomax;
      } else if (MAD == 1) { // BR's smoothed poloidal in-torus
        q = pow(sin(th),3)*pow(r/rin,3.)*exp(-r/400)*rho_av/rhomax;

      } else if (MAD == 2) { // Just the r^3 sin^3 term, possible MAD EHT standard
        q = pow(sin(th),3)*pow(r/rin,3.)*rho_av/rhomax;

      } else if (MAD == 3) { // Gaussian-strength vertical threaded field
        double wid = 2; //Radius of half-maximum. Units of rin
        q = gsl_ran_gaussian_pdf((r/rin)*sin(th), wid/sqrt(2*log(2)));

      } else if (MAD == 4) { // Narayan '12, Penna '12 conditions
        // Former uses rstart=25, rend=810, lam_B=25
        double uc = uu_av - uu_end;
        double ucm = uu_plane_av - uu_end;
        b_buffer = 0; // Note builtin buffer below
        q = pow(sin(th),3)*(uc/(ucm+SMALL) - 0.2) / 0.8;
        //Exclude q outside torus and large q resulting from division by SMALL
        if ( r > rBend || r < rBstart || fabs(q) > 1.e2 ) q = 0;
        NORM_WITH_MAXR = 0; //?

        //if (q != 0 && th > M_PI/2-0.1 && th < M_PI/2+0.1) printf("q_mid is %.10e\n", q);
      } else {
        printf("MAD = %i not supported!\n", MAD);
        exit(-1);
      }
    } else { // TODO How about 2D?
      q = rho_av/rhomax;
    }

    // Apply floor
    q -= b_buffer;

    A[i][j] = 0.;
    if (q > 0.) {
      if (MAD == 7) { // Narayan limit for MAD
        double lam_B = 25;
        double flux_correction = sin( 1/lam_B * (pow(r,2./3) + 15./8*pow(r,-2./5) - pow(rBstart,2./3) - 15./8*pow(rBstart,-2./5)));
        double q_mod = q*flux_correction;
        A[i][j] = q_mod;
      } else {
        A[i][j] = q;
      }
    }
  } // ZSLOOP

  // Calculate B-field and find bsq_max
  double bsq_max = 0.;
  double beta_min = 1e100;
  ZLOOP {
    double X[NDIM];
    coord(i,j,k,CORN,X);
    double r, th;
    bl_coord(X,&r,&th);

    // Flux-ct
    S->P[B1][k][j][i] = -(A[i][j] - A[i][j + 1]
	+ A[i + 1][j] - A[i + 1][j + 1]) /
	(2. * dx[2] * G->gdet[CENT][j][i]);
    S->P[B2][k][j][i] = (A[i][j] + A[i][j + 1]
	     - A[i + 1][j] - A[i + 1][j + 1]) /
	     (2. * dx[1] * G->gdet[CENT][j][i]);

    S->P[B3][k][j][i] = 0.;

    get_state(G, S, i, j, k, CENT);
    if ((r > rBstart && r < rBend) || MAD != 7) {
      double bsq_ij = bsq_calc(S, i, j, k);
      if (bsq_ij > bsq_max) bsq_max = bsq_ij;
      double beta_ij = (gam - 1.)*(S->P[UU][k][j][i])/(0.5*(bsq_ij+SMALL)) ;
      if(beta_ij < beta_min) beta_min = beta_ij ;
    }
  }
  bsq_max = mpi_max(bsq_max);
  beta_min = mpi_min(beta_min);

  double umax_plane = 0;
  for (int i = 0; i < N1TOT; i++) {
      double X[NDIM];
      coord(i,NG,NG,CORN,X);
      double r, th;
      bl_coord(X,&r,&th);
      if ((r > rBstart && r < rBend) || MAD != 7) {
	if (uu_plane[i] > umax_plane) umax_plane = uu_plane[i];
      }
  }

  double norm = 0;
  if (!NORM_WITH_MAXR) {
    // Ratio of max UU, beta
    double beta_act = (gam - 1.) * umax / (0.5 * bsq_max);
    // In plane only
    //double beta_act = (gam - 1.) * umax_plane / (0.5 * bsq_max);
    LOGN("Umax is %.10e", umax);
    LOGN("bsq_max is %.10e", bsq_max);
    LOGN("beta is %.10e", beta_act);
    norm = sqrt(beta_act / beta);
  } else {
    // Beta_min = 100 normalization
    LOGN("Min beta in torus is %f", beta_min);
    norm = sqrt(beta_min / beta) ;
  }

  // Apply normalization
  LOGN("Normalization is %f\n", norm);
  ZLOOP {
    S->P[B1][k][j][i] *= norm ;
    S->P[B2][k][j][i] *= norm ;
  }

  // This adds a central flux based on specifying some BHflux
  // Initialize a net magnetic field inside the initial torus
  ZSLOOP(0, 0, 0, N2, 0, N1) {
    double X[NDIM];
    coord(i,j,k,CORN,X);
    double r,th;
    bl_coord(X, &r, &th);

    A[i][j] = 0.;

    double x = r*sin(th);
    double z = r*cos(th);
    double a_hyp = 20.;
    double b_hyp = 60.;
    double x_hyp = a_hyp*sqrt(1. + pow(z/b_hyp,2));

    double q = (pow(x,2) - pow(x_hyp,2))/pow(x_hyp,2);
    if (x < x_hyp) {
      A[i][j] = 10.*q;
    }
  }

  // Evaluate net flux
  double Phi_proc = 0.;
  ISLOOP(5, N1-1) {
    JSLOOP(0, N2-1) {
      int jglobal = j - NG + global_start[1];
      //int j = N2/2+NG;
      int k = NG;
      if (jglobal == N2TOT / 2) {
        double X[NDIM];
        coord(i, j, k, CENT, X);
        double r, th;
        bl_coord(X, &r, &th);

        if (r < rin) {
          double B2net = (A[i][j] + A[i][j + 1] - A[i + 1][j] - A[i + 1][j + 1]);
          // / (2.*dx[1]*G->gdet[CENT][j][i]);
          Phi_proc += fabs(B2net) * M_PI / N3CPU; // * 2.*dx[1]*G->gdet[CENT][j][i]
        }
      }
    }
  }

  //If left bound in X1.  Note different convention from bhlight!
  if (global_start[0] == 0) {
    JSLOOP(0, N2/2-1) {
      int i = 5 + NG;

      double B1net = -(A[i][j] - A[i][j+1] + A[i+1][j] - A[i+1][j+1]); // /(2.*dx[2]*G->gdet[CENT][j][i]);
      Phi_proc += fabs(B1net)*M_PI/N3CPU;  // * 2.*dx[2]*G->gdet[CENT][j][i]
    }
  }
  double Phi = mpi_reduce(Phi_proc);

  norm = BHflux/(Phi + SMALL);

  ZLOOP {
    // Flux-ct
    S->P[B1][k][j][i] += -norm
        * (A[i][j] - A[i][j + 1] + A[i + 1][j] - A[i + 1][j + 1])
        / (2. * dx[2] * G->gdet[CENT][j][i]);
    S->P[B2][k][j][i] += norm
        * (A[i][j] + A[i][j + 1] - A[i + 1][j] - A[i + 1][j + 1])
        / (2. * dx[1] * G->gdet[CENT][j][i]);
  }

#if ELECTRONS
  init_electrons(G,S);
#endif

  // Enforce boundary conditions
  fixup(G, S);
  set_bounds(G, S);

  LOG("Finished init()");

}

double lfish_calc(double r)
{
  return (((pow(a, 2) - 2. * a * sqrt(r) + pow(r, 2)) *
     ((-2. * a * r *
       (pow(a, 2) - 2. * a * sqrt(r) +
        pow(r,
      2))) / sqrt(2. * a * sqrt(r) + (-3. + r) * r) +
      ((a + (-2. + r) * sqrt(r)) * (pow(r, 3) + pow(a, 2) *
      (2. + r))) / sqrt(1 + (2. * a) / pow (r, 1.5) - 3. / r)))
    / (pow(r, 3) * sqrt(2. * a * sqrt(r) + (-3. + r) * r) *
       (pow(a, 2) + (-2. + r) * r))
      );
}

