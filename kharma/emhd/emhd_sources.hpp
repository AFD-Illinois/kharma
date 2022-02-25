
#pragma once

#include "decs.hpp"

/**
 * Implicit source terms for EMHD
 */
KOKKOS_INLINE_FUNCTION void emhd_implicit_sources(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                                  const Local& dU, const VarMap& m_u)
{
    Real gdet = G.gdet(loc, j, i);
    Real tau = 0. //HFSDAJKHFASDHJLASFD
    dU(m_u.Q)  = -gdet * (P(m_p.Q) / tau);
    dU(m_u.DP) = -gdet * (P(m_p.DP) / tau);
}


KOKKOS_INLINE_FUNCTION void emhd_time_derivative_sources(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                                         const Local& dU, const VarMap& m_u)
{

    // Initializations
    double rho      = P(m_p.RHO);
    double Theta    = S->Theta[k][j][i];
    double bsq      = S->bsq[k][j][i];
    double chi_emhd = S->chi_emhd[k][j][i];
    double nu_emhd  = S->nu_emhd[k][j][i];
    double tau      = S->tau[k][j][i];

    double gdet = G->gdet[loc][j][i];

    // Compute partial derivative of ucov
    double dt_ucov[GR_DIM];
    DLOOP1 {
        double ucov_new = S_new->ucov[mu][k][j][i];
        double ucov_old = S_old->ucov[mu][k][j][i];

        dt_ucov[mu] = (ucov_new - ucov_old) / dt;
    }

    // Compute div of ucon (only temporal part is nonzero)
    double div_ucon = 0;
    DLOOP1 {
        double gcon_t_mu = G->gcon[loc][0][mu][j][i];

        div_ucon += gcon_t_mu * dt_ucov[mu];
    }

    // Compute q0 and delta_P0 (temporal terms)
    double Theta_new, Theta_old, dt_Theta;
    Theta_new = S_new->Theta[k][j][i];
    Theta_old = S_old->Theta[k][j][i];

    dt_Theta = (Theta_new - Theta_old) / dt;

    double q0, deltaP0;
    double bcon_t  = S->bcon[0][k][j][i];

    q0 = -rho * chi_emhd * (bcon_t / sqrt(bsq)) * dt_Theta;
    DLOOP1 {
        double ucon_t  = S->ucon[0][k][j][i];
        double bcon_mu = S->bcon[mu][k][j][i];

        q0 -= rho * chi_emhd * (bcon_mu / sqrt(bsq)) * Theta * ucon_t * dt_ucov[mu];
    }

    deltaP0 = -rho * nu_emhd * div_ucon;
    DLOOP1 {
        double bcon_mu = S->bcon[mu][k][j][i];

        deltaP0 += 3. * rho * nu_emhd * (bcon_t * bcon_mu / bsq) * dt_ucov[mu];
    }

    // Add the time derivative source terms (conduction and viscosity)
    // NOTE: Will have to edit this when higher order terms are considered
    dU(Q)  += gdet * (q0 / tau);
    dU(DP) += gdet * (deltaP0 / tau);
}

// Compute explicit source terms
KOKKOS_INLINE_FUNCTION void emhd_explicit_sources(struct GridGeom *G, struct FluidState *S, int loc,
                                                  int i, int j, int k, double dU_explicit)
{
    // Extended MHD components

    // Initializations

    double rho      = S->P[RHO][k][j][i];
    double Theta    = S->Theta[k][j][i];
    double bsq      = S->bsq[k][j][i];
    double chi_emhd = S->chi_emhd[k][j][i];
    double nu_emhd  = S->nu_emhd[k][j][i];
    double tau      = S->tau[k][j][i];

    double gdet = G->gdet[loc][j][i];

    double grad_ucov[GR_DIM][GR_DIM], grad_Theta[GR_DIM];

    // Compute gradient of ucov and Theta
    gradient_calc(G, S, loc, i, j, k, grad_ucov, grad_Theta);

    // Compute div of ucon (all terms but the time-derivative ones are nonzero)
    double div_ucon = 0;
    DLOOP2 {
        double gcon_mu_nu = G->gcon[loc][mu][nu][j][i];

        div_ucon += gcon_mu_nu * grad_ucov[mu][nu];
    }

    // Compute q0 and deltaP0 (everything but the time-derivative terms)
    double q0, deltaP0;

    DLOOP1 {
        double bcon_mu = S->bcon[mu][k][j][i];

        q0 = -rho * chi_emhd * (bcon_mu / sqrt(bsq)) * grad_Theta[mu];
    }

    DLOOP2 {
        double bcon_mu = S->bcon[mu][k][j][i];
        double ucon_nu = S->ucon[nu][k][j][i];

        q0 -= rho * chi_emhd * (bcon_mu / sqrt(bsq)) * Theta * ucon_nu * grad_ucov[nu][mu];
    }

    deltaP0 = -rho * nu_emhd * div_ucon;
    DLOOP2  {
        double bcon_mu = S->bcon[mu][k][j][i];
        double bcon_nu = S->bcon[nu][k][j][i];

        deltaP0 += 3. * rho * nu_emhd * (bcon_mu * bcon_nu / bsq) * grad_ucov[mu][nu];
    }

    // Add explicit source terms (conduction and viscosity)
    // NOTE: Will have to edit this when higher order terms are considered
    dU(Q)  += gdet * (q0 / tau);
    dU(DP) += gdet * (deltaP0) / tau;
}

// Compute gradient of four velocities and temperature
// Called by emhd_explicit_sources
KOKKOS_INLINE_FUNCTION void gradient_calc(struct GridGeom *G, struct FluidState *S, int loc, int i, int j, int k,
                                          double grad_ucov[GR_DIM][GR_DIM], double grad_Theta[GR_DIM])
{
    // Compute gradient of ucov
    DLOOP1 {
        grad_ucov[0][mu] = 0;

        slope_calc_4vec(S->ucov, mu, 1, i, j, k, grad_ucov[1][mu]);
        slope_calc_4vec(S->ucov, mu, 2, i, j, k, grad_ucov[2][mu]);
        slope_calc_4vec(S->ucov, mu, 3, i, j, k, grad_ucov[3][mu]);
    }

    DLOOP2 {
        for (int gam = 0; gam < GR_DIM; gam++)
            grad_ucov[mu][nu] -= G->conn[gam][mu][nu][j][i] * S->ucov[gam][k][j][i];
    }

    // Compute temperature gradient
    // Time derivative component computed in emhd_time_derivative_sources
    grad_Theta[0] = 0;
    slope_calc_scalar(S->Theta, 1, i, j, k, grad_Theta[1]);
    slope_calc_scalar(S->Theta, 2, i, j, k, grad_Theta[2]);
    slope_calc_scalar(S->Theta, 3, i, j, k, grad_Theta[3]);
}

// Compute slope for 4 vectors
// TODO going to need to either keep or calculate these based on recon choices
KOKKOS_INLINE_FUNCTION void slope_calc_4vec(GridVector u, int component, int dir, int i, int j, int k, double slope)
{
    if (dir == 1)
        slope = SLOPE_ALGO(u[component][k][j][i-2], u[component][k][j][i-1], u[component][k][j][i],
                            u[component][k][j][i+1], u[component][k][j][i+2], dx[dir]);
    if (dir == 2)
        slope = SLOPE_ALGO(u[component][k][j-2][i], u[component][k][j-1][i], u[component][k][j][i],
                            u[component][k][j+1][i], u[component][k][j+2][i], dx[dir]);
    if (dir == 3)
        slope = SLOPE_ALGO(u[component][k-2][j][i], u[component][k-1][j][i], u[component][k][j][i],
                            u[component][k+1][j][i], u[component][k+2][j][i], dx[dir]);
}

// Compute slope for scalars
KOKKOS_INLINE_FUNCTION void slope_calc_scalar(GridDouble T, int dir, int i, int j, int k, double slope)
{
  if (dir == 1) slope = SLOPE_ALGO(T[k][j][i-2], T[k][j][i-1], T[k][j][i], T[k][j][i+1], T[k][j][i+2], dx[dir]);
  if (dir == 2) slope = SLOPE_ALGO(T[k][j-2][i], T[k][j-1][i], T[k][j][i], T[k][j+1][i], T[k][j+2][i], dx[dir]);
  if (dir == 3) slope = SLOPE_ALGO(T[k-2][j][i], T[k-1][j][i], T[k][j][i], T[k+1][j][i], T[k+2][j][i], dx[dir]);
}