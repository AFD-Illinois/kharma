/* 
 *  File: conducting_atmosphere.cpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "spherical_accn.hpp"

TaskStatus InitializeSphericalAccn(MeshBlockData<Real> *mbd, ParameterInput *pin)
{
  
  auto pmb        = mbd->GetBlockPointer();
  GridScalar rho  = mbd->Get("prims.rho").data;
  GridScalar u    = mbd->Get("prims.u").data;
  GridVector uvec = mbd->Get("prims.uvec").data;
  GridVector B_P  = mbd->Get("prims.B").data;

  // Proxy initializations
  auto q  = rho;
  auto dP = rho;

  const bool use_emhd = pmb->packages.AllPackages().count("EMHD");
  bool conduction = false, viscosity = false;

  if (use_emhd) {
    Flag(mbd, "Initializing spherical accretion problem with EMHD");
    
    conduction = pmb->packages.Get("EMHD")->Param<bool>("conduction");
    viscosity  = pmb->packages.Get("EMHD")->Param<bool>("viscosity");

    if (conduction)
      q = mbd->Get("prims.q").data;
    if (viscosity)
      dP = mbd->Get("prims.dP").data;
  }
  else
    Flag(mbd, "Initializing spherical accretion problem");

  // Add problem specific params
  const Real rin       = pin->GetOrAddReal("spherical_accn", "rin", 6.0);
  const Real rho_init  = pin->GetOrAddReal("spherical_accn", "rho_init", 1.0);
  const Real r_bondi   = pin->GetOrAddReal("spherical_accn", "r_bondi", 50.0);
  const Real beta_init = pin->GetOrAddReal("spherical_accn", "beta_init", 1.0);

  // Obtain GRMHD params
  const auto& grmhd_pars = pmb->packages.Get("GRMHD")->AllParams();
  const Real& gam        = grmhd_pars.Get<Real>("gamma");

  // Compute internal energy density from Bondi radius
  const Real cs_init = m::sqrt(2 / r_bondi);
  const Real pg_init = (cs_init * cs_init) * (rho_init / gam);
  const Real u_init  = pg_init / (gam - 1.);

  // Add to package properties, since they continue to be needed on boundaries
  if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rho_init")))
    pmb->packages.Get("GRMHD")->AddParam<Real>("rho_init", rho_init);
  if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("beta_init")))
    pmb->packages.Get("GRMHD")->AddParam<Real>("beta_init", beta_init);
  if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("u_init")))
    pmb->packages.Get("GRMHD")->AddParam<Real>("u_init", u_init);
  
  // Bounds of the domain
  IndexDomain domain = IndexDomain::interior;
  const int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
  const int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
  const int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
  
  // Grid
  const auto& G = pmb->coords;
  const bool use_ks          = G.coords.is_ks();
  const GReal a              = G.coords.get_a();
  const SphBLCoords blcoords = SphBLCoords(a);
  const SphKSCoords kscoords = SphKSCoords(a);

  // Initialize fluid variables
  pmb->par_for("spherical_accn_init",  ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA_3D {
      GReal Xnative[GR_DIM], Xembed[GR_DIM]; 
      G.coord(k, j, i, Loci::center, Xnative);
      G.coord_embed(k, j, i, Loci::center, Xembed);
      GReal r = Xembed[1];

      if (r >= rin) {
        rho(k, j, i) = rho_init;
        u(k, j, i)   = u_init;
        VLOOP uvec(v, k, j, i) = 0.;
        // Note that the velocity primitives defined up there aren't quite right.
        // For a fluid at rest wrt. the normal observer, ucon = {-1/g_tt,0,0,0}. 
        // We need to use this info to obtain the correct values for U1, U2 and U3
        Real ucon[GR_DIM]         = {0};
        Real gcov[GR_DIM][GR_DIM] = {0};
        Real gcon[GR_DIM][GR_DIM] = {0};
        G.gcov(Loci::center, j, i, gcov);
        G.gcon(Loci::center, j, i, gcon);

        ucon[0] = 1./sqrt(-gcov[0][0]);
        ucon[1] = 0.;
        ucon[2] = 0.;
        ucon[3] = 0.;

        double alpha, beta[GR_DIM], gamma;

        // Solve for primitive velocities (utilde)
        alpha = 1/sqrt(-gcon[0][0]);
        gamma = ucon[0] * alpha;

        beta[0] = 0.;
        beta[1] = alpha*alpha*gcon[0][1];
        beta[2] = alpha*alpha*gcon[0][2];
        beta[3] = alpha*alpha*gcon[0][3];

        VLOOP uvec(v, k, j, i) = ucon[v+1] + beta[v+1]*gamma/alpha;

        // EMHD variables
        if (conduction)
            q(k, j, i)  = 0.;
        if (viscosity)
            dP(k, j, i) = 0.;
      }

    }
  );

  return TaskStatus::complete;

}


TaskStatus dirichlet_bc_sph_accn(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse)
{
  Flag(mbd, "Fluid: Applying Dirichlet boundary conditions along radial direction; B-field: Outflow condition");

  auto pmb    = mbd->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const bool use_emhd = pmb->packages.AllPackages().count("EMHD");
  const Real rho_init = pmb->packages.Get("GRMHD")->Param<Real>("rho_init");
  const Real u_init   = pmb->packages.Get("GRMHD")->Param<Real>("u_init");

  GridScalar rho  = mbd->Get("prims.rho").data;
  GridScalar u    = mbd->Get("prims.u").data;
  GridVector uvec = mbd->Get("prims.uvec").data;
  GridVector B_P  = mbd->Get("prims.B").data;
  GridScalar q;
  GridScalar dP;
  if (use_emhd) {
    q  = mbd->Get("prims.q").data;
    dP = mbd->Get("prims.dP").data;
  }

  const auto& G = pmb->coords;
  IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
  IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

  int ref_tmp;
  // For this problem the domain passed to this function WILL be outer_x1 but the if statement
  // acts as a conditional statement
  if (domain == IndexDomain::outer_x1)
    ref_tmp = bounds.GetBoundsI(IndexDomain::interior).e;
  const int ref = ref_tmp;

  pmb->par_for("dirichlet_boundary_conditions_fluid", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA_3D {
      rho(k, j, i) = rho_init;
      u(k, j, i)   = u_init;

      VLOOP uvec(v, k, j, i) = 0.;
      // Note that the velocity primitives defined up there aren't quite right.
      // For a fluid at rest wrt. the normal observer, ucon = {-1/g_tt,0,0,0}. 
      // We need to use this info to obtain the correct values for U1, U2 and U3
      Real ucon[GR_DIM]         = {0};
      Real gcov[GR_DIM][GR_DIM] = {0};
      Real gcon[GR_DIM][GR_DIM] = {0};
      G.gcov(Loci::center, j, i, gcov);
      G.gcon(Loci::center, j, i, gcon);

      ucon[0] = 1./sqrt(-gcov[0][0]);
      ucon[1] = 0.;
      ucon[2] = 0.;
      ucon[3] = 0.;

      double alpha, beta[GR_DIM], gamma;

      // Solve for primitive velocities (utilde)
      alpha = 1/sqrt(-gcon[0][0]);
      gamma = ucon[0] * alpha;

      beta[0] = 0.;
      beta[1] = alpha*alpha*gcon[0][1];
      beta[2] = alpha*alpha*gcon[0][2];
      beta[3] = alpha*alpha*gcon[0][3];

      VLOOP uvec(v, k, j, i) = ucon[v+1] + beta[v+1]*gamma/alpha;

      if (use_emhd) {
        q(k, j, i)  = 0.;
        dP(k, j, i) = 0.;
      }

      // Fluid variables at outer boundary have been set.
      // Now update B-field
      const Real rescale = G.gdet(Loci::center, j, ref) / G.gdet(Loci::center, j, i);
      VLOOP B_P(v, k, j, i) *= rescale;
    }
  );

  Flag(mbd, "Fluid: Dirichlet boundary conditions applied; B-field: Outflow condition applied");
  return TaskStatus::complete;

}
