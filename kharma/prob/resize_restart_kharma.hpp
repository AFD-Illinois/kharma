// Load the grid variables up with primitives from an old KHARMA run
#pragma once

#include "decs.hpp"

#include "mesh/mesh.hpp"

// added by Hyerin (10/07/22)
#include "bondi.hpp"
#include "b_flux_ct.hpp"

/**
 * Read the header of an KHARMA HDF5 restart file, and set appropriate parameters
 * Call this before mesh creation!
 */
void ReadKharmaRestartHeader(std::string fname, std::unique_ptr<ParameterInput>& pin);

/**
 * Read data from an KHARMA restart file. Does not support >1 meshblock in Parthenon
 * 
 * Returns stop time tf of the original simulation, for e.g. replicating regression tests
 */
TaskStatus ReadKharmaRestart(MeshBlockData<Real> *rc, ParameterInput *pin);

// newly added by Hyerin (09/06/22)
TaskStatus SetKharmaRestart(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);

// Hint form resize.hpp
// TODO: (Hyerin) should I do const for x1, x2, x3, var?
KOKKOS_INLINE_FUNCTION void Xtoindex(const GReal XG[GR_DIM],
                                  //Real *x1, Real *x2, Real *x3,
                                   const GridScalar& x1, const GridScalar& x2, const GridScalar& x3,
                                   const hsize_t length[GR_DIM], int& iblock,
                                   int& i, int& j, int& k, GReal del[GR_DIM])
{
    //cout << "Hyerin: entered Xtoindex" <<endl;
    Real dx1, dx2, dx3, dx_sum, dx1_min, dx2_min, dx3_min, dx_sum_min;

    // initialize
    iblock =0;
    i = 0;
    j = 0;
    k = 0;
    dx1_min = (XG[1]-x1(iblock,i))*(XG[1]-x1(iblock,i));
    dx2_min = (XG[2]-x2(iblock,j))*(XG[2]-x2(iblock,j));
    dx3_min = (XG[3]-x3(iblock,k))*(XG[3]-x3(iblock,k));
    dx_sum_min = dx1_min + dx2_min + dx3_min;

    for (int iblocktemp = 0; iblocktemp < length[0]; iblocktemp++) {
        // independently searching for minimum for i,j,k
        for (int itemp = 0; itemp < length[1]; itemp++) {
            dx1 = (XG[1]-x1(iblocktemp,itemp))*(XG[1]-x1(iblocktemp,itemp));
            if (dx1 < dx1_min) {
                dx1_min = dx1;
                i = itemp;
            }
        }
        for (int jtemp = 0; jtemp < length[2]; jtemp++) {
            dx2 = (XG[2]-x2(iblocktemp,jtemp))*(XG[2]-x2(iblocktemp,jtemp));
            if (dx2 < dx2_min) {
                dx2_min = dx2;
                j = jtemp;
            }
        }
        for (int ktemp = 0; ktemp < length[3]; ktemp++) {
            dx3 = (XG[3]-x3(iblocktemp,ktemp))*(XG[3]-x3(iblocktemp,ktemp));
            if (dx3 < dx3_min) {
                dx3_min = dx3;
                k = ktemp;
            }
        }
        dx_sum = (XG[1]-x1(iblocktemp,i))*(XG[1]-x1(iblocktemp,i)) + 
                 (XG[2]-x2(iblocktemp,j))*(XG[2]-x2(iblocktemp,j)) + 
                 (XG[3]-x3(iblocktemp,k))*(XG[3]-x3(iblocktemp,k));
        if (dx_sum < dx_sum_min) {
            dx_sum_min = dx_sum;
            iblock = iblocktemp;
        }
    }

    del[1] = 0.; //(XG[1] - ((i) * dx[1] + startx[1])) / dx[1];
    del[2] = 0.;//(XG[2] - ((j) * dx[2] + startx[2])) / dx[2];
    del[3] = 0.;// (phi   - ((k) * dx[3] + startx[3])) / dx[3];
    if (m::abs(dx1_min/m::pow(XG[1],2.))>1.e-8) printf("Xtoindex: dx2 pretty large = %g at r= %g \n",dx1_min, XG[1]);
    if (m::abs(dx2_min)>1.e-8) printf("Xtoindex: dx2 pretty large = %g at th = %g \n",dx2_min, XG[2]);
    if (m::abs(dx3_min)>1.e-8) printf("Xtoindex: dx2 pretty large = %g at phi = %g \n",dx3_min, XG[3]);
}


KOKKOS_INLINE_FUNCTION void convert_to_utwiddle(const GRCoordinates& G, const CoordinateEmbedding& coords,
                                           const SphBLCoords& bl,  const SphKSCoords& ks, 
                                           const int& k, const int& j, const int& i, Real ucon_bl[GR_DIM], Real u_prim[NVEC])
{
    GReal Xnative[GR_DIM], Xembed[GR_DIM]; //
    G.coord(k, j, i, Loci::center, Xnative);
    G.coord_embed(k, j, i, Loci::center, Xembed);

    // Set u^t to make u^r a 4-vector
    Real gcov_bl[GR_DIM][GR_DIM];
    bl.gcov_embed(Xembed, gcov_bl);
    set_ut(gcov_bl, ucon_bl);

    // Then transform that 4-vector to KS, then to native
    Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
    ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
    coords.con_vec_to_native(Xnative, ucon_ks, ucon_mks);

    Real gcon[GR_DIM][GR_DIM];
    G.gcon(Loci::center, j, i, gcon); //TODO: this causes the memory issue!!
    fourvel_to_prim(gcon, ucon_mks, u_prim);

}

KOKKOS_INLINE_FUNCTION void get_prim_restart_kharma(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                    const SphBLCoords& bl,  const SphKSCoords& ks, 
                    const Real fx1min, const Real fx1max, const Real fnghost, const bool should_fill, const bool is_spherical,
                    const Real gam, const Real rs,  const Real mdot, const Real ur_frac, const Real uphi, const hsize_t length[GR_DIM],
                    const GridScalar& x1, const GridScalar& x2, const GridScalar& x3, const GridScalar& rho, const GridScalar& u, const GridVector& uvec,
                    const GridScalar& x1_fill, const GridScalar& x2_fill, const GridScalar& x3_fill, const GridScalar& rho_fill, const GridScalar& u_fill, const GridVector& uvec_fill,
                    const Real vacuum_logrho, const Real vacuum_log_u_over_rho,
                    const int& k, const int& j, const int& i) 
{
    Real rho_temp, u_temp;
    Real u_prim[NVEC]; //, B_prim[NVEC];
    
    GReal X[GR_DIM];
    G.coord(k, j, i, Loci::center, X);
    GReal del[GR_DIM]; // not really needed now since I am doing nearest neighbor interpolation
    int iblocktemp, itemp, jtemp, ktemp;
    // Interpolate the value at this location from the global grid
    if ((!should_fill) && (X[1]<fx1min)) {// if cannot be read from restart file
        // same as Bondi (06/13/23)
        get_prim_bondi(G, coords, P, m_p, gam, bl, ks, mdot, rs, rs*rs*100, ur_frac, uphi, k, j, i); // get the solution at r_b*100
    }
    // HyerinTODO: if fname_fill exists and smaller.
    else {
        if ((should_fill) && ((X[1]>fx1max)||(X[1]<fx1min))) { // fill with the fname_fill
            //Xtoindex(X, &(x1_fill[0]), &(x2_fill[0]), &(x3_fill[0]), length, iblocktemp, itemp, jtemp, ktemp, del);
            Xtoindex(X, x1_fill, x2_fill, x3_fill, length, iblocktemp, itemp, jtemp, ktemp, del);
            rho_temp = rho_fill(iblocktemp,ktemp,jtemp,itemp);
            u_temp = u_fill(iblocktemp,ktemp,jtemp,itemp);
            VLOOP u_prim[v] = uvec_fill(v,iblocktemp,ktemp,jtemp,itemp);
            //if (include_B) VLOOP B_prim[v] = B_fill(v,iblocktemp,ktemp,jtemp,itemp);
        }
        else { 
            Xtoindex(X, x1, x2, x3, length, iblocktemp, itemp, jtemp, ktemp, del);
            rho_temp = rho(iblocktemp,ktemp,jtemp,itemp);
            u_temp = u(iblocktemp,ktemp,jtemp,itemp);
            VLOOP u_prim[v] = uvec(v,iblocktemp,ktemp,jtemp,itemp);
            //if (include_B) VLOOP B_prim[v] = B(v,iblocktemp,ktemp,jtemp,itemp);
        }
        P(m_p.RHO, k, j, i) = rho_temp;
        P(m_p.UU, k, j, i) = u_temp;
        P(m_p.U1, k, j, i) = u_prim[0]; 
        P(m_p.U2, k, j, i) = u_prim[1];
        P(m_p.U3, k, j, i) = u_prim[2];
    }

}

KOKKOS_INLINE_FUNCTION void get_B_restart_kharma(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                    const SphBLCoords& bl,  const SphKSCoords& ks, 
                    const Real fx1min, const Real fx1max, const bool should_fill,
                    const hsize_t length[GR_DIM],
                    const GridScalar& x1, const GridScalar& x2, const GridScalar& x3, const GridVector& B,
                    const GridScalar& x1_fill, const GridScalar& x2_fill, const GridScalar& x3_fill, const GridVector& B_fill, const GridVector& B_save,
                    const int& k, const int& j, const int& i) 
{
    //Real B_prim[NVEC];
    Real B_cons[NVEC];
    
    GReal X[GR_DIM];
    G.coord(k, j, i, Loci::center, X);
    GReal del[GR_DIM]; // not really needed now since I am doing nearest neighbor interpolation
    int iblocktemp, itemp, jtemp, ktemp;
    // Interpolate the value at this location from the global grid
    if ((!should_fill) && (X[1]<fx1min)) {// if cannot be read from restart file
        // do nothing. just use the initialization from SeedBField
   }
    else if ((should_fill) && ((X[1]>fx1max)||(X[1]<fx1min))) { // fill with the fname_fill
        Xtoindex(X, x1_fill, x2_fill, x3_fill, length, iblocktemp, itemp, jtemp, ktemp, del);
        //VLOOP B_prim[v] = B_fill(v,iblocktemp,ktemp,jtemp,itemp);
        VLOOP B_cons[v] = B_fill(v,iblocktemp,ktemp,jtemp,itemp);
    }
    else { 
        Xtoindex(X, x1, x2, x3, length, iblocktemp, itemp, jtemp, ktemp, del);
        //VLOOP B_prim[v] = B(v,iblocktemp,ktemp,jtemp,itemp);
        VLOOP B_cons[v] = B(v,iblocktemp,ktemp,jtemp,itemp);
    }

    B_save(0, k, j, i) = B_cons[0];
    B_save(1, k, j, i) = B_cons[1];
    B_save(2, k, j, i) = B_cons[2];

}
