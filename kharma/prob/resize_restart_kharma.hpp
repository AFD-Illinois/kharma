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
void ReadKharmaRestartHeader(std::string fname, ParameterInput *pin);

/**
 * Read data from an KHARMA restart file. Does not support >1 meshblock in Parthenon
 * 
 * Returns stop time tf of the original simulation, for e.g. replicating regression tests
 */
TaskStatus ReadKharmaRestart(std::shared_ptr<MeshBlockData<Real>> rc, ParameterInput *pin);

// newly added by Hyerin (09/06/22)
TaskStatus SetKharmaRestart(std::shared_ptr<MeshBlockData<Real>> rc, IndexDomain domain, bool coarse);

// Hint form resize.hpp
// TODO: (Hyerin) should I do const for x1, x2, x3, var?
KOKKOS_INLINE_FUNCTION void Xtoindex(const GReal XG[GR_DIM],
                                  //Real *x1, Real *x2, Real *x3,
                                   const GridScalar& x1, const GridScalar& x2, const GridScalar& x3,
                                   const hsize_t length[GR_DIM], int& iblock,
                                   int& i, int& j, int& k, GReal del[GR_DIM])
{
    //cout << "Hyerin: entered Xtoindex" <<endl;
    Real dx2, dx2_min;

    // initialize
    iblock =0;
    i = 0;
    j = 0;
    k = 0;
    dx2_min = m::pow(XG[1]-x1(iblock,i),2.)+
              m::pow(XG[2]-x2(iblock,j),2.)+
              m::pow(XG[3]-x3(iblock,k),2.);

    for (int iblocktemp = 0; iblocktemp < length[0]; iblocktemp++) {
        for (int itemp = 0; itemp < length[1]; itemp++) {
            for (int jtemp = 0; jtemp < length[2]; jtemp++) {
                for (int ktemp = 0; ktemp < length[3]; ktemp++) {
                    dx2 = m::pow(XG[1]-x1(iblocktemp,itemp),2.)+
                          m::pow(XG[2]-x2(iblocktemp,jtemp),2.)+
                          m::pow(XG[3]-x3(iblocktemp,ktemp),2.);

                    // simplest interpolation (Hyerin 07/26/22)
                    if (dx2<dx2_min){
                        dx2_min=dx2;
                        iblock=iblocktemp;
                        i = itemp;
                        j = jtemp;
                        k = ktemp;
                    }
                }
            }
        }
    }

    del[1] = 0.; //(XG[1] - ((i) * dx[1] + startx[1])) / dx[1];
    del[2] = 0.;//(XG[2] - ((j) * dx[2] + startx[2])) / dx[2];
    del[3] = 0.;// (phi   - ((k) * dx[3] + startx[3])) / dx[3];
    if (m::abs(dx2_min / m::pow(XG[1],2.)) > 1.e-8) printf("Xtoindex: dx2 pretty large = %g at r= %g \n", dx2_min, XG[1]);
}

// TOOD(BSP) these can be merged and moved back into the fn body now

KOKKOS_INLINE_FUNCTION void get_prim_restart_kharma(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                    const Real fx1min, const Real fx1max, const Real fnghost, const bool should_fill, const bool is_spherical, const bool include_B,
                    const Real gam, const Real rs,  const Real mdot, const hsize_t length[GR_DIM],
                    const GridScalar& x1, const GridScalar& x2, const GridScalar& x3, const GridScalar& rho_file, const GridScalar& u_file, const GridVector& uvec_file, const GridVector& B_file,
                    const GridScalar& x1_fill, const GridScalar& x2_fill, const GridScalar& x3_fill, const GridScalar& rho_fill, const GridScalar& u_fill, const GridVector& uvec_fill, const GridVector& B_fill,
                    const int& k, const int& j, const int& i) 
{
    Real rho = 0, u = 0;
    Real u_prim[NVEC] = {0}; //, B_prim[NVEC];

    GReal X[GR_DIM];
    G.coord(k, j, i, Loci::center, X);
    GReal del[GR_DIM]; // not really needed now since I am doing nearest neighbor interpolation
    int iblocktemp, itemp, jtemp, ktemp;

    // Interpolate the value at this location from the global grid
    if ((!should_fill) && (X[1]<fx1min)) {// if cannot be read from restart file
        GReal Xembed[GR_DIM];
        G.coord_embed(k, j, i, Loci::center, Xembed);
        GReal r = Xembed[1];
  
        // copy over smallest radius states
        //Xtoindex(X, x1, x2, x3, length, iblocktemp, itemp, jtemp, ktemp, del);
        iblocktemp = 0; // assuming always this block contains smallest radii?
        itemp = fnghost; // in order to copy over the physical region, not the ghost region
        // (02/08/23) instead in order to set the vacuum homogeneous instead of having theta phi dependence, set j and k values
        jtemp = fnghost;
        ktemp = fnghost;
        rho = rho_file(iblocktemp, ktemp, jtemp, itemp);
        u = u_file(iblocktemp, ktemp, jtemp, itemp);

        // (02/08/23) instead in order to set the vacuum homogeneous instead of having theta phi dependence, set to the bondi radius values (assume r_B ~ r_s**2)
        //Real T_temp = get_T(m::pow(rs,2), C1, C2, n, rs);
        //rho_temp = m::pow(T_temp, n);
        //u_temp = rho_temp * T_temp * n;
                        
        Real rho_tmp, u_tmp, ur;
        get_bondi_soln(r, rs, mdot, gam, rho_tmp, u_tmp, ur);
        Real ucon_bl[GR_DIM] = {0., ur, 0., 0.};
        Real ucon_native[GR_DIM];
        coords.bl_fourvel_to_native(X, ucon_bl, ucon_native);

        // Convert native 4-vector to primitive u-twiddle, see Gammie '04
        Real gcon[GR_DIM][GR_DIM];
        G.gcon(Loci::center, j, i, gcon);
        fourvel_to_prim(gcon, ucon_native, u_prim);

        // printf("Bondi fill location: %g %g %g %g KS: %g %g %g %g\nr: %g T: %g ur: %g\nucon: %g %g %g %g native: %g %g %g %g\nPrims: %g %g %g %g %g\n",
        //         X[0], X[1], X[2], X[3], Xembed[0], Xembed[1], Xembed[2], Xembed[3],
        //         r, T, ur, ucon_bl[0], ucon_bl[1], ucon_bl[2], ucon_bl[3], ucon_native[0], ucon_native[1], ucon_native[2], ucon_native[3],
        //         rho_temp, u_temp, u_prim[0], u_prim[1], u_prim[2]);

   }
    // HyerinTODO: if fname_fill exists and smaller.
    else if ((should_fill) && ((X[1]>fx1max)||(X[1]<fx1min))) { // fill with the fname_fill
        //Xtoindex(X, &(x1_fill[0]), &(x2_fill[0]), &(x3_fill[0]), length, iblocktemp, itemp, jtemp, ktemp, del);
        Xtoindex(X, x1_fill, x2_fill, x3_fill, length, iblocktemp, itemp, jtemp, ktemp, del);
        rho = rho_fill(iblocktemp, ktemp, jtemp, itemp);
        u = u_fill(iblocktemp, ktemp, jtemp, itemp);
        VLOOP u_prim[v] = uvec_fill(v, iblocktemp, ktemp, jtemp, itemp);
        //if (include_B) VLOOP B_prim[v] = B_fill(v,iblocktemp,ktemp,jtemp,itemp);
    }
    else { 
        Xtoindex(X, x1, x2, x3, length, iblocktemp, itemp, jtemp, ktemp, del);
        rho = rho_file(iblocktemp,ktemp,jtemp,itemp);
        u = u_file(iblocktemp,ktemp,jtemp,itemp);
        VLOOP u_prim[v] = uvec_file(v,iblocktemp,ktemp,jtemp,itemp);
        //if (include_B) VLOOP B_prim[v] = B(v,iblocktemp,ktemp,jtemp,itemp);
        //printf("File fill location: %g %g %g %g new index: %d %d %d from old index: (%d) %d %d %d\n",
        //       X[0], X[1], X[2], X[3], k, j, i, iblocktemp, ktemp, jtemp, itemp);
    }
    // if (u_prim[1] > 1 || u_prim[2] > 1) {
    //     printf("Fill prims: %g %g %g %g %g from bondi,fill,file: %d %d %d\n", rho_temp, u_temp, u_prim[0], u_prim[1], u_prim[2], filled_bondi, filled_fill, filled_file);
    // }
    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;
    P(m_p.U1, k, j, i) = u_prim[0]; 
    P(m_p.U2, k, j, i) = u_prim[1];
    P(m_p.U3, k, j, i) = u_prim[2];

}

KOKKOS_INLINE_FUNCTION void get_B_restart_kharma(const GRCoordinates& G, const VariablePack<Real>& U, const VarMap& m_u,
                    const Real fx1min, const Real fx1max, const bool should_fill,
                    const hsize_t length[GR_DIM],
                    const GridScalar& x1, const GridScalar& x2, const GridScalar& x3, const GridVector& B,
                    const GridScalar& x1_fill, const GridScalar& x2_fill, const GridScalar& x3_fill, const GridVector& B_fill,
                    const int& k, const int& j, const int& i) 
{
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
        VLOOP B_cons[v] = B_fill(v,iblocktemp,ktemp,jtemp,itemp);
    }
    else { 
        Xtoindex(X, x1, x2, x3, length, iblocktemp, itemp, jtemp, ktemp, del);
        VLOOP B_cons[v] = B(v,iblocktemp,ktemp,jtemp,itemp);
    }

    VLOOP U(m_u.B1 + v, k, j, i) = B_cons[v];
}
