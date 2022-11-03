/* 
 *  File: interpolation.hpp
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
#pragma once

#include "decs.hpp"

// For using the ipole routines verbatim.
// Automatically wraps in k so we can avoid ghost zones
#define ind_sph(i, j, k) ( (((k)+n3) % n3) * n2 * n1 + (j) * n1 + (i))
#define ind_periodic(i, j, k) ( (((k)+n3) % n3) * n2 * n1 + (((j)+n2) % n2) * n1 + (((i)+n1) % n1) )

/**
 * Routines for interpolating and initializing a KHARMA meshblock from the
 * correct area of a global iharm3d restart file, used in resize_restart.cpp.
 * Doesn't include "Elliptic maid" solver step for eliminating magnetic field
 * divergence, see b_flux_ct for that (as it is divergence-rep dependent)
 */

/**
 *  translates geodesic coordinates to a grid zone and returns offset
 *  for interpolation purposes. integer index corresponds to the zone
 *  center "below" the desired point and del[i] \in [0,1) returns the
 *  offset from that zone center.
 *
 *  0    0.5    1
 *  [     |     ]
 *  A  B  C DE  F
 *
 *  startx = 0.
 *  dx = 0.5
 *
 *  A -> (-1, 0.5)
 *  B -> ( 0, 0.0)
 *  C -> ( 0, 0.5)
 *  D -> ( 0, 0.9)
 *  E -> ( 1, 0.0)
 *  F -> ( 1, 0.5)
 */
KOKKOS_INLINE_FUNCTION void Xtoijk(const GReal XG[GR_DIM],
                                   const GReal startx[GR_DIM],
                                   const GReal dx[GR_DIM],
                                   int& i, int& j, int& k, GReal del[GR_DIM],
                                   bool nearest=false)
{
    // If we ever include ghosts in iharm3d-format restarts, we need to clip phi here
    // GReal phi = fmod(XG[3], stopx[3]);
    // if (phi < 0.0) // TODO adapt for startx3 != 0?
    //     phi += stopx[3];
    GReal phi = XG[3];

    if (nearest) {
        // get the index of the zone we are in: >= left corner?
        i = (int) ((XG[1] - startx[1]) / dx[1] + 1000) - 1000;
        j = (int) ((XG[2] - startx[2]) / dx[2] + 1000) - 1000;
        k = (int) ((phi   - startx[3]) / dx[3] + 1000) - 1000;
    } else {
        // Normal operation
        // get provisional zone index. see note above function for details. note we
        // shift to zone centers because that's where variables are most exact.
        i = (int) ((XG[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
        j = (int) ((XG[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;
        k = (int) ((phi   - startx[3]) / dx[3] - 0.5 + 1000) - 1000;
    }

    // now construct del
    del[1] = (XG[1] - ((i + 0.5) * dx[1] + startx[1])) / dx[1];
    del[2] = (XG[2] - ((j + 0.5) * dx[2] + startx[2])) / dx[2];
    del[3] = (phi   - ((k + 0.5) * dx[3] + startx[3])) / dx[3];
}

KOKKOS_INLINE_FUNCTION void ijktoX(const GReal startx[GR_DIM], const GReal dx[GR_DIM],
                                   const int& i, const int& j, const int& k,
                                   GReal XG[GR_DIM])
{
    // get provisional zone index. see note above function for details. note we
    // shift to zone centers because that's where variables are most exact.
    XG[0] = 0.;
    XG[1] = startx[1] + (i + 0.5) * dx[1];
    XG[2] = startx[2] + (j + 0.5) * dx[2];
    XG[3] = startx[3] + (k + 0.5) * dx[3];
}

/**
 * This interpolates a single-array variable 'var' representing a grid of size 'startx' to 'stopx' in
 * native coordinates, returning its value at location X
 * NOTE: 'startx' must correspond to the grid you are interpolating *from*
 */
KOKKOS_INLINE_FUNCTION Real linear_interp(const GRCoordinates& G, const GReal X[GR_DIM],
                                          const GReal startx[GR_DIM],
                                          const GReal dx[GR_DIM], const bool& is_spherical, const bool& weight_by_gdet,
                                          const int& n3, const int& n2, const int& n1,
                                          const Real *var)
{
    // zone and offset from X
    // Obtain this in
    GReal del[GR_DIM];
    int i, j, k;
    Xtoijk(X, startx, dx, i, j, k, del);

    Real interp;
    if (is_spherical) {
        // For ghost zones, we treat each boundary differently:
        // In X1, repeat first & last zones.
        if (i < 0) { i = 0; del[1] = 0; }
        if (i > n1-2) { i = n1 - 2; del[1] = 1; }
        // In X2, stop completely at the last zone
        // Left side of leftmost segment
        if (j < 0) { j = 0; del[2] = 0; }
        // Right side of rightmost segment.  Phrased this way to not segfault
        if (j > n2-2) { j = n2 - 2; del[2] = 1; }
        // k auto-wraps. So do all indices for periodic boxes.

        if (weight_by_gdet) {
            GReal Xtmp[GR_DIM];
            ijktoX(startx, dx, i, j, k, Xtmp);
            GReal g_ij = G.coords.gdet_native(Xtmp);
            ijktoX(startx, dx, i + 1, j, k, Xtmp);
            GReal g_i1j = G.coords.gdet_native(Xtmp);
            ijktoX(startx, dx, i, j + 1, k, Xtmp);
            GReal g_ij1 = G.coords.gdet_native(Xtmp);
            ijktoX(startx, dx, i + 1, j + 1, k, Xtmp);
            GReal g_i1j1 = G.coords.gdet_native(Xtmp);

            // interpolate in x1 and x2
                interp = var[ind_sph(i    , j    , k)]*g_ij*(1. - del[1])*(1. - del[2]) +
                         var[ind_sph(i    , j + 1, k)]*g_ij1*(1. - del[1])*del[2] +
                         var[ind_sph(i + 1, j    , k)]*g_i1j*del[1]*(1. - del[2]) +
                         var[ind_sph(i + 1, j + 1, k)]*g_i1j1*del[1]*del[2];

            // then interpolate in x3 if we need
            if (n3 > 1) {
                interp = (1. - del[3])*interp +
                        del[3]*(var[ind_sph(i    , j    , k + 1)]*g_ij*(1. - del[1])*(1. - del[2]) +
                                var[ind_sph(i    , j + 1, k + 1)]*g_ij1*(1. - del[1])*del[2] +
                                var[ind_sph(i + 1, j    , k + 1)]*g_i1j*del[1]*(1. - del[2]) +
                                var[ind_sph(i + 1, j + 1, k + 1)]*g_i1j1*del[1]*del[2]);
            }
            interp /= G.coords.gdet_native(X);
        } else {
            // interpolate in x1 and x2
                interp = var[ind_sph(i    , j    , k)]*(1. - del[1])*(1. - del[2]) +
                         var[ind_sph(i    , j + 1, k)]*(1. - del[1])*del[2] +
                         var[ind_sph(i + 1, j    , k)]*del[1]*(1. - del[2]) +
                         var[ind_sph(i + 1, j + 1, k)]*del[1]*del[2];

            // then interpolate in x3 if we need
            if (n3 > 1) {
                interp = (1. - del[3])*interp +
                        del[3]*(var[ind_sph(i    , j    , k + 1)]*(1. - del[1])*(1. - del[2]) +
                                var[ind_sph(i    , j + 1, k + 1)]*(1. - del[1])*del[2] +
                                var[ind_sph(i + 1, j    , k + 1)]*del[1]*(1. - del[2]) +
                                var[ind_sph(i + 1, j + 1, k + 1)]*del[1]*del[2]);
            }
        }
    } else {
        // interpolate in x1 and x2
            interp = var[ind_periodic(i    , j    , k)]*(1. - del[1])*(1. - del[2]) +
                     var[ind_periodic(i    , j + 1, k)]*(1. - del[1])*del[2] +
                     var[ind_periodic(i + 1, j    , k)]*del[1]*(1. - del[2]) +
                     var[ind_periodic(i + 1, j + 1, k)]*del[1]*del[2];

        // then interpolate in x3 if we need
        if (n3 > 1) {
            interp = (1. - del[3])*interp +
                    del[3]*(var[ind_periodic(i    , j    , k + 1)]*(1. - del[1])*(1. - del[2]) +
                            var[ind_periodic(i    , j + 1, k + 1)]*(1. - del[1])*del[2] +
                            var[ind_periodic(i + 1, j    , k + 1)]*del[1]*(1. - del[2]) +
                            var[ind_periodic(i + 1, j + 1, k + 1)]*del[1]*del[2]);
        }
    }

    return interp;
}

