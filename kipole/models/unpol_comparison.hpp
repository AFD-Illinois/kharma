/* 
 *  File: b_flux_ct.hpp
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

#include <memory>

#include <parthenon/parthenon.hpp>

#include "matrix.hpp"
#include "rad_functions.hpp"
#include "tetrad.hpp"

using namespace parthenon;

/**
 * This physics package returns emissivity values for the analytic model used as a comparison point
 * in Gold et al. 2020, the unpolarized EHT GRRT code comparison
 */
class Model_UnpolComp {
    private:
        double freqcgs_;
        double A_, alpha_, height_, l0_, a_;
        double L_unit_, T_unit_, RHO_unit_, B_unit_;
    public:
        /**
         * Set member variables
         */
        void Initialize(ParameterInput *pin);

        //// INTERFACE: Functions called from elsewhere in ipole ////
        KOKKOS_INLINE_FUNCTION double get_model_ne(const GRCoordinates& G, double X[GR_DIM])
        {
            // Matter model defined in Gold et al 2020 section 3
            double Xembed[GR_DIM];
            G.coords.coord_to_embed(X, Xembed);
            double r = Xembed[1], th = Xembed[2];
            double n_exp = 1. / 2 * (pow(r / 10, 2) + pow(height_ * cos(th), 2));

            // Cutoff when result will be ~0
            return (n_exp < 200) ? RHO_unit_ * exp(-n_exp) : 0;
        }

        KOKKOS_INLINE_FUNCTION void get_model_jk(const GRCoordinates& G, double X[GR_DIM], double Kcon[GR_DIM], double *jnuinv, double *knuinv)
        {
            // Emission model defined in Gold et al 2020 section 3
            double n = get_model_ne(G, X);

            double Ucon[GR_DIM], Ucov[GR_DIM], Bcon[GR_DIM], Bcov[GR_DIM];
            get_model_fourv(G, X, Kcon, Ucon, Ucov, Bcon, Bcov);
            double nu = get_fluid_nu(Kcon, Ucov);

            *jnuinv = fmax(n * pow(nu / freqcgs_, -alpha_) / pow(nu, 2), 0);
            *knuinv = fmax((A_ * n * pow(nu / freqcgs_, -(2.5 + alpha_)) + 1.e-54) * nu, 0);
        }

        KOKKOS_INLINE_FUNCTION void get_model_jar(const GRCoordinates& G, double X[GR_DIM], double Kcon[GR_DIM],
                        double *jI, double *jQ, double *jU, double *jV,
                        double *aI, double *aQ, double *aU, double *aV,
                        double *rQ, double *rU, double *rV)
        {
            // Define a model here relating X,K -> j_S, alpha_S, rho_S
            // (and below relating X,K -> u,B 4-vectors)
            // ipole will do the rest

            // Just take the unpolarized emissivity and absorptivity as I,
            // and set the rest to zero
            // Of course, you can be more elaborate
            double j, k;
            get_model_jk(G, X, Kcon, &j, &k);

            *jI = j;
            *jQ = 0;
            *jU = 0;
            *jV = 0;

            *aI = k;
            *aQ = 0;
            *aU = 0;
            *aV = 0;

            *rQ = 0;
            *rU = 0;
            *rV = 0;

            return;
        }

        KOKKOS_INLINE_FUNCTION void get_model_fourv(const GRCoordinates& G, double X[GR_DIM], double Kcon[GR_DIM], double Ucon[GR_DIM], double Ucov[GR_DIM],
                            double Bcon[GR_DIM], double Bcov[GR_DIM])
        {
            double Xembed[GR_DIM];
            G.coords.coord_to_embed(X, Xembed);
            // Note these quantities from Gold et al are in BL!
            // We could have converted the problem to KS, but instead we did this
            double R = Xembed[1] * sin(Xembed[2]);
            double l = (l0_ / (1 + R)) * pow(R, 1 + 0.5);

            // Metrics: BL
            SphKSCoords ks = mpark::get<SphKSCoords>(G.coords.base);
            SphBLCoords bl = SphBLCoords(ks.a);
            double bl_gcov[GR_DIM][GR_DIM], bl_gcon[GR_DIM][GR_DIM];
            bl.gcov_embed(Xembed, bl_gcov);
            invert(&bl_gcov[0][0], &bl_gcon[0][0]);
            // Native (anything KS-based)
            double gcov[GR_DIM][GR_DIM], gcon[GR_DIM][GR_DIM];
            G.coords.gcov_native(X, gcov);
            G.coords.gcon_native(gcov, gcon);

            // Get the normal observer velocity for Ucon/Ucov, in BL coordinates
            double ubar = sqrt(-1. / (bl_gcon[0][0] - 2. * bl_gcon[0][3] * l + bl_gcon[3][3] * l * l));
            double bl_Ucov[GR_DIM] = {-ubar, 0., 0., l * ubar};

            double bl_Ucon[GR_DIM];
            flip_index(bl_Ucov, bl_gcon, bl_Ucon);

            // Transform to KS coordinates,
            double ks_Ucon[GR_DIM];
            ks.vec_from_bl(X, bl_Ucon, ks_Ucon);
            // then to our coordinates,
            G.coords.con_vec_to_native(X, ks_Ucon, Ucon);

            // and grab Ucov
            flip_index(Ucon, gcov, Ucov);

            // This model defines no field in emission, but the field is used for making
            // tetrads so we want it consistent
            Bcon[0] = 0;
            Bcon[1] = 0;
            Bcon[2] = 1;
            Bcon[3] = 0;
            flip_index(Bcon, gcov, Bcov);
        }

        /**
         * This problem defines no field in emission, but we want to control ipole's worst
         * tendencies when making tetrads.  This will return a correct value even for a
         * possible fluid/field model later, too.
         */
        KOKKOS_INLINE_FUNCTION double get_model_b(const GRCoordinates& G, double X[GR_DIM])
        {
            double Ucon[GR_DIM], Bcon[GR_DIM];
            double Ucov[GR_DIM], Bcov[GR_DIM];
            double Kcon[GR_DIM] = {0}; // TODO interface change if we ever need a real one here
            get_model_fourv(G, X, Kcon, Ucon, Ucov, Bcon, Bcov);
            return sqrt(dot(Bcon, Bcov)) * B_unit_;
        }

        //// STUBS: Functions for normal models which we don't use ////
        // Define these to specify a fluid model: e- density/temperature for
        // synchrotron radiation based on an energy distribution
        KOKKOS_INLINE_FUNCTION double get_model_thetae(double X[GR_DIM]) { return 0; }
        KOKKOS_INLINE_FUNCTION void get_model_powerlaw_vals(double X[GR_DIM], double *p, double *n,
                                    double *gamma_min, double *gamma_max, double *gamma_cut) { return; }

        // This is only called for trace file output, and doesn't really apply to analytic models
        KOKKOS_INLINE_FUNCTION void get_model_primitives(double X[GR_DIM], double *p) { return; }
};