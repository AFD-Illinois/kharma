/* 
 *  File: mhdmodes.hpp
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

#include <complex>

#include "decs.hpp"


using namespace std::literals::complex_literals;
using namespace std;
using namespace parthenon;

/**
 * Initialization for different analytic wave modes in magnetized plasma.
 * Note this assumes ideal EOS with gamma=4/3!
 * 
 * @param nmode: type of linear wave, from:
 * 0. Entropy, static mode
 * 1. Slow mode
 * 2. Alfven wave
 * 3. Fast mode
 * 
 * @param dir: direction of wave. 0 = components of each
 *
 * Returns the stopping time corresponding to advection by 1 wavelength
 */
Real InitializeMHDModes(MeshBlock *pmb, GRCoordinates G, GridVars P, GridVector B_P, int nmode, int dir)
{
    // Mean state
    Real rho0 = 1.;
    Real u0 = 1.;
    // TODO try a boosted entropy test with uN0 > 0. Take as arguments?
    Real u10 = 0.;
    Real u20 = 0.;
    Real u30 = 0.;
    // B is set later, see below
    Real B10 = 0.;
    Real B20 = 0.;
    Real B30 = 0.;

    // Wavevector (TODO set on the fly)
    Real k1 = 2. * M_PI;
    Real k2 = 2. * M_PI;
    Real k3 = 2. * M_PI;
    // "Faux-2D" planar waves direction
    // Set to 0 for "full" 3D wave
    if (dir == 1)
        k1 = 0;
    if (dir == 2)
        k2 = 0;
    if (dir == 3)
        k3 = 0;

    Real amp = 1.e-4;

    std::complex<Real> omega;
    Real drho, du, du1, du2, du3, dB1, dB2, dB3;

    // Eigenmode definitions
    if (dir == 0)
    {
        // 3D (1,1,1) wave
        B10 = 1.;
        if (nmode == 0)
        { // Entropy
            omega = 2. * M_PI / 5. * 1.i;
            drho = 1.;
        }
        else if (nmode == 1)
        { // Slow
            omega = 2.35896379113i;
            drho = 0.556500332363;
            du = 0.742000443151;
            du1 = -0.282334999306;
            du2 = 0.0367010491491;
            du3 = 0.0367010491491;
            dB1 = -0.195509141461;
            dB2 = 0.0977545707307;
            dB3 = 0.0977545707307;
        }
        else if (nmode == 2)
        { // Alfven
            omega = -3.44144232573i;
            du2 = -0.339683110243;
            du3 = 0.339683110243;
            dB2 = 0.620173672946;
            dB3 = -0.620173672946;
        }
        else
        { // Fast
            omega = 6.92915162882i;
            drho = 0.481846076323;
            du = 0.642461435098;
            du1 = -0.0832240462505;
            du2 = -0.224080007379;
            du3 = -0.224080007379;
            dB1 = 0.406380545676;
            dB2 = -0.203190272838;
            dB3 = -0.203190272838;
        }
    }
    else
    {
        // 2D (1,1,0), (1,0,1), (0,1,1) wave
        // Constant field direction
        if (dir == 1)
        {
            B20 = 1.;
        }
        else if (dir == 2)
        {
            B30 = 1.;
        }
        else if (dir == 3)
        {
            B10 = 1.;
        }

        if (nmode == 0)
        { // Entropy
            omega = 2. * M_PI / 5. * 1.i;
            drho = 1.;
        }
        else if (nmode == 1)
        { // Slow
            omega = 2.41024185339i;
            drho = 0.558104461559;
            du = 0.744139282078;
            if (dir == 1)
            {
                du2 = -0.277124827421;
                du3 = 0.0630348927707;
                dB2 = -0.164323721928;
                dB3 = 0.164323721928;
            }
            else if (dir == 2)
            {
                du3 = -0.277124827421;
                du1 = 0.0630348927707;
                dB3 = -0.164323721928;
                dB1 = 0.164323721928;
            }
            else if (dir == 3)
            {
                du1 = -0.277124827421;
                du2 = 0.0630348927707;
                dB1 = -0.164323721928;
                dB2 = 0.164323721928;
            }
        }
        else if (nmode == 2)
        { // Alfven
            omega = 3.44144232573i;
            if (dir == 1)
            {
                du1 = 0.480384461415;
                dB1 = 0.877058019307;
            }
            else if (dir == 2)
            {
                du2 = 0.480384461415;
                dB2 = 0.877058019307;
            }
            else if (dir == 3)
            {
                du3 = 0.480384461415;
                dB3 = 0.877058019307;
            }
        }
        else
        { // Fast
            omega = 5.53726217331i;
            drho = 0.476395427447;
            du = 0.635193903263;
            if (dir == 1)
            {
                du2 = -0.102965815319;
                du3 = -0.316873207561;
                dB2 = 0.359559114174;
                dB3 = -0.359559114174;
            }
            else if (dir == 2)
            {
                du3 = -0.102965815319;
                du1 = -0.316873207561;
                dB3 = 0.359559114174;
                dB1 = -0.359559114174;
            }
            else if (dir == 3)
            {
                du1 = -0.102965815319;
                du2 = -0.316873207561;
                dB1 = 0.359559114174;
                dB2 = -0.359559114174;
            }
        }
    }

    // Override end time to be exactly 1 period
    Real tf;
    if (nmode != 0) {
        tf = 2. * M_PI / fabs(omega.imag());
    } else {
        tf = -1;
    }

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for("mhdmodes_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);

            Real mode = amp * cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);
            P(prims::rho, k, j, i) = rho0 + drho * mode;
            P(prims::u, k, j, i) = u0 + du * mode;
            P(prims::u1, k, j, i) = u10 + du1 * mode;
            P(prims::u2, k, j, i) = u20 + du2 * mode;
            P(prims::u3, k, j, i) = u30 + du3 * mode;
            B_P(0, k, j, i) = B10 + dB1 * mode;
            B_P(1, k, j, i) = B20 + dB2 * mode;
            B_P(2, k, j, i) = B30 + dB3 * mode;
        }
    );

    return tf;
}
