/* 
 *  File: coordinate_utils.hpp
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
#include "matrix.hpp"

/**
 * Rotate a set of coordinates 'Xin' by 'angle' about the *y-axis*
 * (chosen so the slice at phi=0 in output will show the desired tilt)
 */
KOKKOS_INLINE_FUNCTION void rotate_polar(const GReal Xin[GR_DIM], const GReal angle, GReal Xout[GR_DIM], const bool spherical=true)
{
    // Make sure we don't break the trivial case
    if (m::abs(angle) < 1e-20) {
        DLOOP1 Xout[mu] = Xin[mu];
        return;
    }

    // There are clever ways to do this, but this way is more flexible and understandable
    // Like everything else in this file, it is not necessarily very fast

    // Convert to cartesian
    GReal Xin_cart[GR_DIM] = {0};
    if (spherical) {
        Xin_cart[1] = Xin[1]*sin(Xin[2])*cos(Xin[3]);
        Xin_cart[2] = Xin[1]*sin(Xin[2])*sin(Xin[3]);
        Xin_cart[3] = Xin[1]*cos(Xin[2]);
    } else {
        DLOOP1 Xin_cart[mu] = Xin[mu];
    }

    // Rotate about the y axis
    GReal R[GR_DIM][GR_DIM] = {0};
    R[0][0] = 1;
    R[1][1] =  cos(angle);
    R[1][3] =  sin(angle);
    R[2][2] =  1;
    R[3][1] = -sin(angle);
    R[3][3] =  cos(angle);

    GReal Xout_cart[GR_DIM] = {0};
    DLOOP2 Xout_cart[mu] += R[mu][nu] * Xin_cart[nu];

    // Convert back
    if (spherical) {
        Xout[0] = Xin[0];
        // This transformation preserves r, we keep the accurate version
        Xout[1] = Xin[1]; //m::sqrt(Xout_cart[1]*Xout_cart[1] + Xout_cart[2]*Xout_cart[2] + Xout_cart[3]*Xout_cart[3]);
        Xout[2] = acos(Xout_cart[3]/Xout[1]);
        if (m::isnan(Xout[2])) { // GCC has some trouble with ~acos(-1)
            if (Xout_cart[3]/Xout[1] < 0)
                Xout[2] = M_PI;
            else
                Xout[2] = 0.0;
        }
        Xout[3] = atan2(Xout_cart[2], Xout_cart[1]);
    } else {
        DLOOP1 Xout[mu] = Xout_cart[mu];
    }
}

/**
 * Set the transformation matrix dXdx for converting vectors from spherical to Cartesian coordinates,
 * including rotation *and* normalization!
 * There exists an analytic inverse, of course, but we just take numerical inverses because they are easy
 */
KOKKOS_INLINE_FUNCTION void set_dXdx_sph2cart(const GReal X[GR_DIM], GReal dXdx[GR_DIM][GR_DIM])
{
    const GReal &r = X[1], &th = X[2], &phi = X[3];
    dXdx[0][0] = 1;
    dXdx[1][1] = sin(th)*cos(phi);
    dXdx[1][2] = r*cos(th)*cos(phi);
    dXdx[1][3] = -r*sin(th)*sin(phi);
    dXdx[2][1] = sin(th)*sin(phi);
    dXdx[2][2] = r*cos(th)*sin(phi);
    dXdx[2][3] = r*sin(th)*cos(phi);
    dXdx[3][1] = cos(th);
    dXdx[3][2] = -r*sin(th);
    dXdx[3][3] = 0;
}

/**
 * Same as rotate_polar but for vectors: rotate about the y-axis
 */
KOKKOS_INLINE_FUNCTION void rotate_polar_vec(const GReal Xin[GR_DIM], const GReal vin[GR_DIM], const GReal angle,
                                             const GReal Xout[GR_DIM], GReal vout[GR_DIM],
                                             const bool spherical=true)
{
    // Make sure we don't break the trivial case
    if (m::abs(angle) < 1e-20) {
        DLOOP1 vout[mu] = vin[mu];
        return;
    }
    
    // Again, there are clever ways to do this by mapping to a spherical surface, etc
    // But this seems much more straightforward, and this is more flexible in letting us
    // define any rotation or translation we want in Cartesian coordinates.

    // Convert to Cartesian
    GReal vin_cart[GR_DIM] = {0};
    if (spherical) {
        // Note we use the *inverse* matrix here
        GReal dXdx[GR_DIM][GR_DIM] = {0};
        set_dXdx_sph2cart(Xin, dXdx);
        DLOOP2 vin_cart[mu] += dXdx[mu][nu]*vin[nu];
    } else {
        DLOOP1 vin_cart[mu] = vin[mu];
    }

    // Rotate about the y axis
    GReal R[GR_DIM][GR_DIM] = {0};
    R[0][0] = 1;
    R[1][1] = cos(angle);
    R[1][3] = sin(angle);
    R[2][2] = 1;
    R[3][1] = -sin(angle);
    R[3][3] = cos(angle);

    GReal vout_cart[GR_DIM] = {0};
    DLOOP2 vout_cart[mu] += R[mu][nu] * vin_cart[nu];

    // Convert back
    if (spherical) {
        // We have to clear vout since it's passed in
        GReal dXdx[GR_DIM][GR_DIM] = {0}, dxdX[GR_DIM][GR_DIM] = {0};
        set_dXdx_sph2cart(Xout, dXdx);
        invert(&dXdx[0][0], &dxdX[0][0]);
        DLOOP1 vout[mu] = 0;
        DLOOP2 vout[mu] += dxdX[mu][nu]*vout_cart[nu];
    } else {
        DLOOP1 vout[mu] = vout_cart[mu];
    }
}

/**
 * Set time component for a consistent 4-velocity given a 3-velocity
 */
KOKKOS_INLINE_FUNCTION void set_ut(const Real gcov[GR_DIM][GR_DIM], Real ucon[GR_DIM])
{
    Real AA, BB, CC;

    AA = gcov[0][0];
    BB = 2. * (gcov[0][1] * ucon[1] +
               gcov[0][2] * ucon[2] +
               gcov[0][3] * ucon[3]);
    CC = 1. + gcov[1][1] * ucon[1] * ucon[1] +
         gcov[2][2] * ucon[2] * ucon[2] +
         gcov[3][3] * ucon[3] * ucon[3] +
         2. * (gcov[1][2] * ucon[1] * ucon[2] +
               gcov[1][3] * ucon[1] * ucon[3] +
               gcov[2][3] * ucon[2] * ucon[3]);

    Real discr = BB * BB - 4. * AA * CC;
    ucon[0] = (-BB - m::sqrt(discr)) / (2. * AA);
}

/**
 * Make primitive velocities u-twiddle out of 4-velocity.  See Gammie '04
 * 
 * This function and set_ut together can turn any desired 3-velocity into a
 * form usable to initialize uvec in KHARMA; see bondi.hpp for usage.
 */
KOKKOS_INLINE_FUNCTION void fourvel_to_prim(const Real gcon[GR_DIM][GR_DIM], const Real ucon[GR_DIM], Real u_prim[NVEC])
{
    Real alpha2 = -1.0 / gcon[0][0];
    // Note gamma/alpha is ucon[0]
    u_prim[0] = ucon[1] + ucon[0] * alpha2 * gcon[0][1];
    u_prim[1] = ucon[2] + ucon[0] * alpha2 * gcon[0][2];
    u_prim[2] = ucon[3] + ucon[0] * alpha2 * gcon[0][3];
}

/**
 * Define metric correction to KS Kerr metric for modified gravity metrics.
 * 
 * The dCS and EdGB metrics are written as exp(M.T) @ gcov_kerr_ks(f(r,th)) @ exp(M),
 * where M(r,th) is the metric correction and f(r,th) is the redefinition of "radius".
 * For GR, f(r,th) == r, but for dCS and EdGB it is rescaled so that radius means
 * radius far away from the black hole. This has to do with the way the metric is
 * transformed from BL to KS by Yiqi.
 * 
 * Here we define the metric correction for dCS.
*/
KOKKOS_INLINE_FUNCTION void metric_correction_dcs(const Real a, const Real zeta, const GReal r, const GReal th, Real gcov_corr[GR_DIM][GR_DIM])
{
    // Get horizon since correction is radius-dependent
    const Real r_eh = (1 + m::sqrt(1 - m::pow(a,2))) + ((72185 * m::pow(a,2) / 96096) + (189049321 * m::pow(a,4) / 931170240)) * zeta;

    // Get cos(th)
    const GReal cth = m::cos(th);

    // Compute correction
    if (r < r_eh) {
        gcov_corr[0][0] = r * (-0.10186551426782008 + 0.2044681509464842*m::pow(cth,2) -
                    0.047577127121860865 * m::pow(cth,4) + r * (0.045274692542072226 -
                    0.09476108773453298 * m::pow(cth,2) + 0.024336657157255144 * m::pow(cth,4)));
        gcov_corr[0][1] = r * (0.045412209337770834 + 0.6948770087146782*m::pow(cth,2) -
                    0.22740730325285005*m::pow(cth,4) + r*(-0.018243440528231694 -
                    0.3195382642179189*m::pow(cth,2) + 0.11136116213415795*m::pow(cth,4)));
        gcov_corr[0][3] = r*(0.9245158222592134 - 1.5540867827958567*m::pow(cth,2) +
                    0.7817181596052425*m::pow(cth,4) - 0.15214719906859933*m::pow(cth,6) +
                    r*(-0.39831379601518213 + 0.6873438631794827*m::pow(cth,2) -
                    0.3629692425201996*m::pow(cth,4) + 0.07393917535589899*m::pow(cth,6)));
        gcov_corr[1][1] = r*(0.016459528849326405 - 0.0877649193211191*m::pow(cth,2) +
                    0.013672933340417168*m::pow(cth,4) + r*(-0.005981133205508223 +
                    0.0397302074948605*m::pow(cth,2) - 0.008047529144189837*m::pow(cth,4)));
        gcov_corr[1][3] = r*(0.08767766008723328 - 0.03217485989886997*m::pow(cth,2) -
                    0.04878094127510386*m::pow(cth,4) - 0.006721858913259437*m::pow(cth,6) +
                    r*(-0.027339833822029428 + 0.0032782178868501132*m::pow(cth,2) +
                    0.021849341728100158*m::pow(cth,4) + 0.002212274207079146*m::pow(cth,6)));
        gcov_corr[2][2] = r*(-0.005891088388619743 - 0.08213157970162656*m::pow(cth,2) +
                    0.04869729617963716*m::pow(cth,4) + r*(0.0012258296731578948 +
                    0.04059152589803598*m::pow(cth,2) - 0.022648614125266385*m::pow(cth,4)));
        gcov_corr[3][3] = r*(0.0779005273559132 - 0.18750384045886062*m::pow(cth,2) +
                    0.07027794119233827*m::pow(cth,4) + r*(-0.03758218403345795 +
                    0.08957607617157008*m::pow(cth,2) - 0.03282515069218463*m::pow(cth,4)));
    }
    else {
        gcov_corr[0][0] = (-3.764802631578948*m::pow(cth,4) + r*(0.20737746740621577*m::pow(cth,2) -1.2411199826309292*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.004397491677732722 + 2.275428565488274*m::pow(cth,2) - 0.25329799985773566*m::pow(cth,4)) + 
                    m::pow(r,8)*(-0.019512591789299767 + 0.02926262957652467*m::pow(cth,2) + 0.010298255966503736*m::pow(cth,4)) + 
                    m::pow(r,6)*(-0.07871641654069922 + 0.018403555854965288*m::pow(cth,2) + 0.011519474758872082*m::pow(cth,4)) + 
                    m::pow(r,5)*(-0.1601638195572365 + 0.011472388631105742*m::pow(cth,2) + 0.01770559700437244*m::pow(cth,4)) + 
                    m::pow(r,7)*(-0.03924720003976625 + 0.02044660472170251*m::pow(cth,2) + 0.02059651193300747*m::pow(cth,4)) + 
                    m::pow(r,4)*(-0.2062015656196536 + 0.3715397076125874*m::pow(cth,2) + 0.030607798809265254*m::pow(cth,4)) + 
                    m::pow(r,3)*(-0.19117702886935253 + 1.0350857068689334*m::pow(cth,2) + 0.07147163859562965*m::pow(cth,4)))/m::pow(r,10);
        gcov_corr[0][1] = (-5.950657894736843*m::pow(cth,4) + r*(0.20903884720779903*m::pow(cth,2) - 4.565478341021519*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.030794723900581383 + 3.383010068231072*m::pow(cth,2) - 2.1399936508080724*m::pow(cth,4)) + 
                    m::pow(r,3)*(-0.1496858869925613 + 3.1029202972978545*m::pow(cth,2) - 0.5753275196143051*m::pow(cth,4)) + 
                    m::pow(r,4)*(-0.09506265415433443 + 1.8595028809086818*m::pow(cth,2) - 0.07821619255002231*m::pow(cth,4)) + 
                    m::pow(r,8)*(0.03680861200741224 + 0.05852525915304934*m::pow(cth,2) + 0.02059651193300747*m::pow(cth,4)) + 
                    m::pow(r,5)*(-0.012651581837955407 + 0.7163988910916779*m::pow(cth,2) + 0.02211495126091475*m::pow(cth,4)) + 
                    m::pow(r,6)*(0.07232253531051423 + 0.22249925201839119*m::pow(cth,2) + 0.023038949517744163*m::pow(cth,4)) + 
                    m::pow(r,7)*(0.05973819193995381 + 0.06827691730484674*m::pow(cth,2) + 0.04119302386601494*m::pow(cth,4)))/m::pow(r,10);
        gcov_corr[0][3] = (2.9753289473684217*m::pow(cth,4) - 2.9753289473684217*m::pow(cth,6) + 
                    r*(-0.3301529518690068*m::pow(cth,2) + 3.1596148319314237*m::pow(cth,4) - 2.8294618800624165*m::pow(cth,6)) + 
                    m::pow(r,2)*(0.008578113558176309 - 1.9548229788672007*m::pow(cth,2) + 3.4050066693919323*m::pow(cth,4) -1.458761804082908*m::pow(cth,6)) + 
                    m::pow(r,3)*(0.3315641345145541 - 2.9057414924554914*m::pow(cth,2) + 3.0361396822695923*m::pow(cth,4) - 0.461962324328655*m::pow(cth,6)) + 
                    m::pow(r,4)*(0.39370132858720774 - 2.167692591796219*m::pow(cth,2) + 1.857264696904211*m::pow(cth,4) - 0.08327343369519986*m::pow(cth,6)) + 
                    m::pow(r,5)*(0.9987748278913282 - 1.840302530914861*m::pow(cth,2) + 0.8372567620733599*m::pow(cth,4) + 0.004270940950172779*m::pow(cth,6)) + 
                    m::pow(r,8)*(0.3226400239145535 - 0.3519026534910782*m::pow(cth,2) + 0.018964373610020936*m::pow(cth,4) + 0.010298255966503736*m::pow(cth,6)) + 
                    m::pow(r,6)*(0.9441539653309928 - 1.222496704008794*m::pow(cth,2) + 0.2668232639189292*m::pow(cth,4) + 0.011519474758872082*m::pow(cth,6)) + 
                    m::pow(r,7)*(0.7174242038941537 - 0.7898827927584792*m::pow(cth,2) + 0.05186207693131773*m::pow(cth,4) + 0.02059651193300747*m::pow(cth,6)))/m::pow(r,10);
        gcov_corr[1][1] = (2.185855263157895*m::pow(cth,4) + m::pow(r,3)*(-0.03416402565299624 - 0.48939094433013075*m::pow(cth,2) - 0.0694329299472212*m::pow(cth,4)) + 
                    m::pow(r,4)*(-0.029868113792055086 - 0.12779951461553218*m::pow(cth,2) - 0.049418835957736065*m::pow(cth,4)) + 
                    m::pow(r,5)*(-0.015665261737899675 + 0.07028917056179607*m::pow(cth,2) - 0.029196090158295766*m::pow(cth,4)) + 
                    m::pow(r,7)*(0.006093802057191114 + 0.0060208618032693405*m::pow(cth,2) - 0.02059651193300747*m::pow(cth,4)) + 
                    m::pow(r,6)*(-0.008285434801374236 + 0.038395483503946296*m::pow(cth,2) - 0.014392098047352914*m::pow(cth,4)) + 
                    m::pow(r,8)*(0.040556426561124084 - 0.05364943007715063*m::pow(cth,2) - 0.010298255966503736*m::pow(cth,4)) + 
                    m::pow(r,2)*(-0.026397232222848656 - 1.1827213707401538*m::pow(cth,2) + 0.16206700830456913*m::pow(cth,4)) + 
                    r*(-0.0016613798015832518*m::pow(cth,2) + 0.7709044110221688*m::pow(cth,4)))/m::pow(r,10);
        gcov_corr[1][3] = (0.11281676413255362*m::pow(cth,2) - 0.22563352826510724*m::pow(cth,4) + 0.11281676413255362*m::pow(cth,6) + 
                    m::pow(r,4)*(-0.07720323170405753 + 0.14576209104746674*m::pow(cth,2) - 0.05991448698276085*m::pow(cth,4) - 0.008644372360648345*m::pow(cth,6)) + 
                    m::pow(r,5)*(-0.03212746825771086 + 0.056741178976613044*m::pow(cth,2) - 0.017099953180093504*m::pow(cth,4) - 0.0075137575388086785*m::pow(cth,6)) + 
                    m::pow(r,6)*(-0.010353864572767907 + 0.017827860455817794*m::pow(cth,2) - 0.002121331434403294*m::pow(cth,4) - 0.005352664448646592*m::pow(cth,6)) + 
                    m::pow(r,8)*(0.141807420167977 - 0.1271761053797147*m::pow(cth,2) - 0.009482186805010468*m::pow(cth,4) - 0.005149127983251868*m::pow(cth,6)) + 
                    m::pow(r,7)*(0.013633334595171994 - 0.011195420057222646*m::pow(cth,2) + 0.002711213445302516*m::pow(cth,4) - 0.005149127983251868*m::pow(cth,6)) + 
                    m::pow(r,3)*(-0.10714813519542485 + 0.21170106044976475*m::pow(cth,2) - 0.10195771531325491*m::pow(cth,4) - 0.002595209941084969*m::pow(cth,6)) + 
                    m::pow(r,2)*(-0.10346977678554554 + 0.234990701941892*m::pow(cth,2) - 0.15957207352714736*m::pow(cth,4) + 0.02805114837080091*m::pow(cth,6)) + 
                    r*(0.0012855403069909927 + 0.07377655505803313*m::pow(cth,2) - 0.15140973103703922*m::pow(cth,4) + 0.07634763567201511*m::pow(cth,6)))/m::pow(r,9);
        gcov_corr[2][2] = (2.185855263157895*m::pow(cth,4) + m::pow(r,8)*(0.0033429657288016426 - 0.024386800500625968*m::pow(cth,2)) + 
                    m::pow(r,3)*(0.006730793813251797 - 0.5711805832626268*m::pow(cth,2) - 0.028538110480973156*m::pow(cth,4)) + 
                    m::pow(r,7)*(0.006907947918770011 - 0.005498612955602743*m::pow(cth,2) - 0.009891183035714287*m::pow(cth,4)) + 
                    m::pow(r,4)*(0.010639314047962048 - 0.2088143702955665*m::pow(cth,2) - 0.008911408117718933*m::pow(cth,4)) + 
                    m::pow(r,5)*(0.013113976137320347 + 0.012730694811356026*m::pow(cth,2) - 0.00041685228307574617*m::pow(cth,4)) + 
                    m::pow(r,6)*(0.009614703564723954 + 0.0025952067717499227*m::pow(cth,2) + 0.0035080403187452754*m::pow(cth,4)) + 
                    m::pow(r,9)*(-0.019512591789299767 + 0.02926262957652467*m::pow(cth,2) + 0.010298255966503736*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.000149323899600326 - 1.2358144829850521*m::pow(cth,2) + 0.18861356442701813*m::pow(cth,4)) + 
                    r*(-0.0016613798015832518*m::pow(cth,2) + 0.7709044110221688*m::pow(cth,4)))/m::pow(r,10);
        gcov_corr[3][3] = (-0.021043834771824328*m::pow(r,8) + 2.185855263157895*m::pow(cth,4) + 
                    m::pow(r,7)*(0.03337541444374187 - 0.031966079480574595*m::pow(cth,2) - 0.009891183035714287*m::pow(cth,4)) + 
                    m::pow(r,6)*(0.09176977844624386 - 0.08243249139825083*m::pow(cth,2) + 0.006380663607226107*m::pow(cth,4)) + 
                    m::pow(r,9)*(-0.019512591789299767 + 0.02926262957652467*m::pow(cth,2) + 0.010298255966503736*m::pow(cth,4)) + 
                    m::pow(r,5)*(0.1962996774206555 - 0.18194549962590242*m::pow(cth,2) + 0.011073640870847584*m::pow(cth,4)) + 
                    m::pow(r,4)*(0.2654430122788289 - 0.4989350764842804*m::pow(cth,2) + 0.026405599840128257*m::pow(cth,4)) + 
                    m::pow(r,3)*(0.2545651668505909 - 0.9160120725078158*m::pow(cth,2) + 0.0684590057268767*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.024124799408067325 - 1.4390317859599984*m::pow(cth,2) + 0.36785539189349736*m::pow(cth,4)) + 
                    r*(-0.2272949080666905*m::pow(cth,2) + 0.9965379392872759*m::pow(cth,4)))/m::pow(r,10);
    }
}

/**
 * And now the correction for EdGB.
*/
KOKKOS_INLINE_FUNCTION void metric_correction_edgb(const Real a, const Real zeta, const GReal r, const GReal th, Real gcov_corr[GR_DIM][GR_DIM])
{
    // Get horizon since correction is radius-dependent
    const Real r_eh = (1 + m::sqrt(1 - m::pow(a,2))) + (-(1117 / 2310) - (3697 * m::pow(a,2) / 5850) + (211270219 * m::pow(a,4) / 1018467450)) * zeta;

    // Get cos(th)
    const GReal cth = m::cos(th);

    // Compute correction
    if (r < r_eh) {
        gcov_corr[0][0] = r * (0.09881663310865521 - 0.06589959486016632*m::pow(cth,2) +
                    0.09483604081290098*m::pow(cth,4) + r * (-0.0508082639442996 +
                    0.03603450867902285*m::pow(cth,2) - 0.04755871830571102*m::pow(cth,4)));
        gcov_corr[0][1] = r * (1.1660157755065736 - 0.923446854438797*m::pow(cth,2) +
                    0.4445882546481567*m::pow(cth,4) + r * (-0.5358018673193612 +
                    0.45031722475873553*m::pow(cth,2) - 0.22156152685943775*m::pow(cth,4)));
        gcov_corr[0][3] = r * (0.26629173567914655 - 0.012224165629961845*m::pow(cth,2) -
                    0.4504719467791443*m::pow(cth,4) + 0.19640437672995958*m::pow(cth,6) +
                    r * (-0.117216423840705 - 0.007938446653934146*m::pow(cth,2) +
                    0.22304710326845398*m::pow(cth,4) - 0.09789223277381481*m::pow(cth,6)));
        gcov_corr[1][1] = r * (-0.21770433903942524 + 0.18701808101289774*m::pow(cth,2) -
                    0.09675918672289041*m::pow(cth,4) + r * (0.1031193586421407 -
                    0.09134547021704825*m::pow(cth,2) + 0.04792976859542179*m::pow(cth,4)));
        gcov_corr[1][3] = r * (-0.05211790724575234 + 0.10282977052005256*m::pow(cth,2) -
                    0.04878574904287365*m::pow(cth,4) - 0.0019261142314265672*m::pow(cth,6) +
                    r * (0.024949669974718935 - 0.04977020053092391*m::pow(cth,2) +
                    0.024521350206178745*m::pow(cth,4) + 0.0002991803500262339*m::pow(cth,6)));
        gcov_corr[2][2] = r * (-0.2456694322274893 + 0.11802793355278088*m::pow(cth,2) -
                    0.06388746774868627*m::pow(cth,4) + r * (0.11088665306540471 -
                    0.06020804162310437*m::pow(cth,2) + 0.033009174906014245*m::pow(cth,4)));
        gcov_corr[3][3] = r * (-0.10427504873012293 - 0.05238594045593648*m::pow(cth,2) -
                    0.034867977237335264*m::pow(cth,4) + r * (0.04443039634840591 +
                    0.02057044808324951*m::pow(cth,2) + 0.01868694191665917*m::pow(cth,4)));
    }
    else {
        gcov_corr[0][0] = (4.291164274322169*m::pow(cth,4) + m::pow(r,3)*(-0.2109264049823971 - 0.727527995711403*m::pow(cth,2) - 0.2289082277890738*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.003213989800925891 - 2.8729270695242883*m::pow(cth,2) - 0.07200152505487534*m::pow(cth,4)) + 
                    m::pow(r,4)*(0.7091703981563269 + 0.03414119433807664*m::pow(cth,2) - 0.039940983006676915*m::pow(cth,4)) + 
                    m::pow(r,8)*(-0.056750235572314435 - 0.010559999053525839*m::pow(cth,2) + 0.00445260686733901*m::pow(cth,4)) + 
                    m::pow(r,7)*(-0.1071071860501588 + 0.02587578792636418*m::pow(cth,2) + 0.00890521373467802*m::pow(cth,4)) + 
                    m::pow(r,5)*(0.3036965654525125 + 0.35876703857586284*m::pow(cth,2) + 0.016769533525680457*m::pow(cth,4)) + 
                    m::pow(r,6)*(0.16405391299415248 + 0.14311498384751617*m::pow(cth,2) + 0.02155839824857682*m::pow(cth,4)) + 
                    r*(0.25435376653421765*m::pow(cth,2) + 0.8575513355876029*m::pow(cth,4)))/m::pow(r,10);
        gcov_corr[0][1] = (6.788644338118022*m::pow(cth,4) + m::pow(r,8)*(0.1847953257927588 - 0.021119998107051677*m::pow(cth,2) + 0.00890521373467802*m::pow(cth,4)) + 
                    m::pow(r,7)*(0.5012698809657435 - 0.017972840146824282*m::pow(cth,2) + 0.01781042746935604*m::pow(cth,4)) + 
                    m::pow(r,6)*(1.1678820823398104 - 0.08665926837000991*m::pow(cth,2) + 0.04311679649715364*m::pow(cth,4)) + 
                    m::pow(r,5)*(1.3968855321134972 - 0.5214758738558606*m::pow(cth,2) + 0.05509367847082749*m::pow(cth,4)) + 
                    m::pow(r,4)*(1.2803751437389894 - 1.9801727988754083*m::pow(cth,2) + 0.11100987467318106*m::pow(cth,4)) + 
                    m::pow(r,3)*(-0.19674673106180973 - 3.631607281837495*m::pow(cth,2) + 0.4584931004015606*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.027577060783827698 - 4.600749569082939*m::pow(cth,2) + 1.927572690712894*m::pow(cth,4)) + 
                    r*(0.30636166840678114*m::pow(cth,2) + 4.428944923865755*m::pow(cth,4)))/m::pow(r,10);
        gcov_corr[0][3] = (-3.394322169059011*m::pow(cth,4) + 3.394322169059011*m::pow(cth,6) + 
                    m::pow(r,8)*(0.12158613671972822 - 0.11102613766620238*m::pow(cth,2) - 0.01501260592086485*m::pow(cth,4) + 0.00445260686733901*m::pow(cth,6)) + 
                    m::pow(r,7)*(0.26003646165384053 - 0.27131451731494105*m::pow(cth,2) + 0.0023728419264224393*m::pow(cth,4) + 0.00890521373467802*m::pow(cth,6)) + 
                    m::pow(r,6)*(0.15034750415271614 - 0.18684369578994947*m::pow(cth,2) + 0.014937793388656521*m::pow(cth,4) + 0.02155839824857682*m::pow(cth,6)) + 
                    m::pow(r,5)*(0.13078293898666954 - 0.09663728782559695*m::pow(cth,2) - 0.05834564172670751*m::pow(cth,4) + 0.024199990565634915*m::pow(cth,6)) + 
                    m::pow(r,4)*(-0.32370570453399117 + 0.8745571659528123*m::pow(cth,2) - 0.5881531198749078*m::pow(cth,4) + 0.0373016584560868*m::pow(cth,6)) + 
                    m::pow(r,3)*(0.3444360776914127 + 0.8053120998423906*m::pow(cth,2) - 1.3136486338971856*m::pow(cth,4) + 0.16390045636338207*m::pow(cth,6)) + 
                    m::pow(r,2)*(0.008855468153400483 + 2.0874440102442646*m::pow(cth,2) - 2.8957358880860653*m::pow(cth,4) + 0.7994364096883999*m::pow(cth,6)) + 
                    r*(-0.3627447439778267*m::pow(cth,2) - 1.5398496895507068*m::pow(cth,4) + 1.9025944335285334*m::pow(cth,6)))/m::pow(r,10);
        gcov_corr[1][1] = (-2.497480063795853*m::pow(cth,4) + r*(-0.052007901872563524*m::pow(cth,2) - 0.7819199040676263*m::pow(cth,4)) + 
                    m::pow(r,2)*(-0.02436307098290181 + 1.7686911221641155*m::pow(cth,2) - 0.17276456721199093*m::pow(cth,4)) + 
                    m::pow(r,5)*(-0.33186954053463574 - 0.04713633522918607*m::pow(cth,2) - 0.028384689943681703*m::pow(cth,4)) + 
                    m::pow(r,6)*(-0.22196976327416232 - 0.025450775001065937*m::pow(cth,2) - 0.02446218735307713*m::pow(cth,4)) + 
                    m::pow(r,4)*(-0.5497817903708182 + 0.27738723097161644*m::pow(cth,2) - 0.010676666072864816*m::pow(cth,4)) + 
                    m::pow(r,7)*(-0.017878650978800745 + 0.01361897560289736*m::pow(cth,2) - 0.00890521373467802*m::pow(cth,4)) + 
                    m::pow(r,8)*(0.0018711207888258974 + 0.022693577087165375*m::pow(cth,2) - 0.00445260686733901*m::pow(cth,4)) + 
                    m::pow(r,3)*(-0.011217539355069367 + 0.7538723496096127*m::pow(cth,2) + 0.04135389749183582*m::pow(cth,4)))/m::pow(r,10);
        gcov_corr[1][3] = (0.10478195488721805*m::pow(cth,2) - 0.2095639097744361*m::pow(cth,4) + 0.10478195488721805*m::pow(cth,6) + 
                    m::pow(r,4)*(-0.07454768676004499 + 0.14076463686934715*m::pow(cth,2) - 0.05788621345855937*m::pow(cth,4) - 0.008330736650742812*m::pow(cth,6)) + 
                    m::pow(r,5)*(-0.03306014626262768 + 0.05858256643029559*m::pow(cth,2) - 0.017984694072708127*m::pow(cth,4) - 0.007537726094959777*m::pow(cth,6)) + 
                    m::pow(r,6)*(-0.014357782202486562 + 0.023325964842828922*m::pow(cth,2) - 0.003890913976466552*m::pow(cth,4) - 0.005077268663875806*m::pow(cth,6)) + 
                    m::pow(r,3)*(-0.10234633569089166 + 0.2020710498436015*m::pow(cth,2) - 0.09710309261452797*m::pow(cth,4) - 0.0026216215381818465*m::pow(cth,6)) + 
                    m::pow(r,7)*(-0.0033719892357476668 + 0.004158778725804516*m::pow(cth,2) + 0.0014395139436126551*m::pow(cth,4) - 0.002226303433669505*m::pow(cth,6)) + 
                    m::pow(r,8)*(0.004042832787549678 - 0.009322832314312596*m::pow(cth,2) + 0.007506302960432425*m::pow(cth,4) - 0.002226303433669505*m::pow(cth,6)) + 
                    m::pow(r,2)*(-0.09979646693535539 + 0.22353023569085523*m::pow(cth,2) - 0.14767107057564424*m::pow(cth,4) + 0.02393730182014443*m::pow(cth,6)) + 
                    r*(0.0015552425331560671 + 0.06452527650736407*m::pow(cth,2) - 0.13371628061419635*m::pow(cth,4) + 0.0676357615736762*m::pow(cth,6)))/m::pow(r,9);
        gcov_corr[2][2] = (-2.497480063795853*m::pow(cth,4) + 
                    m::pow(r,8)*(-0.00041852575015099877 + 0.012133578033639538*m::pow(cth,2)) + 
                    r*(-0.052007901872563524*m::pow(cth,2) - 0.7819199040676263*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.0013914126287246588 + 1.7171821549408626*m::pow(cth,2) - 0.14701008360036444*m::pow(cth,4)) + 
                    m::pow(r,6)*(-0.20399052197974238 - 0.061409257589905675*m::pow(cth,2) - 0.006482946058657265*m::pow(cth,4)) + 
                    m::pow(r,5)*(-0.30359291081514894 - 0.10368959466815979*m::pow(cth,2) - 0.00010806022419483593*m::pow(cth,4)) + 
                    m::pow(r,7)*(-0.0072303365947720705 - 0.007939422645679456*m::pow(cth,2) + 0.001249323593073593*m::pow(cth,4)) + 
                    m::pow(r,9)*(-0.056750235572314435 - 0.010559999053525839*m::pow(cth,2) + 0.00445260686733901*m::pow(cth,4)) + 
                    m::pow(r,4)*(-0.5106316084099219 + 0.19908686704982395*m::pow(cth,2) + 0.028473515888031427*m::pow(cth,4)) + 
                    m::pow(r,3)*(0.028197926131824688 + 0.6750414186358245*m::pow(cth,2) + 0.08076936297872989*m::pow(cth,4))) / m::pow(r,10);
        gcov_corr[3][3] = (0.011715052283488538*m::pow(r,8) - 2.497480063795853*m::pow(cth,4) + 
                    r*(-0.2615718116469996*m::pow(cth,2) - 0.5723559942931902*m::pow(cth,4)) + 
                    m::pow(r,6)*(-0.11989098816006717 - 0.1484125805140812*m::pow(cth,2) - 0.003579156954156954*m::pow(cth,4)) + 
                    m::pow(r,7)*(0.032264426934489464 - 0.04743418617494099*m::pow(cth,2) + 0.001249323593073593*m::pow(cth,4)) + 
                    m::pow(r,9)*(-0.056750235572314435 - 0.010559999053525839*m::pow(cth,2) + 0.00445260686733901*m::pow(cth,4)) + 
                    m::pow(r,5)*(-0.12622090757557206 - 0.29267675432573786*m::pow(cth,2) + 0.011507096193806405*m::pow(cth,4)) + 
                    m::pow(r,2)*(0.02403541117403899 + 1.5335121496365696*m::pow(cth,2) + 0.014015923158614446*m::pow(cth,4)) + 
                    m::pow(r,4)*(-0.2667887550672424 - 0.07866292517738821*m::pow(cth,2) + 0.06238045477256399*m::pow(cth,4)) + 
                    m::pow(r,3)*(0.2672063254894295 + 0.3487429501510368*m::pow(cth,2) + 0.1680594321059128*m::pow(cth,4))) / m::pow(r,10);
    }
}

/**
 * Define the radius correction for dCS.
*/
KOKKOS_INLINE_FUNCTION void radius_correction_dcs(const Real a, const Real zeta, const GReal r, const GReal th, Real &r_new)
{
    // Get horizon since correction is radius-dependent
    const Real r_eh = (1 + m::sqrt(1 - m::pow(a,2))) + ((72185 * m::pow(a,2) / 96096) + (189049321 * m::pow(a,4) / 931170240)) * zeta;

    // Get cos(th)
    const GReal cth = m::cos(th);

    // Compute correction
    if (r < r_eh) {
        r_new = r * (1.020691229419899 - 0.031030208008011838*m::pow(cth,2) -
            0.01092031131121305*m::pow(cth,4) + r * (-0.005485265354929308 +
            0.008226138789940869*m::pow(cth,2) + 0.0028949853140593047*m::pow(cth,4)));
    }
    else {
        r_new = 0.01951259178929976 + r - 0.02926262957652467*m::pow(cth,2) - 0.010298255966503736*m::pow(cth,4);
    }
}

/**
 * Define the radius correction for EdGB.
*/
KOKKOS_INLINE_FUNCTION void radius_correction_edgb(const Real a, const Real zeta, const GReal r, const GReal th, Real &r_new)
{
    // Get horizon since correction is radius-dependent
    const Real r_eh =(1 + m::sqrt(1 - m::pow(a,2))) + (-(1117 / 2310) - (3697 * m::pow(a,2) / 5850) + (211270219 * m::pow(a,4) / 1018467450)) * zeta;

    // Get cos(th)
    const GReal cth = m::cos(th);

    // Compute correction
    if (r < r_eh) {
        r_new = r * (1.0629450418611497 + 0.01171271935304809*m::pow(cth,2) - 
            0.00493864955500953*m::pow(cth,4) + r * (-0.01745401695228727 - 
            0.0032478173991280367*m::pow(cth,2) + 0.0013694370597876334*m::pow(cth,4)));
    }
    else {
        r_new = 0.056750235572314435 + r + 0.01055999905352584*m::pow(cth,2) - 0.00445260686733901*m::pow(cth,4);
    }
}

/**
 * Kerr metric in Kerr-Schild coordinates that accepts a "radial" coordinate and theta
 * at grid zone and computes the metric components
*/
KOKKOS_INLINE_FUNCTION void gcov_kerr_gr(const GReal r, const GReal th, const GReal a, Real gcov[GR_DIM][GR_DIM])
{
    
    const GReal cth = m::cos(th);
    const GReal sth = m::sin(th);
    const GReal sin2 = sth*sth;
    const GReal rho2 = r*r + a*a*cth*cth;

    gcov[0][0] = -1. + 2.*r/rho2;
    gcov[0][1] = 2.*r/rho2;
    gcov[0][2] = 0.;
    gcov[0][3] = -2.*a*r*sin2/rho2;

    gcov[1][0] = 2.*r/rho2;
    gcov[1][1] = 1. + 2.*r/rho2;
    gcov[1][2] = 0.;
    gcov[1][3] = -a*sin2*(1. + 2.*r/rho2);

    gcov[2][0] = 0.;
    gcov[2][1] = 0.;
    gcov[2][2] = rho2;
    gcov[2][3] = 0.;

    gcov[3][0] = -2.*a*r*sin2/rho2;
    gcov[3][1] = -a*sin2*(1. + 2.*r/rho2);
    gcov[3][2] = 0.;
    gcov[3][3] = sin2*(rho2 + a*a*sin2*(1. + 2.*r/rho2));
}