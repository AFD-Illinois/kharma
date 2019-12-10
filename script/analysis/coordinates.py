
import numpy as np

from defs import Met

def coord_to_KS(X, mtype):
  pass

def vec_to_KS(vec, X, mtype):
    """Translate a vector from """
    return np.einsum("i...,ij...", vec, dxdX_KS_to(X, mtype))

def dxdX_to_KS(X, mtype, met_params, koral_rad=False):
    """Get transformation matrix to Kerr-Schild from several different coordinate systems.
    X should be given in Kerr-Schild coordinates."""

    # Play some index games to get the inverse from numpy
    ks_t = np.einsum("ij...->...ij", dxdX_KS_to(X, mtype, met_params, koral_rad))
    return np.einsum("...ij->ij...", np.linalg.inv(ks_t))

def dxdX_KS_to(X, mtype, met_params, koral_rad=False):
    """Get transformation to Kerr-Schild coordinates from another coordinate system.
    X should be given in native coordinates"""

    dxdX = np.zeros((4, 4, *X.shape[1:]))
    
    dxdX[0,0] = 1 # We don't yet use anything _that_ exotic
    if mtype == Met.MINKOWSKI:
        # Handle Minkowski spacetime separately
        raise ValueError("Cannot translate spacetimes!")
    elif mtype == Met.MKS:
        hslope = met_params['hslope']
        dxdX[1, 1] = np.exp(X[1])
        dxdX[2, 2] = np.pi - (hslope - 1.) * np.pi * np.cos(2. * np.pi * X[2])
        dxdX[3, 3] = 1
    elif mtype == Met.FMKS:
        dxdX[1, 1] = np.exp(X[1])
        hslope = met_params['hslope']
        mks_smooth, poly_norm, poly_xt, poly_alpha = met_params['mks_smooth'], met_params['poly_norm'], met_params['poly_xt'], met_params['poly_alpha']
        startx1 = met_params['startx1']
        
        dxdX[2, 1] = -np.exp(mks_smooth * (startx1 - X[1])) * mks_smooth * (np.pi / 2. -
                                                                                   np.pi * X[2] + poly_norm * (
                                                                                               2. * X[2] - 1.) * (1 + (
                    np.power((-1. + 2 * X[2]) / poly_xt, poly_alpha)) / (1 + poly_alpha)) -
                                                                                   1. / 2. * (1. - hslope) * np.sin(
                    2. * np.pi * X[2]))
        dxdX[2, 2] = np.pi + (1. - hslope) * np.pi * np.cos(2. * np.pi * X[2]) + np.exp(
            mks_smooth * (startx1 - X[1])) * (-np.pi +
                                                     2. * poly_norm * (1. + np.power((2. * X[2] - 1.) / poly_xt,
                                                                                     poly_alpha) / (poly_alpha + 1.)) +
                                                     (2. * poly_alpha * poly_norm * (2. * X[2] - 1.) * np.power(
                                                         (2. * X[2] - 1.) / poly_xt, poly_alpha - 1.)) / (
                                                                 (1. + poly_alpha) * poly_xt) -
                                                     (1. - hslope) * np.pi * np.cos(2. * np.pi * X[2]))
        dxdX[3, 3] = 1.
    elif mtype == Met.MKS3:
        # TODO take these as params, bring this in line with above w.r.t function name
        if koral_rad:
            R0=-1.35; H0=0.7; MY1=0.002; MY2=0.02; MP0=1.3
        else:
            # MAD
            #R0=0; H0=0.6; MY1=0.0025; MY2=0.025; MP0=1.2
            #SANE
            R0=-2; H0=0.6; MY1=0.0025; MY2=0.025; MP0=1.2

        dxdX[1,1] = 1./(X[1] - R0)
        dxdX[2, 1] = -((np.power(2, 1 + MP0) * np.power(X[1], -1 + MP0) * MP0 * (MY1 - MY2) * np.arctan(((-2 * X[2] + np.pi) * np.tan((H0 * np.pi) / 2.)) / np.pi)) / 
                       (H0 * np.power(np.power(X[1], MP0) * (1 - 2 * MY1) + np.power(2, 1 + MP0) * (MY1 - MY2), 2) * np.pi))
        dxdX[2, 2] = ( (-2 * np.power(X[1], MP0) * np.tan((H0 * np.pi) / 2.)) / 
                       (H0 * (np.power(X[1], MP0) * (-1 + 2 * MY1) + 
                              np.power(2, 1 + MP0) * (-MY1 + MY2)) * np.pi**2 * (1 + (np.power(-2 * X[2] + np.pi, 2) * np.power(np.tan((H0 * np.pi) / 2.), 2)) / 
                                                                                                             np.pi**2)))
        dxdX[3,3] = 1.
    elif mtype == Met.EKS:
        dxdX[1,1] = 1. / X[1]
        dxdX[2,2] = 1. / np.pi
        dxdX[3,3] = 1.
    else:
        raise ValueError("Unsupported metric type {}!".format(mtype))

    return dxdX
