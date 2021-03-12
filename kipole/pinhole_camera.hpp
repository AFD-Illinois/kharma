

#include <parthenon/parthenon.hpp>

#include "tetrad.hpp"

namespace PinholeCamera
{

/**
 * @brief 
 * 
 * @param swarm 
 * @param pin 
 */
void InitializeGeodesics(Swarm* swarm, ParameterInput *pin);

}

/*
 * Make orthonormal basis for camera frame.
 *
 * e^0 along Ucam
 * e^3 outward (!) along radius vector
 * e^2 toward north pole of coordinate system ("y" in the image plane)
 * e^1 in the remaining direction ("x" in the image plane)
 *
 * This combination measures the final Stokes parameters correctly (IEEE/IAS).
 * These values are then translated if a different convention is to be output.
 *
 * Points the camera so that the angular momentum k_{th,phi} at FOV center is 0
 */

KOKKOS_INLINE_FUNCTION int make_camera_tetrad(const GRCoordinates& G, const GReal Xcam[GR_DIM], double Econ[GR_DIM][GR_DIM],
                                              double Ecov[GR_DIM][GR_DIM])
{
    double Gcov[GR_DIM][GR_DIM], Gcon[GR_DIM][GR_DIM];
    G.coords.gcov_native(Xcam, Gcov);
    G.coords.gcon_native(Gcov, Gcon);
    double Ucam[GR_DIM], Kcon[GR_DIM], trial[GR_DIM];

    // center the camera according to impact parameter, i.e., make it
    // so that Kcontetrad = ( 1, 0, 0, 1 ) corresponds to an outgoing
    // wavevector with zero angular momentum / zero impact parameter.

    // use normal observer velocity. this forces (Gcov.Econ[0])[3] = 0.
    trial[0] = -1.;
    trial[1] = 0.;
    trial[2] = 0.;
    trial[3] = 0.;
    flip_index(trial, Gcon, Ucam);

    // set Kcon (becomes Econ[3][mu]) outward directed with central
    // pixel k_phi = 0. this ensures that a photon with zero impact
    // parameter will be in the center of the field of view.
    trial[0] = 1.;
    trial[1] = 1.;
    trial[2] = 0.;
    trial[3] = 0.;
    flip_index(trial, Gcon, Kcon);

    // set the y camera direction to be parallel to the projected
    // spin axis of the black hole (on the image plane defined to
    // be normal to the Kcon vector above).
    trial[0] = 0.;
    trial[1] = 0.;
    trial[2] = 1.;
    trial[3] = 0.;

    // Make sure to pass back any errors
    return make_plasma_tetrad(Ucam, Kcon, trial, Gcov, Econ, Ecov);
}

/*
 * Make orthonormal basis for camera frame -- old implementation
 *
 * e^0 along Ucam
 * e^3 outward (!) along radius vector
 * e^2 toward north pole of coordinate system ("y" in the image plane)
 * e^1 in the remaining direction ("x" in the image plane)
 *
 * This combination measures the final Stokes parameters correctly (IEEE/IAS).
 * These values are then translated if a different convention is to be output.
 *
 * Points the camera so that the *contravariant wavevector* k^{th,phi} = 0
 */

KOKKOS_INLINE_FUNCTION int make_camera_tetrad_old(const GRCoordinates& G, double X[GR_DIM], double Econ[GR_DIM][GR_DIM],
                                                  double Ecov[GR_DIM][GR_DIM])
{
    double Gcov[GR_DIM][GR_DIM], Gcon[GR_DIM][GR_DIM];
    G.coords.gcov_native(X, Gcov);
    G.coords.gcon_native(Gcov, Gcon);
    double Ucam[GR_DIM], Kcon[GR_DIM], trial[GR_DIM];

    // old centering method

    Ucam[0] = 1.;
    Ucam[1] = 0.;
    Ucam[2] = 0.;
    Ucam[3] = 0.;

    trial[0] = 1.;
    trial[1] = 1.;
    trial[2] = 0.;
    trial[3] = 0.;
    flip_index(trial, Gcon, Kcon);

    trial[0] = 0.;
    trial[1] = 0.;
    trial[2] = 1.;
    trial[3] = 0.;

    // Make sure to pass back any errors
    return make_plasma_tetrad(Ucam, Kcon, trial, Gcov, Econ, Ecov);
}

/*
 * Initialize a geodesic from the camera
 * This takes the parameters struct directly since most of them are
 * camera parameters anyway
 */
KOKKOS_INLINE_FUNCTION void pinhole_K(const GRCoordinates& G, long int i, long int j, int nx, int ny,
                                      double fovx, double fovy, double xoff, double yoff, double rotcam,
                                      const GReal Xcam[GR_DIM], GReal Kcon[GR_DIM])
{
    double Econ[GR_DIM][GR_DIM];
    double Ecov[GR_DIM][GR_DIM];
    double Kcon_tetrad[GR_DIM];

    make_camera_tetrad(G, Xcam, Econ, Ecov);

    // Construct outgoing wavevectors
    // xoff: allow arbitrary offset for e.g. ML training imgs
    // +0.5: project geodesics from px centers
    // xoff/yoff are separated to keep consistent behavior between refinement levels
    double dxoff = (i + 0.5 + xoff - 0.01) / nx - 0.5;
    double dyoff = (j + 0.5 + yoff) / ny - 0.5;
    Kcon_tetrad[0] = 0.;
    Kcon_tetrad[1] = (dxoff * cos(rotcam) - dyoff * sin(rotcam)) * fovx;
    Kcon_tetrad[2] = (dxoff * sin(rotcam) + dyoff * cos(rotcam)) * fovy;
    Kcon_tetrad[3] = 1.;

    /* normalize */
    null_normalize(Kcon_tetrad, 1.);

    /* translate into coordinate frame */
    tetrad_to_coordinate(Econ, Kcon_tetrad, Kcon);
}