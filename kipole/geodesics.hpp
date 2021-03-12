

#include <parthenon/parthenon.hpp>


namespace Geodesics {

/**
 * @brief Initialize the Geodesics package
 * 
 * @param pin parameters
 * @return std::shared_ptr<StateDescriptor> package reference
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

/**
 * @brief 
 * 
 * @param swarm of geodesics
 * @param dl by which to push them, in native coordinates
 */
void PushGeodesics(Swarm* swarm, double dl);


/**
 * @brief Trace a swarm of geodesics backward from the camera to their stopping points
 * Takes starting locations and wavevectors in *units of electron mass frequency*
 * 
 */
TaskStatus TraceGeodesicsBack(Swarm* swarm);

} // Geodesics

/**
 * Choose stepsize according to inverse Kcon, dramatically decreasing the step
 * toward the coordinate pole and EH.
 * 
 * Use the sum of inverses by default; the strict minimum seems to occasionally
 * overstep even for small eps
 * 
 * This stepsize function can be troublesome inside of R = 2M,
 * and should be used cautiously in this region.
 */
#define STEP_STRICT_MIN 0
KOKKOS_INLINE_FUNCTION double stepsize(const GRCoordinates& G, double X[GR_DIM], double Kcon[GR_DIM], double eps)
{
    double deh = fmin(fabs(X[1] - G.x1f(0)), 0.1);
    double dlx1 = eps * (10 * deh) / (fabs(Kcon[1]) + SMALL * SMALL);

    double dpole = fmin(fabs(X[2]), fabs(1. - X[2])); //stopx2
    double dlx2 = eps * dpole / 3 / (fabs(Kcon[2]) + SMALL * SMALL);

    double dlx3 = eps / (fabs(Kcon[3]) + SMALL * SMALL);

    if (STEP_STRICT_MIN) {
        return fmin(fmin(dlx1, dlx2), dlx3);
    } else {
        double idlx1 = 1. / (fabs(dlx1) + SMALL * SMALL);
        double idlx2 = 1. / (fabs(dlx2) + SMALL * SMALL);
        double idlx3 = 1. / (fabs(dlx3) + SMALL * SMALL);

        return 1. / (idlx1 + idlx2 + idlx3);
    }
}

#define THIN_DISK 0
/**
 * @brief Condition for stopping the backward integration (camera->origin) of the photon geodesic
 * 
 * @param X Current position
 * @param Xhalf Half-step beyond current position (for thin-disk stop condition)
 * @param Kcon Current wavevector
 * @return bool whether to stop the integration
 */
KOKKOS_INLINE_FUNCTION bool stop_backward_integration(const GRCoordinates& G, double X[NDIM], double Kcon[NDIM], double r_out)
{
  // TODO thin disk.  Cartesian.

  // Geometric stop conditions
  // TODO try exp()
  SphKSCoords ks = mpark::get<SphKSCoords>(G.coords.base);
  double Xembed[GR_DIM];
  G.coords.coord_to_embed(X, Xembed);

  if ((Xembed[1] > r_out && Kcon[1] < 0.) || // Stop either beyond rmax_geo
       Xembed[1] < (ks.rhor() + 0.0001)) { // Or right near the coordinate center
    return true;
  }

  return false;
}