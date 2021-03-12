
#include "decs.hpp"
#include "constants.hpp"
#include "matrix.hpp"

/* get frequency in fluid frame, in Hz */
KOKKOS_INLINE_FUNCTION double get_fluid_nu(double Kcon[GR_DIM], double Ucov[GR_DIM])
{
    double nu;

    // This is the energy in electron rest-mass units
    nu = -dot(Kcon, Ucov) * ME * CL * CL / HPL;

    if (nu < 0.) {
      nu = 1.;
#if DEBUG
      printf("Fluid nu < 0: %g !", nu);
#endif
    }

    if (isnan(nu)) {
      printf("isnan get_fluid_nu, K: %g %g %g %g\n",
              Kcon[0], Kcon[1], Kcon[2], Kcon[3]);
      printf("isnan get_fluid_nu, U: %g %g %g %g\n",
              Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
    }

    return (nu);
}

/* return angle between magnetic field and wavevector */
KOKKOS_INLINE_FUNCTION double get_bk_angle(double X[GR_DIM], double Kcon[GR_DIM], double Ucov[GR_DIM], double Bcon[GR_DIM], double Bcov[GR_DIM])
{
    double B = sqrt(fabs(dot(Bcon, Bcov)));

    if (B == 0.)
    	return (M_PI / 2.);

    double k = fabs(dot(Kcon, Ucov));

    double mu = dot(Kcon, Bcov) / (k * B);

    if (fabs(mu) > 1.)
	    mu /= fabs(mu);

    if (isnan(mu))
	    printf("isnan get_bk_angle\n");

    return acos(mu);
}
