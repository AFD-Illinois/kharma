




#include <parthenon/parthenon.hpp>

#include "decs.hpp"
#include "matrix.hpp"
#include "phys.hpp"

TaskStatus CalculateCurrent(MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1, const double& dt);

// Return mu, nu component of contravarient Maxwell tensor at grid zone i, j, k
KOKKOS_INLINE_FUNCTION double get_Fcon(const GRCoordinates& G, GridVars P, 
                                        const int& mu, const int& nu, const int& k, const int& j, const int& i)
{
    if (mu == nu) {
        return 0.;
    } else {
        FourVectors Dtmp;
        get_state(G, P, k, j, i, Loci::center, Dtmp);
        double Fcon = 0.;
        for (int kap = 0; kap < NDIM; kap++) {
            for (int lam = 0; lam < NDIM; lam++) {
                Fcon += (-1. / G.gdet(Loci::center, j, i)) * antisym(mu, nu, kap, lam) * Dtmp.ucov[kap] * Dtmp.bcov[lam];
            }
        }

        return Fcon;
    }
}
