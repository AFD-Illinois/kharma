
#include "pinhole_camera.hpp"

#include "constants.hpp"
#include "coordinate_systems.hpp"

void PinholeCamera::InitializeGeodesics(Swarm* swarm, ParameterInput *pin)
{
    int nx = pin->GetInteger("camera", "nx");
    int ny = pin->GetInteger("camera", "ny");

    Real fovx_muas = pin->GetOrAddReal("camera", "fovx", 160.);
    Real fovy_muas = pin->GetOrAddReal("camera", "fovy", 160.);
    Real L_unit = pin->GetReal("units", "L_unit");
    Real Dsource = pin->GetOrAddReal("camera", "Dsource", DM87_PC) * PC;
    Real fovx = fovx_muas * Dsource / L_unit / MUAS_PER_RAD;
    Real fovy = fovy_muas * Dsource / L_unit / MUAS_PER_RAD;

    Real xoff = pin->GetOrAddReal("camera", "xoff", 0.);
    Real yoff = pin->GetOrAddReal("camera", "yoff", 0.);
    Real rotcam = pin->GetOrAddReal("camera", "rotcam", 0.) * M_PI/180.;

    GReal Xembed[GR_DIM] = {1,
                            pin->GetOrAddReal("camera", "r", 1000.),
                            pin->GetOrAddReal("camera", "th", 17.) / 180*M_PI, 
                            pin->GetOrAddReal("camera", "phi", 0.) / 180*M_PI};
    GReal Xcam[GR_DIM];
    
    auto& G = swarm->GetBlockPointer()->coords;
    G.coords.coord_to_native(Xembed, Xcam);

    ParArrayND<bool> new_particles_mask = swarm->AddEmptyParticles(nx*ny);
    //auto &mask = swarm->GetMask().Get();

    auto &t = swarm->GetReal("t").Get();
    auto &x = swarm->GetReal("x").Get();
    auto &y = swarm->GetReal("y").Get();
    auto &z = swarm->GetReal("z").Get();

    auto &k0 = swarm->GetReal("k0").Get();
    auto &k1 = swarm->GetReal("k1").Get();
    auto &k2 = swarm->GetReal("k2").Get();
    auto &k3 = swarm->GetReal("k3").Get();

    auto swarm_d = swarm->GetDeviceContext();

    swarm->GetBlockPointer()->par_for("init_XK", 0, swarm->get_max_active_index(),
        KOKKOS_LAMBDA(const int n) {
            if (new_particles_mask(n)) {
                double Kcon[GR_DIM];
                int i = n / nx;
                int j = n % nx;
                pinhole_K(G, i, j, nx, ny, fovx, fovy, xoff, yoff, rotcam, Xcam, Kcon);
                t(n) = Xcam[0];
                x(n) = Xcam[1];
                y(n) = Xcam[2];
                z(n) = Xcam[3];

                k0(n) = Kcon[0];
                k1(n) = Kcon[1];
                k2(n) = Kcon[2];
                k3(n) = Kcon[3];
            }
        }
    );
}