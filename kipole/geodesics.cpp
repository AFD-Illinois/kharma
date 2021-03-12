
#include "geodesics.hpp"

#include "decs.hpp"

#include "coordinate_systems.hpp"

std::shared_ptr<StateDescriptor> Geodesics::Initialize(ParameterInput *pin)
{
    FLAG("Initializing geodesics package");
    auto pkg = std::make_shared<StateDescriptor>("geodesics");
    Params &params = pkg->AllParams();

    params.Add("max_nstep", pin->GetOrAddInteger("geodesics", "max_nstep", 50000));
    params.Add("eps", pin->GetOrAddReal("geodesics", "eps", 0.01));
    params.Add("r_max", pin->GetOrAddReal("geodesics", "r_max", 1000.0));

    std::string swarm_name = "geodesics";
    Metadata swarm_metadata;
    pkg->AddSwarm(swarm_name, swarm_metadata);
    Metadata m({Metadata::Real});
    pkg->AddSwarmValue("t", swarm_name, m);
    pkg->AddSwarmValue("k0", swarm_name, m);
    pkg->AddSwarmValue("k1", swarm_name, m);
    pkg->AddSwarmValue("k2", swarm_name, m);
    pkg->AddSwarmValue("k3", swarm_name, m);
    pkg->AddSwarmValue("unpol", swarm_name, m);

    m = Metadata({Metadata::Integer});
    pkg->AddSwarmValue("stop", swarm_name, m);
    pkg->AddSwarmValue("nstep", swarm_name, m);

    //std::vector<int> s_4vector = {4};
    //m = Metadata({Metadata::Independent, Metadata::Restart, Metadata::Real}, s_4vector);
    //pkg->AddSwarmValue("kcon", swarm_name, m);

    std::vector<int> s_4tensor = {4, 4};
    m = Metadata({Metadata::Real}, s_4tensor);
    pkg->AddSwarmValue("Nreal", swarm_name, m);
    pkg->AddSwarmValue("Nimag", swarm_name, m);

    FLAG("Initialized");
    return pkg;
}

void Geodesics::PushGeodesics(Swarm* swarm, bool forward=false)
{
    auto &stop = swarm->GetInteger("stop").Get();
    auto &nstep = swarm->GetInteger("nstep").Get();

    auto &t = swarm->GetReal("t").Get();
    auto &x = swarm->GetReal("x").Get();
    auto &y = swarm->GetReal("y").Get();
    auto &z = swarm->GetReal("z").Get();

    auto &k0 = swarm->GetReal("k0").Get();
    auto &k1 = swarm->GetReal("k1").Get();
    auto &k2 = swarm->GetReal("k2").Get();
    auto &k3 = swarm->GetReal("k3").Get();

    auto& G = swarm->GetBlockPointer()->coords;
    double eps = swarm->GetBlockPointer()->packages.Get("geodesics")->Param<Real>("eps");

    swarm->GetBlockPointer()->par_for("init_XK", 0, swarm->get_max_active_index(),
        KOKKOS_LAMBDA(const int n) {
            if (!stop(n)) {
                double lconn[GR_DIM][GR_DIM][GR_DIM];
                double dKcon[GR_DIM];
                double Xh[GR_DIM], Kconh[GR_DIM];

                double Kcon[GR_DIM] = {k0(n), k1(n), k2(n), k3(n)};
                double X[GR_DIM] = {t(n), x(n), y(n), z(n)};
                double dl = (forward ? 1. : -1.) * stepsize(G, X, Kcon, eps);

                // RK2.  TODO symplectic?
                // TODO this can be done with fewer temporaries right?

                // Half-step
                G.coords.conn_native(X, lconn);

                // Advance K
                DLOOP1 dKcon[mu] = 0.;
                DLOOP3 dKcon[mu] -= 0.5 * dl * lconn[mu][nu][lam] * Kcon[nu] * Kcon[lam];
                DLOOP1 Kconh[mu] = Kcon[mu] + dKcon[mu];

                // Advance X
                DLOOP1 Xh[mu] = X[mu] + 0.5 * dl * Kcon[mu];

                // If we switch back to recording geodesics record half-step vals here
                // DLOOP1 {
                //     Xhalf[mu] = Xh[mu];
                //     Kconhalf[mu] = Kconh[mu];
                // }

                // Full step
                G.coords.conn_native(Xh, lconn);

                // Advance K
                DLOOP1 dKcon[mu] = 0.;
                DLOOP3 dKcon[mu] -= dl * lconn[nu][lam][mu] * Kconh[nu] * Kconh[lam];
                k0(n) += dKcon[0];
                k1(n) += dKcon[1];
                k2(n) += dKcon[2];
                k3(n) += dKcon[3];

                // Advance X
                t(n) += dl*Kconh[0];
                x(n) += dl*Kconh[1];
                y(n) += dl*Kconh[2];
                z(n) += dl*Kconh[3];

                // Record stepping
                nstep(n) += 1;
            }
        }
    );
}

TaskStatus Geodesics::TraceGeodesicsBack(Swarm* swarm)
{
    FLAG("Tracing geodesics back");
    auto &stop = swarm->GetInteger("stop").Get();
    auto &nstep = swarm->GetInteger("nstep").Get();

    auto &t = swarm->GetReal("t").Get();
    auto &x = swarm->GetReal("x").Get();
    auto &y = swarm->GetReal("y").Get();
    auto &z = swarm->GetReal("z").Get();

    auto &k0 = swarm->GetReal("k0").Get();
    auto &k1 = swarm->GetReal("k1").Get();
    auto &k2 = swarm->GetReal("k2").Get();
    auto &k3 = swarm->GetReal("k3").Get();

    auto& G = swarm->GetBlockPointer()->coords;
    Real r_max = swarm->GetBlockPointer()->packages.Get("geodesics")->Param<Real>("r_max");
    int max_nstep = swarm->GetBlockPointer()->packages.Get("geodesics")->Param<int>("max_nstep");

    FLAG("Starting backward integration");

    // Integrate backwards
    int total_nstep = 0;
    while (total_nstep < max_nstep - 1) {
        total_nstep++;
        // Push backward from camera by one step along each geodesic
        PushGeodesics(swarm);

        // Mark which should be evolved no further; check whether any geodesics are left
        int n_remaining = 0;
        Kokkos::Sum<int> n_rem_sum(n_remaining);
        swarm->GetBlockPointer()->par_reduce("check_stop_integrate", 0, swarm->get_max_active_index(),
            KOKKOS_LAMBDA(const int n, int& local) {
                double Kcon[GR_DIM] = {k0(n), k1(n), k2(n), k3(n)};
                double X[GR_DIM] = {t(n), x(n), y(n), z(n)};
                stop(n) |= stop_backward_integration(G, X, Kcon, r_max);
                local += !stop(n);
            }
        , n_rem_sum);
        printf("n_remaining %d\n", n_remaining);

        // Break if we're done...
        if (n_remaining == 0) break;
        // ... or on signal
        if (SignalHandler::CheckSignalFlags() != 0) {
            return TaskStatus::fail;
        }
    }
    auto nstep_h = nstep.GetHostMirrorAndCopy();
    for (int n=0; n < swarm->get_max_active_index(); ++n) {
        printf("%f ", nstep_h(n));
    }

    FLAG("Finished backward integration");
    return TaskStatus::complete;
}
