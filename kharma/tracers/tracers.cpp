/* 
 *  File: tracers.cpp
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
#include "tracers.hpp"

#include "Kokkos_Random.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

// *************************************************//
// define the tracer particles package, including  *//
// initialization and update functions.            *//
// *************************************************//

std::shared_ptr<KHARMAPackage> Tracers::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Tracers");

    int num_tracers = pin->GetOrAddReal("tracers", "num_tracers", 100);
    pkg->AddParam<>("num_tracers", num_tracers);

    // Initialize random number generator pool
    int rng_seed = pin->GetOrAddInteger("tracers", "rng_seed", 1273);
    pkg->AddParam<>("rng_seed", rng_seed);
    RNGPool rng_pool(rng_seed);
    pkg->AddParam<>("rng_pool", rng_pool);

    // Add swarm of tracer particles
    std::string swarm_name = "tracers";
    Metadata swarm_metadata({Metadata::Provides, Metadata::None});
    pkg->AddSwarm(swarm_name, swarm_metadata);
    Metadata real_swarmvalue_metadata({Metadata::Real});
    pkg->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

    pkg->EstimateTimestepBlock = EstimateTimestepBlock;

    return pkg;
}

void Tracers::InitParticles(MeshBlock *pmb, ParameterInput *pin)
{
    auto &tr_pkg = pmb->packages.Get("Tracers");
    auto &mbd = pmb->meshblock_data.Get();
    auto &swarm = pmb->swarm_data.Get()->Get("tracers");
    const auto num_tracers = tr_pkg->Param<int>("num_tracers");
    auto rng_pool = tr_pkg->Param<RNGPool>("rng_pool");

    const int ndim = pmb->pmy_mesh->ndim;

    const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    auto coords = pmb->coords;

    const Real &x_min = pmb->coords.Xf<1>(ib.s);
    const Real &y_min = pmb->coords.Xf<2>(jb.s);
    const Real &z_min = pmb->coords.Xf<3>(kb.s);
    const Real &x_max = pmb->coords.Xf<1>(ib.e + 1);
    const Real &y_max = pmb->coords.Xf<2>(jb.e + 1);
    const Real &z_max = pmb->coords.Xf<3>(kb.e + 1);

    const auto mesh_size = pmb->pmy_mesh->mesh_size;
    const Real x_min_mesh = mesh_size.xmin(X1DIR);
    const Real y_min_mesh = mesh_size.xmin(X2DIR);
    const Real z_min_mesh = mesh_size.xmin(X3DIR);
    const Real x_max_mesh = mesh_size.xmax(X1DIR);
    const Real y_max_mesh = mesh_size.xmax(X2DIR);
    const Real z_max_mesh = mesh_size.xmax(X3DIR);

    // TODO FIXME
    // This will be called after rho, u, uvec are initialized,
    // so you can work from real densities/take any averages/etc
    Real advected_mean = 1.0;
    Real advected_amp = 1.0;
    Real kwave = 1.0;

    // Calculate fraction of total tracer particles on this meshblock by integrating the
    // advected profile over both the mesh and this meshblock. Tracer number follows number
    // = advected*volume.
    Real number_meshblock =
        advected_mean * (x_max - x_min) -
        advected_amp / kwave * (cos(kwave * x_max) - cos(kwave * x_min));
    number_meshblock *= (y_max - y_min) * (z_max - z_min);
    Real number_mesh = advected_mean * (x_max_mesh - x_min_mesh);
    number_mesh -=
        advected_amp / kwave * (cos(kwave * x_max_mesh) - cos(kwave * x_min_mesh));
    number_mesh *= (y_max_mesh - y_min_mesh) * (z_max_mesh - z_min_mesh);

    int num_tracers_meshblock = std::round(num_tracers * number_meshblock / number_mesh);
    int gid = pmb->gid;

    ParArrayND<int> new_indices;
    swarm->AddEmptyParticles(num_tracers_meshblock, new_indices);

    auto &x = swarm->Get<Real>("x").Get();
    auto &y = swarm->Get<Real>("y").Get();
    auto &z = swarm->Get<Real>("z").Get();
    auto &id = swarm->Get<int>("id").Get();

    auto swarm_d = swarm->GetDeviceContext();
    // This hardcoded implementation should only used in PGEN and not during runtime
    // addition of particles as indices need to be taken into account.
    pmb->par_for("CreateParticles", 0, num_tracers_meshblock - 1,
        KOKKOS_LAMBDA(const int n) {
            auto rng_gen = rng_pool.get_state();

            // Rejection sample the x position
            Real val;
            do {
                x(n) = x_min + rng_gen.drand() * (x_max - x_min);
                val = advected_mean + advected_amp * sin(2. * M_PI * x(n));
            } while (val < rng_gen.drand() * (advected_mean + advected_amp));

            y(n) = y_min + rng_gen.drand() * (y_max - y_min);
            z(n) = z_min + rng_gen.drand() * (z_max - z_min);
            id(n) = num_tracers * gid + n;

            rng_pool.free_state(rng_gen);
        }
    );
}

Real Tracers::EstimateTimestepBlock(MeshBlockData<Real> *mbd)
{
    auto pmb = mbd->GetBlockPointer();
    auto pkg = pmb->packages.Get("advection_package");

    const auto &vx = pkg->Param<Real>("vx");
    const auto &vy = pkg->Param<Real>("vy");
    const auto &vz = pkg->Param<Real>("vz");

    // Assumes a grid with constant dx, dy, dz within a block
    const Real &dx_i = pmb->coords.Dxc<1>(0);
    const Real &dx_j = pmb->coords.Dxc<2>(0);
    const Real &dx_k = pmb->coords.Dxc<3>(0);

    Real min_dt = dx_i / std::abs(vx + TINY_NUMBER);
    min_dt = std::min(min_dt, dx_j / std::abs(vy + TINY_NUMBER));
    min_dt = std::min(min_dt, dx_k / std::abs(vz + TINY_NUMBER));

    // No CFL number for particles
    return min_dt;
}

TaskStatus Tracers::Advect(MeshBlock *pmb, const StagedIntegrator *integrator)
{
    auto swarm = pmb->swarm_data.Get()->Get("tracers");
    auto adv_pkg = pmb->packages.Get("advection_package");

    int max_active_index = swarm->GetMaxActiveIndex();

    Real dt = integrator->dt;

    auto &x = swarm->Get<Real>("x").Get();
    auto &y = swarm->Get<Real>("y").Get();
    auto &z = swarm->Get<Real>("z").Get();

    const auto &vx = adv_pkg->Param<Real>("vx");
    const auto &vy = adv_pkg->Param<Real>("vy");
    const auto &vz = adv_pkg->Param<Real>("vz");

    auto swarm_d = swarm->GetDeviceContext();
    pmb->par_for("Tracer advection", 0, max_active_index,
        KOKKOS_LAMBDA(const int n) {
            if (swarm_d.IsActive(n)) {
                x(n) += vx * dt;
                y(n) += vy * dt;
                z(n) += vz * dt;

                bool on_current_mesh_block = true;
                swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
            }
        }
    );

    return TaskStatus::complete;
}

TaskStatus Tracers::Deposit(MeshBlock *pmb)
{
    auto swarm = pmb->swarm_data.Get()->Get("tracers");

    // Meshblock geometry
    const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    // again using scalar dx_D for assuming a uniform grid in this example
    const Real &dx_i = pmb->coords.Dxc<1>(0);
    const Real &dx_j = pmb->coords.Dxf<2>(0);
    const Real &dx_k = pmb->coords.Dxf<3>(0);
    const Real &minx_i = pmb->coords.Xf<1>(ib.s);
    const Real &minx_j = pmb->coords.Xf<2>(jb.s);
    const Real &minx_k = pmb->coords.Xf<3>(kb.s);

    const auto &x = swarm->Get<Real>("x").Get();
    const auto &y = swarm->Get<Real>("y").Get();
    const auto &z = swarm->Get<Real>("z").Get();
    auto swarm_d = swarm->GetDeviceContext();

    auto &tracer_dep = pmb->meshblock_data.Get()->Get("tracer_deposition").data;
    // Reset particle count
    pmb->par_for(
        "ZeroParticleDep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) { tracer_dep(k, j, i) = 0.; });

    const int ndim = pmb->pmy_mesh->ndim;

    pmb->par_for("DepositTracers", 0, swarm->GetMaxActiveIndex(),
        KOKKOS_LAMBDA(const int n) {
            if (swarm_d.IsActive(n)) {
                int i = static_cast<int>(std::floor((x(n) - minx_i) / dx_i) + ib.s);
                int j = 0;
                if (ndim > 1) {
                    j = static_cast<int>(std::floor((y(n) - minx_j) / dx_j) + jb.s);
                }
                int k = 0;
                if (ndim > 2) {
                    k = static_cast<int>(std::floor((z(n) - minx_k) / dx_k) + kb.s);
                }

                // For testing in this example we make sure the indices are correct
                //   if (i >= ib.s && i <= ib.e && j >= jb.s && j <= jb.e && k >= kb.s &&
                //       k <= kb.e) {
                    Kokkos::atomic_add(&tracer_dep(k, j, i), 1.0);
                //   } else {
                //     PARTHENON_FAIL("Particle outside of active region during deposition.");
                //   }
            }
        }
    );

    return TaskStatus::complete;
}
