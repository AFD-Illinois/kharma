// Tools for Containers -- sorry, MeshBlockDatas (??)
#pragma once

#include <parthenon/parthenon.hpp>

using namespace parthenon;

/**
 * In order to abstract over time-integration schemes (RK#, etc, etc) Parthenon introduces "containers"
 * A container is a full copy of system state -- all variables.
 * Each stage fills values for dU/dt independent of time integration scheme, and then the UpdateContainer function
 * fills the next stage based on existing stages and dU/dt
 * (or rather its task does, namely the succinctly-named BlockStageNamesIntegratorTask)
 */
TaskStatus UpdateContainer(std::shared_ptr<MeshBlock>& pmb, int stage,
                           std::vector<std::string>& stage_name,
                           Integrator* integrator);

/**
 * Quick function to just copy a variable by name from one container to the next
 */
TaskStatus CopyField(std::string& var, std::shared_ptr<MeshBlockData<Real>>& rc0, std::shared_ptr<MeshBlockData<Real>>& rc1);