// Tools for Containers

//#include "driver/driver.hpp"
#include "driver/multistage.hpp"
#include "interface/container.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"


using namespace parthenon;

/**
 * In order to abstract over time-integration schemes (RK#, etc, etc) Parthenon introduces "containers"
 * A container is a full copy of system state -- all variables.
 * Each stage fills values for dU/dt independent of time integration scheme, and then the UpdateContainer function
 * fills the next stage based on existing stages and dU/dt
 * (or rather its task does, namely the succinctly-named BlockStageNamesIntegratorTask)
 */
TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string>& stage_name,
                           Integrator* integrator);

// Tools for adding container-based tasks to Parthenon integrator stage descriptions
using ContainerTaskFunc = std::function<TaskStatus(Container<Real>&)>;
using TwoContainerTaskFunc = std::function<TaskStatus(Container<Real>&, Container<Real>&)>;

class ContainerTask : public BaseTask {
 public:
  ContainerTask(TaskID id, ContainerTaskFunc func,
                TaskID dep, Container<Real> rc)
    : BaseTask(id,dep), _func(func), _cont(rc) {}
  TaskStatus operator () () { return _func(_cont); }
 private:
  ContainerTaskFunc _func;
  Container<Real> _cont;
};

class TwoContainerTask : public BaseTask {
 public:
  TwoContainerTask(TaskID id, TwoContainerTaskFunc func,
                   TaskID dep, Container<Real> rc1, Container<Real> rc2)
    : BaseTask(id,dep), _func(func), _cont1(rc1), _cont2(rc2) {}
  TaskStatus operator () () { return _func(_cont1, _cont2); }
 private:
  TwoContainerTaskFunc _func;
  Container<Real> _cont1;
  Container<Real> _cont2;
};

// Couple of functions for working with container-based tasks.
TaskID AddContainerTask(TaskList& tl, ContainerTaskFunc func, TaskID dep, Container<Real>& rc)
{
    return tl.AddTask<ContainerTask>(func,dep,rc);
}
TaskID AddTwoContainerTask(TaskList& tl, TwoContainerTaskFunc f, TaskID dep, Container<Real>& rc1, Container<Real>& rc2)
{
    return tl.AddTask<TwoContainerTask>(f,dep,rc1,rc2);
}
TaskID AddUpdateTask(TaskList& tl, MeshBlock* pmb, int stage, const std::vector<std::string>& stage_name, Integrator* integrator, BlockStageNamesIntegratorTaskFunc f, TaskID dep)
{
    return tl.AddTask<BlockStageNamesIntegratorTask>(f, dep, pmb, stage, stage_name, integrator);
}