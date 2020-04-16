// Tools for adding 

#include "driver/driver.hpp"
#include "driver/multistage.hpp"
#include "interface/Container.hpp"
#include "interface/StateDescriptor.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"

using namespace parthenon;

TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string>& stage_name,
                           Integrator* integrator);

using ContainerTaskFunc = std::function<TaskStatus(Container<Real>&)>;
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
using TwoContainerTaskFunc =
  std::function<TaskStatus(Container<Real>&, Container<Real>&)>;
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