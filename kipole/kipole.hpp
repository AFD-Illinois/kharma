
#include <parthenon/parthenon.hpp>

/**
 * General preferences for KIPOLE
 */
namespace KIPOLE {

    /**
     * This exists to mess with the Parameters struct in between parsing
     * and constructing the mesh.  It will die when Init() gets split
     */
    Properties_t ProcessProperties(std::unique_ptr<ParameterInput>& pin);

    /**
     * Load the model and emission packages
     */
    Packages_t ProcessPackages(std::unique_ptr<ParameterInput>& pin);

    /**
     * @brief 
     * 
     * @param pmb 
     * @param pin 
     */
    void LoadProblem(MeshBlock *pmb, ParameterInput *pin);

    /**
     * Fill any arrays that are calculated only for output
     * 
     * This becomes a member function (!) of MeshBlock and is called for each block
     */
    void FillOutput(MeshBlock *pmb, ParameterInput *pin);
}