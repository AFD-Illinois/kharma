

#include "kipole.hpp"

#include "constants.hpp"
#include "geodesics.hpp"
#include "unpol_comparison.hpp"

Packages_t KIPOLE::ProcessPackages(std::unique_ptr<ParameterInput>& pin)
{
    Packages_t packages;

    packages.Add(Geodesics::Initialize(pin.get()));
    //packages.Add();

    return std::move(packages);
}

Properties_t KIPOLE::ProcessProperties(std::unique_ptr<ParameterInput>& pin)
{
    // TODO somehow only parse the coordinate system once, so we can know exactly whether we're spherical/modified
    // So far every non-null transform is exp(x1) but who knows
    std::string cb = pin->GetString("coordinates", "base");
    std::string ctf = pin->GetOrAddString("coordinates", "transform", "null");
    if (ctf != "null") {
        int n1tot = pin->GetInteger("parthenon/mesh", "nx1");
        GReal Rout = pin->GetReal("coordinates", "r_out");
        Real a = pin->GetReal("coordinates", "a");
        // TODO make the following independent of exponentiated r
        GReal Rhor = 1 + sqrt(1 - a*a);
        GReal x1max = log(Rout);
        // Set Rin such that we have 5 zones completely inside the event horizon
        // If xeh = log(Rhor), xin = log(Rin), and xout = log(Rout),
        // then we want xeh = xin + 5.5 * (xout - xin) / N1TOT:
        GReal x1min = (n1tot * log(Rhor) / 5.5 - x1max) / (-1. + n1tot / 5.5);
        if (x1min < 0.0) {
            throw std::invalid_argument("Not enough radial zones were specified to put 5 zones inside EH!");
        }
        //cerr << "Setting x1min: " << x1min << " x1max " << x1max << " based on BH with a=" << a << endl;
        pin->SetReal("parthenon/mesh", "x1min", x1min);
        pin->SetReal("parthenon/mesh", "x1max", x1max);
    }
    // Assumption: if we're in a spherical system...
    if (cb == "spherical_ks" || cb == "ks" || cb == "spherical_bl" || cb == "bl" || cb == "spherical_minkowski") {
        // ...then we definitely want spherical boundary conditions
        // TODO only set all this if it isn't already
        pin->SetString("parthenon/mesh", "ix1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ox1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ix2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ox2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ix3_bc", "periodic");
        pin->SetString("parthenon/mesh", "ox3_bc", "periodic");

        // We also know the bounds for most transforms in spherical.  Set them.
        if (ctf == "none") {
            pin->SetReal("parthenon/mesh", "x2min", 0.0);
            pin->SetReal("parthenon/mesh", "x2max", M_PI);
            pin->SetReal("parthenon/mesh", "x3min", 0.0);
            pin->SetReal("parthenon/mesh", "x3max", 2*M_PI);
        } else if (ctf == "modified" || ctf == "mks" || ctf == "funky" || ctf == "fmks") {
            pin->SetReal("parthenon/mesh", "x2min", 0.0);
            pin->SetReal("parthenon/mesh", "x2max", 1.0);
            pin->SetReal("parthenon/mesh", "x3min", 0.0);
            pin->SetReal("parthenon/mesh", "x3max", 2*M_PI);
        } // TODO any other transforms/systems
    }

    // Derive units we actually need
    double MBH = pin->GetOrAddReal("units", "MBH", 4.e6) * MSUN;
    //pin->AddParameter("units", "MBH_cgs", MBH);
    double L_unit = GNEWT * MBH / (CL * CL);
    pin->SetReal("units", "L_unit", L_unit);
    pin->SetReal("units", "T_unit", L_unit / CL);
    
    // TODO rephrase based on M_unit for not Gold et al
    double RHO_unit = pin->GetOrAddReal("units", "RHO_unit", 3.e-18);
    pin->SetReal("units", "RHO_unit", RHO_unit);
    pin->SetReal("units", "B_unit", CL * sqrt(4. * M_PI * RHO_unit));
}

void KIPOLE::LoadProblem(MeshBlock *pmb, ParameterInput *pin)
{
    // Any like, mesh loading we want to do
}

// void KIPOLE::FillOutput(MeshBlock *pmb, ParameterInput *pin) {
    
// }