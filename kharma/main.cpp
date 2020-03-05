//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
// (C) Ben Prather and Illinois AFD group, let's keep this train going
// Don't worry, we're still releasing it under BSD 3-clause
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"

// Local headers
#include "mpi.hpp"

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief KHARMA main function

int main(int argc, char *argv[]) {
  int mpi_err = mpi_init(argc, argv);
  if (mpi_err) {return mpi_err;}
  // KOKKOS INIT HERE...

  if (argc != 2) {
    if (Globals::my_rank == 0) {
      std::cout << "\nUsage: " << argv[0] << " input_file\n"
        << "\tTry this input file:\n"
        << "\tparthenon/example/parthinput.example"
        << std::endl;
    }
    return 0;
  }

  std::string inputFileName = argv[1];
  ParameterInput pin;
  IOWrapper inputFile;
  inputFile.Open(inputFileName.c_str(), IOWrapper::FileMode::read);
  pin.LoadFromFile(inputFile);
  inputFile.Close();

  if (Globals::my_rank == 0) {
    std::cout << "\ninput file = " << inputFileName << std::endl;
    if (pin.DoesParameterExist("mesh","nx1")) {
      std::cout << "nx1 = " << pin.GetInteger("mesh","nx1") << std::endl;
    }
    if (pin.DoesParameterExist("mesh","x1min")) {
      std::cout << "x1min = " << pin.GetReal("mesh","x1min") << std::endl;
    }
    if (pin.DoesParameterExist("mesh","x1max")) {
      std::cout << "x1max = " << pin.GetReal("mesh","x1max") << std::endl;
    }
    if (pin.DoesParameterExist("mesh", "ix1_bc")) {
      std::cout << "x1 inner boundary condition = "
        << pin.GetString("mesh","ix1_bc") << std::endl;
    }
    if (pin.DoesParameterExist("mesh", "ox1_bc")) {
      std::cout << "x1 outer boundary condition = "
        << pin.GetString("mesh","ox1_bc") << std::endl;
    }
  }

  std::vector<std::shared_ptr<MaterialPropertiesInterface>> mats;
  std::map<std::string, std::shared_ptr<StateDescriptor>> physics;
  Mesh m(&pin, mats, physics, [](Container<Real> &) {});

  mpi_finalize();
}
