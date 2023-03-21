/* 
 *  File: kharma_package.cpp
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
#include "kharma_package.hpp"

#include "types.hpp"

// PHYSICS-RELATED
// TODO take & accumulate TaskStatus?  Useful for ::incomplete if we ever want to do that
// TODO Several of these are unused & commented, but will be used as I meshify different drivers.
//      Then, I can work on meshifying packages by degrees

TaskStatus Packages::FixFlux(MeshData<Real> *md)
{
    Flag("Fixing fluxes on mesh");
    for (auto &package : md->GetMeshPointer()->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (kpackage->FixFlux != nullptr)
                kpackage->FixFlux(md);
        }
    }
    Flag("Fixed");
    return TaskStatus::complete;
}

// TaskStatus Packages::BlockPtoU(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse)
// {
//     Flag("Getting conserved variables on block");
//     for (auto &package : mbd->GetBlockPointer()->packages.AllPackages()) {
//         if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
//             if (kpackage->BlockPtoU != nullptr)
//                 kpackage->BlockPtoU(mbd, domain, coarse);
//         }
//     }
//     Flag("Done");
//     return TaskStatus::complete;
// }
// TaskStatus Packages::MeshPtoU(MeshData<Real> *md, IndexDomain domain, bool coarse)
// {
//     for (int i=0; i < md->NumBlocks(); ++i)
//         PtoU(md->GetBlockData(i).get(), domain, coarse);
//     return TaskStatus::complete;
// }

TaskStatus Packages::BlockUtoP(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse)
{
    Flag("Recovering primitive variables");
    for (auto &package : mbd->GetBlockPointer()->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (kpackage->BlockUtoP != nullptr)
                kpackage->BlockUtoP(mbd, domain, coarse);
        }
    }
    Flag("Recovered");
    return TaskStatus::complete;
}
TaskStatus Packages::MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockUtoP(md->GetBlockData(i).get(), domain, coarse);
    return TaskStatus::complete;
}

TaskStatus Packages::BlockUtoPExceptMHD(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Recovering primitive variables on boundaries");
    // We need to re-fill the primitive variables on the physical boundaries,
    // since the driver has already called UtoP for the step.
    // However, this does *not* apply to the GRMHD variables, as the boundary call
    // used/filled their primitive values.  Instead, they will need a PtoU call
    auto pmb = rc->GetBlockPointer();
    for (auto &package : pmb->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (package.first != "GRMHD" && package.first != "Inverter") {
                if (kpackage->BlockUtoP != nullptr)
                    kpackage->BlockUtoP(rc, domain, coarse);
            }
        }
    }
    Flag(rc, "Recovered");
    return TaskStatus::complete;
}
TaskStatus Packages::MeshUtoPExceptMHD(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockUtoPExceptMHD(md->GetBlockData(i).get(), domain, coarse);
    return TaskStatus::complete;
}

TaskStatus Packages::AddSource(MeshData<Real> *md, MeshData<Real> *mdudt)
{
    Flag("Adding source terms");
    for (auto &package : md->GetMeshPointer()->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (kpackage->AddSource != nullptr)
                kpackage->AddSource(md, mdudt);
        }
    }
    Flag("Added");
    return TaskStatus::complete;
}

TaskStatus Packages::BlockApplyPrimSource(MeshBlockData<Real> *rc)
{
    Flag("Applying primitive source terms");
    for (auto &package : rc->GetBlockPointer()->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (kpackage->BlockApplyPrimSource != nullptr)
                kpackage->BlockApplyPrimSource(rc);
        }
    }
    Flag("Added");
    return TaskStatus::complete;
}

// TODO will these need to be done on coarse versions?
TaskStatus Packages::BlockApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain)
{
    Flag("Applying floors");
    auto pmb = mbd->GetBlockPointer();
    auto pkgs = pmb->packages.AllPackages();

    // Apply the version from "Floors" package first
    if (pkgs.count("Floors")) {
        KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(pkgs.at("Floors").get());
        // We *want* to crash on null deref if this kpackage is null, something would be wrong
        if (kpackage->BlockApplyFloors != nullptr)
            kpackage->BlockApplyFloors(mbd, domain);
    }
    // Then anything else
    for (auto &package : mbd->GetBlockPointer()->packages.AllPackages()) {
        if (package.first != "Floors") {
            if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
                if (kpackage->BlockApplyFloors != nullptr)
                    kpackage->BlockApplyFloors(mbd, domain);
            }
        }
    }
    Flag("Applied");

    return TaskStatus::complete;
}
TaskStatus Packages::MeshApplyFloors(MeshData<Real> *md, IndexDomain domain)
{
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockApplyFloors(md->GetBlockData(i).get(), domain);
    return TaskStatus::complete;
}

// GENERAL CALLBACKS
// TODO this will need to be mesh'd too
void Packages::UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin)
{
    Flag("Filling output arrays");
    for (auto &package : pmb->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (kpackage->BlockUserWorkBeforeOutput != nullptr)
                kpackage->BlockUserWorkBeforeOutput(pmb, pin);
        }
    }
    Flag("Filled");
}

void Packages::PreStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    Flag("Pre-step package work");
    for (auto &package : pmesh->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (kpackage->MeshPreStepUserWorkInLoop != nullptr)
                kpackage->MeshPreStepUserWorkInLoop(pmesh, pin, tm);
        }
    }
    Flag("Done pre-step package work");
}

void Packages::PostStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    Flag("Post-step package work");
    for (auto &package : pmesh->packages.AllPackages()) {
        if (KHARMAPackage *kpackage = dynamic_cast<KHARMAPackage*>(package.second.get())) {
            if (kpackage->MeshPostStepUserWorkInLoop != nullptr)
                kpackage->MeshPostStepUserWorkInLoop(pmesh, pin, tm);
        }
    }
}

void Packages::PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    // Parthenon's version of this has a bug, but I would probably subclass it anyway.
    // very useful to have a single per-step spot to control any routine print statements
    const auto& md = pmesh->mesh_data.GetOrAdd("base", 0).get();
    if (md->NumBlocks() > 0) {
        for (auto &package : pmesh->packages.AllPackages()) {
            if (package.second->PostStepDiagnosticsMesh != nullptr)
                package.second->PostStepDiagnosticsMesh(tm, md);
        }
    }
}

