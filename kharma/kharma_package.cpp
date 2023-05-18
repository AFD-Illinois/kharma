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
// TODO continue meshification until all is mesh

TaskStatus Packages::FixFlux(MeshData<Real> *md)
{
    Flag("FixFlux");
    auto kpackages = md->GetMeshPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->FixFlux != nullptr) {
            Flag("FixFlux_"+kpackage.first);
            kpackage.second->FixFlux(md);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag("BlockUtoP");
    auto kpackages = rc->GetBlockPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->BlockUtoP != nullptr) {
            Flag("BlockUtoP_"+kpackage.first);
            kpackage.second->BlockUtoP(rc, domain, coarse);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}
TaskStatus Packages::MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    Flag("MeshUtoP");
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockUtoP(md->GetBlockData(i).get(), domain, coarse);
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::BlockUtoPExceptMHD(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag("BlockUtoPExceptMHD");
    // We need to re-fill the primitive variables on the physical boundaries,
    // since the driver has already called UtoP for the step.
    // However, this does *not* apply to the GRMHD variables, as the boundary call
    // used/filled their primitive values.  Instead, they will need a PtoU call
    auto kpackages = rc->GetBlockPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.first != "GRMHD" && kpackage.first != "Inverter") {
            if (kpackage.second->BlockUtoP != nullptr) {
                Flag("BlockUtoPExceptMHD_"+kpackage.first);
                kpackage.second->BlockUtoP(rc, domain, coarse);
                EndFlag();
            }
        }
    }
    EndFlag();
    return TaskStatus::complete;
}
TaskStatus Packages::MeshUtoPExceptMHD(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    Flag("MeshUtoPExceptMHD");
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockUtoPExceptMHD(md->GetBlockData(i).get(), domain, coarse);
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::AddSource(MeshData<Real> *md, MeshData<Real> *mdudt)
{
    Flag("AddSource");
    auto kpackages = md->GetMeshPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->AddSource != nullptr) {
            Flag("AddSource_"+kpackage.first);
            kpackage.second->AddSource(md, mdudt);
            EndFlag();
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::BlockApplyPrimSource(MeshBlockData<Real> *rc)
{
    // TODO print only if there's calls inside?
    Flag("BlockApplyPrimSource");
    auto kpackages = rc->GetBlockPointer()->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->BlockApplyPrimSource != nullptr) {
            kpackage.second->BlockApplyPrimSource(rc);
        }
    }
    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Packages::BlockApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain)
{
    Flag("BlockApplyFloors");
    auto pmb = mbd->GetBlockPointer();
    auto pkgs = pmb->packages.AllPackages();

    // Apply the version from "Floors" package first
    if (pkgs.count("Floors")) {
        KHARMAPackage *pkpackage = pmb->packages.Get<KHARMAPackage>("Floors");
        if (pkpackage->BlockApplyFloors != nullptr) {
            Flag("BlockApplyFloors_Floors");
            pkpackage->BlockApplyFloors(mbd, domain);
            EndFlag();
        }
    }
    // Then anything else
    auto kpackages = pmb->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.first != "Floors") {
            if (kpackage.second->BlockApplyFloors != nullptr) {
                Flag("BlockApplyFloors_"+kpackage.first);
                kpackage.second->BlockApplyFloors(mbd, domain);
                EndFlag();
            }
        }
    }
    EndFlag();

    return TaskStatus::complete;
}
TaskStatus Packages::MeshApplyFloors(MeshData<Real> *md, IndexDomain domain)
{
    Flag("MeshApplyFloors");
    for (int i=0; i < md->NumBlocks(); ++i)
        BlockApplyFloors(md->GetBlockData(i).get(), domain);
    EndFlag();
    return TaskStatus::complete;
}

// GENERAL CALLBACKS
// TODO this will need to be mesh'd too
void Packages::UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin)
{
    Flag("UserWorkBeforeOutput");
    auto kpackages = pmb->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->BlockUserWorkBeforeOutput != nullptr) {
            Flag("UserWorkBeforeOutput_"+kpackage.first);
            kpackage.second->BlockUserWorkBeforeOutput(pmb, pin);
            EndFlag();
        }
    }
    EndFlag();
}

void Packages::PreStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    Flag("PreStepUserWorkInLoop");
    auto kpackages = pmesh->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->MeshPreStepUserWorkInLoop != nullptr) {
            Flag("PreStepUserWorkInLoop_"+kpackage.first);
            kpackage.second->MeshPreStepUserWorkInLoop(pmesh, pin, tm);
            EndFlag();
        }
    }
    EndFlag();
}

void Packages::PostStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    Flag("PostStepUserWorkInLoop");
    auto kpackages = pmesh->packages.AllPackagesOfType<KHARMAPackage>();
    for (auto kpackage : kpackages) {
        if (kpackage.second->MeshPostStepUserWorkInLoop != nullptr) {
            Flag("PostStepUserWorkInLoop_"+kpackage.first);
            kpackage.second->MeshPostStepUserWorkInLoop(pmesh, pin, tm);
            EndFlag();
        }
    }
    EndFlag();
}

void Packages::PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    // Parthenon's version of this has a bug, but I would probably subclass it anyway.
    // very useful to have a single per-step spot to control any routine print statements
    Flag("PostStepDiagnostics");
    const auto& md = pmesh->mesh_data.GetOrAdd("base", 0).get();
    if (md->NumBlocks() > 0) {
        for (auto &package : pmesh->packages.AllPackages()) {
            if (package.second->PostStepDiagnosticsMesh != nullptr) {
                Flag("PostStepDiagnostics_"+package.first);
                package.second->PostStepDiagnosticsMesh(tm, md);
                EndFlag();
            }
        }
    }
    EndFlag();
}

