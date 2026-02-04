//===-- ComputeAndInsertCFIs.cpp - Compute and Insert CFIs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass computes and inserts Call Frame Information (CFI) instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ComputeAndInsertCFIsPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "compute-and-insert-cfis"

namespace {

class ComputeAndInsertCFIsLegacyPass : public MachineFunctionPass {
public:
  static char ID;

  ComputeAndInsertCFIsLegacyPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Compute and Insert CFIs"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

class ComputeAndInsertCFIs {
public:
  ComputeAndInsertCFIs() {}
  bool runOnMachineFunction(MachineFunction &MF);
};

} // namespace

char ComputeAndInsertCFIsLegacyPass::ID = 0;

INITIALIZE_PASS(ComputeAndInsertCFIsLegacyPass, DEBUG_TYPE,
                "Compute and Insert CFIs", false, false)

PreservedAnalyses
ComputeAndInsertCFIPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  if (!ComputeAndInsertCFIs().runOnMachineFunction(MF))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

FunctionPass *llvm::createComputeAndInsertCFIs() {
  return new ComputeAndInsertCFIsLegacyPass();
}

bool ComputeAndInsertCFIsLegacyPass::runOnMachineFunction(MachineFunction &MF) {
  return ComputeAndInsertCFIs().runOnMachineFunction(MF);
}

bool ComputeAndInsertCFIs::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  // TODO: Implement the pass logic here.
  return Changed;
}
