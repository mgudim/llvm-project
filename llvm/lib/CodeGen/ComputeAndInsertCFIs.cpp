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
#include "llvm/CodeGen/ReachingDefAnalysis.h"
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

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ReachingDefInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class ComputeAndInsertCFIs {
public:
  ComputeAndInsertCFIs(ReachingDefInfo &RDI) : RDI(RDI) {}
  bool runOnMachineFunction(MachineFunction &MF);

private:
  ReachingDefInfo &RDI;
};

} // namespace

char ComputeAndInsertCFIsLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(ComputeAndInsertCFIsLegacyPass, DEBUG_TYPE,
                      "Compute and Insert CFIs", false, false)
INITIALIZE_PASS_DEPENDENCY(ReachingDefInfoWrapperPass)
INITIALIZE_PASS_END(ComputeAndInsertCFIsLegacyPass, DEBUG_TYPE,
                    "Compute and Insert CFIs", false, false)

PreservedAnalyses
ComputeAndInsertCFIPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  ReachingDefInfo &RDI = MFAM.getResult<ReachingDefAnalysis>(MF);
  if (!ComputeAndInsertCFIs(RDI).runOnMachineFunction(MF))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

FunctionPass *llvm::createComputeAndInsertCFIs() {
  return new ComputeAndInsertCFIsLegacyPass();
}

bool ComputeAndInsertCFIsLegacyPass::runOnMachineFunction(MachineFunction &MF) {
  ReachingDefInfo &RDI = getAnalysis<ReachingDefInfoWrapperPass>().getRDI();
  return ComputeAndInsertCFIs(RDI).runOnMachineFunction(MF);
}

bool ComputeAndInsertCFIs::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  // TODO: Implement the pass logic here.
  return Changed;
}
