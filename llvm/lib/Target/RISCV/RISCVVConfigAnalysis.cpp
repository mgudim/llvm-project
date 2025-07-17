//===- RISCVVConfigAnalysis --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the RISCV analysis of vector unit config.
//===----------------------------------------------------------------------===//

#include "RISCVVConfigAnalysis.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-vconfig-analysis"

void RISCVVConfigInfo::compute(const MachineFunction &MF) {
}

void RISCVVConfigInfo::clear() {
}

RISCVVConfigAnalysis::Result
RISCVVConfigAnalysis::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  return RISCVVConfigInfo();
}

char RISCVVConfigWrapperPass::ID = 0;


INITIALIZE_PASS_BEGIN(RISCVVConfigWrapperPass, DEBUG_TYPE,
                      "RISCV Vector Config Analysis", false, true)
INITIALIZE_PASS_END(RISCVVConfigWrapperPass, DEBUG_TYPE,
                    "RISCV Vector Config Analysis", false, true)

RISCVVConfigWrapperPass::RISCVVConfigWrapperPass()
    : MachineFunctionPass(ID) {}

void RISCVVConfigWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool RISCVVConfigWrapperPass::runOnMachineFunction(MachineFunction &MF) {
  Result = RISCVVConfigInfo();
  Result.compute(MF);
  return false;
}
