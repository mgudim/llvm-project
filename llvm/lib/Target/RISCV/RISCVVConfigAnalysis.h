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

#ifndef LLVM_LIB_TARGET_RISCV_RISCVVCONFIGANALYSIS_H
#define LLVM_LIB_TARGET_RISCV_RISCVVCONFIGANALYSIS_H

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>
#include <vector>
using namespace llvm;

namespace llvm {

class RISCVVConfigInfo {
public:
  RISCVVConfigInfo() {}
  void compute(const MachineFunction &MF);
  void clear();
};

class RISCVVConfigAnalysis
    : public AnalysisInfoMixin<RISCVVConfigAnalysis> {
  friend AnalysisInfoMixin<RISCVVConfigAnalysis>;
  static AnalysisKey Key;

public:
  using Result = RISCVVConfigInfo;
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

class RISCVVConfigWrapperPass : public MachineFunctionPass {
  RISCVVConfigInfo Result;
public:
  static char ID;

  RISCVVConfigWrapperPass();

  void getAnalysisUsage(AnalysisUsage &) const override;
  bool runOnMachineFunction(MachineFunction &) override;
  void releaseMemory() override { Result.clear(); }
  RISCVVConfigInfo &getResult() { return Result; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVVCONFIGANALYSIS_H
