//===- llvm/CodeGen/ComputeAndInsertCFIsPass.h ------------------*- C++ -*-===//
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

#ifndef LLVM_CODEGEN_COMPUTEANDINSERTCFISPASS_H
#define LLVM_CODEGEN_COMPUTEANDINSERTCFISPASS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class MachineFunction;

struct ComputeAndInsertCFIPass : public PassInfoMixin<ComputeAndInsertCFIPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_COMPUTEANDINSERTCFISPASS_H
