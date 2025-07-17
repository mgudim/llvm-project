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

static bool isLMUL1OrSmaller(RISCVVType::VLMUL LMUL) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(LMUL);
  return Fractional || LMul == 1;
}

bool RISCVVConfigInfo::areCompatibleVTYPEs(uint64_t CurVType, uint64_t NewVType,
                                const DemandedFields &Used) {
  switch (Used.SEW) {
  case DemandedFields::SEWNone:
    break;
  case DemandedFields::SEWEqual:
    if (RISCVVType::getSEW(CurVType) != RISCVVType::getSEW(NewVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqual:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqualAndLessThan64:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType) ||
        RISCVVType::getSEW(NewVType) >= 64)
      return false;
    break;
  }

  switch (Used.LMUL) {
  case DemandedFields::LMULNone:
    break;
  case DemandedFields::LMULEqual:
    if (RISCVVType::getVLMUL(CurVType) != RISCVVType::getVLMUL(NewVType))
      return false;
    break;
  case DemandedFields::LMULLessThanOrEqualToM1:
    if (!isLMUL1OrSmaller(RISCVVType::getVLMUL(NewVType)))
      return false;
    break;
  }

  if (Used.SEWLMULRatio) {
    auto Ratio1 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(CurVType),
                                              RISCVVType::getVLMUL(CurVType));
    auto Ratio2 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(NewVType),
                                              RISCVVType::getVLMUL(NewVType));
    if (Ratio1 != Ratio2)
      return false;
  }

  if (Used.TailPolicy && RISCVVType::isTailAgnostic(CurVType) !=
                             RISCVVType::isTailAgnostic(NewVType))
    return false;
  if (Used.MaskPolicy && RISCVVType::isMaskAgnostic(CurVType) !=
                             RISCVVType::isMaskAgnostic(NewVType))
    return false;
  return true;
}

bool VSETVLIInfo::hasCompatibleVTYPE(const DemandedFields &Used,
                        const VSETVLIInfo &Require) const {
  return RISCVVConfigInfo::areCompatibleVTYPEs(Require.encodeVTYPE(), encodeVTYPE(), Used);
}

bool RISCVVConfigInfo::haveVectorOp() {
  return HaveVectorOp;
}

void RISCVVConfigInfo::compute(const MachineFunction &MF) {
  //assert(BlockInfo.empty() && "Expect empty block infos");
  //BlockInfo.resize(MF.getNumBlockIDs());

  //bool HaveVectorOp = false;

  //// Phase 1 - determine how VL/VTYPE are affected by the each block.
  //for (const MachineBasicBlock &MBB : MF) {
  //  VSETVLIInfo TmpStatus;
  //  HaveVectorOp |= computeVLVTYPEChanges(MBB, TmpStatus);
  //  // Initial exit state is whatever change we found in the block.
  //  BlockData &BBInfo = BlockInfo[MBB.getNumber()];
  //  BBInfo.Exit = TmpStatus;
  //  LLVM_DEBUG(dbgs() << "Initial exit state of " << printMBBReference(MBB)
  //                    << " is " << BBInfo.Exit << "\n");

  //}

  //// If we didn't find any instructions that need VSETVLI, we're done.
  //if (!HaveVectorOp) {
  //  BlockInfo.clear();
  //  return false;
  //}

  //// Phase 2 - determine the exit VL/VTYPE from each block. We add all
  //// blocks to the list here, but will also add any that need to be revisited
  //// during Phase 2 processing.
  //for (const MachineBasicBlock &MBB : MF) {
  //  WorkList.push(&MBB);
  //  BlockInfo[MBB.getNumber()].InQueue = true;
  //}
  //while (!WorkList.empty()) {
  //  const MachineBasicBlock &MBB = *WorkList.front();
  //  WorkList.pop();
  //  computeIncomingVLVTYPE(MBB);
  //}

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
