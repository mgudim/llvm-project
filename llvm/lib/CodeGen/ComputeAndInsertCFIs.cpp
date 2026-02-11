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

#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/ComputeAndInsertCFIsPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

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
  ComputeAndInsertCFIs(MachineFunction &MF, ReachingDefInfo &RDI)
      : MF(MF), RDI(RDI), TII(*MF.getSubtarget().getInstrInfo()),
        TRI(*MF.getSubtarget().getRegisterInfo()),
        TFI(*MF.getSubtarget().getFrameLowering()), CSRs(MF, TRI) {}
  bool runOnMachineFunction(MachineFunction &MF);

private:
  MachineFunction &MF;
  ReachingDefInfo &RDI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  const TargetFrameLowering &TFI;

  class CSROrdering {
  public:
    CSROrdering(const MachineFunction &MF, const TargetRegisterInfo &TRI);
    SmallVector<int> &getOrderedCSRs();
    /// What is the dwarf register at index `Idx in this ordering?
    int getDwarfRegFromOrderIdx(unsigned Idx);
    /// What is the index of the dwarf register `Reg`?
    unsigned getOrderIdxFromDwarfReg(int DwarfReg);

  private:
    SmallVector<int> OrderedCSRs;
    SmallDenseMap<int, unsigned> DwarfRegToOrdIdxMap;
  };

  CSROrdering CSRs;

  class PreservedValueInfo {
  public:
    enum Kind { Invalid, Register, CFAOffset };
    Kind K = Invalid;

  private:
    union {
      // Dwarf register number
      unsigned Reg;
      // CFA offset
      int64_t Offset;
    };
    int64_t AddedImm = 0;

  public:
    PreservedValueInfo() {}

    static PreservedValueInfo createCFAOffset(int64_t Offset) {
      PreservedValueInfo PVInfo;
      PVInfo.K = Kind::CFAOffset;
      PVInfo.Offset = Offset;
      return PVInfo;
    }

    static PreservedValueInfo createRegister(unsigned Reg) {
      PreservedValueInfo PVInfo;
      PVInfo.K = Kind::Register;
      PVInfo.Reg = Reg;
      return PVInfo;
    }

    bool isValid() const { return K != Kind::Invalid; }

    unsigned getRegister() const {
      assert(K == Kind::Register);
      return Reg;
    }

    int64_t getOffset() const {
      assert(K == Kind::CFAOffset);
      return Offset;
    }

    int64_t getAddedImm() const { return AddedImm; }
    void setAddedImm(int64_t NewAddedImm) { AddedImm = NewAddedImm; }

    bool operator==(const PreservedValueInfo &RHS) const {
      if (K != RHS.K)
        return false;
      if (getAddedImm() != RHS.getAddedImm())
        return false;
      switch (K) {
      case Kind::Invalid:
        return true;
      case Kind::Register:
        return getRegister() == RHS.getRegister();
      case Kind::CFAOffset:
        return getOffset() == RHS.getOffset();
      }
      llvm_unreachable("Unknown PreservedValueInfo Kind!");
    }
    bool operator!=(const PreservedValueInfo &RHS) const {
      return !(*this == RHS);
    }
    void dump(raw_ostream &OS) const {
      switch (K) {
      case Kind::Invalid:
        OS << "Invalid";
        break;
      case Kind::Register:
        OS << "In Dwarf register: " << Reg;
        break;
      case Kind::CFAOffset:
        OS << "At CFA offset: " << Offset;
        break;
      }
      if (getAddedImm() != 0)
        OS << ", AddedImm = " << getAddedImm();
    }
  };

  class CFIState {
  public:
    CFIState() {}
    CFIState(unsigned NumOfCSRs) { CSRPVInfos.resize(NumOfCSRs); }

    int64_t getCFAOffset() const;
    void setCFAOffset(int64_t CFAOffset_);
    int getCFARegister() const;
    void setCFARegister(int CFARegister_);
    PreservedValueInfo &getPVInfoForDwarfReg(int DwarfReg, CSROrdering &CSRs);
    void setPVInfoForDwarfReg(unsigned DwarfReg, PreservedValueInfo &NewPVInfo,
                              CSROrdering &CSRs);

  private:
    int64_t CFAOffset = -1;
    int CFARegister = -1;
    SmallVector<PreservedValueInfo> CSRPVInfos;
  };

  class AllCFIStates {
  public:
    void preAllocate(unsigned NumOfCSRs, unsigned NumOfMBBs);
    SmallVector<CFIState> CFIStatesOnEntry;
    SmallVector<CFIState> CFIStatesOnExit;

    CFIState &getCFIStateAtMBBEntry(const MachineBasicBlock &MBB);
    CFIState &getCFIStateAtMBBExit(const MachineBasicBlock &MBB);
  };
  AllCFIStates AllCFIs;

  SmallVector<MachineBasicBlock *> ReturnMBBs;

  struct CFIBuildInfo {
    MachineBasicBlock *MBB;
    MachineInstr *InsertAfterMI; // nullptr means insert at MBB.begin()
    DebugLoc DL;
    unsigned CFIIndex;
  };
  SmallVector<CFIBuildInfo> CFIBuildInfos;

  class CFIGenerator {
  public:
    virtual ~CFIGenerator() {}
    virtual unsigned generateCFI(
      const PreservedValueInfo &PrevPVInfo,
      const PreservedValueInfo &NewPVInfo
    );
  };

  class CFACFIGenerator : public CFIGenerator {
  public:
    CFACFIGenerator(MachineFunction &MF_) : MF(MF_) {}
    unsigned generateCFI(
      const PreservedValueInfo &PrevPVInfo,
      const PreservedValueInfo &NewPVInfo
    ) override;
  private:
    MachineFunction &MF;
  };

  class CSRCFIGenerator : public CFIGenerator {
  public:
    CSRCFIGenerator(MachineFunction &MF_, unsigned CSRDwarfNum_) : MF(MF_), CSRDwarfNum(CSRDwarfNum_) {}
    unsigned generateCFI(
      const PreservedValueInfo &PrevPVInfo,
      const PreservedValueInfo &NewPVInfo
    ) override;
  private:
    MachineFunction &MF;
    unsigned CSRDwarfNum;
  };

  // For debug only.
  SmallSet<int, 16> InterestingRegs;

  void init();
  void trackPVBottomUpAndRecordCFIs(
    const PreservedValueInfo &ValueToTrack,
    const MachineBasicBlock &MBB, 
    const CFIGenerator &CFIGen
  );
  void trackCFIStateBottomUp();
  void trackCSRBottomUp(unsigned CSRDwarfNum, Register RegToTrack,
                        MachineBasicBlock &MBB,
                        SmallSet<MachineBasicBlock *, 16> &Visited);
  void trackCFABottomUp(
    Register RegToTrack, 
    MachineBasicBlock &MBB,
    SmallSet<MachineBasicBlock *, 16> &Visited
  );

  bool adjustCFIsToMBBLayout();
  unsigned addCFIFrameInstrForCSR(unsigned DwarfReg,
                                  const PreservedValueInfo &PVInfo);
  void insertCFIInstruction(const CFIBuildInfo &Info);
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
  if (!ComputeAndInsertCFIs(MF, RDI).runOnMachineFunction(MF))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

FunctionPass *llvm::createComputeAndInsertCFIs() {
  return new ComputeAndInsertCFIsLegacyPass();
}

bool ComputeAndInsertCFIsLegacyPass::runOnMachineFunction(MachineFunction &MF) {
  ReachingDefInfo &RDI = getAnalysis<ReachingDefInfoWrapperPass>().getRDI();
  return ComputeAndInsertCFIs(MF, RDI).runOnMachineFunction(MF);
}

bool ComputeAndInsertCFIs::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "ComputeAndInsertCFIs running on " << MF.getName() << "\n");
  if (!MF.getSubtarget().emitCFIAfterFE())
    return false;

  init();
  trackCFIStateBottomUp();

  bool InsertedCFIInstr = !CFIBuildInfos.empty();
  for (const CFIBuildInfo &Info : CFIBuildInfos) {
    insertCFIInstruction(Info);
  }

  InsertedCFIInstr |= adjustCFIsToMBBLayout();

  return InsertedCFIInstr;
}

void ComputeAndInsertCFIs::trackCFIStateBottomUp() {
  for (MachineBasicBlock *RetMBB : ReturnMBBs) {
    for (int CSRDwarfNum : CSRs.getOrderedCSRs()) {
      LLVM_DEBUG(
        dbgs() << "Tracking CSR DWARF number " << CSRDwarfNum <<
        " from the return MBB " << printMBBReference(*RetMBB) << "\n";
      );
      std::optional<MCRegister> Reg = TRI.getLLVMRegNum(CSRDwarfNum, true);
      assert(
          Reg.has_value() &&
          "Dwarf register does not have a corresponding LLVM register value!");
      SmallSet<MachineBasicBlock *, 16> Visited;
      trackCSRBottomUp(CSRDwarfNum, Register(Reg.value()), *RetMBB, Visited);
    }
    SmallSet<MachineBasicBlock *, 16> Visited;
    trackCFABottomUp(TFI.getFinalCFARegister(MF), *RetMBB, Visited);
  }
  
  LLVM_DEBUG(dbgs() << "Computed CFI states for each basic block:\n";
             for (int CSRDwarfNum : CSRs.getOrderedCSRs()) {
              if (!InterestingRegs.contains(CSRDwarfNum))
                 continue;
               dbgs() << "CSR Dwarf#" << CSRDwarfNum << ":\n";
               for (const MachineBasicBlock &MBB : MF) {
                 dbgs() << "Block: " << printMBBReference(MBB) << "\n";
                 dbgs() << "Entry: ";
                AllCFIs.getCFIStateAtMBBEntry(MBB)
                    .getPVInfoForDwarfReg(CSRDwarfNum, CSRs)
                    .dump(dbgs());
                 dbgs() << "\n";
                 dbgs() << "Exit: ";
                AllCFIs.getCFIStateAtMBBExit(MBB)
                    .getPVInfoForDwarfReg(CSRDwarfNum, CSRs)
                    .dump(dbgs());
                 dbgs() << "\n";
               }
             });
}

void ComputeAndInsertCFIs::trackCSRBottomUp(
    unsigned CSRDwarfNum, Register RegToTrack, MachineBasicBlock &MBB,
    SmallSet<MachineBasicBlock *, 16> &Visited) {
  Visited.insert(&MBB);
  PreservedValueInfo &PVInfo =
      AllCFIs.getCFIStateAtMBBEntry(MBB).getPVInfoForDwarfReg(CSRDwarfNum, CSRs);
  PVInfo =
      AllCFIs.getCFIStateAtMBBExit(MBB).getPVInfoForDwarfReg(CSRDwarfNum, CSRs);

  LLVM_DEBUG(dbgs() << "Tracking register: " << printReg(RegToTrack, &TRI)
                    << " in " << printMBBReference(MBB) << "\n");

  MachineInstr *LocalReachingDef =
      RDI.getLocalLiveOutMIDef(&MBB, RegToTrack,
                               /*IgnoreNonLiveOut=*/false);
  if (LocalReachingDef)
    InterestingRegs.insert(CSRDwarfNum);

  while (LocalReachingDef) {
    LLVM_DEBUG(dbgs() << "Move of CSR value: " << *LocalReachingDef << "in "
                      << printMBBReference(MBB) << "\n");
    const DebugLoc &DL = LocalReachingDef->getDebugLoc();
    Register FrameReg = MCRegister::NoRegister;
    int FrameIndex = std::numeric_limits<int>::min();
    PreservedValueInfo NewPVInfo;

    if (Register StoredReg =
            TII.isStoreToStackSlotPostFE(*LocalReachingDef, FrameIndex)) {
      assert(FrameIndex == RegToTrack.stackSlotIndex());
      StackOffset Offset = TFI.getFrameIndexReference(MF, FrameIndex, FrameReg);
      assert(Offset.getScalable() == 0 &&
             "Scalable offsets are not supported yet!");
      assert(PVInfo.K == PreservedValueInfo::CFAOffset &&
             PVInfo.getOffset() == Offset.getFixed() && "Wrong CFIState!");

      RegToTrack = StoredReg;
      NewPVInfo = PreservedValueInfo::createRegister(
          TRI.getDwarfRegNum(RegToTrack, true));
    } else if (Register LoadedReg = TII.isLoadFromStackSlotPostFE(
                   *LocalReachingDef, FrameIndex)) {
      assert(LoadedReg == RegToTrack);
      assert(PVInfo.K == PreservedValueInfo::Register &&
             PVInfo.getRegister() == TRI.getDwarfRegNum(RegToTrack, true) &&
             "Wrong CFIState!");

      StackOffset Offset = TFI.getFrameIndexReference(MF, FrameIndex, FrameReg);
      assert(Offset.getScalable() == 0 &&
             "Scalable offsets are not supported yet!");

      RegToTrack = Register::index2StackSlot(FrameIndex);
      NewPVInfo = PreservedValueInfo::createCFAOffset(Offset.getFixed());

    } else if (auto DstSrc = TII.isCopyInstr(*LocalReachingDef)) {
      Register DstReg = DstSrc->Destination->getReg();
      Register SrcReg = DstSrc->Source->getReg();
      assert(DstReg == RegToTrack);
      assert(PVInfo.K == PreservedValueInfo::Register &&
             PVInfo.getRegister() == TRI.getDwarfRegNum(RegToTrack, true) &&
             "Wrong CFIState!");

      RegToTrack = SrcReg;
      NewPVInfo = PreservedValueInfo::createRegister(
          TRI.getDwarfRegNum(RegToTrack, true));
    } else
      llvm_unreachable("Unexpected instruction");

    unsigned CFIIndex = addCFIFrameInstrForCSR(CSRDwarfNum, PVInfo);
    CFIBuildInfos.push_back({&MBB, LocalReachingDef, DL, CFIIndex});

    PVInfo = NewPVInfo;
    LocalReachingDef = RDI.getReachingLocalMIDef(LocalReachingDef, RegToTrack);
    LLVM_DEBUG(dbgs() << "Tracking register: " << printReg(RegToTrack, &TRI)
                      << "\n");
  }

  for (MachineBasicBlock *PredMBB : MBB.predecessors()) {
    if (Visited.contains(PredMBB))
      continue;
    AllCFIs.getCFIStateAtMBBExit(*PredMBB).getPVInfoForDwarfReg(CSRDwarfNum,
                                                                CSRs) = PVInfo;
    trackCSRBottomUp(CSRDwarfNum, RegToTrack, *PredMBB, Visited);
  }
}

void ComputeAndInsertCFIs::trackCFABottomUp(
  Register RegToTrack, 
  MachineBasicBlock &MBB,
  SmallSet<MachineBasicBlock *, 16> &Visited
) {
  Visited.insert(&MBB);
  CFIState &ExitState = AllCFIs.getCFIStateAtMBBExit(MBB);
  int CFARegDwarfNum = ExitState.getCFARegister();
  int64_t CFAOffset = ExitState.getCFAOffset();
  
  LLVM_DEBUG(dbgs() << "Tracking CFA register: " << printReg(RegToTrack, &TRI)
                    << " with offset " << CFAOffset
                    << " in " << printMBBReference(MBB) << "\n");
  
  MachineInstr *LocalReachingDef =
      RDI.getLocalLiveOutMIDef(&MBB, RegToTrack,
                               /*IgnoreNonLiveOut=*/false);
  
  while (LocalReachingDef) {
    LLVM_DEBUG(dbgs() << "CFA definition: " << *LocalReachingDef << " in "
                      << printMBBReference(MBB) << "\n");
    const DebugLoc &DL = LocalReachingDef->getDebugLoc();
    unsigned CFIIndex = std::numeric_limits<unsigned>::max();
    // TODO:
    // Handle saving CFA register to stack?
    if (auto DstSrc = TII.isCopyInstr(*LocalReachingDef)) {
      Register DstReg = DstSrc->Destination->getReg();
      Register SrcReg = DstSrc->Source->getReg();
      assert(DstReg == RegToTrack && "Unexpected destination register");

      CFARegDwarfNum = TRI.getDwarfRegNum(DstReg, true);

      CFIIndex = MF.addFrameInst(
          MCCFIInstruction::createDefCfaRegister(nullptr, CFARegDwarfNum));

      RegToTrack = SrcReg;
    } else if (auto RegImm = TII.isAddImmediate(*LocalReachingDef, RegToTrack)) {
      // At any point we must have that:
      //
      //  `cfa_offset + cfa_reg = cfa` (1)
      //
      // Suppose we have an instruction which defines CFA:
      //
      //  `dst = add src, imm` (2)
      //
      // The `cfa_register` before the instruction mush have been `src` and `cfa_register`
      // after the instruction is `dst`.
      // Let us call the value of `cfa_offset` before this instruction `offset_before`
      // and the value after the instruction `offset_after`.
      // We know the `offset_after` and want to find `offset_before`.
      //
      // From (1) we have:
      //
      // `dst + offset_after = src + offset_before`
      //
      // so
      //
      // `offset_before = (dst - src) + offset_after` (3)
      //
      // But from (2):
      //
      //  `dst - src = imm` (4)
      //
      // Putting (3) and (4) together we get our desired update rule:
      //
      //  `offset_before = imm + offset_after`
      int64_t Imm = RegImm->Imm;

      if (Register SrcReg = RegImm->Reg; SrcReg != RegToTrack) {
        CFARegDwarfNum = TRI.getDwarfRegNum(RegToTrack, true);

        CFIIndex = MF.addFrameInst(MCCFIInstruction::cfiDefCfa(
          nullptr, CFARegDwarfNum, CFAOffset));

        RegToTrack = SrcReg;
      } else {

        // we're adjusting by `offset_after - offset_before = -imm`.
        CFIIndex = MF.addFrameInst(
          MCCFIInstruction::createAdjustCfaOffset(nullptr, -Imm));
      }

      CFAOffset += Imm;
    } else
      llvm_unreachable("Invalid CFA definition!");

    CFIBuildInfos.push_back({&MBB, LocalReachingDef, DL, CFIIndex});
    
    LocalReachingDef = RDI.getReachingLocalMIDef(LocalReachingDef, RegToTrack);
    LLVM_DEBUG(dbgs() << "Tracking CFA register: " << printReg(RegToTrack, &TRI)
                      << " with offset " << CFAOffset << "\n");
  }

  CFIState EntryState = AllCFIs.getCFIStateAtMBBEntry(MBB);
  EntryState.setCFARegister(CFARegDwarfNum);
  EntryState.setCFAOffset(CFAOffset);
  
  // Propagate to predecessor blocks
  for (MachineBasicBlock *PredMBB : MBB.predecessors()) {
    if (Visited.contains(PredMBB))
      continue;
    CFIState &PredExitState = AllCFIs.getCFIStateAtMBBExit(*PredMBB);
    PredExitState.setCFARegister(EntryState.getCFARegister());
    PredExitState.setCFAOffset(EntryState.getCFAOffset());
    trackCFABottomUp(RegToTrack, *PredMBB, Visited);
  }
}

void ComputeAndInsertCFIs::insertCFIInstruction(const CFIBuildInfo &Info) {
  MachineBasicBlock *MBB = Info.MBB;
  if (Info.InsertAfterMI) {
    BuildMI(*MBB, std::next(Info.InsertAfterMI->getIterator()), Info.DL,
            TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(Info.CFIIndex);
  } else {
    BuildMI(*MBB, MBB->begin(), Info.DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(Info.CFIIndex);
  }
}

unsigned ComputeAndInsertCFIs::CFACFIGenerator::generateCFI(
  const PreservedValueInfo &PrevPVInfo,
  const PreservedValueInfo &NewPVInfo
) {
  switch (NewPVInfo.K) {
    case PreservedValueInfo::Register: {
      unsigned NewCFAReg = NewPVInfo.getRegister();
      if (
        (PrevPVInfo.K != PreservedValueInfo::Register) ||
        (PrevPVInfo.getRegister() != NewPVInfo.getRegister())
      ) {
        if (NewPVInfo.getAddedImm() == 0)
          return MF.addFrameInst(
              MCCFIInstruction::createDefCfaRegister(nullptr, NewCFAReg));
        return MF.addFrameInst(MCCFIInstruction::cfiDefCfa(
          nullptr, NewCFAReg, NewPVInfo.getAddedImm()));
      }
    }
    default:
      // TODO: handle CFA saved to stack.
      llvm_unreachable("Do not know how to generate CFI for new CFA");
  }
}

unsigned ComputeAndInsertCFIs::CSRCFIGenerator::generateCFI(
  const PreservedValueInfo &PrevPVInfo,
  const PreservedValueInfo &NewPVInfo
) {
}

unsigned
ComputeAndInsertCFIs::addCFIFrameInstrForCSR(unsigned DwarfReg,
                                             const PreservedValueInfo &PVInfo) {
  switch (PVInfo.K) {
  case PreservedValueInfo::CFAOffset: {
    return MF.addFrameInst(MCCFIInstruction::createOffset(
        nullptr, DwarfReg, PVInfo.getOffset()));
  }
  case PreservedValueInfo::Register: {
    return MF.addFrameInst(MCCFIInstruction::createRegister(
        nullptr, DwarfReg, PVInfo.getRegister()));
  }
  default:
    llvm_unreachable("Invalid PreservedValueInfo!");
  }
}

ComputeAndInsertCFIs::CSROrdering::CSROrdering(
    const MachineFunction &MF, const TargetRegisterInfo &TRI) {
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();

  for (unsigned i = 0; CSRegs[i]; ++i) {
    // Use the EH dwarf reg number. This is correct as long as target emits CFIs
    // using EH Dwarf numbers (which seems to be the case for all targets).
    auto DwarfRegNum = TRI.getDwarfRegNum(CSRegs[i], /*isEH = */ true);
    OrderedCSRs.push_back(DwarfRegNum);
    DwarfRegToOrdIdxMap[DwarfRegNum] = i;
  }
}

SmallVector<int> &ComputeAndInsertCFIs::CSROrdering::getOrderedCSRs() {
  return OrderedCSRs;
}

int ComputeAndInsertCFIs::CSROrdering::getDwarfRegFromOrderIdx(unsigned Idx) {
  assert(Idx < OrderedCSRs.size() && "Wrong order index for a register.");
  return OrderedCSRs[Idx];
}

unsigned
ComputeAndInsertCFIs::CSROrdering::getOrderIdxFromDwarfReg(int DwarfReg) {
  auto Lookup = DwarfRegToOrdIdxMap.find(DwarfReg);
  assert(Lookup != DwarfRegToOrdIdxMap.end() &&
         "Register order index not found.");
  return Lookup->second;
}

int64_t ComputeAndInsertCFIs::CFIState::getCFAOffset() const {
  return CFAOffset;
}

void ComputeAndInsertCFIs::CFIState::setCFAOffset(int64_t CFAOffset_) {
  CFAOffset = CFAOffset_;
}

int ComputeAndInsertCFIs::CFIState::getCFARegister() const {
  return CFARegister;
}

void ComputeAndInsertCFIs::CFIState::setCFARegister(int CFARegister_) {
  CFARegister = CFARegister_;
}

ComputeAndInsertCFIs::PreservedValueInfo &
ComputeAndInsertCFIs::CFIState::getPVInfoForDwarfReg(int DwarfReg,
                                                     CSROrdering &CSRs) {
  assert(DwarfReg >= 0 && "Negative Dwarf register!");
  return CSRPVInfos[CSRs.getOrderIdxFromDwarfReg(DwarfReg)];
}

void ComputeAndInsertCFIs::CFIState::setPVInfoForDwarfReg(
    unsigned DwarfReg, PreservedValueInfo &NewPVInfo, CSROrdering &CSRs) {
  CSRPVInfos[CSRs.getOrderIdxFromDwarfReg(DwarfReg)] = NewPVInfo;
}

ComputeAndInsertCFIs::CFIState &
ComputeAndInsertCFIs::AllCFIStates::getCFIStateAtMBBEntry(
    const MachineBasicBlock &MBB) {
  return CFIStatesOnEntry[MBB.getNumber()];
}

ComputeAndInsertCFIs::CFIState &
ComputeAndInsertCFIs::AllCFIStates::getCFIStateAtMBBExit(
    const MachineBasicBlock &MBB) {
  return CFIStatesOnExit[MBB.getNumber()];
}

void ComputeAndInsertCFIs::AllCFIStates::preAllocate(unsigned NumOfCSRs,
                                                     unsigned NumOfMBBs) {
  CFIState DefaultState(NumOfCSRs);
  CFIStatesOnEntry = SmallVector<CFIState>(NumOfMBBs, DefaultState);
  CFIStatesOnExit = SmallVector<CFIState>(NumOfMBBs, DefaultState);
}

void ComputeAndInsertCFIs::init() {
  unsigned NumOfCSRs = CSRs.getOrderedCSRs().size();
  assert(NumOfCSRs &&
         "initCSROrder should be called before creating CFIState.");
  unsigned NumOfMBBs = MF.getNumBlockIDs();
  AllCFIs.preAllocate(NumOfCSRs, NumOfMBBs);

  int EntyMBBID = MF.front().getNumber();
  assert(EntyMBBID >= 0 && "MBB not in machine function.");
  CFIState &EntryState = AllCFIs.getCFIStateAtMBBEntry(MF.front());

  int InitialCFARegDwarfNum = TRI.getDwarfRegNum(TFI.getInitialCFARegister(MF), true);
  assert(InitialCFARegDwarfNum >= 0 && "Negative Dwarf register!");
  EntryState.setCFARegister(InitialCFARegDwarfNum);
  EntryState.setCFAOffset(TFI.getInitialCFAOffset(MF));

  for (int CSRDwarfNum : CSRs.getOrderedCSRs()) {
    assert(CSRDwarfNum >= 0 && "Negative Dwarf register!");

    EntryState.getPVInfoForDwarfReg(CSRDwarfNum, CSRs) =
        PreservedValueInfo::createRegister((unsigned)CSRDwarfNum);
  }

  int FinalCFARegDwarfNum = TRI.getDwarfRegNum(TFI.getFinalCFARegister(MF), true);
  assert(FinalCFARegDwarfNum >= 0 && "Negative Dwarf register!");
  int64_t FinalCFAOffset = TFI.getFinalCFAOffset(MF);
  for (const MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;
    ReturnMBBs.push_back(const_cast<MachineBasicBlock *>(&MBB));
    CFIState &RetCFIState = AllCFIs.getCFIStateAtMBBExit(MBB);
    RetCFIState = EntryState;
    RetCFIState.setCFARegister(FinalCFARegDwarfNum);
    RetCFIState.setCFAOffset(FinalCFAOffset);
  }
}

bool ComputeAndInsertCFIs::adjustCFIsToMBBLayout() {
  CFIState PrevCFIState = AllCFIs.getCFIStateAtMBBExit(MF.front());
  bool InsertedCFIInstr = false;
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == &MF.front())
      continue;

    CFIState EntryState = AllCFIs.getCFIStateAtMBBEntry(MBB);
    for (int CSRDwarfNum : CSRs.getOrderedCSRs()) {
      PreservedValueInfo &PrevPVInfo =
          PrevCFIState.getPVInfoForDwarfReg(CSRDwarfNum, CSRs);
      PreservedValueInfo HasToBePVInfo =
          AllCFIs.getCFIStateAtMBBEntry(MBB).getPVInfoForDwarfReg(
              CSRDwarfNum, CSRs);
      if (PrevPVInfo == HasToBePVInfo)
        continue;

      unsigned CFIIndex = std::numeric_limits<unsigned>::max();
      switch (HasToBePVInfo.K) {
      case PreservedValueInfo::CFAOffset: {
        CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
            nullptr, CSRDwarfNum, HasToBePVInfo.getOffset()));
        break;
      }
      case PreservedValueInfo::Register: {
        CFIIndex = MF.addFrameInst(MCCFIInstruction::createRegister(
            nullptr, CSRDwarfNum, HasToBePVInfo.getRegister()));
        break;
      }
      default:
        llvm_unreachable("Invalid PreservedValueInfo!");
      }
      auto MBBI = MBB.begin();
      DebugLoc DL = MBB.findDebugLoc(MBBI);
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    }
  }
  return InsertedCFIInstr;
}
