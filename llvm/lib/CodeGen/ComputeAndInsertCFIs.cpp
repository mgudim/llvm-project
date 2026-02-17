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
    int64_t ImmToAdd = 0;

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

    int64_t getImmToAdd() const { return ImmToAdd; }
    void setImmToAdd(int64_t NewImmToAdd) { ImmToAdd = NewImmToAdd; }

    bool operator==(const PreservedValueInfo &RHS) const {
      if (K != RHS.K)
        return false;
      if (getImmToAdd() != RHS.getImmToAdd())
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
      if (getImmToAdd() != 0)
        OS << ", ImmToAdd = " << getImmToAdd();
    }
  };

  static constexpr int CFADwarfNum = std::numeric_limits<int>::min();

  class CFIState {
  public:
    CFIState() {}
    CFIState(unsigned NumOfCSRs) { CSRPVInfos.resize(NumOfCSRs); }

    int64_t getCFAOffset() const;
    void setCFAOffset(int64_t CFAOffset_);
    int getCFARegister() const;
    void setCFARegister(int CFARegister_);
    PreservedValueInfo &getPVInfoForDwarfReg(int DwarfNum, CSROrdering &CSRs);

  private:
    PreservedValueInfo CFAPVInfo;
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
    unsigned generateCFI(
      const PreservedValueInfo &PrevPVInfo,
      const PreservedValueInfo &NewPVInfo
    ) const;

    CFIGenerator(MachineFunction &MF_, int DwarfNum_)
        : MF(MF_), DwarfNum(DwarfNum_) {}

  private:
    MachineFunction &MF;
    int DwarfNum;
  };

  // For debug only.
  SmallSet<int, 16> InterestingRegs;

  void init();
  void trackCFIStateBottomUp();
  bool trackPVBottomUpAndGenCFIs(
    int DwarfNum,
    Register RegToTrack,
    MachineBasicBlock &MBB, 
    const CFIGenerator &CFIGen,
    SmallSet<MachineBasicBlock *, 16> &Visited
  );

  bool adjustCFIsToMBBLayout();
  void insertCFIInstruction(const CFIBuildInfo &Info);
  void printCFIStatesForReg(int CSRDwarfNum);
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
      CFIGenerator CFIGen(MF, CSRDwarfNum);
      if (trackPVBottomUpAndGenCFIs(CSRDwarfNum, Register(Reg.value()), *RetMBB, CFIGen, Visited))
        InterestingRegs.insert(CSRDwarfNum);
    }
    // Now track CFA
    CFIGenerator CFIGen(MF, CFADwarfNum);
    SmallSet<MachineBasicBlock *, 16> Visited;
    LLVM_DEBUG(
      dbgs() << "Tracking CFA from the return MBB " << printMBBReference(*RetMBB) << "\n";
    );
    if (trackPVBottomUpAndGenCFIs(ComputeAndInsertCFIs::CFADwarfNum, TFI.getFinalCFARegister(MF), *RetMBB, CFIGen, Visited))
      InterestingRegs.insert(CFADwarfNum);
  }
  
  LLVM_DEBUG(dbgs() << "Computed CFI states for each basic block:\n";
             for (int CSRDwarfNum : CSRs.getOrderedCSRs()) {
              if (InterestingRegs.contains(CSRDwarfNum))
               printCFIStatesForReg(CSRDwarfNum);
             }
            if (InterestingRegs.contains(CFADwarfNum))
             printCFIStatesForReg(CFADwarfNum);
             );
}

bool ComputeAndInsertCFIs::trackPVBottomUpAndGenCFIs(
  int DwarfNum,
  Register RegToTrack,
  MachineBasicBlock &MBB, 
  const CFIGenerator &CFIGen,
  SmallSet<MachineBasicBlock *, 16> &Visited
) {
  Visited.insert(&MBB);
  PreservedValueInfo CurrentPVInfo = AllCFIs.getCFIStateAtMBBExit(MBB).getPVInfoForDwarfReg(DwarfNum, CSRs);
  PreservedValueInfo PVInfoBeforeInstr = CurrentPVInfo;
  LLVM_DEBUG(dbgs() << "Tracking register: " << printReg(RegToTrack, &TRI)
                    << " in " << printMBBReference(MBB) << "\n");

  MachineInstr *LocalReachingDef =
      RDI.getLocalLiveOutMIDef(&MBB, RegToTrack,
                               /*IgnoreNonLiveOut=*/false);
  bool InsertedCFIs = LocalReachingDef;

  while (LocalReachingDef) {
    LLVM_DEBUG(dbgs() << "Move of value: " << *LocalReachingDef << "in "
                      << printMBBReference(MBB) << "\n");
    const DebugLoc &DL = LocalReachingDef->getDebugLoc();
    int FrameIndex = std::numeric_limits<int>::min();

    if (Register StoredReg =
            TII.isStoreToStackSlotPostFE(*LocalReachingDef, FrameIndex)) {
      assert(FrameIndex == RegToTrack.stackSlotIndex());
      StackOffset Offset = TFI.getFrameIndexReferenceFromSP(MF, FrameIndex);
      assert(Offset.getScalable() == 0 &&
             "Scalable offsets are not supported yet!");

      RegToTrack = StoredReg;
      PVInfoBeforeInstr = PreservedValueInfo::createRegister(
          TRI.getDwarfRegNum(RegToTrack, true));
    } else if (Register LoadedReg = TII.isLoadFromStackSlotPostFE(
                   *LocalReachingDef, FrameIndex)) {
      assert(LoadedReg == RegToTrack);
      StackOffset Offset = TFI.getFrameIndexReferenceFromSP(MF, FrameIndex);
      assert(Offset.getScalable() == 0 &&
             "Scalable offsets are not supported yet!");

      RegToTrack = Register::index2StackSlot(FrameIndex);
      PVInfoBeforeInstr = PreservedValueInfo::createCFAOffset(Offset.getFixed());
    } else if (auto DstSrc = TII.isCopyInstr(*LocalReachingDef)) {
      Register DstReg = DstSrc->Destination->getReg();
      Register SrcReg = DstSrc->Source->getReg();
      assert(DstReg == RegToTrack);

      RegToTrack = SrcReg;
      PVInfoBeforeInstr = PreservedValueInfo::createRegister(
          TRI.getDwarfRegNum(RegToTrack, true));
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
      RegToTrack = RegImm->Reg;
      PVInfoBeforeInstr = PreservedValueInfo::createRegister(TRI.getDwarfRegNum(RegToTrack, true));
      PVInfoBeforeInstr.setImmToAdd(CurrentPVInfo.getImmToAdd() + RegImm->Imm);
    }
    LLVM_DEBUG(dbgs() << "Generating CFI:\n";
    dbgs() << "State before instruction:\n";
    PVInfoBeforeInstr.dump(dbgs());
    dbgs() << "\nState after instruction:\n";
    CurrentPVInfo.dump(dbgs());
    dbgs() << "\n";
    );
    unsigned CFIIndex = CFIGen.generateCFI(PVInfoBeforeInstr, CurrentPVInfo);
    CFIBuildInfos.push_back({&MBB, LocalReachingDef, DL, CFIIndex});
    LocalReachingDef = RDI.getReachingLocalMIDef(LocalReachingDef, RegToTrack);
    CurrentPVInfo = PVInfoBeforeInstr;
  }

  AllCFIs.getCFIStateAtMBBEntry(MBB).getPVInfoForDwarfReg(DwarfNum,
                                                              CSRs) = CurrentPVInfo;
  for (MachineBasicBlock *PredMBB : MBB.predecessors()) {
    if (Visited.contains(PredMBB))
      continue;
    AllCFIs.getCFIStateAtMBBExit(*PredMBB).getPVInfoForDwarfReg(DwarfNum,
                                                                CSRs) = CurrentPVInfo;
    InsertedCFIs |= trackPVBottomUpAndGenCFIs(DwarfNum, RegToTrack, *PredMBB, CFIGen, Visited);
  }
  return InsertedCFIs;
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

unsigned ComputeAndInsertCFIs::CFIGenerator::generateCFI(
    const PreservedValueInfo &PrevPVInfo, const PreservedValueInfo &NewPVInfo) const {
  assert(PrevPVInfo != NewPVInfo && "No CFI change");

  if (DwarfNum == ComputeAndInsertCFIs::CFADwarfNum) {
    switch (NewPVInfo.K) {
    case PreservedValueInfo::Register: {
      unsigned NewCFAReg = NewPVInfo.getRegister();
      if ((PrevPVInfo.K != PreservedValueInfo::Register) ||
          (PrevPVInfo.getRegister() != NewPVInfo.getRegister())) {
        if (NewPVInfo.getImmToAdd() == 0)
          return MF.addFrameInst(
              MCCFIInstruction::createDefCfaRegister(nullptr, NewCFAReg));
        return MF.addFrameInst(
            MCCFIInstruction::cfiDefCfa(nullptr, NewCFAReg,
                                       NewPVInfo.getImmToAdd()));
      }
      return MF.addFrameInst(MCCFIInstruction::createAdjustCfaOffset(
          nullptr, NewPVInfo.getImmToAdd() - PrevPVInfo.getImmToAdd()));
    }
    default:
      // TODO: handle CFA saved to stack.
      llvm_unreachable("Do not know how to generate CFI for new CFA");
    }
  }

  assert(NewPVInfo.getImmToAdd() == 0 && "Non zero immediate added to CSR value");
  const unsigned CSRDwarfNum = static_cast<unsigned>(DwarfNum);
  switch (NewPVInfo.K) {
  case PreservedValueInfo::CFAOffset:
    return MF.addFrameInst(MCCFIInstruction::createOffset(
        nullptr, CSRDwarfNum, NewPVInfo.getOffset()));
  case PreservedValueInfo::Register:
    if (NewPVInfo.getRegister() == CSRDwarfNum)
      return MF.addFrameInst(
          MCCFIInstruction::createRestore(nullptr, CSRDwarfNum));
    return MF.addFrameInst(MCCFIInstruction::createRegister(
        nullptr, CSRDwarfNum, NewPVInfo.getRegister()));
  default:
    llvm_unreachable("Do not know how to generate CFI for new CSR PVInfo");
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

ComputeAndInsertCFIs::PreservedValueInfo &
ComputeAndInsertCFIs::CFIState::getPVInfoForDwarfReg(int DwarfNum,
                                                     CSROrdering &CSRs) {
  if (DwarfNum == CFADwarfNum)
    return CFAPVInfo;

  assert(DwarfNum>= 0 && "Negative Dwarf register!");
  return CSRPVInfos[CSRs.getOrderIdxFromDwarfReg(DwarfNum)];
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
  PreservedValueInfo &CFAPVInfo = EntryState.getPVInfoForDwarfReg(ComputeAndInsertCFIs::CFADwarfNum, CSRs);

  CFAPVInfo = PreservedValueInfo::createRegister(InitialCFARegDwarfNum);
  CFAPVInfo.setImmToAdd(TFI.getInitialCFAOffset(MF));

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
    PreservedValueInfo &RetCFAPVInfo = RetCFIState.getPVInfoForDwarfReg(ComputeAndInsertCFIs::CFADwarfNum, CSRs);
    RetCFAPVInfo = PreservedValueInfo::createRegister(FinalCFARegDwarfNum);
    RetCFAPVInfo.setImmToAdd(FinalCFAOffset);
  }
}

bool ComputeAndInsertCFIs::adjustCFIsToMBBLayout() {
  LLVM_DEBUG(dbgs() << "Inserting more CFIs to adjust to block layout\n";);

  CFIState PrevCFIState = AllCFIs.getCFIStateAtMBBExit(MF.front());
  bool InsertedCFIInstr = false;
  SmallVector<int> AllDwarfNums = CSRs.getOrderedCSRs();
  AllDwarfNums.push_back(CFADwarfNum);
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == &MF.front())
      continue;

    CFIState EntryState = AllCFIs.getCFIStateAtMBBEntry(MBB);
    for (int DwarfNum : AllDwarfNums) {
      PreservedValueInfo &PrevPVInfo =
          PrevCFIState.getPVInfoForDwarfReg(DwarfNum, CSRs);
      PreservedValueInfo HasToBePVInfo =
          AllCFIs.getCFIStateAtMBBEntry(MBB).getPVInfoForDwarfReg(
              DwarfNum, CSRs);

      if (PrevPVInfo == HasToBePVInfo)
        continue;

      CFIGenerator CFIGen(MF, DwarfNum);
      unsigned CFIIndex = CFIGen.generateCFI(PrevPVInfo, HasToBePVInfo);
      auto MBBI = MBB.begin();
      DebugLoc DL = MBB.findDebugLoc(MBBI);
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    }
    PrevCFIState = AllCFIs.getCFIStateAtMBBExit(MBB);
  }
  return InsertedCFIInstr;
}

void ComputeAndInsertCFIs::printCFIStatesForReg(int CSRDwarfNum) {
  if (CSRDwarfNum == CFADwarfNum) {
    dbgs() << "CFA:\n";
  } else {
    dbgs() << "CSR Dwarf#" << CSRDwarfNum << ":\n";
  }
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
}
