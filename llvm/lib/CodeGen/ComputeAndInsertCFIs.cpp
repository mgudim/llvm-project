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

  class CSRSavedLocation {
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

  public:
    CSRSavedLocation() {}

    static CSRSavedLocation createCFAOffset(int64_t Offset) {
      CSRSavedLocation Loc;
      Loc.K = Kind::CFAOffset;
      Loc.Offset = Offset;
      return Loc;
    }

    static CSRSavedLocation createRegister(unsigned Reg) {
      CSRSavedLocation Loc;
      Loc.K = Kind::Register;
      Loc.Reg = Reg;
      return Loc;
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

    bool operator==(const CSRSavedLocation &RHS) const {
      if (K != RHS.K)
        return false;
      switch (K) {
      case Kind::Invalid:
        return true;
      case Kind::Register:
        return getRegister() == RHS.getRegister();
      case Kind::CFAOffset:
        return getOffset() == RHS.getOffset();
      }
      llvm_unreachable("Unknown CSRSavedLocation Kind!");
    }
    bool operator!=(const CSRSavedLocation &RHS) const {
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
    }
  };

  class CFIState {
  public:
    CFIState() {}
    CFIState(unsigned NumOfCSRs) { CSRLocations.resize(NumOfCSRs); }

    int64_t getCFAOffset() const;
    void setCFAOffset(int64_t CFAOffset_);
    Register getCFARegister() const;
    void setCFARegister(Register CFARegister_);
    CSRSavedLocation &getLocationForDwarfReg(int DwarfReg, CSROrdering &CSRs);
    void setLocationForDwarfReg(unsigned DwarfReg,
                                CSRSavedLocation &NewLocation, CSROrdering &CSRs);

  private:
    int64_t CFAOffset = -1;
    Register CFARegister = MCRegister::NoRegister;
    SmallVector<CSRSavedLocation> CSRLocations;
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

  // For debug only.
  SmallSet<int, 16> InterestingRegs;

  void init();
  void trackCFIStateBottomUp();
  void trackRegBottomUp(unsigned CSRDwarfNum, Register Reg,
                        MachineBasicBlock &MBB,
                        SmallSet<MachineBasicBlock *, 16> &Visited);
  bool adjustCFIsToMBBLayout();
  unsigned addCFIFrameInstrForReg(unsigned DwarfReg,
                                  const CSRSavedLocation &Location);
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

  // Check if the target wants to emit CFI after front-end
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
      trackRegBottomUp(CSRDwarfNum, Register(Reg.value()), *RetMBB, Visited);
    }
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
                    .getLocationForDwarfReg(CSRDwarfNum, CSRs)
                    .dump(dbgs());
                 dbgs() << "\n";
                 dbgs() << "Exit: ";
                AllCFIs.getCFIStateAtMBBExit(MBB)
                    .getLocationForDwarfReg(CSRDwarfNum, CSRs)
                    .dump(dbgs());
                 dbgs() << "\n";
               }
             });
}

void ComputeAndInsertCFIs::trackRegBottomUp(
    unsigned CSRDwarfNum, Register RegToTrack, MachineBasicBlock &MBB,
    SmallSet<MachineBasicBlock *, 16> &Visited) {
  Visited.insert(&MBB);
  CSRSavedLocation &Location =
      AllCFIs.getCFIStateAtMBBEntry(MBB).getLocationForDwarfReg(CSRDwarfNum, CSRs);
  Location =
      AllCFIs.getCFIStateAtMBBExit(MBB).getLocationForDwarfReg(CSRDwarfNum, CSRs);

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
    CSRSavedLocation NewLocation;

    if (Register StoredReg =
            TII.isStoreToStackSlotPostFE(*LocalReachingDef, FrameIndex)) {
      assert(FrameIndex == RegToTrack.stackSlotIndex());
      StackOffset Offset = TFI.getFrameIndexReference(MF, FrameIndex, FrameReg);
      assert(Offset.getScalable() == 0 &&
             "Scalable offsets are not supported yet!");
      assert(Location.K == CSRSavedLocation::CFAOffset &&
             Location.getOffset() == Offset.getFixed() && "Wrong CFIState!");

      RegToTrack = StoredReg;
      NewLocation = CSRSavedLocation::createRegister(
          TRI.getDwarfRegNum(RegToTrack, true));
    } else if (Register LoadedReg = TII.isLoadFromStackSlotPostFE(
                   *LocalReachingDef, FrameIndex)) {
      assert(LoadedReg == RegToTrack);
      assert(Location.K == CSRSavedLocation::Register &&
             Location.getRegister() == TRI.getDwarfRegNum(RegToTrack, true) &&
             "Wrong CFIState!");

      StackOffset Offset = TFI.getFrameIndexReference(MF, FrameIndex, FrameReg);
      assert(Offset.getScalable() == 0 &&
             "Scalable offsets are not supported yet!");

      RegToTrack = Register::index2StackSlot(FrameIndex);
      NewLocation = CSRSavedLocation::createCFAOffset(Offset.getFixed());

    } else if (auto DstSrc = TII.isCopyInstr(*LocalReachingDef)) {
      Register DstReg = DstSrc->Destination->getReg();
      Register SrcReg = DstSrc->Source->getReg();
      assert(DstReg == RegToTrack);
      assert(Location.K == CSRSavedLocation::Register &&
             Location.getRegister() == TRI.getDwarfRegNum(RegToTrack, true) &&
             "Wrong CFIState!");

      RegToTrack = SrcReg;
      NewLocation = CSRSavedLocation::createRegister(
          TRI.getDwarfRegNum(RegToTrack, true));
    } else
      llvm_unreachable("Unexpected instruction");

    unsigned CFIIndex = addCFIFrameInstrForReg(CSRDwarfNum, Location);
    CFIBuildInfos.push_back({&MBB, LocalReachingDef, DL, CFIIndex});

    Location = NewLocation;
    LocalReachingDef = RDI.getReachingLocalMIDef(LocalReachingDef, RegToTrack);
    LLVM_DEBUG(dbgs() << "Tracking register: " << printReg(RegToTrack, &TRI)
                      << "\n");
  }

  for (MachineBasicBlock *PredMBB : MBB.predecessors()) {
    if (!Visited.contains(PredMBB)) {
      AllCFIs.getCFIStateAtMBBExit(*PredMBB).getLocationForDwarfReg(
          CSRDwarfNum, CSRs) = Location;
      trackRegBottomUp(CSRDwarfNum, RegToTrack, *PredMBB, Visited);
    }
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

unsigned
ComputeAndInsertCFIs::addCFIFrameInstrForReg(unsigned DwarfReg,
                                             const CSRSavedLocation &Location) {
  switch (Location.K) {
  case CSRSavedLocation::CFAOffset: {
    return MF.addFrameInst(MCCFIInstruction::createOffset(
        nullptr, DwarfReg, Location.getOffset()));
    break;
  }
  case CSRSavedLocation::Register: {
    return MF.addFrameInst(MCCFIInstruction::createRegister(
        nullptr, DwarfReg, Location.getRegister()));
    break;
  }
  default:
    llvm_unreachable("Invalid CSRSavedLocation!");
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

Register ComputeAndInsertCFIs::CFIState::getCFARegister() const {
  return CFARegister;
}

void ComputeAndInsertCFIs::CFIState::setCFARegister(Register CFARegister_) {
  CFARegister = CFARegister_;
}

ComputeAndInsertCFIs::CSRSavedLocation &
ComputeAndInsertCFIs::CFIState::getLocationForDwarfReg(int DwarfReg, CSROrdering &CSRs) {
  assert(DwarfReg >= 0 && "Negative Dwarf register!");
  return CSRLocations[CSRs.getOrderIdxFromDwarfReg(DwarfReg)];
}

void ComputeAndInsertCFIs::CFIState::setLocationForDwarfReg(
    unsigned DwarfReg, CSRSavedLocation &NewLocation, CSROrdering &CSRs) {
  CSRLocations[CSRs.getOrderIdxFromDwarfReg(DwarfReg)] = NewLocation;
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
  for (int CSRDwarfNum : CSRs.getOrderedCSRs()) {
    assert(CSRDwarfNum >= 0 && "Negative Dwarf register!");

    EntryState.getLocationForDwarfReg(CSRDwarfNum, CSRs) =
        CSRSavedLocation::createRegister((unsigned)CSRDwarfNum);
  }

  for (const MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;
    ReturnMBBs.push_back(const_cast<MachineBasicBlock *>(&MBB));
    AllCFIs.getCFIStateAtMBBExit(MBB) = EntryState;
  }
  // TODO: Set initial CFAOffset and CFARegister
}

bool ComputeAndInsertCFIs::adjustCFIsToMBBLayout() {
  CFIState PrevCFIState = AllCFIs.getCFIStateAtMBBExit(MF.front());
  bool InsertedCFIInstr = false;
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == &MF.front())
      continue;

    CFIState EntryState = AllCFIs.getCFIStateAtMBBEntry(MBB);
    for (int CSRDwarfNum : CSRs.getOrderedCSRs()) {
      CSRSavedLocation &PrevLoc =
          PrevCFIState.getLocationForDwarfReg(CSRDwarfNum, CSRs);
      CSRSavedLocation HasToBeLoc =
          AllCFIs.getCFIStateAtMBBEntry(MBB).getLocationForDwarfReg(
              CSRDwarfNum, CSRs);
      if (PrevLoc == HasToBeLoc)
        continue;

      unsigned CFIIndex = std::numeric_limits<unsigned>::max();
      switch (HasToBeLoc.K) {
      case CSRSavedLocation::CFAOffset: {
        CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
            nullptr, CSRDwarfNum, HasToBeLoc.getOffset()));
        break;
      }
      case CSRSavedLocation::Register: {
        CFIIndex = MF.addFrameInst(MCCFIInstruction::createRegister(
            nullptr, CSRDwarfNum, HasToBeLoc.getRegister()));
        break;
      }
      default:
        llvm_unreachable("Invalid CSRSavedLocation!");
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
