// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve2 -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve2 -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefix=CPP-CHECK
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64 -target-feature +sve2 -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64 -target-feature +sve2 -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefix=CPP-CHECK

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

// CHECK-LABEL: @test_svadalp_s16_z(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 8 x i16> zeroinitializer
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[TMP1]], <vscale x 16 x i8> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP2]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s16_zu10__SVBool_tu11__SVInt16_tu10__SVInt8_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 8 x i16> zeroinitializer
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[TMP1]], <vscale x 16 x i8> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP2]]
//
svint16_t test_svadalp_s16_z(svbool_t pg, svint16_t op1, svint8_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s16,_z,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s32_z(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 4 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[TMP1]], <vscale x 8 x i16> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP2]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s32_zu10__SVBool_tu11__SVInt32_tu11__SVInt16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 4 x i32> zeroinitializer
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[TMP1]], <vscale x 8 x i16> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP2]]
//
svint32_t test_svadalp_s32_z(svbool_t pg, svint32_t op1, svint16_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s32,_z,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s64_z(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 2 x i64> zeroinitializer
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[TMP1]], <vscale x 4 x i32> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP2]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s64_zu10__SVBool_tu11__SVInt64_tu11__SVInt32_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 2 x i64> zeroinitializer
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[TMP1]], <vscale x 4 x i32> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP2]]
//
svint64_t test_svadalp_s64_z(svbool_t pg, svint64_t op1, svint32_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s64,_z,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u16_z(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 8 x i16> zeroinitializer
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[TMP1]], <vscale x 16 x i8> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP2]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u16_zu10__SVBool_tu12__SVUint16_tu11__SVUint8_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 8 x i16> zeroinitializer
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[TMP1]], <vscale x 16 x i8> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP2]]
//
svuint16_t test_svadalp_u16_z(svbool_t pg, svuint16_t op1, svuint8_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u16,_z,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u32_z(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 4 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[TMP1]], <vscale x 8 x i16> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP2]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u32_zu10__SVBool_tu12__SVUint32_tu12__SVUint16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 4 x i32> zeroinitializer
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[TMP1]], <vscale x 8 x i16> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP2]]
//
svuint32_t test_svadalp_u32_z(svbool_t pg, svuint32_t op1, svuint16_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u32,_z,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u64_z(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 2 x i64> zeroinitializer
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[TMP1]], <vscale x 4 x i32> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP2]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u64_zu10__SVBool_tu12__SVUint64_tu12__SVUint32_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = select <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 2 x i64> zeroinitializer
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[TMP1]], <vscale x 4 x i32> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP2]]
//
svuint64_t test_svadalp_u64_z(svbool_t pg, svuint64_t op1, svuint32_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u64,_z,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s16_m(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s16_mu10__SVBool_tu11__SVInt16_tu10__SVInt8_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svadalp_s16_m(svbool_t pg, svint16_t op1, svint8_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s16,_m,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s32_m(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s32_mu10__SVBool_tu11__SVInt32_tu11__SVInt16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svadalp_s32_m(svbool_t pg, svint32_t op1, svint16_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s32,_m,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s64_m(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s64_mu10__SVBool_tu11__SVInt64_tu11__SVInt32_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svadalp_s64_m(svbool_t pg, svint64_t op1, svint32_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s64,_m,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u16_m(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u16_mu10__SVBool_tu12__SVUint16_tu11__SVUint8_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svadalp_u16_m(svbool_t pg, svuint16_t op1, svuint8_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u16,_m,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u32_m(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u32_mu10__SVBool_tu12__SVUint32_tu12__SVUint16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svadalp_u32_m(svbool_t pg, svuint32_t op1, svuint16_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u32,_m,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u64_m(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u64_mu10__SVBool_tu12__SVUint64_tu12__SVUint32_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svadalp_u64_m(svbool_t pg, svuint64_t op1, svuint32_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u64,_m,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s16_x(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s16_xu10__SVBool_tu11__SVInt16_tu10__SVInt8_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svadalp_s16_x(svbool_t pg, svint16_t op1, svint8_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s16,_x,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s32_x(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s32_xu10__SVBool_tu11__SVInt32_tu11__SVInt16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svadalp_s32_x(svbool_t pg, svint32_t op1, svint16_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s32,_x,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_s64_x(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_s64_xu10__SVBool_tu11__SVInt64_tu11__SVInt32_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svadalp_s64_x(svbool_t pg, svint64_t op1, svint32_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_s64,_x,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u16_x(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u16_xu10__SVBool_tu12__SVUint16_tu11__SVUint8_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP1:%.*]], <vscale x 16 x i8> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svadalp_u16_x(svbool_t pg, svuint16_t op1, svuint8_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u16,_x,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u32_x(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u32_xu10__SVBool_tu12__SVUint32_tu12__SVUint16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP1:%.*]], <vscale x 8 x i16> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svadalp_u32_x(svbool_t pg, svuint32_t op1, svuint16_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u32,_x,)(pg, op1, op2);
}

// CHECK-LABEL: @test_svadalp_u64_x(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
// CPP-CHECK-LABEL: @_Z18test_svadalp_u64_xu10__SVBool_tu12__SVUint64_tu12__SVUint32_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP1:%.*]], <vscale x 4 x i32> [[OP2:%.*]])
// CPP-CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svadalp_u64_x(svbool_t pg, svuint64_t op1, svuint32_t op2)
{
  return SVE_ACLE_FUNC(svadalp,_u64,_x,)(pg, op1, op2);
}
