// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "info.h"
#include "nccl_common.h"

#include "comms/utils/commSpecs.h"

/**
 * @file DataTypeConv.h
 * @brief Provides type conversion utilities between NCCL's native types and
 * Meta's shared communication library types.
 *
 * This header contains conversion functions for translating between NCCL enums
 * and Meta's shared communication library enums. It also includes static
 * assertions to verify that the enum values match exactly, ensuring that static
 * casting can be used safely without performance overhead during conversion.
 */

namespace meta::comms {

// We intentionally don't provide comm -> nccl conversion functions because
// we might have more enums in CommPattern/CommFunc compared to ncclPattern_t
// and ncclFunc_t, there is no way to safely convert them back.

inline CommPattern ncclToCommPattern(ncclPattern_t ncclPattern) {
  return static_cast<CommPattern>(ncclPattern);
}

constexpr CommFunc ncclToCommFunc(ncclFunc_t ncclFunc) {
  switch (ncclFunc) {
    case ::ncclFuncBroadcast:
      return CommFunc::Broadcast;
    case ::ncclFuncReduce:
      return CommFunc::Reduce;
    case ::ncclFuncAllGather:
      return CommFunc::AllGather;
    case ::ncclFuncReduceScatter:
      return CommFunc::ReduceScatter;
    case ::ncclFuncAllReduce:
      return CommFunc::AllReduce;
    case ::ncclFuncSendRecv:
      return CommFunc::SendRecv;
    case ::ncclFuncSend:
      return CommFunc::Send;
    case ::ncclFuncRecv:
      return CommFunc::Recv;
    default:
      return CommFunc::NumFuncs;
  }
}

inline commDataType_t ncclToCommDataType(ncclDataType_t ncclDataType) {
  return static_cast<commDataType_t>(ncclDataType);
}

inline commRedOp_t ncclToCommRedOp(ncclRedOp_t ncclRedOp) {
  switch (ncclRedOp) {
    case ::ncclSum:
      return commSum;
    case ::ncclProd:
      return commProd;
    case ::ncclMax:
      return commMax;
    case ::ncclMin:
      return commMin;
    case ::ncclAvg:
      return commAvg;
    default:
      return commNumOps;
  }
}

inline CommAlgo ncclToCommAlgo(int ncclAlgo) {
  return static_cast<CommAlgo>(ncclAlgo);
}

inline CommProtocol ncclToCommProtocol(int ncclProtocol) {
  return static_cast<CommProtocol>(ncclProtocol);
}

// The code below is to check whether the value of Comm* enums match the nccl*
// enums. This is to ensure we can directly use static cast to convert between
// the two enums, eliminating performance overhead from the conversion.

/******************************************************************************
 * Begin of static check to ensure CommPattern enum matches ncclPattern enum  *
 *****************************************************************************/
static_assert(
    static_cast<int>(CommPattern::Ring) == static_cast<int>(::ncclPatternRing),
    "CommPattern::Ring must match ncclPatternRing");

static_assert(
    static_cast<int>(CommPattern::RingTwice) ==
        static_cast<int>(::ncclPatternRingTwice),
    "CommPattern::RingTwice must match ncclPatternRingTwice");

static_assert(
    static_cast<int>(CommPattern::PipelineFrom) ==
        static_cast<int>(::ncclPatternPipelineFrom),
    "CommPattern::PipelineFrom must match ncclPatternPipelineFrom");

static_assert(
    static_cast<int>(CommPattern::PipelineTo) ==
        static_cast<int>(::ncclPatternPipelineTo),
    "CommPattern::PipelineTo must match ncclPatternPipelineTo");

static_assert(
    static_cast<int>(CommPattern::TreeUp) ==
        static_cast<int>(::ncclPatternTreeUp),
    "CommPattern::TreeUp must match ncclPatternTreeUp");

static_assert(
    static_cast<int>(CommPattern::TreeDown) ==
        static_cast<int>(::ncclPatternTreeDown),
    "CommPattern::TreeDown must match ncclPatternTreeDown");

static_assert(
    static_cast<int>(CommPattern::TreeUpDown) ==
        static_cast<int>(::ncclPatternTreeUpDown),
    "CommPattern::TreeUpDown must match ncclPatternTreeUpDown");

static_assert(
    static_cast<int>(CommPattern::CollnetChain) ==
        static_cast<int>(::ncclPatternCollnetChain),
    "CommPattern::CollnetChain must match ncclPatternCollnetChain");

static_assert(
    static_cast<int>(CommPattern::CollnetDirect) ==
        static_cast<int>(::ncclPatternCollnetDirect),
    "CommPattern::CollnetDirect must match ncclPatternCollnetDirect");

static_assert(
    static_cast<int>(CommPattern::Nvls) == static_cast<int>(::ncclPatternNvls),
    "CommPattern::Nvls must match ncclPatternNvls");

static_assert(
    static_cast<int>(CommPattern::NvlsTree) ==
        static_cast<int>(::ncclPatternNvlsTree),
    "CommPattern::NvlsTree must match ncclPatternNvlsTree");

static_assert(
    static_cast<int>(CommPattern::PatUp) ==
        static_cast<int>(::ncclPatternPatUp),
    "CommPattern::PatUp must match ncclPatternPatUp");

static_assert(
    static_cast<int>(CommPattern::PatDown) ==
        static_cast<int>(::ncclPatternPatDown),
    "CommPattern::PatDown must match ncclPatternPatDown");

static_assert(
    static_cast<int>(CommPattern::Send) == static_cast<int>(::ncclPatternSend),
    "CommPattern::Send must match ncclPatternSend");

static_assert(
    static_cast<int>(CommPattern::Recv) == static_cast<int>(::ncclPatternRecv),
    "CommPattern::Recv must match ncclPatternRecv");

/******************************************************************************
 * Begin of static check to ensure commDataType_t enum matches ncclDataType_t *
 *****************************************************************************/
static_assert(
    static_cast<int>(commInt8) == static_cast<int>(::ncclInt8),
    "commInt8 must match ncclInt8");

static_assert(
    static_cast<int>(commUint8) == static_cast<int>(::ncclUint8),
    "commUint8 must match ncclUint8");

static_assert(
    static_cast<int>(commInt32) == static_cast<int>(::ncclInt32),
    "commInt32 must match ncclInt32");

static_assert(
    static_cast<int>(commUint32) == static_cast<int>(::ncclUint32),
    "commUint32 must match ncclUint32");

static_assert(
    static_cast<int>(commInt64) == static_cast<int>(::ncclInt64),
    "commInt64 must match ncclInt64");

static_assert(
    static_cast<int>(commUint64) == static_cast<int>(::ncclUint64),
    "commUint64 must match ncclUint64");

static_assert(
    static_cast<int>(commFloat16) == static_cast<int>(::ncclFloat16),
    "commFloat16 must match ncclFloat16");

static_assert(
    static_cast<int>(commFloat32) == static_cast<int>(::ncclFloat32),
    "commFloat32 must match ncclFloat32");

static_assert(
    static_cast<int>(commFloat64) == static_cast<int>(::ncclFloat64),
    "commFloat64 must match ncclFloat64");

static_assert(
    static_cast<int>(commBfloat16) == static_cast<int>(::ncclBfloat16),
    "commBfloat16 must match ncclBfloat16");

static_assert(
    static_cast<int>(commNumTypes) == static_cast<int>(::ncclNumTypes),
    "commNumTypes must match ncclNumTypes");

/******************************************************************************
 * Begin of static check to ensure CommAlgo enum matches NCCL algorithm       *
 *****************************************************************************/
static_assert(
    static_cast<int>(CommAlgo::Tree) == static_cast<int>(NCCL_ALGO_TREE),
    "CommAlgo::Tree must match NCCL_ALGO_TREE");

static_assert(
    static_cast<int>(CommAlgo::Ring) == static_cast<int>(NCCL_ALGO_RING),
    "CommAlgo::Ring must match NCCL_ALGO_RING");

static_assert(
    static_cast<int>(CommAlgo::CollNetDirect) ==
        static_cast<int>(NCCL_ALGO_COLLNET_DIRECT),
    "CommAlgo::CollNetDirect must match NCCL_ALGO_COLLNET_DIRECT");

static_assert(
    static_cast<int>(CommAlgo::CollNetChain) ==
        static_cast<int>(NCCL_ALGO_COLLNET_CHAIN),
    "CommAlgo::CollNetChain must match NCCL_ALGO_COLLNET_CHAIN");

static_assert(
    static_cast<int>(CommAlgo::NVLS) == static_cast<int>(NCCL_ALGO_NVLS),
    "CommAlgo::NVLS must match NCCL_ALGO_NVLS");

static_assert(
    static_cast<int>(CommAlgo::NVLSTree) ==
        static_cast<int>(NCCL_ALGO_NVLS_TREE),
    "CommAlgo::NVLSTree must match NCCL_ALGO_NVLS_TREE");

static_assert(
    static_cast<int>(CommAlgo::PAT) == static_cast<int>(NCCL_ALGO_PAT),
    "CommAlgo::PAT must match NCCL_ALGO_PAT");

/******************************************************************************
 * Begin of static check to ensure CommProtocol enum matches NCCL protocol    *
 *****************************************************************************/
static_assert(
    static_cast<int>(CommProtocol::LL) == static_cast<int>(NCCL_PROTO_LL),
    "CommProtocol::LL must match NCCL_PROTO_LL");

static_assert(
    static_cast<int>(CommProtocol::LL128) == static_cast<int>(NCCL_PROTO_LL128),
    "CommProtocol::LL128 must match NCCL_PROTO_LL128");

static_assert(
    static_cast<int>(CommProtocol::Simple) ==
        static_cast<int>(NCCL_PROTO_SIMPLE),
    "CommProtocol::Simple must match NCCL_PROTO_SIMPLE");

} // namespace meta::comms
