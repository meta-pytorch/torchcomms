// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/tests/CollectiveTestSuite.h"

// AllGather layout + verification traits for the generic CollectiveTestSuite
// harness, plus the per-collective fixture-base alias leaves derive from. Each
// AllGather algo leaf derives `AllGatherTestSuiteBase` and overrides the algo
// dispatch hooks.
struct AllGatherTraits {
  using Algo = enum NCCL_ALLGATHER_ALGO;
  static constexpr bool kHasOp = false;

  static std::string algoName(Algo algo) {
    return allGatherAlgoName(algo);
  }

  static bool support(CtranComm* comm, Algo algo) {
    return ctranAllGatherSupport(comm, algo);
  }

  static BufferPlan plan(
      size_t count,
      int nranks,
      int rank,
      bool inPlace,
      size_t offsetBytes,
      size_t typeBytes) {
    const size_t sendBytes = count * typeBytes;
    const size_t recvBytes = sendBytes * static_cast<size_t>(nranks);
    BufferPlan p;
    p.recvAllocBytes = recvBytes + offsetBytes;
    p.recvUserOffsetBytes = offsetBytes;
    p.sendInitElems = count;
    p.resultElems = count * static_cast<size_t>(nranks);
    if (inPlace) {
      p.sendAliasesRecv = true;
      p.sendUserOffsetBytes =
          offsetBytes + static_cast<size_t>(rank) * sendBytes;
    } else {
      p.sendAllocBytes = sendBytes + offsetBytes;
      p.sendUserOffsetBytes = offsetBytes;
    }
    return p;
  }

  static void buildExpected(
      void* expected,
      commDataType_t datatype,
      size_t count,
      int nranks,
      int /*rank*/,
      cudaStream_t stream) {
    switch (datatype) {
      case commFloat32:
        launchInitGatherExpectedTyped(
            static_cast<float*>(expected), count, nranks, /*rep=*/0, stream);
        return;
      case commFloat16:
        launchInitGatherExpectedTyped(
            static_cast<__half*>(expected), count, nranks, /*rep=*/0, stream);
        return;
      default:
        ADD_FAILURE() << "unsupported datatype " << datatype;
        return;
    }
  }

  static void checkResult(
      const void* got,
      const void* expected,
      commDataType_t datatype,
      size_t resultElems,
      int /*nranks*/,
      cudaStream_t stream,
      size_t count,
      int rank,
      bool inPlace,
      int repeat) {
    // AG is a pure copy - checksum-based bitwise-equality suffices.
    uint64_t actualChecksum = ctran::testsuite::computeRawBitsChecksumForType(
        got, datatype, resultElems, stream);
    uint64_t expectedChecksum = ctran::testsuite::computeRawBitsChecksumForType(
        expected, datatype, resultElems, stream);
    ASSERT_EQ(actualChecksum, expectedChecksum)
        << "AG output checksum mismatch: sendcount=" << count
        << " rank=" << rank << " datatype=" << commDataTypeToString(datatype)
        << " inPlace=" << inPlace << " repeat=" << repeat;
  }
};

using AllGatherTestSuiteBase = CollectiveTestSuite<AllGatherTraits>;
