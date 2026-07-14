// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/tests/CollectiveTestSuite.h"

// AllReduce layout + verification traits for the generic CollectiveTestSuite
// harness, plus the per-collective fixture-base alias leaves derive from. Each
// AllReduce algo leaf (`AllReduce<Algo>TestSuite.cc`) derives
// `AllReduceTestSuiteBase` and overrides the algo dispatch hooks.
struct AllReduceTraits {
  using Algo = enum NCCL_ALLREDUCE_ALGO;
  static constexpr bool kHasOp = true;

  static std::string algoName(Algo algo) {
    return allReduceAlgoName(algo);
  }

  static bool support(CtranComm* comm, Algo algo) {
    return ctranAllReduceSupport(comm, algo);
  }

  static BufferPlan plan(
      size_t count,
      int /*nranks*/,
      int /*rank*/,
      bool inPlace,
      size_t offsetBytes,
      size_t typeBytes) {
    const size_t bytes = count * typeBytes;
    BufferPlan p;
    p.sendAllocBytes = bytes + offsetBytes;
    p.sendUserOffsetBytes = offsetBytes;
    p.sendInitElems = count;
    p.resultElems = count;
    if (inPlace) {
      p.recvAliasesSend = true;
      p.recvUserOffsetBytes = offsetBytes; // recv == send.
    } else {
      p.recvAllocBytes = bytes + offsetBytes;
      p.recvUserOffsetBytes = offsetBytes;
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
        launchInitExpectedKernel(
            static_cast<float*>(expected), count, nranks, /*rep=*/0, stream);
        return;
      case commFloat16:
        launchInitExpectedKernel(
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
      int nranks,
      cudaStream_t stream,
      size_t count,
      int rank,
      bool inPlace,
      int repeat) {
    double maxDelta = ctran::testsuite::computeMaxDeltaForType(
        got, expected, datatype, resultElems, stream);
    double threshold = ctran::testsuite::thresholdForDatatype(datatype, nranks);
    ASSERT_LT(maxDelta, threshold)
        << "maxDelta=" << maxDelta << " threshold=" << threshold
        << " count=" << count << " rank=" << rank
        << " datatype=" << commDataTypeToString(datatype)
        << " inPlace=" << inPlace << " repeat=" << repeat;
  }
};

using AllReduceTestSuiteBase = CollectiveTestSuite<AllReduceTraits>;
