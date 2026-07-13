// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/tests/CollectiveTestSuite.h"

// ReduceScatter layout + verification traits for the generic
// CollectiveTestSuite harness, plus the per-collective fixture-base alias
// leaves derive from. Each ReduceScatter algo leaf derives
// `ReduceScatterTestSuiteBase` and overrides the algo dispatch hooks.
struct ReduceScatterTraits {
  using Algo = enum NCCL_REDUCESCATTER_ALGO;
  static constexpr bool kHasOp = true;

  static std::string algoName(Algo algo) {
    return reduceScatterAlgoName(algo);
  }

  static bool support(CtranComm* comm, Algo algo) {
    return ctranReduceScatterSupport(comm, algo);
  }

  static BufferPlan plan(
      size_t count,
      int nranks,
      int rank,
      bool inPlace,
      size_t offsetBytes,
      size_t typeBytes) {
    const size_t recvBytes = count * typeBytes;
    const size_t sendBytes = recvBytes * static_cast<size_t>(nranks);
    BufferPlan p;
    p.sendAllocBytes = sendBytes + offsetBytes;
    p.sendUserOffsetBytes = offsetBytes;
    p.sendInitElems = count * static_cast<size_t>(nranks);
    p.resultElems = count;
    if (inPlace) {
      p.recvAliasesSend = true;
      p.recvUserOffsetBytes =
          offsetBytes + static_cast<size_t>(rank) * recvBytes;
    } else {
      p.recvAllocBytes = recvBytes + offsetBytes;
      p.recvUserOffsetBytes = offsetBytes;
    }
    return p;
  }

  static void buildExpected(
      void* expected,
      commDataType_t datatype,
      size_t count,
      int nranks,
      int rank,
      cudaStream_t stream) {
    const size_t baseIdx = static_cast<size_t>(rank) * count;
    switch (datatype) {
      case commFloat32:
        launchInitScatterExpectedKernel(
            static_cast<float*>(expected),
            count,
            nranks,
            /*rep=*/0,
            baseIdx,
            stream);
        return;
      case commFloat16:
        launchInitScatterExpectedKernel(
            static_cast<__half*>(expected),
            count,
            nranks,
            /*rep=*/0,
            baseIdx,
            stream);
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
        << "RS maxDelta=" << maxDelta << " threshold=" << threshold
        << " recvcount=" << count << " rank=" << rank
        << " datatype=" << commDataTypeToString(datatype)
        << " inPlace=" << inPlace << " repeat=" << repeat;
  }
};

using ReduceScatterTestSuiteBase = CollectiveTestSuite<ReduceScatterTraits>;
