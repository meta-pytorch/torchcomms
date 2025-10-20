// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iomanip>
#include <iostream>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevPerfUTBase.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/testinfra/TestUtils.h"

void CtranDistAlgoDevPerfTestBase::SetUp() {
  // keep log as minimal
  setenv("NCCL_DEBUG", "ERROR", 1);

  CtranDistAlgoDevTest::SetUp();

  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaEventCreateWithFlags(&start, cudaEventDefault));
  CUDACHECK_TEST(cudaEventCreateWithFlags(&end, cudaEventDefault));
}

void CtranDistAlgoDevPerfTestBase::TearDown() {
  CUDACHECK_TEST(cudaStreamDestroy(stream));
  CUDACHECK_TEST(cudaEventDestroy(start));
  CUDACHECK_TEST(cudaEventDestroy(end));
  CtranDistAlgoDevTest::TearDown();
}

template <typename T>
KernelElem* CtranDistAlgoDevPerfTestBase::prepareReduceElem(
    size_t count,
    size_t numElems,
    int nGroups,
    bool localRankSrcs,
    int nSrcs,
    void** dstBases,
    int nDsts,
    bool barrierLastElem) {
  KernelElem* reduceELemList = nullptr;
  COMMCHECK_TEST(comm_->ctranComm_->ctran_->gpe->allocKernelElems(
      numElems, nGroups, &reduceELemList));

  // In-place reduce from all local ranks for each elem. Each elem starts at
  // different offset
  const int localRank = comm_->ctranComm_->statex_->localRank();

  const int vectorNBytes = count * sizeof(T);
  const int localRankNSrcs = localRankSrcs ? nSrcs : 1;

  auto elem = reduceELemList;
  size_t srcOffset = 0;
  size_t dstOffset = 0;
  while (elem) {
    elem->reduce.count = count;
    elem->reduce.nsrcs = nSrcs;
    if (localRankSrcs) {
      for (int i = 0; i < nSrcs; i++) {
        char* srcBase = reinterpret_cast<char*>(ipcMem_->getBase());
        elem->reduce.srcs[i] = srcBase + srcOffset + i * vectorNBytes;
      }
    } else {
      for (int i = 0; i < nSrcs; i++) {
        char* srcBase = i == localRank
            ? reinterpret_cast<char*>(ipcMem_->getBase())
            : reinterpret_cast<char*>(ipcRemMem_.at(i)->getBase());
        // Each rank starts from different portion similar to reduceScatter
        size_t srcStart = count * numElems * localRank * sizeof(T);
        elem->reduce.srcs[i] = srcBase + srcStart + srcOffset;
      }
    }
    elem->reduce.ndsts = nDsts;
    for (int i = 0; i < nDsts; i++) {
      elem->reduce.dsts[i] = reinterpret_cast<char*>(dstBases[i]) + dstOffset;
    }
    srcOffset += localRankNSrcs * vectorNBytes;
    dstOffset += nDsts * vectorNBytes;
    // Mark each step as final since each of them handles different portion
    elem->reduce.isFinal = true;

    if (elem->next == nullptr) {
      // Flush at last element at each step, so that host side can read
      elem->reduce.flushMem = true;
      // Optionally barrier at last element so that host side can read
      // remote
      // ranks' results too
      if (barrierLastElem) {
        elem->reduce.barrier = true;
      }
    }

    elem = elem->next;
  }
  return reduceELemList;
}

template <typename T>
void CtranDistAlgoDevPerfTestBase::startBenchmark(
    std::string_view kernName,
    std::function<void(KernelElem*, cudaStream_t)> kernelLaunchWrapper,
    std::function<void(KernelElem*)> postKernelWorkFn,
    std::function<void(KernelElem*)> postIterWorkFn,
    CtranDistAlgoDevPerfTestParams param) {
  const auto& [testType, numElems, inplace, barrierLastElem, nGroups, beginCount, endCount, op, warmup, iters, localRankNSrcs, nDsts] =
      param;

  const int localRank = comm_->ctranComm_->statex_->localRank();
  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

  if (comm_->ctranComm_->statex_->rank() == 0) {
    std::cout << std::string(100, '-') << std::endl;
  }

  bool localRankSrcs = localRankNSrcs > 1;
  int nSrcs = localRankSrcs ? localRankNSrcs : nLocalRanks;

  for (size_t count = beginCount; count <= endCount; count *= 2) {
    size_t totalSrcCount = count * numElems * nSrcs;
    size_t totalDstCount = count * numElems * nDsts;
    size_t bytesPerRank = count * sizeof(T);
    size_t totalSrcBytes = totalSrcCount * sizeof(T);
    size_t totalDstBytes = totalDstCount * sizeof(T);
    initIpcBufs<T>(totalSrcCount, totalDstCount);
    // Use ipcBuf as source of reduce, and localBuf as destination
    assignVal<T>(localBuf_, totalDstCount, rand());
    assignVal<T>(ipcBuf_, totalSrcCount, localRank, true);
    // Ensure data has been stored before IPC access
    intraNodeBarrier(comm_);

    void* dstBases[kMaxNVectors];
    // inplace reduce is only valid for 1 of the nDsts
    if (inplace) {
      size_t ipcMemOffset = 0;
      if (!localRankSrcs) {
        // For in-place case, each rank reduces to different portion
        ipcMemOffset = count * numElems * localRank * sizeof(T);
      }
      dstBases[0] = reinterpret_cast<char*>(ipcMem_->getBase()) + ipcMemOffset;
    } else {
      dstBases[0] = localBuf_;
    }
    for (int i = 1; i < nDsts; ++i) {
      dstBases[i] = reinterpret_cast<char*>(localBuf_) + i * count * sizeof(T);
    }

    KernelElem* reduceELemList = prepareReduceElem<T>(
        count,
        numElems,
        nGroups,
        localRankSrcs,
        nSrcs,
        dstBases,
        nDsts,
        barrierLastElem);

    float timeMs = 0.0;
    for (int i = 0; i < warmup + iters; ++i) {
      CUDACHECK_TEST(cudaEventRecord(start, stream));
      kernelLaunchWrapper(reduceELemList, stream);
      CUDACHECK_TEST(cudaEventRecord(end, stream));

      // any post work on host-side to be done after kernel launch
      postKernelWorkFn(reduceELemList);

      // blocking call to measure kernel time
      // TODO: support b2b async kernels?
      CUDACHECK_TEST(cudaStreamSynchronize(stream));
      if (i >= warmup) {
        float iterTimeMs = 0.0;
        CUDACHECK_TEST(cudaEventElapsedTime(&iterTimeMs, start, end));
        timeMs += iterTimeMs;
      }

      // any post work on host-side to be done after kernel is finished in the
      // end of each iteration, e.g., cleanup elements
      postIterWorkFn(reduceELemList);
    }

    if (comm_->ctranComm_->statex_->rank() == 0) {
      auto timeUsPerIter = (timeMs * 1000) / iters;
      std::cout << "[" << kernName << "-" << typeid(T).name() << "] Rank-["
                << comm_->ctranComm_->statex_->rank() << "]";
      std::cout << ", nSrcs " << nSrcs;
      if (localRankSrcs) {
        std::cout << " (" << localRankNSrcs << " per rank, across 1 ranks)";
      } else {
        std::cout << " (1 per rank, across " << nLocalRanks << " ranks)";
      }
      std::cout << ", nDsts " << nDsts << ", redOp " << commOpToString(op)
                << ", nGroups " << nGroups << ", count " << std::setw(9)
                << count << ", nbytesPerRank " << std::setw(9) << bytesPerRank
                << " => " << std::fixed << std::setprecision(2) << std::setw(9)
                << timeUsPerIter << " us per kernel, " << std::fixed
                << std::setprecision(2) << (totalSrcBytes / (timeUsPerIter))
                << " MB/s read " << (totalDstBytes / (timeUsPerIter))
                << " MB/s write" << std::endl;
    }
    // ensure everyone is done before freeing IPC buffer
    intraNodeBarrier(comm_);
    freeIpcBufs();
  }
  if (comm_->ctranComm_->statex_->rank() == 0) {
    std::cout << std::string(100, '-') << std::endl;
  }
}

// TODO: add more types when needed
DECLAR_ALGO_PERF_UT_FUNCS(int);
