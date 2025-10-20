// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/algos/tests/CtranDistAlgoDevUTBase.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevUTKernels.h"
#include "comms/utils/commSpecs.h"

using CtranDistAlgoDevPerfTestParams = std::tuple<
    ElemTestType /*CompleteOrFree*/,
    int /*numKernElems*/,
    bool /*inplace*/,
    bool /*barrierLastElem*/,
    unsigned int /*nGroups*/,
    size_t /*beginCount*/,
    size_t /*endCount*/,
    commRedOp_t /*redOp*/,
    int /*warmup*/,
    int /*iters*/,
    int /*localRankNSrcs*/,
    int /*nDsts*/>;

class CtranDistAlgoDevPerfTestBase : public CtranDistAlgoDevTest {
 public:
  void SetUp() override;
  void TearDown() override;

  /* prepare kernel elements for the reduce kernel
   * @param[in]  count               Number of elements to be reduced
   * @param[in]  numElems            Number of ctran kernel elements
   * @param[in]  nGroups             Number of groups per element
   * @param[in]  localRankSrcs       Whether src buffers are local to the rank
   * @param[in]  nSrcs               Number of src buffers
   * @param[in]  dstBases            Destination buffers
   * @param[in]  nDsts               Number of dst buffers
   * @param[in]  barrierLastElem     Whether to perform barrier at the last
   * element
   * @return                         Pointer to the kernel elements
   */
  template <typename T>
  KernelElem* prepareReduceElem(
      size_t count,
      size_t numElems,
      int nGroups,
      bool localRankSrcs,
      int nSrcs,
      void** dstBases,
      int nDsts,
      bool barrierLastElem);

  /* start benchmark for the reduce kernel and report the performance
   * @param[in]  kernName             Name of the kernel
   * @param[in]  kernelLaunchWrapper  Function to launch the kernel
   * @param[in]  postKernelWorkFn     Function to perform post-kernel work
   * @param[in]  postIterWorkFn       Function to perform post-iteration work
   * @param[in]  param                Parameters for the benchmark, including a
   * range of sizes, number of iterations, etc.
   */
  template <typename T>
  void startBenchmark(
      std::string_view kernName,
      std::function<void(KernelElem*, cudaStream_t)> kernelLaunchWrapper,
      std::function<void(KernelElem*)> postKernelWorkFn,
      std::function<void(KernelElem*)> postIterWorkFn,
      CtranDistAlgoDevPerfTestParams param);

 protected:
  static constexpr size_t kMaxNVectors = 8;

  cudaStream_t stream{nullptr};
  cudaEvent_t start{nullptr};
  cudaEvent_t end{nullptr};

  const struct CommLogData dummyLogMetaData_ = {
      0,
      0xfaceb00c12345678 /*Dummy placeholder value for commHash*/,
      "perfTestComm",
      0,
      0};
};

#define DECLAR_ALGO_PERF_UT_FUNCS(T)                                      \
  template void CtranDistAlgoDevPerfTestBase::startBenchmark<T>(          \
      std::string_view kernName,                                          \
      std::function<void(KernelElem*, cudaStream_t)> kernelLaunchWrapper, \
      std::function<void(KernelElem*)> postKernelWorkFn,                  \
      std::function<void(KernelElem*)> postIterWorkFn,                    \
      CtranDistAlgoDevPerfTestParams param)
