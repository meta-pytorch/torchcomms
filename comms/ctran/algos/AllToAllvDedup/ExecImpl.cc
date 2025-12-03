// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/ExecCommon.h"

namespace ctran::alltoallvdedup {
const bool kIsExec = true;

template <typename T>
extern __global__ void ncclKernelAllToAllvDedup(
    int* flag,
    CtranAlgoDeviceState* devState,
    ExecKernArgs args);
extern __global__ void ncclKernelAllToAllvDedupPrepareReset(
    ExecKernArgs args,
    int nRanks,
    int nLocalRanks);
extern __global__ void ncclKernelAllToAllvDedupPrepare(
    CtranAlgoDeviceState* devState,
    ExecKernArgs args,
    PrepareConfig config,
    const int roles);

namespace {
void* alltoallv_dedup_dpKerns[commNumTypes] = {
    (void*)ncclKernelAllToAllvDedup<int8_t>,
    (void*)ncclKernelAllToAllvDedup<uint8_t>,
    (void*)ncclKernelAllToAllvDedup<int32_t>,
    (void*)ncclKernelAllToAllvDedup<uint32_t>,
    (void*)ncclKernelAllToAllvDedup<int64_t>,
    (void*)ncclKernelAllToAllvDedup<uint64_t>,
    (void*)ncclKernelAllToAllvDedup<half>,
    (void*)ncclKernelAllToAllvDedup<float>,
    (void*)ncclKernelAllToAllvDedup<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__)
    (void*)ncclKernelAllToAllvDedup<__nv_bfloat16>,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    (void*)ncclKernelAllToAllvDedup<__nv_fp8_e4m3>,
    (void*)ncclKernelAllToAllvDedup<__nv_fp8_e5m2>,
#endif
};

commResult_t execGpeFn(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto op = opGroup[0].get();
  auto comm = op->comm_;
  thOpCount = op->opCount;
  thMyRank = comm->statex_->rank();

  auto statex = comm->statex_.get();
  const auto myRank = statex->rank();

  CtranAlgoLogger logger(
      AlgoImpl::algoName(AlgoImpl::Phase::kExec), op->opCount, comm);
  CtranMapper* mapper = comm->ctran_->mapper.get();
  const int nNodes = statex->nNodes();
  const int nLocalRanks = statex->nLocalRanks();

  auto ctran_trace_logger = reinterpret_cast<utils::TraceLogger*>(
      op->alltoallv_dedup_exec.ctran_trace_logger);
  // Always create traceRecord for code simplicity, all recording should be
  // no-op if trace is disabled
  auto ts = std::make_unique<utils::TraceRecord>(
      fmt::format("allToAllvDedupExec_{}", thOpCount), myRank);
  setCommonTraceMetadata(ts.get(), op, kIsExec);

  ExecCtx ctx = {
      .pArgs = *reinterpret_cast<PersistArgs*>(op->alltoallv_dedup_exec.pArgs),
      .resource =
          reinterpret_cast<ResourceRef*>(op->alltoallv_dedup_exec.algoResource),
      .config =
          reinterpret_cast<PersistConfig*>(op->alltoallv_dedup_exec.algoConfig),
      .commStatex = statex,
      .mapper = mapper,
      .ts = ts.get(),
      .opCount = op->opCount};

  // Initialize progressState
  ProgressState state = ProgressState(nNodes, nLocalRanks);

  FB_COMMCHECK(updateProgressXNodeSendState(ctx, state, kIsExec));
  FB_COMMCHECK(updateProgressXNodeRecvState(ctx, state, kIsExec));

  do {
    FB_COMMCHECK(progressXNodeSend(ctx, state, kIsExec));

    FB_COMMCHECK(progressXNodeRecv(ctx, state, kIsExec));

    // FIXME: disable kernel side profiling for now. To be fixed in separate
    // DIFF.
    // profileIntraFwd(ctx, state, kIsExec);
    // profileRecvCopy(ctx, state, kIsExec);
    // profileIntraRecvCopy(ctx, state, kIsExec);

    // NOTE: we don't track the completion of profile-only intraFwd and
    // recvCopy; for any imcomplete profiling events, use the end time of GPE as
    // event end. This should be sufficient to capture the overhead.
  } while (state.numPendingSendNodes > 0 || state.numPendingRecvNodes > 0);

  // progress main loop has ensured trans completion but not all syncs. We
  // explicitly wait for all sync to complete because: 1) ensure request object
  // completion within this op otherwise may mis-complete next op if request
  // memory is reused; 2) algorithm level ensure remote chunks are all freed so
  // next dedup can always start from remote chunk 0 for simplicity
  FB_COMMCHECK(waitSyncComplete(ctx, state, kIsExec));

  ctran_trace_logger->addTraceRecord(std::move(ts));
  return commSuccess;
}

commResult_t launchPrepareKernel(
    const ExecKernArgs& args,
    const PrepareConfig& config,
    const ncclx::CommStateX* statex,
    const CtranAlgoDeviceState* devState_d,
    cudaStream_t stream,
    const int roles = static_cast<int>(PrepareRole::kPrepareAll)) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nRanks = statex->nRanks();

  unsigned int numThreads = config.numThreads;
  unsigned int numBlocks = config.numSendIdxWorkers +
      config.numIntraFwdIdxWorkers + config.numFwdIxWorkers +
      config.numRecvIdxWorkers + config.numRecvOffsetWorkers +
      config.numRecvRedIdxWorkers + config.numResetSyncWorkers;

  // first reset all tmp indices
  {
    std::array<void*, 3> kernelArgs;
    kernelArgs.at(0) = (void*)&args;
    kernelArgs.at(1) = (void*)&nRanks;
    kernelArgs.at(2) = (void*)&nLocalRanks;
    dim3 grid = {1, 1, 1};
    dim3 blocks = {numThreads, 1, 1};
    FB_CUDACHECK(cudaLaunchKernel(
        reinterpret_cast<void*>(ncclKernelAllToAllvDedupPrepareReset),
        grid,
        blocks,
        kernelArgs.data(),
        0,
        stream));
  }

  // prepare tmp indices
  {
    std::array<void*, 4> kernelArgs;
    kernelArgs.at(0) = (void*)&devState_d;
    kernelArgs.at(1) = (void*)&args;
    kernelArgs.at(2) = (void*)&config;
    kernelArgs.at(3) = (void*)&roles;

    dim3 grid = {numBlocks, 1, 1};
    dim3 blocks = {numThreads, 1, 1};
    FB_CUDACHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(ncclKernelAllToAllvDedupPrepare),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(CtranAlgoDeviceState)));
    FB_CUDACHECK(cudaLaunchKernel(
        reinterpret_cast<void*>(ncclKernelAllToAllvDedupPrepare),
        grid,
        blocks,
        kernelArgs.data(),
        sizeof(CtranAlgoDeviceState),
        stream));
  }
  return commSuccess;
}

void setupWorkerGroups(PersistConfig& config) {
  // Overwrite config when launching exec and combine, so kernel can use the
  // same field for both kernels.
  config.numSendGroups =
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS;
  config.numSendWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP;
  config.numFwdWorkers = NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS;
  config.numRecvGroups =
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS;
  config.numRecvWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP;
  config.numIntraFwdWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_FWD_NUM_THREAD_BLOCKS;
  config.numIntraRecvWorkers =
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_RECV_NUM_THREAD_BLOCKS;
}
} // namespace

void setupPrepareKernelConfig(
    const ExecKernArgs& args,
    const ncclx::CommStateX* statex,
    PrepareConfig& config) {
  const auto nNodes = statex->nNodes();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nRanks = statex->nRanks();

  config.numSendIdxWorkers = nNodes;
  config.numIntraFwdIdxWorkers = nLocalRanks;
  config.numFwdIxWorkers = nNodes;
  config.numRecvIdxWorkers = nRanks;
  config.numRecvOffsetWorkers = nRanks * args.pArgs.numRecvBuckets;
  config.numRecvRedIdxWorkers = nRanks;
  config.numResetSyncWorkers = 1;
  config.numThreads = NCCL_CTRAN_ALLTOALLV_DEDUP_RESET_THREAD_BLOCK_SIZE;
}

commResult_t launchPrepareForTest(
    const ExecKernArgs& args,
    const PrepareConfig& config,
    const ncclx::CommStateX* statex,
    const CtranAlgoDeviceState* dState,
    cudaStream_t stream,
    const int role) {
  return launchPrepareKernel(args, config, statex, dState, stream, role);
}

commResult_t AlgoImpl::exec(const ExecArgs& execArgs, const uint64_t opCount) {
  // prepare kernel config for self and NVL copies
  KernelConfig kernelConfig = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV_DEDUP,
      stream_,
      algoName(Phase::kExec),
      opCount);
  ExecKernArgs kernArgs;

  // Assign worker groups based on data distribution
  PersistConfig execConfig = config_;
  setupWorkerGroups(execConfig);

  setupKernelConfig(ctran_, execConfig, kernelConfig);
  setupExecKernelArgs(
      pArgs, execConfig, execArgs, resource_.get(), opCount, kernArgs);
  kernelConfig.algoArgs = &kernArgs; // copied to kernel within submit()

  const auto statex = comm_->statex_.get();
  // Launch prepare kernel
  PrepareConfig config;
  setupPrepareKernelConfig(kernArgs, statex, config);
  FB_COMMCHECK(launchPrepareKernel(
      kernArgs, config, statex, ctran_->algo->getDevState(), stream_));

  // Launch combine kernel with GPE
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  setupGpeOp(
      pArgs,
      execArgs,
      opCount,
      resource_->getRef(),
      config_,
      comm_,
      opGroup,
      ctran_trace_logger.get());

  FB_COMMCHECK(ctran_->gpe->submit(
      std::move(opGroup),
      execGpeFn,
      kernelConfig,
      reinterpret_cast<void*>(alltoallv_dedup_dpKerns[pArgs.datatype])));

  return commSuccess;
}

} // namespace ctran::alltoallvdedup
