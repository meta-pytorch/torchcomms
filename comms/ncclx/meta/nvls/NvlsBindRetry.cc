// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/nvls/NvlsBindRetry.h"

#include <chrono>
#include <thread>

#include "bootstrap.h"
#include "comm.h"
#include "cudawrap.h"
#include "param.h"
#include "utils.h"

#include "comms/utils/memtrace/MemoryTrace.h"

namespace {

NCCL_PARAM(NvlsBindRetryCount, "NVLS_BIND_RETRY_COUNT", 10);
NCCL_PARAM(NvlsBindRetryBackoffMs, "NVLS_BIND_RETRY_BACKOFF_MS", 1000);

} // namespace

namespace ncclx::nvls {

ncclResult_t collectiveBindResult(
    const ncclComm* comm,
    CUresult localResult,
    CUresult* collectiveResult) {
  CUresult localResults[NCCL_MAX_LOCAL_RANKS]{};
  localResults[comm->localRank] = localResult;
  NCCLCHECK(bootstrapIntraNodeAllGather(
      comm->bootstrap,
      comm->localRankToRank,
      comm->localRank,
      comm->localRanks,
      localResults,
      sizeof(CUresult)));

  // Every rank selects the first fatal result in local-rank order; absent a
  // fatal result, CUDA 802 dominates success so all ranks make the same choice.
  *collectiveResult = CUDA_SUCCESS;
  for (int i = 0; i < comm->localRanks; ++i) {
    const CUresult result = localResults[i];
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_SYSTEM_NOT_READY) {
      *collectiveResult = result;
      break;
    }
    if (result == CUDA_ERROR_SYSTEM_NOT_READY) {
      *collectiveResult = result;
    }
  }
  return ncclSuccess;
}

ncclResult_t prepareBindRetry(
    ncclComm* comm,
    CUresult localResult,
    CUresult collectiveResult,
    int64_t bindAttempt,
    size_t ucsize,
    void** ucptr,
    CUmemGenericAllocationHandle* ucHandle,
    CUmemGenericAllocationHandle* mcHandle,
    int* allocMcHandle,
    bool* retried) {
  *retried = false;
  const int64_t retryCount = ncclParamNvlsBindRetryCount();
  const int64_t retryBackoffMs = ncclParamNvlsBindRetryBackoffMs();
  if (collectiveResult != CUDA_ERROR_SYSTEM_NOT_READY ||
      bindAttempt >= retryCount) {
    return ncclSuccess;
  }

  const char* localErrorString =
      "success (a peer rank failed the multicast bind)";
  if (localResult != CUDA_SUCCESS) {
    (void)pfn_cuGetErrorString(localResult, &localErrorString);
  }
  WARN(
      "NVLS multicast bind of size %ld did not succeed on all local ranks (this rank CUDA error %d '%s'); tearing down and retrying (attempt %lld/%lld) after %lldms. This is usually a transient Fabric Manager stall forming the NVSwitch multicast team.",
      ucsize,
      localResult,
      localErrorString,
      static_cast<long long>(bindAttempt + 1),
      static_cast<long long>(retryCount),
      static_cast<long long>(retryBackoffMs));

  NCCLCHECK(ncclMemUntrack(comm->memManager, *ucptr, ucsize));
  meta::comms::memtrace::recordFree(
      comm->logMetaData,
      "nvlsAllocateMem",
      "cuMemRelease",
      reinterpret_cast<uintptr_t>(*ucptr),
      ucsize);
  CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(*ucptr), ucsize));
  CUCHECK(cuMemRelease(*ucHandle));
  CUCHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(*ucptr), ucsize));
  CUCHECK(cuMemRelease(*mcHandle));

  // Clear ownership before the barrier so a later failure cannot send released
  // resources through the caller's cleanup labels.
  *ucptr = nullptr;
  *allocMcHandle = 0;
  NCCLCHECK(bootstrapIntraNodeBarrier(
      comm->bootstrap,
      comm->localRankToRank,
      comm->localRank,
      comm->localRanks,
      comm->localRankToRank[0]));

  if (retryBackoffMs > 0) {
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(retryBackoffMs));
  }
  *retried = true;
  return ncclSuccess;
}

} // namespace ncclx::nvls
