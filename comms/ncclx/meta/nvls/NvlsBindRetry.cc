// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/nvls/NvlsBindRetry.h"
#include "meta/nvls/NvlsBindWatchdog.h"

#include <chrono>
#include <thread>

#include "bootstrap.h"
#include "comm.h"
#include "cudawrap.h"
#include "param.h"
#include "proxy.h"
#include "utils.h"

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 32, 0)
#include "comms/utils/memtrace/MemoryTrace.h"
#endif

namespace {

NCCL_PARAM(NvlsBindRetryCount, "NVLS_BIND_RETRY_COUNT", 10);
NCCL_PARAM(NvlsBindRetryBackoffMs, "NVLS_BIND_RETRY_BACKOFF_MS", 1000);

} // namespace

namespace ncclx::nvls {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
NvlsBootstrapOps gNvlsBootstrap{
    bootstrapIntraNodeAllGather,
    bootstrapIntraNodeBarrier,
    bootstrapIntraNodeBroadcast,
    ncclProxyClientGetFdBlocking};

ncclResult_t collectiveBindResult(
    const ncclComm* comm,
    CUresult localResult,
    CUresult* collectiveResult) {
  CUresult localResults[NCCL_MAX_LOCAL_RANKS]{};
  localResults[comm->localRank] = localResult;
  NCCLCHECK(gNvlsBootstrap.allGather(
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

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 32, 0)
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
  NCCLCHECK(gNvlsBootstrap.barrier(
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

#else
namespace {

ncclResult_t collectiveCleanupResult(
    const ncclComm* comm,
    const char* operation,
    CUresult localResult,
    CUresult contextResult) {
  CUresult result = CUDA_SUCCESS;
  NCCLCHECK(collectiveBindResult(comm, localResult, &result));
  if (result == CUDA_SUCCESS) {
    return ncclSuccess;
  }
  const char* errStr = "unknown error";
  (void)pfn_cuGetErrorString(result, &errStr);
  if (contextResult == CUDA_SUCCESS) {
    WARN(
        "NVLS multicast retry cleanup (%s) failed: CUDA error %d '%s'",
        operation,
        result,
        errStr);
  } else {
    // The cleanup failure returns before the retry/fatal report below, so the
    // triggering map error is logged here; otherwise the root cause never
    // appears at default log level on this double fault.
    const char* contextStr = "unknown error";
    (void)pfn_cuGetErrorString(contextResult, &contextStr);
    WARN(
        "NVLS multicast retry cleanup (%s) failed: CUDA error %d '%s' (triggered by map error %d '%s')",
        operation,
        result,
        errStr,
        contextResult,
        contextStr);
  }
  printCudaDriverErrorHint(result);
  return ncclUnhandledCudaError;
}

} // namespace

ncclResult_t multicastMapWithRetry(
    const ncclComm* comm,
    CUdeviceptr base,
    size_t size,
    CUmemGenericAllocationHandle mcHandle,
    int64_t attempt,
    bool* retry,
    int* mapped) {
  *retry = false;
  const CUresult localResult =
      multicastMapWithWatchdog(comm, base, size, mcHandle);
  *mapped = localResult == CUDA_SUCCESS;
  CUresult result = CUDA_SUCCESS;
  NCCLCHECK(collectiveBindResult(comm, localResult, &result));
  if (result == CUDA_SUCCESS) {
    return ncclSuccess;
  }
  // A peer failed: undo this rank's map so the caller can release the range.
  CUresult unmapResult = CUDA_SUCCESS;
  if (*mapped) {
    unmapResult = CUPFN(cuMemUnmap(base, size));
    if (unmapResult == CUDA_SUCCESS) {
      *mapped = 0;
    }
  }
  NCCLCHECK(collectiveCleanupResult(comm, "cuMemUnmap", unmapResult, result));

  const char* errStr = "unknown error";
  (void)pfn_cuGetErrorString(result, &errStr);
  const int64_t retryCount = ncclParamNvlsBindRetryCount();
  if (result == CUDA_ERROR_SYSTEM_NOT_READY && attempt < retryCount) {
    WARN(
        "NVLS multicast map of size %zu did not succeed on all local ranks (CUDA error %d '%s', this rank CUDA error %d); rebuilding the multicast group and retrying (attempt %lld/%lld) after %lldms. This is usually a transient Fabric Manager stall forming the NVSwitch multicast team.",
        size,
        result,
        errStr,
        localResult,
        static_cast<long long>(attempt + 1),
        static_cast<long long>(retryCount),
        static_cast<long long>(ncclParamNvlsBindRetryBackoffMs()));
    *retry = true;
    return ncclSuccess;
  }

  WARN("Cuda failure %d '%s'", result, errStr);
  printCudaDriverErrorHint(result);
  ERR(ncclUnhandledCudaError,
      "Failed to map NVLink SHARP (NVLS) Multicast memory of size %zu : CUDA error %d '%s'.\nThis is usually caused by a system or configuration error in the Fabric Manager or NVSwitches.\nDisable NVLS (NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.",
      size,
      result,
      errStr);
  return ncclUnhandledCudaError;
}

ncclResult_t finishNvlsTeamRetry(
    const ncclComm* comm,
    const char* cleanupOp,
    CUresult cleanupResult) {
  NCCLCHECK(
      collectiveCleanupResult(comm, cleanupOp, cleanupResult, CUDA_SUCCESS));
  NCCLCHECK(gNvlsBootstrap.barrier(
      comm->bootstrap,
      comm->localRankToRank,
      comm->localRank,
      comm->localRanks,
      comm->localRankToRank[0]));
  const int64_t retryBackoffMs = ncclParamNvlsBindRetryBackoffMs();
  if (retryBackoffMs > 0) {
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(retryBackoffMs));
  }
  return ncclSuccess;
}
#endif

} // namespace ncclx::nvls
