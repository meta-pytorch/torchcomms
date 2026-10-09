// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/nvls/NvlsBindWatchdog.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "meta/nvls/StuckWatchdog.h"

#include "comm.h"
#include "cudawrap.h"
#include "debug.h"
#include "os.h"
#include "param.h"
#include "utils.h"

namespace {

NCCL_PARAM(NvlsBindWatchdogSec, "NVLS_BIND_WATCHDOG_SEC", 30);

constexpr int kHostnameLen = 256;

struct NvlsBindWatchdogState {
  uint64_t startNs{0};
  uint64_t commHash{0};
  uint64_t pid{0};
  int rank{-1};
  int nRanks{-1};
  int localRank{-1};
  int localRanks{-1};
  int cudaDev{-1};
  int nvlsChannels{-1};
  size_t inputSize{0};
  size_t ucsize{0};
  size_t mcsize{0};
  CUmemGenericAllocationHandle mcHandle{};
  CUmemGenericAllocationHandle ucHandle{};
  const void* comm{nullptr};
  std::string commDesc{"unknown"};
  // The guarded driver call, named in every log line.
  const char* op{"cuMulticastBindMem"};
  int64_t watchdogSec{0};
  char hostname[kHostnameLen]{};
};

void initHostname(char* hostname) {
  if (getHostName(hostname, kHostnameLen, '.') != ncclSuccess) {
    std::snprintf(hostname, kHostnameLen, "unknown");
  }
  hostname[kHostnameLen - 1] = '\0';
}

void fillWatchdogState(
    NvlsBindWatchdogState* state,
    const ncclComm* comm,
    size_t inputSize,
    size_t ucsize,
    size_t mcsize,
    CUmemGenericAllocationHandle mcHandle,
    CUmemGenericAllocationHandle ucHandle,
    int64_t watchdogSec) {
  initHostname(state->hostname);
  state->inputSize = inputSize;
  state->ucsize = ucsize;
  state->mcsize = mcsize;
  state->mcHandle = mcHandle;
  state->ucHandle = ucHandle;
  state->watchdogSec = watchdogSec;
  state->pid = ncclOsGetPid();
  state->startNs = clockNano();
  state->comm = comm;
  if (comm != nullptr) {
    state->rank = comm->rank;
    state->nRanks = comm->nRanks;
    state->localRank = comm->localRank;
    state->localRanks = comm->localRanks;
    state->cudaDev = comm->cudaDev;
    state->nvlsChannels = comm->nvlsChannels;
    state->commHash = comm->logMetaData.commHash;
    state->commDesc = comm->logMetaData.commDesc;
  }
}

void logBindState(
    const char* event,
    const NvlsBindWatchdogState& state,
    int cuResult) {
  const uint64_t elapsedMs = (clockNano() - state.startNs) / 1000000;
  INFO(
      NCCL_INIT | NCCL_NVLS,
      "NVLS %s %s cuResult %d elapsedMs %llu host %s pid %llu rank %d/%d localRank %d/%d cudaDev %d nvlsChannels %d comm %p commHash %llx commDesc %s inputSize %zu ucsize %zu mcsize %zu mcHandle 0x%llx ucHandle 0x%llx watchdogSec %lld",
      state.op,
      event,
      cuResult,
      static_cast<unsigned long long>(elapsedMs),
      state.hostname,
      static_cast<unsigned long long>(state.pid),
      state.rank,
      state.nRanks,
      state.localRank,
      state.localRanks,
      state.cudaDev,
      state.nvlsChannels,
      state.comm,
      static_cast<unsigned long long>(state.commHash),
      state.commDesc.c_str(),
      state.inputSize,
      state.ucsize,
      state.mcsize,
      static_cast<unsigned long long>(state.mcHandle),
      static_cast<unsigned long long>(state.ucHandle),
      static_cast<long long>(state.watchdogSec));
}

void logBindStuck(const NvlsBindWatchdogState& state) {
  const uint64_t elapsedMs = (clockNano() - state.startNs) / 1000000;
  WARN(
      "NVLS %s STUCK elapsedMs %llu host %s pid %llu rank %d/%d localRank %d/%d cudaDev %d nvlsChannels %d comm %p commHash %llx commDesc %s inputSize %zu ucsize %zu mcsize %zu mcHandle 0x%llx ucHandle 0x%llx. Check Fabric Manager/NVSwitch logs for multicast team setup errors such as stale or missing GPU handles after a Fabric Manager restart; affected GPUs or the partition may need a GPU reset before NVLS jobs can run safely.",
      state.op,
      static_cast<unsigned long long>(elapsedMs),
      state.hostname,
      static_cast<unsigned long long>(state.pid),
      state.rank,
      state.nRanks,
      state.localRank,
      state.localRanks,
      state.cudaDev,
      state.nvlsChannels,
      state.comm,
      static_cast<unsigned long long>(state.commHash),
      state.commDesc.c_str(),
      state.inputSize,
      state.ucsize,
      state.mcsize,
      static_cast<unsigned long long>(state.mcHandle),
      static_cast<unsigned long long>(state.ucHandle));
}

template <typename Call>
CUresult runUnderWatchdog(NvlsBindWatchdogState& state, Call call) {
  ncclx::nvls::StuckWatchdog watchdog{
      std::chrono::seconds(state.watchdogSec),
      [&state] { logBindStuck(state); },
      [&state] {
        NCCL_NAMED_THREAD_START_EXT(
            "NVLSBindWatch", state.rank, state.commHash, state.commDesc);
      }};
  if (watchdog.launchFailed()) {
    WARN(
        "NVLS %s watchdog thread launch failed for rank %d localRank %d cudaDev %d: %s",
        state.op,
        state.rank,
        state.localRank,
        state.cudaDev,
        watchdog.launchError().c_str());
  }

  logBindState("START", state, -1);
  CUresult err = call();
  const bool joined = watchdog.finish();

  logBindState("RETURN", state, static_cast<int>(err));

  if (!joined) {
    // Fatal: the watchdog's destructor terminates the process; see finish().
    WARN(
        "NVLS %s watchdog thread join failed for rank %d localRank %d cudaDev %d: %s",
        state.op,
        state.rank,
        state.localRank,
        state.cudaDev,
        watchdog.joinError().c_str());
  }
  return err;
}

} // namespace

namespace ncclx::nvls {

CUresult multicastBindMemWithWatchdog(
    const ncclComm* comm,
    size_t inputSize,
    size_t ucsize,
    size_t mcsize,
    CUmemGenericAllocationHandle mcHandle,
    CUmemGenericAllocationHandle ucHandle,
    size_t mcOffset,
    size_t memOffset) {
  NvlsBindWatchdogState state;
  fillWatchdogState(
      &state,
      comm,
      inputSize,
      ucsize,
      mcsize,
      mcHandle,
      ucHandle,
      ncclParamNvlsBindWatchdogSec());
  return runUnderWatchdog(state, [&] {
    return CUPFN(
        cuMulticastBindMem(mcHandle, mcOffset, ucHandle, memOffset, ucsize, 0));
  });
}

ncclResult_t multicastMapWithWatchdog(
    const ncclComm* comm,
    CUdeviceptr base,
    size_t size,
    CUmemGenericAllocationHandle mcHandle) {
  NvlsBindWatchdogState state;
  fillWatchdogState(
      &state, comm, size, 0, size, mcHandle, 0, ncclParamNvlsBindWatchdogSec());
  state.op = "cuMemMap(multicast)";
  const CUresult err = runUnderWatchdog(
      state, [&] { return CUPFN(cuMemMap(base, size, 0, mcHandle, 0)); });
  if (err != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    (void)pfn_cuGetErrorString(err, &errStr);
    WARN("Cuda failure %d '%s'", err, errStr);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 0)
    printCudaDriverErrorHint(err);
#endif
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

} // namespace ncclx::nvls
