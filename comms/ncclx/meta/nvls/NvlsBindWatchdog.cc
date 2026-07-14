// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/nvls/NvlsBindWatchdog.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

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
  std::mutex mutex;
  std::condition_variable cv;
  bool done{false};
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
      "NVLS cuMulticastBindMem %s cuResult %d elapsedMs %llu host %s pid %llu rank %d/%d localRank %d/%d cudaDev %d nvlsChannels %d comm %p commHash %llx commDesc %s inputSize %zu ucsize %zu mcsize %zu mcHandle 0x%llx ucHandle 0x%llx watchdogSec %lld",
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
      "NVLS cuMulticastBindMem STUCK elapsedMs %llu host %s pid %llu rank %d/%d localRank %d/%d cudaDev %d nvlsChannels %d comm %p commHash %llx commDesc %s inputSize %zu ucsize %zu mcsize %zu mcHandle 0x%llx ucHandle 0x%llx. Check Fabric Manager/NVSwitch logs for multicast team setup errors such as stale or missing GPU handles after a Fabric Manager restart; affected GPUs or the partition may need a GPU reset before NVLS jobs can run safely.",
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

void bindWatchdog(NvlsBindWatchdogState* state) {
  NCCL_NAMED_THREAD_START_EXT(
      "NVLSBindWatch", state->rank, state->commHash, state->commDesc);
  std::unique_lock<std::mutex> lock(state->mutex);
  while (!state->done) {
    if (state->cv.wait_for(
            lock, std::chrono::seconds(state->watchdogSec), [state] {
              return state->done;
            })) {
      break;
    }
    logBindStuck(*state);
  }
}

} // namespace

namespace ncclx::nvls {

CUresult multicastBindMemWithWatchdog(
    const ncclComm* comm,
    size_t inputSize,
    size_t ucsize,
    size_t mcsize,
    CUmemGenericAllocationHandle mcHandle,
    CUmemGenericAllocationHandle ucHandle) {
  const int64_t watchdogSec = ncclParamNvlsBindWatchdogSec();
  NvlsBindWatchdogState state;
  fillWatchdogState(
      &state, comm, inputSize, ucsize, mcsize, mcHandle, ucHandle, watchdogSec);

  std::thread watchdogThread;
  if (watchdogSec > 0) {
    try {
      watchdogThread = std::thread(bindWatchdog, &state);
    } catch (const std::exception& ex) {
      WARN(
          "NVLS cuMulticastBindMem watchdog thread launch failed for rank %d localRank %d cudaDev %d: %s",
          state.rank,
          state.localRank,
          state.cudaDev,
          ex.what());
    }
  }

  logBindState("START", state, -1);
  CUresult err = CUPFN(cuMulticastBindMem(mcHandle, 0, ucHandle, 0, ucsize, 0));
  if (watchdogThread.joinable()) {
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.done = true;
    }
    state.cv.notify_one();
  }

  logBindState("RETURN", state, static_cast<int>(err));

  if (watchdogThread.joinable()) {
    const ncclResult_t joinRes = ncclThreadJoin(watchdogThread);
    if (joinRes != ncclSuccess) {
      WARN(
          "NVLS cuMulticastBindMem watchdog thread join failed for rank %d localRank %d cudaDev %d: %d",
          state.rank,
          state.localRank,
          state.cudaDev,
          joinRes);
    }
  }

  return err;
}

} // namespace ncclx::nvls
