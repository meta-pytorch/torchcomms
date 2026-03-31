// Copyright (c) Meta Platforms, Inc. and affiliates.

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllToAll/DeviceAllToAllvPipesImpl.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include <algorithm>
#include <cstdlib>

namespace ctran::device_alltoallv_pipes {

// ---------------------------------------------------------------------------
// CollectiveConfig constructor — resolves all settings up front.
// Priority: per-collective hint > env var > CVAR > built-in default
// ---------------------------------------------------------------------------
CollectiveConfig::CollectiveConfig(
    int nLocalRanks,
    const std::unordered_map<std::string, std::string>* hints_ptr) {
  auto hintInt = [&](const std::string& key) -> int {
    if (hints_ptr) {
      auto it = hints_ptr->find(key);
      if (it != hints_ptr->end()) {
        try {
          return std::stoi(it->second);
        } catch (...) {
        }
      }
    }
    return -1;
  };

  auto hintBool = [&](const std::string& key) -> int {
    if (hints_ptr) {
      auto it = hints_ptr->find(key);
      if (it != hints_ptr->end()) {
        const auto& v = it->second;
        if (v == "1" || v == "true")
          return 1;
        if (v == "0" || v == "false")
          return 0;
      }
    }
    return -1; // not set
  };

  // === Step 1: blockScheduling ===
  // hint > NCCL_CTRAN_DA2A_BLOCK_SCHEDULING env > default(false/warp)
  int bs = hintBool("blockScheduling");
  if (bs >= 0) {
    blockScheduling = (bs == 1);
  } else {
    const char* env = getenv("NCCL_CTRAN_DA2A_BLOCK_SCHEDULING");
    blockScheduling = (env && std::atoi(env) == 1);
  }

  // === Step 2: LL128 threshold (CVAR-only) ===
  // Note: per-collective opt-in/out via hints could be added here in the
  // future if needed.
  ll128ThresholdBytes = static_cast<size_t>(NCCL_CTRAN_DA2A_LL128_THRESHOLD);

  // === Step 3: numBlocks ===
  unsigned int defaultBlocks =
      static_cast<unsigned int>(std::max(1, nLocalRanks * 2));

  // hint > env > default (resolved once for both protocols)
  int nb = hintInt("numBlocks");
  if (nb >= 0) {
    numBlocks = static_cast<unsigned int>(nb);
  } else {
    const char* env = getenv("NCCL_CTRAN_DA2A_NBLOCKS");
    if (env) {
      numBlocks = static_cast<unsigned int>(std::atoi(env));
    } else {
      numBlocks = defaultBlocks;
    }
  }

  // Align to CGA cluster size when LL128 is disabled
  // (volatile stores bypass L1, cluster alignment not beneficial).
  if (ll128ThresholdBytes == 0) {
    unsigned int clusterSize = NCCL_CTRAN_CGA_CLUSTER_SIZE;
    if (clusterSize > 1 && numBlocks % clusterSize != 0) {
      numBlocks = ((numBlocks + clusterSize - 1) / clusterSize) * clusterSize;
    }
  }

  // If block scheduling requested but not enough blocks,
  // fall back to warp scheduling.
  if (blockScheduling &&
      numBlocks < static_cast<unsigned int>(nLocalRanks * 2)) {
    blockScheduling = false;
  }

  // === Step 4: numThreads ===
  // hint > env > CVAR > default(512)
  int nt = hintInt("numThreads");
  if (nt >= 0) {
    numThreads = static_cast<unsigned int>(nt);
  } else {
    const char* env = getenv("NCCL_CTRAN_DA2A_NTHREADS");
    if (env) {
      numThreads = static_cast<unsigned int>(std::atoi(env));
    } else if (NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE > 0) {
      numThreads =
          static_cast<unsigned int>(NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE);
    } else {
      numThreads = 512;
    }
  }
}

// ---------------------------------------------------------------------------
// setupKernelConfig — populates KernelConfig + KernArgs from comm state
// and the fully-resolved CollectiveConfig.
// ---------------------------------------------------------------------------
commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    commDataType_t datatype,
    CtranComm* comm,
    KernelConfig& config,
    ctran::device_alltoallv_pipes::KernArgs& kernArgs,
    int64_t sendcountsMultiplier,
    int64_t recvcountsMultiplier,
    const CollectiveConfig& collConfig) {
  const auto statex = comm->statex_.get();

  kernArgs.sendbuff = sendbuff;
  kernArgs.recvbuff = recvbuff;
  kernArgs.elementSize = commTypeSize(datatype);
  kernArgs.myRank = statex->rank();
  kernArgs.nLocalRanks = statex->nLocalRanks();

  // Device pointers to split sizes
  kernArgs.sendcounts_d = sendcounts_d;
  kernArgs.recvcounts_d = recvcounts_d;

  // Scaling factors for multi-dimensional tensors
  kernArgs.sendcountsMultiplier = sendcountsMultiplier;
  kernArgs.recvcountsMultiplier = recvcountsMultiplier;

  // Build local rank -> global rank mapping
  if (kernArgs.nLocalRanks > CTRAN_MAX_NVL_PEERS) {
    return commInternalError;
  }
  for (int lr = 0; lr < kernArgs.nLocalRanks; lr++) {
    kernArgs.localRankToGlobalRank[lr] = statex->localRankToRank(lr);
  }

  // Set transport array from MultiPeerTransport
  kernArgs.transports = comm->getMultiPeerTransportsPtr();

  // LL128 threshold from pre-resolved config
  kernArgs.ll128ThresholdBytes = collConfig.ll128ThresholdBytes;

  // All config is pre-resolved in CollectiveConfig — just apply it.
  kernArgs.useBlockGroup = collConfig.blockScheduling;
  config.numBlocks = collConfig.numBlocks;
  config.numThreads = collConfig.numThreads;

  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.algoArgs = &kernArgs;

  return commSuccess;
}

} // namespace ctran::device_alltoallv_pipes

#endif // ENABLE_PIPES
