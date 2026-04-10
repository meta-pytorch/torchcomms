// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/CtranPipes.h"

#include <set>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PIPES)

#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/collectives/AllToAllvAutoTuneConfig.h"
#include "comms/pipes/ll128/Ll128Packet.cuh"

commResult_t ctranInitializePipes(CtranComm* comm) {
  if (!NCCL_CTRAN_USE_PIPES) {
    return commSuccess;
  }
  try {
    // Create a non-owning shared_ptr wrapper for bootstrap.
    // SAFETY: multiPeerTransport_ must be destroyed before bootstrap_ in
    // CtranComm::destroy() to avoid dangling reference.
    auto bootstrapPtr = std::shared_ptr<meta::comms::IBootstrap>(
        comm->bootstrap_.get(),
        [](meta::comms::IBootstrap*) {}); // no-op deleter

    comms::pipes::MultiPeerTransportConfig config{};

    // NVL config: per-comm hint > CVAR default
    const auto& pc = comm->config_.pipesConfig;

    config.nvlConfig.pipelineDepth =
        static_cast<size_t>(NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH);

    config.nvlConfig.dataBufferSize = static_cast<size_t>(
        NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE / config.nvlConfig.pipelineDepth);

    config.nvlConfig.chunkSize = (pc.nvlChunkSize > 0)
        ? static_cast<size_t>(pc.nvlChunkSize)
        : static_cast<size_t>(NCCL_CTRAN_PIPES_NVL_CHUNK_SIZE);

    config.nvlConfig.useDualStateBuffer = (pc.useDualStateBuffer >= 0)
        ? (pc.useDualStateBuffer == 1)
        : NCCL_CTRAN_PIPES_USE_DUAL_STATE_BUFFER;

    // LL128 buffer allocation for DeviceAllToAllv
    if (NCCL_CTRAN_DA2A_LL128_THRESHOLD > 0) {
      if (NCCL_CTRAN_DA2A_LL128_BUFFER_SIZE > 0) {
        config.nvlConfig.ll128BufferSize = NCCL_CTRAN_DA2A_LL128_BUFFER_SIZE;
      } else {
        config.nvlConfig.ll128BufferSize =
            comms::pipes::ll128_buffer_size(256 * 1024);
      }
      CLOGF(
          INFO,
          "Pipes LL128 buffer size configured (size={} per peer)",
          config.nvlConfig.ll128BufferSize);
    }

    // IBGDA config (ordered to match MultipeerIbgdaTransportConfig fields)
    config.ibgdaConfig.cudaDevice = comm->statex_->cudaDev();
    if (NCCL_IB_GID_INDEX >= 0) {
      config.ibgdaConfig.gidIndex = static_cast<int>(NCCL_IB_GID_INDEX);
    }
    if (!NCCL_IB_ADDR_FAMILY.empty()) {
      config.ibgdaConfig.addressFamily = (NCCL_IB_ADDR_FAMILY == "IPV4")
          ? comms::pipes::AddressFamily::IPV4
          : comms::pipes::AddressFamily::IPV6;
    }
    // Pass raw NCCL_IB_HCA string to ibgdaConfig; NicDiscovery's ibHcaParser
    // handles prefix semantics and port suffixes internally.
    if (!NCCL_IB_HCA.empty()) {
      std::string hcaStr = NCCL_IB_HCA_PREFIX;
      for (size_t i = 0; i < NCCL_IB_HCA.size(); ++i) {
        if (i > 0) {
          hcaStr += ',';
        }
        hcaStr += NCCL_IB_HCA[i];
      }
      config.ibgdaConfig.ibHca = std::move(hcaStr);
    }
    config.ibgdaConfig.dataBufferSize = NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE;
    config.ibgdaConfig.qpDepth = NCCL_CTRAN_IBGDA_QP_DEPTH;
    if (NCCL_IB_TIMEOUT != NCCL_IB_TIMEOUT_DEFAULTCVARVALUE) {
      config.ibgdaConfig.timeout = static_cast<uint8_t>(NCCL_IB_TIMEOUT);
    }
    if (NCCL_IB_RETRY_CNT != NCCL_IB_RETRY_CNT_DEFAULTCVARVALUE) {
      config.ibgdaConfig.retryCount = static_cast<uint8_t>(NCCL_IB_RETRY_CNT);
    }
    if (NCCL_IB_TC != NCCL_IB_TC_DEFAULTCVARVALUE) {
      config.ibgdaConfig.trafficClass = static_cast<uint8_t>(NCCL_IB_TC);
    }
    if (NCCL_IB_SL != NCCL_IB_SL_DEFAULTCVARVALUE) {
      config.ibgdaConfig.serviceLevel = static_cast<uint8_t>(NCCL_IB_SL);
    }
    if (NCCL_CTRAN_IBGDA_MIN_RNR_TIMER !=
        NCCL_CTRAN_IBGDA_MIN_RNR_TIMER_DEFAULTCVARVALUE) {
      config.ibgdaConfig.minRnrTimer =
          static_cast<uint8_t>(NCCL_CTRAN_IBGDA_MIN_RNR_TIMER);
    }
    if (NCCL_CTRAN_IBGDA_RNR_RETRY !=
        NCCL_CTRAN_IBGDA_RNR_RETRY_DEFAULTCVARVALUE) {
      config.ibgdaConfig.rnrRetry =
          static_cast<uint8_t>(NCCL_CTRAN_IBGDA_RNR_RETRY);
    }

    // IBGDA device alltoallv staging: start from autotune defaults, then
    // allow CVARs to override individual fields.
    {
      comms::pipes::AllToAllvAutoTuneConfig autoDefaults;
      config.ibgdaSetupConfig.dataBufferSize =
          autoDefaults.ibgdaInit.stagingBufferSize;
      config.ibgdaSetupConfig.chunkSize = autoDefaults.ibgdaInit.ibgdaChunkSize;
      config.ibgdaSetupConfig.pipelineDepth =
          autoDefaults.ibgdaInit.pipelineDepth;
    }
    if (NCCL_CTRAN_PIPES_DEVICE_ALLTOALLV_STAGING_BUFFER_SIZE > 0) {
      config.ibgdaSetupConfig.dataBufferSize =
          NCCL_CTRAN_PIPES_DEVICE_ALLTOALLV_STAGING_BUFFER_SIZE;
    }
    if (NCCL_CTRAN_PIPES_DEVICE_ALLTOALLV_CHUNK_SIZE > 0) {
      config.ibgdaSetupConfig.chunkSize =
          NCCL_CTRAN_PIPES_DEVICE_ALLTOALLV_CHUNK_SIZE;
    }
    if (NCCL_CTRAN_PIPES_DEVICE_ALLTOALLV_PIPELINE_DEPTH > 0) {
      config.ibgdaSetupConfig.pipelineDepth =
          static_cast<int>(NCCL_CTRAN_PIPES_DEVICE_ALLTOALLV_PIPELINE_DEPTH);
    }

    config.disableIb = NCCL_CTRAN_PIPES_DISABLE_IB;
    config.topoConfig.p2pDisable = NCCL_P2P_DISABLE ||
        NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal;

    // TEST_IBGDA_SINGLE_NODE=1: Force all peers to IBGDA even on single node.
    // Data travels through real NICs (GPU→PCIe→NIC→IB switch→NIC→PCIe→GPU).
    // For testing IBGDA alltoallv correctness without multi-node cluster.
    const char* forceIbgdaEnv = std::getenv("TEST_IBGDA_SINGLE_NODE");
    if (forceIbgdaEnv && std::string(forceIbgdaEnv) == "1") {
      config.forceIbgda = true;
      config.disableIb = false; // Override — can't force IBGDA with IB disabled
    }

    // Topology config: MNNVL mode and overrides
    config.topoConfig.mnnvlMode =
        static_cast<comms::pipes::MnnvlMode>(NCCL_MNNVL_ENABLE);

    // Guard against H100 Grand Teton returning NVML fabric info
    // (state=COMPLETED) without actual cross-node NVLink (MNNVL) capability.
    // The FABRIC handle export/import probe (same check used by ncclx's
    // ncclMnnvlCheck Gate 7 and CommStateX's isCuMemFabricEnabled) is the only
    // reliable way to distinguish real MNNVL (GB200) from false positives.
    if (config.topoConfig.mnnvlMode != comms::pipes::MnnvlMode::kDisabled &&
        !ctran::utils::isCuMemFabricEnabled()) {
      CLOGF(
          INFO,
          "CTRAN-PIPES: FABRIC handle probe failed — disabling MNNVL Tier 1 "
          "topology detection (falling back to same-host peer access)");
      config.topoConfig.mnnvlMode = comms::pipes::MnnvlMode::kDisabled;
    }

    if (NCCL_MNNVL_UUID != -1) {
      config.topoConfig.mnnvlUuid = NCCL_MNNVL_UUID;
    }
    if (NCCL_MNNVL_CLIQUE_ID != -1) {
      config.topoConfig.mnnvlCliqueId = static_cast<int>(NCCL_MNNVL_CLIQUE_ID);
    }

    // Derive maxChannelsPerPeer from autotune table (worst-case with
    // numIbgdaPeers=1 — gives the maximum possible channel count, safe to
    // oversize since unused channels are inert).
    config.maxChannelsPerPeer = comms::pipes::getMaxIbgdaChannelsPerPeer(1);

    comm->multiPeerTransport_ =
        std::make_unique<comms::pipes::MultiPeerTransport>(
            comm->statex_->rank(),
            comm->statex_->nRanks(),
            comm->statex_->cudaDev(),
            bootstrapPtr,
            config);
    CLOGF(
        INFO,
        "Pipes MultiPeerTransport initialized: nvlPeers={}, ibgdaPeers={}, p2pDisable={}",
        comm->multiPeerTransport_->nvl_n_ranks() - 1,
        comm->multiPeerTransport_->ibgda_peer_ranks().size(),
        config.topoConfig.p2pDisable);

    // Populate AllToAllvAutoTuneConfig with in-code defaults.
    // CVAR overrides take priority over the in-code defaults.
    auto& autoTune = comm->alltoallvAutoTuneConfig_;

    // NVL init-time defaults (from lookup table), overridden by CVARs
    autoTune.nvlInit.nvlChunkSize = config.nvlConfig.chunkSize;
    autoTune.nvlInit.nvlDataBufferSize =
        config.nvlConfig.dataBufferSize * config.nvlConfig.pipelineDepth;
    autoTune.nvlInit.pipelineDepth =
        static_cast<int>(config.nvlConfig.pipelineDepth);

    // IBGDA init-time defaults (from lookup table), overridden by CVARs
    autoTune.ibgdaInit.stagingBufferSize =
        config.ibgdaSetupConfig.dataBufferSize;
    autoTune.ibgdaInit.pipelineDepth = config.ibgdaSetupConfig.pipelineDepth;
    autoTune.ibgdaInit.ibgdaChunkSize = config.ibgdaSetupConfig.chunkSize;
  } catch (const std::exception& e) {
    CLOGF(ERR, "Failed to initialize Pipes MultiPeerTransport: {}", e.what());
    return commInternalError;
  }

  // Wire staging buffers and build nvlTransports now that both CtranAlgo
  // (SharedResource) and MultiPeerTransport have been created.
  return ctranInitPipesResources(comm->ctran_->algo.get());
}

// Verify that ctran (CommStateX) and pipes (MultiPeerTransport) have a
// consistent view of the NVL peer group. This is critical because
// ctranInitPipesResources() wires ctran's SharedResource staging buffers
// (indexed by statex local rank) as external data buffers to pipes (indexed
// by NVL local rank). A mismatch would cause buffer cross-wiring.
//
// Both systems assign NVL local indices by sorting global ranks:
//   - statex: CommStateX::localRank() returns position in sorted host group
//   - pipes:  TopologyDiscovery sorts nvlGroupGlobalRanks then assigns i
//
// Checks performed:
//   1. Group sizes match (nLocalRanks == nvlNRanks)
//   2. Peer count matches (nvlPeerRanks.size() == nLocalRanks - 1)
//   3. Forward: every statex local rank exists in pipes with the same NVL
//      local index (verifies identical ordering)
//   4. Reverse: every pipes NVL peer exists in statex's local group
//      (together with #3, proves set equality)
//
// Aborts on any mismatch since continuing would corrupt communication.
void validatePipesCtranConsistency(CtranComm* comm) {
  auto* statex = comm->statex_.get();
  auto* mpt = comm->multiPeerTransport_.get();
  int nLocalRanks = statex->nLocalRanks();
  auto localRankToRanks = statex->localRankToRanks();
  int nvlNRanks = mpt->nvl_n_ranks();
  FB_CHECKABORT(
      nLocalRanks == nvlNRanks,
      "CTRAN-PIPES: nLocalRanks ({}) != nvlNRanks ({}). "
      "External staging buffer wiring requires matching rank groups.",
      nLocalRanks,
      nvlNRanks);

  const auto& nvlPeerRanks = mpt->nvl_peer_ranks();
  FB_CHECKABORT(
      static_cast<int>(nvlPeerRanks.size()) == nLocalRanks - 1,
      "CTRAN-PIPES: nvlPeerRanks size ({}) != nLocalRanks - 1 ({}). "
      "Peer rank sets must match.",
      nvlPeerRanks.size(),
      nLocalRanks - 1);

  // Build set of global ranks from statex's local group for reverse lookup.
  std::set<int> statexLocalRanks(
      localRankToRanks.begin(), localRankToRanks.end());

  // Check forward: every statex local rank is in pipes' NVL group,
  // and the NVL local index agrees.
  for (int i = 0; i < nLocalRanks; i++) {
    int globalRank = localRankToRanks[i];
    int nvlLocalFromStatex = statex->localRank(globalRank);
    int nvlLocalFromPipes = mpt->global_to_nvl_local(globalRank);
    FB_CHECKABORT(
        nvlLocalFromStatex == nvlLocalFromPipes,
        "CTRAN-PIPES: NVL local rank mismatch for global rank {}. "
        "statex->localRank()={} vs global_to_nvl_local()={}",
        globalRank,
        nvlLocalFromStatex,
        nvlLocalFromPipes);
  }

  // Check reverse: every pipes NVL peer is in statex's local group.
  for (int peerGlobalRank : nvlPeerRanks) {
    FB_CHECKABORT(
        statexLocalRanks.count(peerGlobalRank) > 0,
        "CTRAN-PIPES: Pipes NVL peer rank {} not found in statex local "
        "group. The two systems disagree on which GPUs are NVL-connected.",
        peerGlobalRank);
  }
}

commResult_t ctranInitPipesResources(CtranAlgo* algo) {
  auto* comm = algo->comm_;
  if (!comm->multiPeerTransport_) {
    return commSuccess;
  }

  auto* statex = comm->statex_.get();
  int localRank = statex->localRank();

  // Wire SharedResource staging buffers to MultiPeerTransport as external
  // data buffers, then exchange. This lets MultiPeerNvlTransport manage
  // ChunkState and signal buffers internally while reusing the staging
  // buffers already allocated and IPC-shared via SharedResource.
  FB_CHECKABORT(
      algo->sharedRes_ != nullptr,
      "CTRAN-PIPES: SharedResource must be initialized before "
      "ctranInitPipesResources");

  int nvlNRanks = comm->multiPeerTransport_->nvl_n_ranks();

  // Wire staging buffers only when there are NVL peers. When P2P is disabled
  // (NCCL_P2P_DISABLE=1), nvlNRanks == 1 (self only) while nLocalRanks may
  // be larger. No NVL peers means no staging buffers to wire; communication
  // falls back to IBGDA for all peers including intra-node.
  if (nvlNRanks > 1) {
    validatePipesCtranConsistency(comm);

    // Build per-NVL-rank buffer spans. DeviceSpan is non-assignable (const
    // pointer member), so we construct the vectors in NVL local rank order.
    const auto bufSize = static_cast<uint32_t>(algo->devState_.bufSize);
    std::vector<comms::pipes::DeviceSpan<char>> localSpans;
    std::vector<comms::pipes::DeviceSpan<char>> remoteSpans;
    localSpans.reserve(nvlNRanks);
    remoteSpans.reserve(nvlNRanks);

    for (int nvl = 0; nvl < nvlNRanks; nvl++) {
      if (nvl == localRank) {
        localSpans.emplace_back(nullptr, 0u);
        remoteSpans.emplace_back(nullptr, 0u);
        continue;
      }
      // Map NVL local rank back to statex local rank index (same value since
      // both systems assign indices in sorted global rank order).
      localSpans.emplace_back(
          static_cast<char*>(algo->devState_.localStagingBufsMap[nvl]),
          bufSize);
      remoteSpans.emplace_back(
          static_cast<char*>(algo->devState_.remoteStagingBufsMap[nvl]),
          bufSize);
    }

    comms::pipes::ExternalStagingBuffers externalBufs;
    externalBufs.localBuffers = std::move(localSpans);
    externalBufs.remoteBuffers = std::move(remoteSpans);

    comm->multiPeerTransport_->setExternalNvlDataBuffers(
        std::move(externalBufs));
  }

  comm->multiPeerTransport_->exchange();

  return commSuccess;
}

#else

commResult_t ctranInitializePipes(CtranComm* comm) {
  return commSuccess;
}

commResult_t ctranInitPipesResources(CtranAlgo* algo) {
  return commSuccess;
}

#endif // defined(ENABLE_PIPES)
