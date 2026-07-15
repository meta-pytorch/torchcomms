// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/CtranPipes.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <set>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterDirectIbConfig.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PRIMS)

#include "comms/prims/trace/PipesTrace.h"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/transport/ll128/Ll128Packet.cuh"

namespace {

bool ctranPipesTraceEnabled() {
  return NCCL_CTRAN_PIPES_TRACE_ENABLE;
}

int ctranPipesNvlMaxNumChannels() {
  return std::max(1, NCCL_CTRAN_MAX_NBLOCKS);
}

size_t alignDown(size_t value, size_t alignment) {
  return (value / alignment) * alignment;
}

} // namespace

commResult_t ctran::ctranPreparePipesTrace(
    CtranComm* comm,
    comms::prims::PipesTraceHandle& trace) {
  trace = {};
  if (!ctranPipesTraceEnabled()) {
    return commSuccess;
  }
  const uint32_t ringSize = comms::prims::PipesTrace::normalizeRingSize(
      NCCL_CTRAN_PIPES_TRACE_RING_SIZE);
  if (ringSize == 0) {
    return commSuccess;
  }

  if (comm->pipesTrace_ == nullptr) {
    comm->pipesTrace_ = std::make_unique<comms::prims::PipesTrace>();
  }
  comm->pipesTrace_->ensure(
      ringSize,
      std::chrono::milliseconds(NCCL_CTRAN_PIPES_TRACE_POLL_INTERVAL_MS));
  trace = comm->pipesTrace_->deviceHandle();
  return commSuccess;
}

commResult_t ctranInitializePipes(CtranComm* comm) {
  if (!NCCL_CTRAN_USE_PIPES) {
    CLOGF(INFO, "CTRAN-PRIMS: initialization skipped; prims are disabled");
    return commSuccess;
  }
  try {
    CLOGF(
        INFO,
        "CTRAN-PRIMS: initialization started rank={} nRanks={} cudaDev={}",
        comm->statex_->rank(),
        comm->statex_->nRanks(),
        comm->statex_->cudaDev());

    // Create a non-owning shared_ptr wrapper for bootstrap.
    // SAFETY: multiPeerTransport_ must be destroyed before bootstrap_ in
    // CtranComm::destroy() to avoid dangling reference.
    auto bootstrapPtr = std::shared_ptr<meta::comms::IBootstrap>(
        comm->bootstrap_.get(),
        [](meta::comms::IBootstrap*) {}); // no-op deleter

    const auto& pc = comm->config_.pipesConfig;
    comms::prims::MultiPeerTransportConfig config{};

    config.nvlConfig.pipelineDepth =
        static_cast<size_t>(NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH);

    const bool hierAgOverlapEnabled =
        NCCL_CTRAN_HIER_AG_OVERLAP_ENABLE && comm->statex_->nLocalRanks() > 1;
    const size_t nvlSharedDevbufSize =
        ctranEffectiveP2pNvlSharedDevbufSize(comm->statex_->nLocalRanks());
    const size_t nvlDataBufferSize = static_cast<size_t>(
        nvlSharedDevbufSize / config.nvlConfig.pipelineDepth);

    config.nvlConfig.maxNumChannels = ctranPipesNvlMaxNumChannels();
    config.nvlConfig.perChannelSize = alignDown(
        nvlDataBufferSize /
            static_cast<size_t>(config.nvlConfig.maxNumChannels),
        16);

    // LL128 buffer allocation for DeviceAllToAllv
    if (NCCL_CTRAN_DA2A_LL128_THRESHOLD > 0) {
      if (NCCL_CTRAN_DA2A_LL128_BUFFER_SIZE > 0) {
        config.nvlConfig.ll128BufferSize = NCCL_CTRAN_DA2A_LL128_BUFFER_SIZE;
      } else {
        config.nvlConfig.ll128BufferSize =
            comms::prims::ll128_buffer_size(256 * 1024);
      }
      CLOGF(
          INFO,
          "Prims LL128 buffer size configured (size={} per peer)",
          config.nvlConfig.ll128BufferSize);
    }

    // IB config (ordered to match MultipeerIbTransportConfig fields)
    config.ibConfig.cudaDevice = comm->statex_->cudaDev();
    if (NCCL_IB_GID_INDEX >= 0) {
      config.ibConfig.gidIndex = static_cast<int>(NCCL_IB_GID_INDEX);
    }
    if (!NCCL_IB_ADDR_FAMILY.empty()) {
      config.ibConfig.addressFamily = (NCCL_IB_ADDR_FAMILY == "IPV4")
          ? comms::prims::AddressFamily::IPV4
          : comms::prims::AddressFamily::IPV6;
    }
    // Pass raw NCCL_IB_HCA string to ibConfig; NicDiscovery's ibHcaParser
    // handles prefix semantics and port suffixes internally.
    if (!NCCL_IB_HCA.empty()) {
      std::string hcaStr = NCCL_IB_HCA_PREFIX;
      for (size_t i = 0; i < NCCL_IB_HCA.size(); ++i) {
        if (i > 0) {
          hcaStr += ',';
        }
        hcaStr += NCCL_IB_HCA[i];
      }
      config.ibConfig.ibHca = std::move(hcaStr);
    }
    uint64_t ibgdaDataBufferSize = (pc.ibgdaDataBufferSize > 0)
        ? static_cast<size_t>(pc.ibgdaDataBufferSize)
        : static_cast<size_t>(NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE);
    if (hierAgOverlapEnabled && NCCL_CTRAN_HIER_AG_IBGDA_DATA_BUFFER_SIZE > 0) {
      ibgdaDataBufferSize = std::max(
          ibgdaDataBufferSize, NCCL_CTRAN_HIER_AG_IBGDA_DATA_BUFFER_SIZE);
    }
    config.ibConfig.dataBufferSize = static_cast<size_t>(ibgdaDataBufferSize);
    config.ibConfig.qpDepth = NCCL_CTRAN_IBGDA_QP_DEPTH;
    if (NCCL_IB_TIMEOUT != NCCL_IB_TIMEOUT_DEFAULTCVARVALUE) {
      config.ibConfig.timeout = static_cast<uint8_t>(NCCL_IB_TIMEOUT);
    }
    if (NCCL_IB_RETRY_CNT != NCCL_IB_RETRY_CNT_DEFAULTCVARVALUE) {
      config.ibConfig.retryCount = static_cast<uint8_t>(NCCL_IB_RETRY_CNT);
    }
    if (NCCL_IB_TC != NCCL_IB_TC_DEFAULTCVARVALUE) {
      config.ibConfig.trafficClass = static_cast<uint8_t>(NCCL_IB_TC);
    }
    if (NCCL_IB_SL != NCCL_IB_SL_DEFAULTCVARVALUE) {
      config.ibConfig.serviceLevel = static_cast<uint8_t>(NCCL_IB_SL);
    }
    if (NCCL_CTRAN_IBGDA_MIN_RNR_TIMER !=
        NCCL_CTRAN_IBGDA_MIN_RNR_TIMER_DEFAULTCVARVALUE) {
      config.ibConfig.minRnrTimer =
          static_cast<uint8_t>(NCCL_CTRAN_IBGDA_MIN_RNR_TIMER);
    }
    if (NCCL_CTRAN_IBGDA_RNR_RETRY !=
        NCCL_CTRAN_IBGDA_RNR_RETRY_DEFAULTCVARVALUE) {
      config.ibConfig.rnrRetry =
          static_cast<uint8_t>(NCCL_CTRAN_IBGDA_RNR_RETRY);
    }
    config.ibConfig.ibLazyConnect = pc.ibLazyConnect;
    config.ibConfig.materializePeerTimeoutMs =
        NCCL_CTRAN_IBGDA_MATERIALIZE_PEER_TIMEOUT_MS;
    if (NCCL_CTRAN_IB_MAX_GROUPS <= 0) {
      CLOGF(
          ERR,
          "NCCL_CTRAN_IB_MAX_GROUPS must be positive, got {}",
          NCCL_CTRAN_IB_MAX_GROUPS);
      return commInvalidArgument;
    }
    if (NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC <= 0) {
      CLOGF(
          ERR,
          "NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC must be positive, got {}",
          NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC);
      return commInvalidArgument;
    }
    config.ibConfig.maxGroups = static_cast<int>(NCCL_CTRAN_IB_MAX_GROUPS);
    config.ibConfig.qpsPerConnection =
        static_cast<int>(NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC);

    const bool directIbReduceScatter =
        NCCL_REDUCESCATTER_ALGO == NCCL_REDUCESCATTER_ALGO::ctdirect_ib;
    if (directIbReduceScatter) {
      config.ibConfig.perChannelSize =
          ctran::reducescatter::direct_ib::kPerChannelSize;
      config.ibConfig.max_num_channels =
          ctran::reducescatter::direct_ib::kMaxNumBlocks;
      config.ibConfig.pipelineDepth =
          ctran::reducescatter::direct_ib::kPipelineDepth;
      config.ibConfig.qpsPerConnection =
          ctran::reducescatter::direct_ib::kQpsPerConnection;
      config.ibConfig.maxGroups =
          ctran::reducescatter::direct_ib::kMaxNumBlocks;
      config.ibConfig.dataBufferSize =
          config.ibConfig.fixedChannelDataBufferSize();
      CLOGF(
          INFO,
          "Direct IB ReduceScatter pins IB config: perChannelSize={}, maxNumChannels={}, pipelineDepth={}, qpsPerConnection={}, maxGroups={}, dataBufferSize={}",
          config.ibConfig.perChannelSize,
          config.ibConfig.max_num_channels,
          config.ibConfig.pipelineDepth,
          config.ibConfig.qpsPerConnection,
          config.ibConfig.maxGroups,
          config.ibConfig.dataBufferSize);
    }

    if (NCCL_CTRAN_IBGDA_SENDRECV_ENABLE || directIbReduceScatter) {
      if (config.ibConfig.dataBufferSize == 0) {
        CLOGF(
            ERR,
            "NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1 requires a positive "
            "IBGDA data-buffer size via NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE "
            "or the per-communicator IBGDA data-buffer override");
        return commInvalidArgument;
      }
      if (!directIbReduceScatter &&
          NCCL_CTRAN_IBGDA_SENDRECV_PIPELINE_DEPTH <= 0) {
        CLOGF(
            ERR,
            "NCCL_CTRAN_IBGDA_SENDRECV_PIPELINE_DEPTH must be positive, got {}",
            NCCL_CTRAN_IBGDA_SENDRECV_PIPELINE_DEPTH);
        return commInvalidArgument;
      }
      if (!directIbReduceScatter) {
        if (config.ibConfig.dataBufferSize %
                static_cast<std::size_t>(config.ibConfig.maxGroups) !=
            0) {
          CLOGF(
              ERR,
              "IBGDA data-buffer size {} must be divisible by maxGroups {}",
              config.ibConfig.dataBufferSize,
              config.ibConfig.maxGroups);
          return commInvalidArgument;
        }
        config.ibConfig.perChannelSize = config.ibConfig.dataBufferSize /
            static_cast<std::size_t>(config.ibConfig.maxGroups);
        config.ibConfig.max_num_channels = config.ibConfig.maxGroups;
        config.ibConfig.pipelineDepth =
            static_cast<int>(NCCL_CTRAN_IBGDA_SENDRECV_PIPELINE_DEPTH);
      }
      CLOGF(
          INFO,
          "Prims IBGDA sendRecv configured: perChannelSize={}, maxNumChannels={}, pipelineDepth={}, dataBufferSize={}, directIbReduceScatter={}",
          config.ibConfig.perChannelSize,
          config.ibConfig.max_num_channels,
          config.ibConfig.pipelineDepth,
          config.ibConfig.dataBufferSize,
          directIbReduceScatter);
    }

    if (NCCL_CTRAN_PIPES_IB_MODE == NCCL_CTRAN_PIPES_IB_MODE::ibrc) {
      config.ibMode = comms::prims::IbBackendMode::kIbrc;
    }
    config.disableIb = NCCL_CTRAN_PIPES_DISABLE_IB;
    config.topoConfig.p2pDisable = NCCL_P2P_DISABLE ||
        NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal;

    // Topology config: MNNVL mode and overrides
    config.topoConfig.mnnvlMode =
        static_cast<comms::prims::MnnvlMode>(NCCL_MNNVL_ENABLE);
    config.topoConfig.logicalNvlRanks = comm->statex_->localRankToRanks();

    CLOGF(
        INFO,
        "CTRAN-PRIMS: config prepared rank={} nvlPipelineDepth={} nvlSharedDevbufSize={} nvlDataBufferSize={} nvlMaxNumChannels={} nvlPerChannelSize={} hierAgOverlapEnabled={} disableIb={} p2pDisable={} mnnvlMode={} ibgdaDataBufferSize={} ibgdaQpDepth={} ibLazyConnect={} materializePeerTimeoutMs={}",
        comm->statex_->rank(),
        config.nvlConfig.pipelineDepth,
        nvlSharedDevbufSize,
        nvlDataBufferSize,
        config.nvlConfig.maxNumChannels,
        config.nvlConfig.perChannelSize,
        hierAgOverlapEnabled,
        config.disableIb,
        config.topoConfig.p2pDisable,
        static_cast<int>(config.topoConfig.mnnvlMode),
        config.ibConfig.dataBufferSize,
        config.ibConfig.qpDepth,
        config.ibConfig.ibLazyConnect,
        config.ibConfig.materializePeerTimeoutMs);

    CLOGF(
        INFO,
        "CTRAN-PRIMS: full config prepared rank={} logicalNvlRanks={}",
        comm->statex_->rank(),
        config.topoConfig.logicalNvlRanks
            ? config.topoConfig.logicalNvlRanks->size()
            : 0);

    // Guard against H100 Grand Teton returning NVML fabric info
    // (state=COMPLETED) without actual cross-node NVLink (MNNVL) capability.
    // The FABRIC handle export/import probe (same check used by ncclx's
    // ncclMnnvlCheck Gate 7 and CommStateX's isCuMemFabricEnabled) is the only
    // reliable way to distinguish real MNNVL (GB200) from false positives.
    if (config.topoConfig.mnnvlMode != comms::prims::MnnvlMode::kDisabled &&
        !ctran::utils::isCuMemFabricEnabled()) {
      CLOGF(
          INFO,
          "CTRAN-PRIMS: FABRIC handle probe failed — disabling MNNVL Tier 1 "
          "topology detection (falling back to same-host peer access)");
      config.topoConfig.mnnvlMode = comms::prims::MnnvlMode::kDisabled;
    }

    if (NCCL_MNNVL_UUID != -1) {
      config.topoConfig.mnnvlUuid = NCCL_MNNVL_UUID;
    }
    if (NCCL_MNNVL_CLIQUE_ID != -1) {
      config.topoConfig.mnnvlCliqueId = static_cast<int>(NCCL_MNNVL_CLIQUE_ID);
    }

    CLOGF(
        INFO,
        "CTRAN-PRIMS: constructing MultiPeerTransport rank={}",
        comm->statex_->rank());
    comm->multiPeerTransport_ =
        std::make_unique<comms::prims::MultiPeerTransport>(
            comm->statex_->rank(),
            comm->statex_->nRanks(),
            comm->statex_->cudaDev(),
            bootstrapPtr,
            config);
    CLOGF(
        INFO,
        "Prims MultiPeerTransport initialized: nvlPeers={}, ibPeers={}, p2pDisable={}",
        comm->multiPeerTransport_->nvl_n_ranks() - 1,
        comm->multiPeerTransport_->ib_peer_ranks().size(),
        config.topoConfig.p2pDisable);
  } catch (const std::exception& e) {
    CLOGF(ERR, "Failed to initialize Prims MultiPeerTransport: {}", e.what());
    return commInternalError;
  }

  // Wire staging buffers and build nvlTransports now that both CtranAlgo
  // (SharedResource) and MultiPeerTransport have been created.
  CLOGF(
      INFO,
      "CTRAN-PRIMS: starting resource initialization rank={}",
      comm->statex_->rank());
  auto ret = ctranInitPipesResources(comm->ctran_->algo.get());
  CLOGF(
      INFO,
      "CTRAN-PRIMS: resource initialization finished rank={} status={}",
      comm->statex_->rank(),
      static_cast<int>(ret));
  return ret;
}

// Verify that ctran (CommStateX) and prims (MultiPeerTransport) have a
// consistent view of the NVL peer group. This is critical because
// ctranInitPipesResources() wires ctran's SharedResource staging buffers
// (indexed by statex local rank) as external data buffers to prims (indexed
// by NVL local rank). A mismatch would cause buffer cross-wiring.
//
// Both systems assign NVL local indices by sorting global ranks:
//   - statex: CommStateX::localRank() returns position in sorted host group
//   - prims:  TopologyDiscovery sorts nvlGroupGlobalRanks then assigns i
//
// Checks performed:
//   1. Group sizes match (nLocalRanks == nvlNRanks)
//   2. Peer count matches (nvlPeerRanks.size() == nLocalRanks - 1)
//   3. Forward: every statex local rank exists in prims with the same NVL
//      local index (verifies identical ordering)
//   4. Reverse: every prims NVL peer exists in statex's local group
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
      "CTRAN-PRIMS: nLocalRanks ({}) != nvlNRanks ({}). "
      "External staging buffer wiring requires matching rank groups.",
      nLocalRanks,
      nvlNRanks);

  const auto& nvlPeerRanks = mpt->nvl_peer_ranks();
  FB_CHECKABORT(
      static_cast<int>(nvlPeerRanks.size()) == nLocalRanks - 1,
      "CTRAN-PRIMS: nvlPeerRanks size ({}) != nLocalRanks - 1 ({}). "
      "Peer rank sets must match.",
      nvlPeerRanks.size(),
      nLocalRanks - 1);

  // Build set of global ranks from statex's local group for reverse lookup.
  std::set<int> statexLocalRanks(
      localRankToRanks.begin(), localRankToRanks.end());

  // Check forward: every statex local rank is in prims' NVL group,
  // and the NVL local index agrees.
  for (int i = 0; i < nLocalRanks; i++) {
    int globalRank = localRankToRanks[i];
    int nvlLocalFromStatex = statex->localRank(globalRank);
    int nvlLocalFromPipes = mpt->global_to_nvl_local(globalRank);
    FB_CHECKABORT(
        nvlLocalFromStatex == nvlLocalFromPipes,
        "CTRAN-PRIMS: NVL local rank mismatch for global rank {}. "
        "statex->localRank()={} vs global_to_nvl_local()={}",
        globalRank,
        nvlLocalFromStatex,
        nvlLocalFromPipes);
  }

  // Check reverse: every prims NVL peer is in statex's local group.
  for (int peerGlobalRank : nvlPeerRanks) {
    FB_CHECKABORT(
        statexLocalRanks.count(peerGlobalRank) > 0,
        "CTRAN-PRIMS: Prims NVL peer rank {} not found in statex local "
        "group. The two systems disagree on which GPUs are NVL-connected.",
        peerGlobalRank);
  }
}

commResult_t ctranInitPipesResources(CtranAlgo* algo) {
  auto* comm = algo->comm_;
  if (!comm->multiPeerTransport_) {
    CLOGF(
        INFO,
        "CTRAN-PRIMS: resource initialization skipped; MultiPeerTransport is not initialized");
    return commSuccess;
  }

  auto* statex = comm->statex_.get();
  int localRank = statex->localRank();
  CLOGF(
      INFO,
      "CTRAN-PRIMS: resource initialization started rank={} localRank={} nLocalRanks={} nRanks={} cudaDev={}",
      statex->rank(),
      localRank,
      statex->nLocalRanks(),
      statex->nRanks(),
      statex->cudaDev());

  // Wire SharedResource staging buffers to MultiPeerTransport as external
  // data buffers, then exchange. This lets MultiPeerNvlTransport manage
  // its signal/channel-state buffers internally while reusing the staging
  // buffers already allocated and IPC-shared via SharedResource.
  FB_CHECKABORT(
      algo->sharedRes_ != nullptr,
      "CTRAN-PRIMS: SharedResource must be initialized before "
      "prims resource initialization");

  int nvlNRanks = comm->multiPeerTransport_->nvl_n_ranks();
  CLOGF(
      INFO,
      "CTRAN-PRIMS: resource topology rank={} nvlNRanks={} nvlPeers={} ibPeers={}",
      statex->rank(),
      nvlNRanks,
      nvlNRanks - 1,
      comm->multiPeerTransport_->ib_peer_ranks().size());

  // Wire staging buffers only when there are NVL peers. When P2P is disabled
  // (NCCL_P2P_DISABLE=1), nvlNRanks == 1 (self only) while nLocalRanks may
  // be larger. No NVL peers means no staging buffers to wire; communication
  // falls back to IBGDA for all peers including intra-node.
  if (nvlNRanks > 1) {
    CLOGF(
        INFO,
        "CTRAN-PRIMS: validating ctran/prims consistency rank={}",
        statex->rank());
    validatePipesCtranConsistency(comm);
    CLOGF(
        INFO,
        "CTRAN-PRIMS: ctran/prims consistency validated rank={}",
        statex->rank());

    // Build per-NVL-rank buffer spans. DeviceSpan is non-assignable (const
    // pointer member), so we construct the vectors in NVL local rank order.
    const auto bufSize = static_cast<uint32_t>(algo->devState_.bufSize);
    CLOGF(
        INFO,
        "CTRAN-PRIMS: building external NVL staging spans rank={} bufSize={} nvlNRanks={}",
        statex->rank(),
        bufSize,
        nvlNRanks);
    std::vector<comms::prims::DeviceSpan<char>> localSpans;
    std::vector<comms::prims::DeviceSpan<char>> remoteSpans;
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
      CLOGF(
          INFO,
          "CTRAN-PRIMS: wiring NVL staging span rank={} nvlLocalRank={} localBuf={} remoteBuf={} size={}",
          statex->rank(),
          nvl,
          algo->devState_.localStagingBufsMap[nvl],
          algo->devState_.remoteStagingBufsMap[nvl],
          bufSize);
      localSpans.emplace_back(
          static_cast<char*>(algo->devState_.localStagingBufsMap[nvl]),
          bufSize);
      remoteSpans.emplace_back(
          static_cast<char*>(algo->devState_.remoteStagingBufsMap[nvl]),
          bufSize);
    }

    comms::prims::ExternalStagingBuffers externalBufs;
    externalBufs.localBuffers = std::move(localSpans);
    externalBufs.remoteBuffers = std::move(remoteSpans);

    CLOGF(
        INFO,
        "CTRAN-PRIMS: setting external NVL data buffers rank={}",
        statex->rank());
    comm->multiPeerTransport_->setExternalNvlDataBuffers(
        std::move(externalBufs));
    CLOGF(
        INFO,
        "CTRAN-PRIMS: external NVL data buffers set rank={}",
        statex->rank());
  } else {
    CLOGF(
        INFO,
        "CTRAN-PRIMS: no NVL peers; skipping external staging buffer wiring rank={}",
        statex->rank());
  }

  CLOGF(
      INFO,
      "CTRAN-PRIMS: starting MultiPeerTransport exchange rank={}",
      statex->rank());
  comm->multiPeerTransport_->exchange();
  CLOGF(
      INFO,
      "CTRAN-PRIMS: MultiPeerTransport exchange finished rank={}",
      statex->rank());

  CLOGF(
      INFO,
      "CTRAN-PRIMS: resource initialization finished rank={}",
      statex->rank());
  return commSuccess;
}

#else

commResult_t ctranInitializePipes(CtranComm* comm) {
  return commSuccess;
}

commResult_t ctranInitPipesResources(CtranAlgo* algo) {
  return commSuccess;
}

#endif // defined(ENABLE_PRIMS)
