// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/CtranPipes.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <set>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/common/OrderedWorkStreamGuard.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranLogger.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

bool ctranPrimsEnabled(const CtranComm* comm) {
  const auto enablePrims = comm->config_.primsConfig.enablePrims;
  return enablePrims < 0 ? NCCL_CTRAN_USE_PIPES : enablePrims != 0;
}

#if defined(ENABLE_PRIMS)

#include "comms/prims/trace/PipesTrace.h"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/transport/ll128/Ll128Packet.cuh"

namespace {

bool ctranPipesTraceEnabled() {
  return NCCL_CTRAN_PIPES_TRACE_ENABLE;
}

// Resolves the per-communicator override first, MCCL_MAX_NBLOCKS second. As
// with the CVAR this is both the NVL channel count and the collective
// launch-geometry block cap -- see ctranPrimsResolvedMaxBlocks().
// Clamped into int range before narrowing: an int64 hint of 2^32 would
// otherwise truncate to 0 and 2^31 to INT_MIN, both silently collapsing to a
// single channel.
int ctranPipesNvlMaxNumChannels(const ctranPrimsConfig& pc) {
  const int64_t resolved = std::min<int64_t>(
      ctranPrimsResolvedMaxBlocks(pc), std::numeric_limits<int>::max());
  return std::max(1, static_cast<int>(resolved));
}

size_t alignedPerChannelSize(
    size_t totalSize,
    size_t maxChannels,
    size_t pipelineDepth) {
  constexpr size_t kDataAlignment = 16;
  if (maxChannels == 0 || pipelineDepth == 0 ||
      pipelineDepth > std::numeric_limits<size_t>::max() / kDataAlignment) {
    return 0;
  }

  const size_t alignment = kDataAlignment * pipelineDepth;
  return (totalSize / maxChannels / alignment) * alignment;
}

} // namespace

commResult_t ctran::ctranBuildMultimemNvlTransportConfig(
    const ctranPrimsConfig& config,
    size_t bufferSize,
    int nLocalRanks,
    comms::prims::MultimemNvlTransportConfig& multimemConfig) {
  const size_t pipelineDepth = MCCL_NVL_MULTIMEM_PIPELINE_DEPTH;
  constexpr size_t kMaxPipelineDepth = std::min(
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
      std::numeric_limits<size_t>::max() / 16);
  if (pipelineDepth == 0 || pipelineDepth > kMaxPipelineDepth) {
    CLOGF(
        ERR,
        "MCCL_NVL_MULTIMEM_PIPELINE_DEPTH must be in [1, {}], got {}",
        kMaxPipelineDepth,
        pipelineDepth);
    return commInvalidArgument;
  }

  const int64_t resolvedMaxChannels = ctranPrimsResolvedMaxChannels(config);
  const int64_t resolvedMaxBlocks = ctranPrimsResolvedMaxBlocks(config);
  const size_t maxChannels =
      resolvedMaxChannels > 0 ? static_cast<size_t>(resolvedMaxChannels) : 0;
  const size_t maxBlocks =
      resolvedMaxBlocks > 0 ? static_cast<size_t>(resolvedMaxBlocks) : 0;
  const size_t perChannelSize =
      alignedPerChannelSize(bufferSize, maxChannels, pipelineDepth);
  const auto candidate = comms::prims::make_multimem_nvl_transport_config({
      .perChannelSize = perChannelSize,
      .pipelineDepth = pipelineDepth,
      .maxChannels = maxChannels,
      .maxBlocks = maxBlocks,
      .userSignalCount = 1,
  });
  const auto validation = comms::prims::validate_multimem_nvl_transport_config(
      candidate, nLocalRanks);
  if (!validation) {
    CLOGF(
        ERR,
        "CTRAN-PRIMS: invalid NVL multimem config: {} (bufferSize={} perChannelSize={} pipelineDepth={} maxChannels={} maxBlocks={})",
        validation.errorMessage,
        bufferSize,
        candidate.perChannelSize,
        candidate.pipelineDepth,
        candidate.maxChannels,
        candidate.maxBlocks);
    return commInvalidArgument;
  }

  multimemConfig = candidate;
  return commSuccess;
}

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
      std::chrono::milliseconds(NCCL_CTRAN_PIPES_TRACE_POLL_INTERVAL_MS),
      nullptr,
      static_cast<uint32_t>(comm->statex_->rank()));
  trace = comm->pipesTrace_->deviceHandle();
  return commSuccess;
}

commResult_t ctranInitializePipes(CtranComm* comm) {
  if (!ctranPrimsEnabled(comm)) {
    CTRAN_LOG(INFO, "CTRAN-PRIMS: initialization skipped; prims are disabled");
    return commSuccess;
  }
  comms::prims::PipesTraceHandle trace;
  FB_COMMCHECK(ctran::ctranPreparePipesTrace(comm, trace));
  try {
    CTRAN_LOG(
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

    const auto& pc = comm->config_.primsConfig;
    comms::prims::MultiPeerTransportConfig config{};

    config.nvlConfig.pipelineDepth =
        static_cast<size_t>(NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH);

    const bool hierAgOverlapEnabled =
        NCCL_CTRAN_HIER_AG_OVERLAP_ENABLE && comm->statex_->nLocalRanks() > 1;
    const size_t nvlSharedDevbufSize =
        ctranEffectiveP2pNvlSharedDevbufSize(comm->statex_->nLocalRanks());
    config.nvlConfig.maxNumChannels = ctranPipesNvlMaxNumChannels(pc);
    const size_t nvlMaxNumChannels =
        static_cast<size_t>(config.nvlConfig.maxNumChannels);
    config.nvlConfig.perChannelSize = alignedPerChannelSize(
        nvlSharedDevbufSize, nvlMaxNumChannels, config.nvlConfig.pipelineDepth);
    if (config.nvlConfig.perChannelSize == 0) {
      CLOGF(
          ERR,
          "CTRAN-PRIMS: invalid NVL config; sharedDevbufSize={} maxNumChannels={} pipelineDepth={} cannot produce aligned perChannelSize",
          nvlSharedDevbufSize,
          config.nvlConfig.maxNumChannels,
          config.nvlConfig.pipelineDepth);
      return commInvalidArgument;
    }
    const size_t nvlDataBufferSize =
        nvlMaxNumChannels * config.nvlConfig.perChannelSize;

    // The multimem staging window is independent of the P2P shared devbuf.
    // A larger window means fewer staging rounds, which is the dominant
    // cnvlmm throughput lever at large sizes. A zero size disables multimem.
    const size_t multimemDevbufSize =
        static_cast<size_t>(MCCL_NVL_MULTIMEM_BUFSIZE);
    if (comm->statex_->nLocalRanks() > 2 && multimemDevbufSize > 0) {
      comms::prims::MultimemNvlTransportConfig multimemConfig{};
      if (const auto result = ctran::ctranBuildMultimemNvlTransportConfig(
              pc,
              multimemDevbufSize,
              comm->statex_->nLocalRanks(),
              multimemConfig);
          result != commSuccess) {
        return result;
      }
      config.nvlConfig.enableMultimem = true;
      config.nvlConfig.multimem = multimemConfig;
    }

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
    const bool channelsFromHint = pc.maxChannels > 0;
    const char* const channelsSource =
        channelsFromHint ? "primsConfig.maxChannels" : "MCCL_MAX_NCHANNELS";
    const int64_t requestedMaxChannels = ctranPrimsResolvedMaxChannels(pc);
    if (requestedMaxChannels <= 0 ||
        requestedMaxChannels >
            static_cast<int64_t>(std::numeric_limits<int>::max())) {
      CLOGF(
          ERR,
          "max channels must be in [1, {}], got {} (from {})",
          std::numeric_limits<int>::max(),
          requestedMaxChannels,
          channelsSource);
      return commInvalidArgument;
    }
    const int maxChannels = static_cast<int>(requestedMaxChannels);
    // Each knob resolves per-communicator hint first, global CVAR second. The
    // source is carried into every diagnostic below so a bad value points at
    // the setting that produced it rather than at an internal constant.
    const bool depthFromHint = pc.channelPipelineDepth > 0;
    const char* const depthSource = depthFromHint
        ? "primsConfig.channelPipelineDepth"
        : "MCCL_CHANNEL_PIPELINE_DEPTH";
    // Range-check before narrowing: an int64 hint of 2^32+8 would otherwise
    // truncate to a silently-different depth of 8, and 2^32 would truncate to 0
    // and be reported as "got 0" rather than as the value the user set.
    const int64_t requestedPipelineDepth = depthFromHint
        ? pc.channelPipelineDepth
        : static_cast<int64_t>(MCCL_CHANNEL_PIPELINE_DEPTH);
    if (requestedPipelineDepth <= 0 ||
        requestedPipelineDepth >
            static_cast<int64_t>(std::numeric_limits<int>::max())) {
      CLOGF(
          ERR,
          "channel pipeline depth must be in [1, {}], got {} (from {})",
          std::numeric_limits<int>::max(),
          requestedPipelineDepth,
          depthSource);
      return commInvalidArgument;
    }
    const int channelPipelineDepth = static_cast<int>(requestedPipelineDepth);

    // Both sources are per-channel, per-direction, so the total is always an
    // exact multiple of the channel count -- no divisibility check needed.
    const bool bufferFromHint = pc.channelBufferSize > 0;
    const char* const bufferSource = bufferFromHint
        ? "primsConfig.channelBufferSize"
        : "MCCL_CHANNEL_BUFFER_SIZE";
    const size_t perDirectionChannelBuffer = bufferFromHint
        ? static_cast<size_t>(pc.channelBufferSize)
        : static_cast<size_t>(MCCL_CHANNEL_BUFFER_SIZE);
    if (perDirectionChannelBuffer >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(maxChannels)) {
      CLOGF(
          ERR,
          "channel buffer size {} (from {}) overflows total size for {} channels (from {})",
          perDirectionChannelBuffer,
          bufferSource,
          maxChannels,
          channelsSource);
      return commInvalidArgument;
    }
    config.ibConfig.dataBufferSize =
        perDirectionChannelBuffer * static_cast<size_t>(maxChannels);
    config.ibConfig.qpDepth = MCCL_IB_QP_DEPTH;
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
    if (NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC <= 0) {
      CLOGF(
          ERR,
          "NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC must be positive, got {}",
          NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC);
      return commInvalidArgument;
    }
    config.ibConfig.maxGroups = maxChannels;
    config.ibConfig.qpsPerConnection =
        static_cast<int>(NCCL_CTRAN_IB_QPS_PER_BLOCK_PER_NIC);
    switch (MCCL_IBGDA_RELIABLE_DOORBELL_MODE) {
      case MCCL_IBGDA_RELIABLE_DOORBELL_MODE::auto_:
        break;
      case MCCL_IBGDA_RELIABLE_DOORBELL_MODE::enable:
        config.ibConfig.enableReliableDoorbell = true;
        break;
      case MCCL_IBGDA_RELIABLE_DOORBELL_MODE::disable:
        config.ibConfig.enableReliableDoorbell = false;
        break;
    }

    if (config.ibConfig.dataBufferSize == 0) {
      CLOGF(
          ERR,
          "send/recv requires a positive staging size via MCCL_CHANNEL_BUFFER_SIZE or primsConfig.channelBufferSize");
      return commInvalidArgument;
    }
    const auto pipelineDepth = static_cast<size_t>(channelPipelineDepth);
    if (perDirectionChannelBuffer % pipelineDepth != 0) {
      CLOGF(
          ERR,
          "IB per-direction channel buffer {} (from {}) must be divisible by pipeline depth {}",
          perDirectionChannelBuffer,
          bufferSource,
          channelPipelineDepth);
      return commInvalidArgument;
    }
    // MultiPeerIbTransport enforces these too, but it throws
    // std::invalid_argument which the catch below turns into commInternalError
    // with no mention of the offending setting. Now that both values are
    // settable per communicator, check them here so a bad hint is reported as
    // what it is.
    const size_t channelChunkSize = perDirectionChannelBuffer / pipelineDepth;
    auto check16ByteAligned =
        [](const char* what, size_t value, const std::string& source) {
          if (value >= 16 && value % 16 == 0) {
            return true;
          }
          CLOGF(
              ERR,
              "IB {} must be >= 16 and 16-byte aligned, got {} (from {})",
              what,
              value,
              source);
          return false;
        };
    // The chunk is derived from both knobs, so name both: a bad depth must not
    // report the buffer as the culprit.
    if (!check16ByteAligned(
            "per-direction channel buffer",
            perDirectionChannelBuffer,
            bufferSource) ||
        !check16ByteAligned(
            "channel chunk (buffer / pipeline depth)",
            channelChunkSize,
            fmt::format("{} / {}", bufferSource, depthSource))) {
      return commInvalidArgument;
    }

    config.ibConfig.perChannelSize = perDirectionChannelBuffer;
    config.ibConfig.max_num_channels = config.ibConfig.maxGroups;
    config.ibConfig.pipelineDepth = channelPipelineDepth;
    CLOGF(
        INFO,
        "Prims IB sendRecv configured: rank={}, commDesc={}, perChannelSize={} (from {}), channelChunkSize={}, maxNumChannels={}, pipelineDepth={} (from {}), dataBufferSize={}",
        comm->statex_->rank(),
        comm->config_.commDesc,
        config.ibConfig.perChannelSize,
        bufferSource,
        channelChunkSize,
        config.ibConfig.max_num_channels,
        config.ibConfig.pipelineDepth,
        depthSource,
        config.ibConfig.dataBufferSize);

    if (MCCL_IB_MODE == MCCL_IB_MODE::ibrc) {
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
        "CTRAN-PRIMS: config prepared rank={} nvlPipelineDepth={} nvlSharedDevbufSize={} nvlDataBufferSize={} nvlMaxNumChannels={} nvlPerChannelSize={} enableMultimem={} multimemPerChannelSize={} multimemPipelineDepth={} multimemMaxChannels={} multimemMaxBlocks={} hierAgOverlapEnabled={} disableIb={} p2pDisable={} mnnvlMode={} ibgdaDataBufferSize={} ibgdaQpDepth={} ibLazyConnect={}",
        comm->statex_->rank(),
        config.nvlConfig.pipelineDepth,
        nvlSharedDevbufSize,
        nvlDataBufferSize,
        config.nvlConfig.maxNumChannels,
        config.nvlConfig.perChannelSize,
        config.nvlConfig.enableMultimem,
        config.nvlConfig.enableMultimem
            ? config.nvlConfig.multimem.perChannelSize
            : 0,
        config.nvlConfig.enableMultimem
            ? config.nvlConfig.multimem.pipelineDepth
            : 0,
        config.nvlConfig.enableMultimem ? config.nvlConfig.multimem.maxChannels
                                        : 0,
        config.nvlConfig.enableMultimem ? config.nvlConfig.multimem.maxBlocks
                                        : 0,
        hierAgOverlapEnabled,
        config.disableIb,
        config.topoConfig.p2pDisable,
        static_cast<int>(config.topoConfig.mnnvlMode),
        config.ibConfig.dataBufferSize,
        config.ibConfig.qpDepth,
        config.ibConfig.ibLazyConnect);

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

    CTRAN_LOG(
        INFO,
        "CTRAN-PRIMS: constructing MultiPeerTransport rank={}",
        comm->statex_->rank());
    comm->multiPeerTransport_ =
        std::make_unique<comms::prims::MultiPeerTransport>(
            comm->statex_->rank(),
            comm->statex_->nRanks(),
            comm->statex_->cudaDev(),
            bootstrapPtr,
            config,
            std::nullopt,
            comm->getAbort());
    auto primsOrderedWorkStreamGuard =
        std::make_unique<ctran::algos::OrderedWorkStreamGuard>();
    primsOrderedWorkStreamGuard->init(
        comm->logMetaData_, false /* synchronizeEagerAfterCapturedWork */);
    comm->primsOrderedWorkStreamGuard_ = std::move(primsOrderedWorkStreamGuard);
    CTRAN_LOG(
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
  CTRAN_LOG(
      INFO,
      "CTRAN-PRIMS: starting resource initialization rank={}",
      comm->statex_->rank());
  auto ret = ctranInitPipesResources(comm->ctran_->algo.get());
  CTRAN_LOG(
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

  // External staging buffers are indexed by Ctran's host-local rank. A
  // physical NVLink clique can be smaller than that group, in which case the
  // transport must retain its internally allocated buffers.
  const bool canUseExternalNvlBuffers =
      nvlNRanks > 1 && nvlNRanks == statex->nLocalRanks();
  if (canUseExternalNvlBuffers) {
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
  } else if (nvlNRanks <= 1) {
    CLOGF(
        INFO,
        "CTRAN-PRIMS: no NVL peers; skipping external staging buffer wiring rank={}",
        statex->rank());
  } else {
    CLOGF(
        INFO,
        "CTRAN-PRIMS: physical NVLink group size {} differs from Ctran local group size {}; using internal staging buffers rank={}",
        nvlNRanks,
        statex->nLocalRanks(),
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
