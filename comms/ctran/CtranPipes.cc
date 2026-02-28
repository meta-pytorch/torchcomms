// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/CtranPipes.h"

#include "comms/ctran/CtranComm.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PIPES)

#include "comms/pipes/MultiPeerTransport.h"

commResult_t ctranInitializePipes(CtranComm* comm) {
  if (!NCCL_CTRAN_USE_PIPES) {
    return commSuccess;
  }
  try {
    // Create a non-owning shared_ptr wrapper for bootstrap.
    // SAFETY: multiPeerTransport_ must be destroyed before bootstrap_ in
    // CtranComm::destroy() to avoid dangling reference.
    auto bootstrapPtr = std::shared_ptr<ctran::bootstrap::IBootstrap>(
        comm->bootstrap_.get(),
        [](ctran::bootstrap::IBootstrap*) {}); // no-op deleter

    comms::pipes::MultiPeerTransportConfig config{};

    // NVL config: use cvars consistent with CtranAlgo's P2P NVL transport
    config.nvlConfig.dataBufferSize = NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE /
        NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH;
    config.nvlConfig.chunkSize = NCCL_CTRAN_PIPES_NVL_CHUNK_SIZE;
    config.nvlConfig.pipelineDepth = NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH;

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
    config.ibgdaConfig.signalCount = NCCL_CTRAN_IBGDA_SIGNAL_COUNT;
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

    // Topology config: MNNVL mode and overrides
    config.topoConfig.mnnvlMode =
        static_cast<comms::pipes::MnnvlMode>(NCCL_MNNVL_ENABLE);
    if (NCCL_MNNVL_UUID != -1) {
      config.topoConfig.mnnvlUuid = NCCL_MNNVL_UUID;
    }
    if (NCCL_MNNVL_CLIQUE_ID != -1) {
      config.topoConfig.mnnvlCliqueId = static_cast<int>(NCCL_MNNVL_CLIQUE_ID);
    }

    comm->multiPeerTransport_ =
        std::make_unique<comms::pipes::MultiPeerTransport>(
            comm->statex_->rank(),
            comm->statex_->nRanks(),
            comm->statex_->cudaDev(),
            bootstrapPtr,
            config);
    comm->multiPeerTransport_->exchange();
    CLOGF(INFO, "Pipes MultiPeerTransport initialized");
  } catch (const std::exception& e) {
    CLOGF(ERR, "Failed to initialize Pipes MultiPeerTransport: {}", e.what());
    return commInternalError;
  }
  return commSuccess;
}

#else

commResult_t ctranInitializePipes(CtranComm* comm) {
  return commSuccess;
}

#endif // defined(ENABLE_PIPES)
