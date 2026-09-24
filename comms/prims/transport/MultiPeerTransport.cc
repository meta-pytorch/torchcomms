// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/MultiPeerTransport.h"

#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
// On AMD, HIPify renames `cuda*` runtime calls to `hip*`; pull in the HIP
// runtime so those symbols resolve. The CUDA driver-API path
// (`CudaDriverLazy.h` + `cuMem*`) is unavailable; the corresponding code
// paths in this file are guarded by `#ifndef __HIP_PLATFORM_AMD__`.
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>

#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/prims/bootstrap/NvlBootstrapAdapter.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/topology/TopologyDiscovery.h"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportInternal.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace comms::prims {

namespace {

#define CUDA_CHECK(cmd)                                                    \
  do {                                                                     \
    cudaError_t err = (cmd);                                               \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(                                            \
          std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + \
          __FILE__ + ":" + std::to_string(__LINE__));                      \
    }                                                                      \
  } while (0)

#define CU_CHECK(cmd)                                                          \
  do {                                                                         \
    CUresult err = (cmd);                                                      \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr = nullptr;                                            \
      pfn_cuGetErrorString(err, &errStr);                                      \
      throw std::runtime_error(                                                \
          std::string("CUDA driver error: ") + (errStr ? errStr : "unknown") + \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));                 \
    }                                                                          \
  } while (0)

comms::fault_tolerance::AbortDevice makeAbortDeviceHandle(
    const std::shared_ptr<comms::fault_tolerance::Abort>& abort,
    int deviceId) {
  if (abort == nullptr || !abort->isEnabled()) {
    return comms::fault_tolerance::AbortDevice{};
  }
  meta::comms::CudaDeviceGuard guard{deviceId};
  return abort->getDeviceHandle();
}

} // namespace

MultiPeerTransport::MultiPeerTransport(
    int myRank,
    int nRanks,
    int deviceId,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultiPeerTransportConfig& config,
    std::optional<TopologyResult> topo,
    std::shared_ptr<comms::fault_tolerance::Abort> abort)
    : myRank_(myRank),
      nRanks_(nRanks),
      deviceId_(deviceId),
      bootstrap_(std::move(bootstrap)),
      abort_(std::move(abort)),
      abortDevice_(makeAbortDeviceHandle(abort_, deviceId_)) {
  if (!topo.has_value()) {
    TopologyDiscovery topoDiscovery;
    topo = topoDiscovery.discover(
        myRank_, nRanks_, deviceId_, *bootstrap_, config.topoConfig);
  }
  initFromTopology(std::move(*topo), config);
}

void MultiPeerTransport::initFromTopology(
    TopologyResult topo,
    const MultiPeerTransportConfig& config) {
  nvlPeerRanks_ = std::move(topo.nvlPeerRanks);
  globalToNvlLocal_ = std::move(topo.globalToNvlLocal);

  // Derive fields from the slim TopologyResult.
  nvlNRanks_ = static_cast<int>(nvlPeerRanks_.size()) + 1;
  nvlLocalRank_ = globalToNvlLocal_.at(myRank_);

  typePerRank_.resize(nRanks_);

  if (config.disableIb) {
    // NVL-only mode: validate all non-self peers are NVL-reachable, then
    // force every non-self rank to P2P_NVL. IBGDA is never constructed.
    COMMS_LOG(
        DBG,
        "MultiPeerTransport: rank {} IBGDA disabled by config, NVL-only mode",
        myRank_);

    for (int r = 0; r < nRanks_; ++r) {
      if (r == myRank_) {
        typePerRank_.at(r) = TransportType::SELF;
      } else if (globalToNvlLocal_.count(r)) {
        typePerRank_.at(r) = TransportType::P2P_NVL;
      } else {
        throw std::runtime_error(
            "MultiPeerTransport: IBGDA disabled but rank " + std::to_string(r) +
            " is not NVL-reachable from rank " + std::to_string(myRank_) +
            ". All ranks must be in the same NVL domain when "
            "IB transport is disabled.");
      }
    }
    // ibPeerRanks_ stays empty; ibgdaTransport_ stays nullptr.
  } else {
    const auto ibTransportType = config.ibMode == IbBackendMode::kIbrc
        ? TransportType::P2P_IBRC
        : TransportType::P2P_IBGDA;
    for (int r = 0; r < nRanks_; ++r) {
      if (r == myRank_) {
        typePerRank_.at(r) = TransportType::SELF;
      } else if (globalToNvlLocal_.count(r)) {
        typePerRank_.at(r) = TransportType::P2P_NVL;
      } else {
        typePerRank_.at(r) = ibTransportType;
      }
    }

    for (int r = 0; r < nRanks_; ++r) {
      if (typePerRank_.at(r) == TransportType::P2P_IBGDA ||
          typePerRank_.at(r) == TransportType::P2P_IBRC) {
        ibPeerRanks_.push_back(r);
      }
    }
  }

  // Log topology summary (init-time, once per communicator).
  {
    int nvlCount = 0;
    int ibgdaCount = 0;
    int ibrcCount = 0;
    for (int r = 0; r < nRanks_; ++r) {
      if (typePerRank_[r] == TransportType::P2P_NVL) {
        ++nvlCount;
      } else if (typePerRank_[r] == TransportType::P2P_IBGDA) {
        ++ibgdaCount;
      } else if (typePerRank_[r] == TransportType::P2P_IBRC) {
        ++ibrcCount;
      }
    }
    COMMS_LOG(
        DBG,
        "MultiPeerTransport: rank {}/{} topology: {} NVL peers, {} IBGDA peers, {} IBRC peers",
        myRank_,
        nRanks_,
        nvlCount,
        ibgdaCount,
        ibrcCount);
  }
  for (int r = 0; r < nRanks_; ++r) {
    COMMS_LOG(
        DBG,
        "MultiPeerTransport: rank {} -> rank {}: {}",
        myRank_,
        r,
        transport_type_name(typePerRank_[r]));
  }

  // Create NVLink sub-transport with NvlBootstrapAdapter
  if (!nvlPeerRanks_.empty()) {
    std::vector<int> localRankToCommRank(nvlNRanks_);
    for (const auto& [globalRank, nvlLocal] : globalToNvlLocal_) {
      localRankToCommRank[nvlLocal] = globalRank;
    }

    nvlBootstrapAdapter_ = std::make_shared<NvlBootstrapAdapter>(
        bootstrap_, std::move(localRankToCommRank));

    nvlTransport_ = std::make_unique<MultiPeerNvlTransport>(
        nvlLocalRank_,
        nvlNRanks_,
        deviceId_,
        nvlBootstrapAdapter_,
        config.nvlConfig);
    COMMS_LOG(
        DBG,
        "MultiPeerTransport: rank {} created NVL sub-transport, nvlNRanks={} nvlLocalRank={}",
        myRank_,
        nvlNRanks_,
        nvlLocalRank_);
  }

  // Create the IB sub-transport — the universal fallback for all non-NVL peers.
  // Exactly one backend is built, selected by config.ibMode (kIbgda default;
  // kIbrc selects the CPU-proxy backend).
  if (!config.disableIb && !ibPeerRanks_.empty()) {
    auto ibConfig = config.ibConfig;
    ibConfig.cudaDevice = deviceId_;
    if (config.ibMode == IbBackendMode::kIbrc) {
      // IBRC's device waits sit on the CPU proxy, so the backend needs the
      // handle itself; IBGDA takes one per call on the wait APIs instead.
      ibrcTransport_ = std::make_unique<MultipeerIbrcTransport>(
          myRank_,
          nRanks_,
          bootstrap_,
          ibConfig,
          abortDevice_,
          abort_ ? std::function<bool()>(
                       [abort = abort_] { return abort->isAborted(); })
                 : nullptr);
      COMMS_LOG(
          DBG,
          "MultiPeerTransport: rank {} created IBRC sub-transport for {} peers",
          myRank_,
          ibPeerRanks_.size());
    } else {
      ibgdaTransport_ = std::make_unique<MultipeerIbgdaTransport>(
          myRank_, nRanks_, bootstrap_, ibConfig);
      COMMS_LOG(
          DBG,
          "MultiPeerTransport: rank {} created IBGDA sub-transport for {} peers",
          myRank_,
          ibPeerRanks_.size());
    }
  }
}

MultiPeerTransport::~MultiPeerTransport() {
  free_device_handle();
  static_cast<void>(detail::releaseTransportForProcessLifetimeIfQuarantined(
      ibgdaTransport_, ibgda_resources_quarantined()));
  // IBRC teardown must run to stop its progress thread. Its cleanup retains the
  // provider resources and backing allocations when rollback was quarantined.
}

std::optional<int> MultiPeerTransport::ibgda_max_groups() const {
  if (!ibgdaTransport_) {
    return std::nullopt;
  }
  return ibgdaTransport_->maxGroups();
}

std::optional<int> MultiPeerTransport::ibgda_pipeline_depth() const {
  if (!ibgdaTransport_) {
    return std::nullopt;
  }
  return ibgdaTransport_->pipelineDepth();
}

std::optional<int> MultiPeerTransport::ib_max_num_channels() const {
  if (ibgdaTransport_) {
    return ibgdaTransport_->maxNumChannels();
  }
  if (ibrcTransport_) {
    return ibrcTransport_->maxNumChannels();
  }
  return std::nullopt;
}

std::optional<int> MultiPeerTransport::nvl_max_num_channels() const {
  if (nvlTransport_) {
    return nvlTransport_->maxNumChannels();
  }
  return std::nullopt;
}

void MultiPeerTransport::setExternalNvlDataBuffers(
    ExternalStagingBuffers externalStagingBuffers) {
  if (nvlTransport_) {
    nvlTransport_->setExternalDataBuffers(std::move(externalStagingBuffers));
  }
}

void MultiPeerTransport::exchange() {
  if (exchangeState_ == ExchangeState::kFailed) {
    throw std::runtime_error("MultiPeerTransport: exchange previously failed");
  }
#ifndef __HIP_PLATFORM_AMD__
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "MultiPeerTransport::exchange: failed to initialize CUDA driver API");
  }
#endif

  if (nvlTransport_) {
    nvlTransport_->exchange();
  }
  if (ibgdaTransport_) {
    ibgdaTransport_->exchange();
  }
  if (ibrcTransport_) {
    ibrcTransport_->exchange();
  }

  build_device_handle(/*allowAllocation=*/true);
  exchangeState_ = ExchangeState::kExchanged;
}

void MultiPeerTransport::prepareExchange() {
  if (exchangeState_ == ExchangeState::kFailed) {
    throw std::runtime_error("MultiPeerTransport: exchange previously failed");
  }
  if (exchangeState_ == ExchangeState::kPrepared ||
      exchangeState_ == ExchangeState::kExchanged) {
    return;
  }

  try {
#ifndef __HIP_PLATFORM_AMD__
    // CUDA driver-API init is required for the cuMem-based fabric / POSIX-FD
    // exchange paths. On AMD only the cudaIpc (hipIpc) path is available, so
    // no driver-API init is needed.
    if (cuda_driver_lazy_init() != 0) {
      throw std::runtime_error(
          "MultiPeerTransport::exchange: failed to initialize CUDA driver API");
    }
#endif

    if (nvlTransport_) {
      nvlTransport_->prepareExchange();
    }
    if (ibgdaTransport_) {
      ibgdaTransport_->prepareExchange();
    }

    transportsHost_.reserve(static_cast<std::size_t>(nRanks_));
    const std::size_t arrayBytes =
        static_cast<std::size_t>(nRanks_) * sizeof(Transport);
    CUDA_CHECK(cudaMalloc(&transportsGpu_, arrayBytes));
  } catch (...) {
    exchangeState_ = ExchangeState::kFailed;
    rollbackPreparedExchange();
    throw;
  }

  exchangeState_ = ExchangeState::kPrepared;
}

void MultiPeerTransport::exchangePrepared() {
  if (exchangeState_ == ExchangeState::kFailed) {
    throw std::runtime_error("MultiPeerTransport: exchange previously failed");
  }
  if (exchangeState_ == ExchangeState::kExchanged) {
    return;
  }
  if (exchangeState_ != ExchangeState::kPrepared) {
    throw std::logic_error(
        "MultiPeerTransport::exchangePrepared called before prepareExchange");
  }

  COMMS_LOG(
      DBG,
      "MultiPeerTransport: rank {} exchange() nvl={} ibgda={} ibrc={}",
      myRank_,
      nvlTransport_ ? "yes" : "no",
      ibgdaTransport_ ? "yes" : "no",
      ibrcTransport_ ? "yes" : "no");

  try {
    if (nvlTransport_) {
      nvlTransport_->exchangePrepared();
    }
    if (ibgdaTransport_) {
      ibgdaTransport_->exchangePrepared();
    }
    if (ibrcTransport_) {
      ibrcTransport_->exchange();
    }

    build_device_handle(/*allowAllocation=*/false);
  } catch (...) {
    exchangeState_ = ExchangeState::kFailed;
    // Keep exchanged CUDA-IPC exports alive until the owner has signaled the
    // communicator failure. Freeing them here can block on peers that have not
    // yet closed their imported mappings, preventing the abort from firing.
    throw;
  }
  exchangeState_ = ExchangeState::kExchanged;
}

TransportType MultiPeerTransport::get_transport_type(int peerRank) const {
  return typePerRank_[peerRank];
}

bool MultiPeerTransport::is_nvl_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_NVL;
}

bool MultiPeerTransport::is_ibgda_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_IBGDA;
}

P2pNvlTransportDevice MultiPeerTransport::get_p2p_nvl_transport_device(
    int globalPeerRank) const {
  if (!nvlTransport_) {
    throw std::runtime_error(
        "get_p2p_nvl_transport_device: NVL transport not available");
  }
  int nvlLocalPeerRank = globalToNvlLocal_.at(globalPeerRank);
  return nvlTransport_->getP2pTransportDevice(nvlLocalPeerRank);
}

P2pIbgdaTransportDevice* MultiPeerTransport::get_p2p_ibgda_transport_device(
    int globalPeerRank) {
  requireIbTransportUsable();
  if (!ibgdaTransport_) {
    throw std::runtime_error(
        "get_p2p_ibgda_transport_device: IBGDA transport not available (nRanks == 1?)");
  }
  return detail::runWithProcessLifetimeQuarantineOnFailure(
      *ibgdaTransport_,
      [globalPeerRank](auto& transport) {
        return transport.getP2pTransportDevice(globalPeerRank);
      },
      [this](std::string_view context) { quarantineIbgdaTransport(context); });
}

Transport* /*nullable*/ MultiPeerTransport::get_nvl_transports_array() const {
  if (!nvlTransport_) {
    return nullptr;
  }
  return nvlTransport_->getDeviceTransports().data();
}

bool MultiPeerTransport::has_multimem_nvl_transport() const {
  // nvlTransport_ is legitimately null in normal builds without an NVL domain
  // (e.g. nRanks == 1), same as get_nvl_transports_array() above; the null
  // check is not masking an invariant.
  return nvlTransport_ && nvlTransport_->hasMultimemNvlTransport();
}

bool MultiPeerTransport::initialize_multimem_nvl_transport() const {
  return nvlTransport_ &&
      nvlTransport_->initializeMultimemNvlTransportIfEligible();
}

MultimemNvlTransportDevice
MultiPeerTransport::get_multimem_nvl_transport_device() const {
  // The getter is local after collective initialization succeeds.
  if (!has_multimem_nvl_transport()) {
    throw std::runtime_error(
        "MultiPeerTransport: multimem NVL transport is not initialized");
  }
  return nvlTransport_->getMultimemNvlTransportDevice();
}

P2pSelfTransportDevice MultiPeerTransport::get_p2p_self_transport_device()
    const {
  return P2pSelfTransportDevice{};
}

MultiPeerDeviceHandle MultiPeerTransport::get_device_handle(
    const std::vector<int>& peers) {
  requireIbTransportUsable();
  if (!deviceHandleBuilt_) {
    throw std::runtime_error(
        "MultiPeerTransport::get_device_handle(peers) called before exchange()");
  }
  if (!peers.empty()) {
    materializePeers(peers);
  }
  return MultiPeerDeviceHandle{
      myRank_,
      nRanks_,
      {transportsGpu_, static_cast<uint32_t>(nRanks_)},
      abortDevice_,
      static_cast<int>(nvlPeerRanks_.size()),
      static_cast<int>(ibPeerRanks_.size()),
  };
}

bool MultiPeerTransport::is_lazy_mode() const {
  return true;
}

void MultiPeerTransport::materializePeers(const std::vector<int>& peers) {
  requireIbTransportUsable();
  auto materializeOn = [&](auto& ibTransport) {
    for (int peer : peers) {
      if (peer >= 0 && peer < nRanks_ && peer != myRank_ &&
          (typePerRank_[peer] == TransportType::P2P_IBGDA ||
           typePerRank_[peer] == TransportType::P2P_IBRC)) {
        ibTransport->queuePeerForMaterialization(peer);
      }
    }
    ibTransport->connectPeers();
  };
  if (ibgdaTransport_) {
    for (int peer : peers) {
      if (peer >= 0 && peer < nRanks_ && peer != myRank_ &&
          typePerRank_[peer] == TransportType::P2P_IBGDA) {
        ibgdaTransport_->queuePeerForMaterialization(peer);
      }
    }
    connectIbgdaPeers();
  } else if (ibrcTransport_) {
    detail::runWithProcessLifetimeQuarantineOnFailure(
        *ibrcTransport_,
        [&](auto&) { materializeOn(ibrcTransport_); },
        [this](std::string_view context) {
          quarantineIbgdaTransport(context);
        });
  }
}

void MultiPeerTransport::connectPeers() {
  requireIbTransportUsable();
  if (ibgdaTransport_) {
    connectIbgdaPeers();
  } else if (ibrcTransport_) {
    detail::runWithProcessLifetimeQuarantineOnFailure(
        *ibrcTransport_,
        [](auto& transport) { transport.connectPeers(); },
        [this](std::string_view context) {
          quarantineIbgdaTransport(context);
        });
  }
}

void MultiPeerTransport::requireIbTransportUsable() const {
  if (ibgda_resources_quarantined()) {
    throw std::runtime_error(
        "MultiPeerTransport: IB transport is poisoned after an unsafe cleanup "
        "failure; retry is not supported");
  }
}

void MultiPeerTransport::connectIbgdaPeers() {
  detail::runWithProcessLifetimeQuarantineOnFailure(
      *ibgdaTransport_,
      [](auto& transport) { transport.connectPeers(); },
      [this](std::string_view context) { quarantineIbgdaTransport(context); });
}

void MultiPeerTransport::quarantineIbgdaTransport(
    std::string_view context) noexcept {
  if (ibgdaTransport_ == nullptr && ibrcTransport_ == nullptr) {
    return;
  }
  if (ibgdaResourcesQuarantined_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  if (abort_ != nullptr) {
    try {
      static_cast<void>(abort_->setAbort(
          comms::fault_tolerance::AbortReason::NETWORK_ERROR, context));
    } catch (const std::exception& ex) {
      COMMS_LOG(
          ERR,
          "MultiPeerTransport: failed to publish IB quarantine abort: {}",
          ex.what());
    } catch (...) {
      COMMS_LOG(
          ERR, "MultiPeerTransport: failed to publish IB quarantine abort");
    }
  }
  COMMS_LOG(
      ERR,
      "MultiPeerTransport: poisoned this communicator and will retain its "
      "IB transport for process lifetime after unsafe cleanup: {}",
      context);
}

IbgdaLocalBuffer MultiPeerTransport::localRegisterIbgdaBuffer(
    void* ptr,
    size_t size) {
  requireIbTransportUsable();
  if (ibgdaTransport_) {
    return detail::runWithProcessLifetimeQuarantineOnFailure(
        *ibgdaTransport_,
        [ptr, size](auto& transport) {
          return transport.registerBuffer(ptr, size);
        },
        [this](std::string_view context) {
          quarantineIbgdaTransport(context);
        });
  }
  if (ibrcTransport_) {
    return detail::runWithProcessLifetimeQuarantineOnFailure(
        *ibrcTransport_,
        [ptr, size](auto& transport) {
          return transport.registerBuffer(ptr, size);
        },
        [this](std::string_view context) {
          quarantineIbgdaTransport(context);
        });
  }
  throw std::runtime_error(
      "localRegisterIbgdaBuffer: IB transport not available");
}

IbBufferRegistration MultiPeerTransport::registerIbBufferRange(
    void* ptr,
    std::size_t size) {
  requireIbTransportUsable();
  if (ibgdaTransport_) {
    return ibgdaTransport_->registerIbBufferRange(ptr, size);
  }
  if (ibrcTransport_) {
    return ibrcTransport_->registerIbBufferRange(ptr, size);
  }
  throw std::runtime_error("registerIbBufferRange: IB transport not available");
}

void MultiPeerTransport::deregisterIbBufferRange(
    IbBufferRegistration& registration) {
  if (ibgda_resources_quarantined()) {
    if (ibgdaTransport_) {
      ibgdaTransport_->retainIbBufferRangeForProcessLifetime(registration);
    } else if (ibrcTransport_) {
      ibrcTransport_->retainIbBufferRangeForProcessLifetime(registration);
    }
    return;
  }
  if (ibgdaTransport_) {
    ibgdaTransport_->deregisterIbBufferRange(registration);
    return;
  }
  if (ibrcTransport_) {
    ibrcTransport_->deregisterIbBufferRange(registration);
    return;
  }
  throw std::runtime_error(
      "deregisterIbBufferRange: IB transport not available");
}

bool MultiPeerTransport::localDeregisterIbgdaBuffer(void* ptr) noexcept {
  try {
    if (ibgda_resources_quarantined()) {
      return false;
    }
    bool deregistered = false;
    if (ibgdaTransport_) {
      deregistered = ibgdaTransport_->deregisterBuffer(ptr);
    } else if (ibrcTransport_) {
      deregistered = ibrcTransport_->deregisterBuffer(ptr);
    } else {
      return false;
    }
    if (!deregistered) {
      quarantineIbgdaTransport("cached MR deregistration failed");
    }
    return deregistered;
  } catch (const std::exception& ex) {
    quarantineIbgdaTransport("cached MR deregistration threw");
    COMMS_LOG(ERR, "Failed to deregister IB buffer {}: {}", ptr, ex.what());
    return false;
  } catch (...) {
    quarantineIbgdaTransport("cached MR deregistration threw");
    COMMS_LOG(ERR, "Failed to deregister IB buffer {}: unknown exception", ptr);
    return false;
  }
}

std::vector<IbgdaRemoteBuffer> MultiPeerTransport::exchangeIbgdaBuffer(
    const IbgdaLocalBuffer& localBuf) {
  requireIbTransportUsable();
  if (ibgdaTransport_) {
    return ibgdaTransport_->exchangeBuffer(localBuf);
  }
  if (ibrcTransport_) {
    return ibrcTransport_->exchangeBuffer(localBuf);
  }
  throw std::runtime_error("exchangeIbgdaBuffer: IB transport not available");
}

P2pIbrcHostWriter MultiPeerTransport::getHostWriter(
    int peerRank,
    uint32_t queueIndex) const {
  if (ibrcTransport_) {
    return ibrcTransport_->getHostWriter(peerRank, queueIndex);
  }
  throw std::runtime_error(
      "getHostWriter: IBRC transport not available (build with ibMode=kIbrc)");
}

int MultiPeerTransport::ibNumNics() const {
  if (ibrcTransport_) {
    return ibrcTransport_->numNics();
  }
  throw std::runtime_error(
      "ibNumNics: IBRC transport not available (build with ibMode=kIbrc)");
}

std::size_t MultiPeerTransport::hostLaneCapacity(int peerRank) const {
  if (ibrcTransport_) {
    return ibrcTransport_->hostLaneCapacity(peerRank);
  }
  throw std::runtime_error(
      "hostLaneCapacity: IBRC transport not available (build with ibMode=kIbrc)");
}

P2pIbrcHostLanes MultiPeerTransport::getHostLanes(int peerRank, int numLanes)
    const {
  if (ibrcTransport_) {
    return ibrcTransport_->getHostLanes(peerRank, numLanes);
  }
  throw std::runtime_error(
      "getHostLanes: IBRC transport not available (build with ibMode=kIbrc)");
}

IbgdaLocalBuffer MultiPeerTransport::allocateIbCounterBuffer(
    std::size_t size,
    void** hostPtr) {
  requireIbTransportUsable();
  *hostPtr = nullptr;
  if (ibrcTransport_) {
    void* host = nullptr;
    void* device = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
    std::memset(host, 0, size);
    *hostPtr = host;
    return IbgdaLocalBuffer(device, NetworkLKeys{});
  }
  if (ibgdaTransport_) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    CUDA_CHECK(cudaMemset(ptr, 0, size));
    return IbgdaLocalBuffer(ptr, NetworkLKeys{});
  }
  throw std::runtime_error(
      "allocateIbCounterBuffer: IB transport not available");
}

IbgdaLocalBuffer MultiPeerTransport::registerIbCounterBuffer(
    const IbgdaLocalBuffer& buffer,
    std::size_t size) {
  requireIbTransportUsable();
  if (ibgdaTransport_) {
    return detail::runWithProcessLifetimeQuarantineOnFailure(
        *ibgdaTransport_,
        [ptr = buffer.ptr, size](auto& transport) {
          return transport.registerBuffer(ptr, size);
        },
        [this](std::string_view context) {
          quarantineIbgdaTransport(context);
        });
  }
  if (ibrcTransport_) {
    return buffer;
  }
  throw std::runtime_error(
      "registerIbCounterBuffer: IB transport not available");
}

bool MultiPeerTransport::freeIbCounterBuffer(
    IbgdaLocalBuffer& buffer,
    void*& hostPtr) noexcept {
  if (buffer.ptr == nullptr) {
    return true;
  }
  if (ibgda_resources_quarantined()) {
    return false;
  }
  if (buffer.lkey_per_device.size > 0 &&
      !localDeregisterIbgdaBuffer(buffer.ptr)) {
    return false;
  }
  if (hostPtr != nullptr) {
    (void)cudaFreeHost(hostPtr);
    hostPtr = nullptr;
  } else {
    (void)cudaFree(buffer.ptr);
  }
  buffer = IbgdaLocalBuffer{};
  return true;
}

MultiPeerTransport::NvlMemMode MultiPeerTransport::detectNvlMemMode(
    void* ptr) const {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error("detectNvlMemMode: CUDA driver not available");
  }

  CUmemGenericAllocationHandle handle;
  CUresult ret = pfn_cuMemRetainAllocationHandle(&handle, ptr);
  if (ret == CUDA_ERROR_INVALID_VALUE) {
    return NvlMemMode::kCudaIpc;
  }
  if (ret != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    pfn_cuGetErrorString(ret, &errStr);
    throw std::runtime_error(
        std::string("detectNvlMemMode: cuMemRetainAllocationHandle failed: ") +
        (errStr ? errStr : "unknown"));
  }

  CUmemAllocationProp prop = {};
  CUresult propRet = pfn_cuMemGetAllocationPropertiesFromHandle(&prop, handle);
  pfn_cuMemRelease(handle);
  if (propRet != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    pfn_cuGetErrorString(propRet, &errStr);
    throw std::runtime_error(
        std::string(
            "detectNvlMemMode: cuMemGetAllocationPropertiesFromHandle failed: ") +
        (errStr ? errStr : "unknown"));
  }

  if (prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
    return NvlMemMode::kFabric;
  }
  if (prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    return NvlMemMode::kPosixFd;
  }
  throw std::runtime_error(
      "exchangeNvlBuffer: cuMem buffer lacks both CU_MEM_HANDLE_TYPE_FABRIC "
      "and CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR. "
      "Allocate with at least one shareable handle type.");
#else
  return NvlMemMode::kCudaIpc;
#endif
}

std::vector<void*> MultiPeerTransport::exchangeNvlBuffer(
    void* localPtr,
    bool* localHandlePossiblyExposed) {
  if (!nvlBootstrapAdapter_ || nvlNRanks_ <= 1) {
    throw std::runtime_error(
        "exchangeNvlBuffer: NVL transport not available or single rank");
  }

  NvlMemMode mode = detectNvlMemMode(localPtr);

  if (mode == NvlMemMode::kFabric || mode == NvlMemMode::kPosixFd) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
    throw std::runtime_error("exchangeNvlBuffer: VMM path requires CUDA 12.3+");
#else
    if (cuda_driver_lazy_init() != 0) {
      throw std::runtime_error("exchangeNvlBuffer: CUDA driver not available");
    }

    int cudaDev = 0;
    CUdevice cuDev;
    CUDA_CHECK(cudaGetDevice(&cudaDev));
    CU_CHECK(pfn_cuDeviceGet(&cuDev, cudaDev));

    // Retain the external VMM allocation's physical handle via
    // CuMemAllocation::retain -- its destructor releases the retain reference
    // on both the success path and any exception unwind from
    // nvlMemExchangeVmm (allGather / import / map failures), preventing a
    // silent driver-refcount leak. The factory also queries the buffer's
    // real allocated size via cuMemGetAddressRange -- nvlMemExchangeVmm
    // needs a granularity-multiple size for peer-side cuMemAddressReserve.
    auto phys = CuMemAllocation::retain(localPtr);

    auto pm = nvlMemExchangeVmm(
        *nvlBootstrapAdapter_,
        nvlLocalRank_,
        nvlNRanks_,
        cuDev,
        phys->handle(),
        localPtr,
        phys->size(),
        /*preferFabric=*/mode == NvlMemMode::kFabric,
        localHandlePossiblyExposed);

    std::vector<void*> mappedPtrs = pm.peerPtrs;
    nvlExchangeRecords_[localPtr] = NvlExchangeRecord{mode, std::move(pm)};
    return mappedPtrs;
#endif
  }

  auto pm = nvlMemExchangeCudaIpc(
      *nvlBootstrapAdapter_,
      nvlLocalRank_,
      nvlNRanks_,
      localPtr,
      localHandlePossiblyExposed);
  std::vector<void*> mappedPtrs = pm.peerPtrs;
  nvlExchangeRecords_[localPtr] =
      NvlExchangeRecord{NvlMemMode::kCudaIpc, std::move(pm)};
  return mappedPtrs;
}

void MultiPeerTransport::unmapNvlBuffers(const std::vector<void*>& mappedPtrs) {
  // Find the exchange record by the self entry (localPtr).
  void* localPtr = (nvlLocalRank_ >= 0 &&
                    nvlLocalRank_ < static_cast<int>(mappedPtrs.size()))
      ? mappedPtrs[nvlLocalRank_]
      : nullptr;

  auto it =
      localPtr ? nvlExchangeRecords_.find(localPtr) : nvlExchangeRecords_.end();
  if (it == nvlExchangeRecords_.end()) {
    return;
  }

  auto& record = it->second;
  if (record.mode == NvlMemMode::kFabric ||
      record.mode == NvlMemMode::kPosixFd) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
    if (cuda_driver_lazy_init() == 0) {
      // Tear down the peer VAs. Each CuMemMapping co-owns its imported peer
      // CuMemAllocation (via keepAlive), so clearing the mappings runs
      // cuMemUnmap + cuMemAddressFree and then releases the imported physical
      // handle -- the unmap-VA-then-release-handle ordering is preserved
      // without separate handle bookkeeping.
      record.mem.vmmMappings.clear();
    }
#endif
  } else {
    // cudaIpc path: close every non-self peer handle.
    for (int rank = 0; rank < static_cast<int>(record.mem.peerPtrs.size());
         ++rank) {
      if (rank == nvlLocalRank_ || record.mem.peerPtrs[rank] == nullptr) {
        continue;
      }
      cudaError_t err = cudaIpcCloseMemHandle(record.mem.peerPtrs[rank]);
      if (err != cudaSuccess) {
        fprintf(
            stderr,
            "MultiPeerTransport::unmapNvlBuffers: "
            "cudaIpcCloseMemHandle failed for rank %d: %s\n",
            rank,
            cudaGetErrorString(err));
      }
    }
  }

  nvlExchangeRecords_.erase(it);
}

void MultiPeerTransport::build_device_handle(bool allowAllocation) {
  if (!allowAllocation &&
      (transportsGpu_ == nullptr || transportsHost_.capacity() < nRanks_)) {
    throw std::logic_error(
        "MultiPeerTransport::build_device_handle called before prepareExchange");
  }

  transportsHost_.clear();
  if (allowAllocation) {
    transportsHost_.reserve(static_cast<std::size_t>(nRanks_));
  }
  for (int r = 0; r < nRanks_; ++r) {
    switch (typePerRank_[r]) {
      case TransportType::SELF:
        transportsHost_.emplace_back(P2pSelfTransportDevice{});
        break;

      case TransportType::P2P_NVL: {
        int nvlLocal = globalToNvlLocal_.at(r);
        P2pNvlTransportDevice nvlDev =
            nvlTransport_->buildP2pTransportDevice(nvlLocal);
        transportsHost_.emplace_back(nvlDev);
        break;
      }

      case TransportType::P2P_IBGDA: {
        P2pIbgdaTransportDevice* devPtr = ibgdaTransport_
            ? ibgdaTransport_->getP2pTransportDeviceSlot(r)
            : nullptr;
        transportsHost_.emplace_back(devPtr);
        break;
      }

      case TransportType::P2P_IBRC: {
        P2pIbrcTransportDevice* devPtr = ibrcTransport_
            ? ibrcTransport_->getP2pTransportDeviceSlot(r)
            : nullptr;
        transportsHost_.emplace_back(devPtr);
        break;
      }
    }
  }

  const std::size_t arrayBytes =
      static_cast<std::size_t>(nRanks_) * sizeof(Transport);
  if (allowAllocation && transportsGpu_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&transportsGpu_, arrayBytes));
  }
  CUDA_CHECK(cudaMemcpy(
      transportsGpu_,
      transportsHost_.data(),
      arrayBytes,
      cudaMemcpyHostToDevice));
  transportsHost_.clear();

  deviceHandleBuilt_ = true;
}

void MultiPeerTransport::free_device_handle() {
  if (transportsGpu_) {
    (void)cudaFree(transportsGpu_);
    transportsGpu_ = nullptr;
  }
  transportsHost_.clear();
  deviceHandleBuilt_ = false;
}

void MultiPeerTransport::rollbackPreparedExchange() noexcept {
  free_device_handle();
  std::vector<Transport>().swap(transportsHost_);
  ibrcTransport_.reset();
  ibgdaTransport_.reset();
  nvlTransport_.reset();
  nvlBootstrapAdapter_.reset();
}

} // namespace comms::prims
