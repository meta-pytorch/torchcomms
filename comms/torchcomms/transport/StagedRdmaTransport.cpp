// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "StagedRdmaTransport.h"

#include <unistd.h>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>

#include <comms/utils/CudaRAII.h>

#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/synchronization/CallOnce.h>

#include <climits>

#include <comms/ctran/ibverbx/IbvPd.h>
#include <comms/ctran/ibverbx/Ibverbx.h>
#include <comms/ctran/utils/CudaWrap.h>
#include <comms/utils/cvars/nccl_cvars.h>

#include <fmt/core.h>
#include <folly/logging/xlog.h>

// ibverbx wraps all libibverbs types in its own namespace
using namespace ibverbx; // NOLINT(google-build-using-namespace)

namespace {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
folly::once_flag initOnceFlag;

void initEnvironment() {
  folly::call_once(initOnceFlag, [] { ncclCvarInit(); });
}

#define CUDA_CHECK(cmd)                     \
  do {                                      \
    auto err = (cmd);                       \
    if (err != cudaSuccess) {               \
      throw std::runtime_error(             \
          fmt::format(                      \
              "CUDA error {} at {}:{}: {}", \
              static_cast<int>(err),        \
              __FILE__,                     \
              __LINE__,                     \
              cudaGetErrorString(err)));    \
    }                                       \
  } while (0)

// ---------------------------------------------------------------------------
// QP configuration constants — matches production values in IbvQpUtils.cc
// and nccl_cvars.yaml defaults.
// ---------------------------------------------------------------------------

// RDMA port and addressing
constexpr uint8_t kPortNum = 1; // Standard IB port number
constexpr int kGidIndex = 3; // RoCEv2 GID index (Ethernet)

// QP capacity
constexpr int kTotalQps = 1; // 1 for H100; increase for GB200
constexpr int kMaxMsgCntPerQp = 4; // Single-buffer protocol headroom
constexpr int kMaxSge = 1; // One scatter-gather entry per WR

// QP transport — aligned with NCCL_IB_* cvar defaults
constexpr uint8_t kTimeout = 20; // ACK timeout (~4.2 s)
constexpr uint8_t kRetryCnt = 7; // Transport retries (7 = infinite)
constexpr uint8_t kRnrRetryCnt = 7; // RNR retries (7 = infinite)
constexpr uint8_t kMinRnrTimer = 12; // RNR NAK timer
constexpr uint8_t kMaxRdAtomic = 1; // Outstanding RDMA read/atomic
constexpr uint8_t kHopLimit = 255; // GRH hop limit

// Value the client writes to the server's readyToSendFlag_ via RDMA_WRITE
// to signal that it has finished copying data out of its staging buffer.
static uint64_t kRecvReadyValue = 1;

// Serialize local connection info (business card + GID + port + MTU + staging)
// to JSON. Staging info allows the peer to populate peerStaging_ during
// connectQp() for subsequent one-sided RDMA operations.
std::string serializeConnectionInfo(
    const ibverbx::IbvVirtualQpBusinessCard& busCard,
    uint64_t subnetPrefix,
    uint64_t interfaceId,
    uint8_t port,
    ibv_mtu mtu,
    const torch::comms::StagingRendezvousInfo& staging) {
  folly::dynamic obj = folly::dynamic::object;
  obj["busCard"] = busCard.serialize();
  obj["subnetPrefix"] = static_cast<int64_t>(subnetPrefix);
  obj["interfaceId"] = static_cast<int64_t>(interfaceId);
  obj["port"] = port;
  obj["mtu"] = static_cast<int>(mtu);

  // Staging buffer info
  folly::dynamic stagingObj = folly::dynamic::object;
  stagingObj["addr"] = static_cast<int64_t>(staging.stagingBuf.addr);
  stagingObj["rkey"] = static_cast<int64_t>(staging.stagingBuf.rkey);
  stagingObj["size"] = static_cast<int64_t>(staging.stagingBuf.size);
  obj["stagingBuf"] = std::move(stagingObj);

  // recvReady info (only present on server side)
  if (staging.recvReady) {
    folly::dynamic flagObj = folly::dynamic::object;
    flagObj["addr"] = static_cast<int64_t>(staging.recvReady->addr);
    flagObj["rkey"] = static_cast<int64_t>(staging.recvReady->rkey);
    flagObj["size"] = static_cast<int64_t>(staging.recvReady->size);
    obj["recvReady"] = std::move(flagObj);
  }

  return folly::toJson(obj);
}

struct ConnectionInfo {
  ibverbx::IbvVirtualQpBusinessCard busCard;
  uint64_t subnetPrefix;
  uint64_t interfaceId;
  uint8_t port;
  ibv_mtu mtu;
  torch::comms::StagingRendezvousInfo staging;
};

ConnectionInfo deserializeConnectionInfo(const std::string& json) {
  auto obj = folly::parseJson(json);
  auto busCard =
      ibverbx::IbvVirtualQpBusinessCard::deserialize(obj["busCard"].asString());
  if (!busCard) {
    throw std::runtime_error(
        "Failed to deserialize IbvVirtualQpBusinessCard: " +
        busCard.error().errStr);
  }

  torch::comms::StagingRendezvousInfo staging;
  if (obj.count("stagingBuf")) {
    auto& sb = obj["stagingBuf"];
    staging.stagingBuf.addr = static_cast<uintptr_t>(sb["addr"].asInt());
    staging.stagingBuf.rkey = static_cast<uint32_t>(sb["rkey"].asInt());
    staging.stagingBuf.size = static_cast<size_t>(sb["size"].asInt());
  }
  if (obj.count("recvReady")) {
    auto& rr = obj["recvReady"];
    torch::comms::StagingRendezvousInfo::BufferInfo readyInfo;
    readyInfo.addr = static_cast<uintptr_t>(rr["addr"].asInt());
    readyInfo.rkey = static_cast<uint32_t>(rr["rkey"].asInt());
    readyInfo.size = static_cast<size_t>(rr["size"].asInt());
    staging.recvReady = readyInfo;
  }

  return ConnectionInfo{
      .busCard = std::move(*busCard),
      .subnetPrefix = static_cast<uint64_t>(obj["subnetPrefix"].asInt()),
      .interfaceId = static_cast<uint64_t>(obj["interfaceId"].asInt()),
      .port = static_cast<uint8_t>(obj["port"].asInt()),
      .mtu = static_cast<ibv_mtu>(obj["mtu"].asInt()),
      .staging = std::move(staging),
  };
}

} // namespace

namespace torch::comms {

// --- StagedBuffer ---

StagedBuffer::StagedBuffer(
    size_t size,
    int cudaDev,
    ibverbx::IbvPd& pd,
    StagedTransferConfig::StagingMode mode)
    : size_(size),
      cudaDev_(cudaDev),
      isGpu_(mode == StagedTransferConfig::StagingMode::GPU) {
  if (isGpu_) {
    CUDA_CHECK(cudaSetDevice(cudaDev));
    CUDA_CHECK(cudaMalloc(&buf_, size));

    dmabufFd_ = ctran::utils::getCuMemDmaBufFd(buf_, size);
    if (dmabufFd_ < 0) {
      cudaFree(buf_);
      throw std::runtime_error("Failed to get dmabuf fd for GPU buffer");
    }

    auto maybeMr = pd.regDmabufMr(
        /*offset=*/0,
        size,
        reinterpret_cast<uintptr_t>(buf_),
        dmabufFd_,
        static_cast<ibv_access_flags>(
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ));
    if (!maybeMr) {
      close(dmabufFd_);
      cudaFree(buf_);
      throw std::runtime_error(
          "Failed to register dmabuf MR: " + maybeMr.error().errStr);
    }
    mr_.emplace(std::move(*maybeMr));
  } else {
    int ret = posix_memalign(&buf_, 4096, size);
    if (ret != 0) {
      throw std::runtime_error(
          fmt::format("posix_memalign failed: {}", strerror(ret)));
    }

    auto maybeMr = pd.regMr(
        buf_,
        size,
        static_cast<ibv_access_flags>(
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ));
    if (!maybeMr) {
      free(buf_);
      throw std::runtime_error(
          "Failed to register CPU staging MR: " + maybeMr.error().errStr);
    }
    mr_.emplace(std::move(*maybeMr));
  }
}

StagedBuffer::~StagedBuffer() {
  mr_.reset();
  if (dmabufFd_ >= 0) {
    close(dmabufFd_);
  }
  if (buf_) {
    if (isGpu_) {
      cudaFree(buf_);
    } else {
      free(buf_);
    }
  }
}

StagedBuffer::StagedBuffer(StagedBuffer&& other) noexcept
    : buf_(other.buf_),
      size_(other.size_),
      cudaDev_(other.cudaDev_),
      isGpu_(other.isGpu_),
      dmabufFd_(other.dmabufFd_),
      mr_(std::move(other.mr_)) {
  other.buf_ = nullptr;
  other.dmabufFd_ = -1;
}

StagedBuffer& StagedBuffer::operator=(StagedBuffer&& other) noexcept {
  if (this != &other) {
    mr_.reset();
    if (dmabufFd_ >= 0) {
      close(dmabufFd_);
    }
    if (buf_) {
      if (isGpu_) {
        cudaFree(buf_);
      } else {
        free(buf_);
      }
    }

    buf_ = other.buf_;
    size_ = other.size_;
    cudaDev_ = other.cudaDev_;
    isGpu_ = other.isGpu_;
    dmabufFd_ = other.dmabufFd_;
    mr_ = std::move(other.mr_);

    other.buf_ = nullptr;
    other.dmabufFd_ = -1;
  }
  return *this;
}

// --- StagedRdmaTransportBase ---

// Process-global CUDA stream pool — one stream per device, lazily created.
// Transports share the stream to avoid per-instance cudaStreamCreate overhead.
static cudaStream_t getSharedStagedStream(int cudaDev) {
  static std::mutex mu;
  static std::unordered_map<int, meta::comms::CudaStream> streams;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = streams.find(cudaDev);
    if (it != streams.end()) {
      return it->second.get();
    }
    CUDA_CHECK(cudaSetDevice(cudaDev));
    auto [inserted, ok] = streams.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(cudaDev),
        std::forward_as_tuple(cudaStreamNonBlocking));
    XLOGF(INFO, "Created shared staged RDMA stream for cudaDev={}", cudaDev);
    return inserted->second.get();
  }
}

StagedRdmaTransportBase::StagedRdmaTransportBase(
    int cudaDev,
    folly::EventBase* evb,
    StagedTransferConfig config)
    : cudaDev_(cudaDev), config_(config), evb_(evb) {
  initEnvironment();
}

StagedRdmaTransportBase::~StagedRdmaTransportBase() {
  if (stream_) {
    // Sync to ensure pending cudaMemcpyAsync completes before staging
    // buffer is freed. Don't destroy — stream is shared (process lifetime).
    cudaStreamSynchronize(stream_);
  }
}

int32_t StagedRdmaTransportBase::getDeviceId() const {
  if (!pd_.has_value()) {
    throw std::runtime_error(
        "getDeviceId() called before setupLocalTransport()");
  }
  return pd_->getDeviceId();
}

void StagedRdmaTransportBase::initIbResources() {
  // Initialize CUDA driver PFN symbols (only needed for GPU staging mode
  // which uses getCuMemDmaBufFd for dmabuf export)
  if (config_.stagingMode == StagedTransferConfig::StagingMode::GPU) {
    auto cudaInitResult = ctran::utils::commCudaLibraryInit();
    if (cudaInitResult != commSuccess) {
      throw std::runtime_error("Failed to initialize CUDA library for PFN");
    }
  }

  // Initialize ibverbx (loads libibverbs symbols)
  auto ibvInitResult = ibverbx::ibvInit();
  if (!ibvInitResult) {
    throw std::runtime_error(
        "Failed to initialize ibverbx: " + ibvInitResult.error().errStr);
  }

  // 1. Get IB device list and pick the one matching our CUDA device.
  auto maybeDevices =
      ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  if (!maybeDevices) {
    throw std::runtime_error(
        "Failed to get IB device list: " + maybeDevices.error().errStr);
  }
  auto& devices = *maybeDevices;
  size_t devIdx =
      static_cast<size_t>(cudaDev_) * NCCL_CTRAN_IB_DEVICES_PER_RANK;
  if (devIdx >= devices.size()) {
    throw std::runtime_error(
        fmt::format(
            "CUDA device {} maps to IB device index {} "
            "(NCCL_CTRAN_IB_DEVICES_PER_RANK={}), but only {} IB devices available",
            cudaDev_,
            devIdx,
            NCCL_CTRAN_IB_DEVICES_PER_RANK,
            devices.size()));
  }
  device_.emplace(std::move(devices.at(devIdx)));

  // 2. Allocate protection domain
  auto maybePd = device_->allocPd();
  if (!maybePd) {
    throw std::runtime_error(
        "Failed to allocate PD: " + maybePd.error().errStr);
  }
  pd_.emplace(std::move(*maybePd));

  // 3. Create virtual CQ
  int cqe = 2 * kTotalQps * kMaxMsgCntPerQp;
  auto maybeVcq = device_->createVirtualCq(cqe, nullptr, nullptr, 0);
  if (!maybeVcq) {
    throw std::runtime_error(
        "Failed to create virtual CQ: " + maybeVcq.error().errStr);
  }
  vcq_.emplace(std::move(*maybeVcq));

  // 4. Create virtual QP with DQPLB load balancing
  ibv_qp_init_attr initAttr = {};
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.sq_sig_all = 0;
  auto& physCqs = vcq_->getPhysicalCqsRef();
  initAttr.send_cq = physCqs.at(0).cq();
  initAttr.recv_cq = physCqs.at(0).cq();
  initAttr.cap.max_send_wr = kMaxMsgCntPerQp;
  initAttr.cap.max_recv_wr = kMaxMsgCntPerQp;
  initAttr.cap.max_send_sge = kMaxSge;
  initAttr.cap.max_recv_sge = kMaxSge;

  if (config_.stagingBufSize > static_cast<size_t>(INT_MAX)) {
    throw std::runtime_error(
        fmt::format(
            "stagingBufSize {} exceeds INT_MAX for VirtualQP maxMsgSize",
            config_.stagingBufSize));
  }

  auto maybeVqp = pd_->createVirtualQp(
      kTotalQps,
      &initAttr,
      &*vcq_,
      kMaxMsgCntPerQp,
      static_cast<int>(config_.stagingBufSize),
      ibverbx::LoadBalancingScheme::DQPLB);
  if (!maybeVqp) {
    throw std::runtime_error(
        "Failed to create virtual QP: " + maybeVqp.error().errStr);
  }
  vqp_.emplace(std::move(*maybeVqp));

  // 5. Transition QP to INIT
  ibv_qp_attr initQpAttr = {};
  initQpAttr.qp_state = IBV_QPS_INIT;
  initQpAttr.qp_access_flags = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  initQpAttr.pkey_index = 0;
  initQpAttr.port_num = kPortNum;

  auto initResult = vqp_->modifyVirtualQp(
      &initQpAttr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (!initResult) {
    throw std::runtime_error(
        "Failed to transition QP to INIT: " + initResult.error().errStr);
  }

  // 6. Create staging buffer (CPU or GPU depending on config)
  stagingBuf_.emplace(
      config_.stagingBufSize, cudaDev_, *pd_, config_.stagingMode);

  // CUDA stream is created lazily by ensureCudaStream() on first use in
  // send()/recv(). This avoids GPU memory allocation at setup time, which
  // is critical for CPU staging mode where CUDA may not be needed at all.
}

void StagedRdmaTransportBase::ensureCudaStream() {
  if (!stream_) {
    stream_ = getSharedStagedStream(cudaDev_);
  }
}

void StagedRdmaTransportBase::connectQp(const std::string& peerConnInfo) {
  auto peer = deserializeConnectionInfo(peerConnInfo);

  // Store peer's staging info for use in send/recv
  peerStaging_ = std::move(peer.staging);

  // Transition QP: INIT → RTR
  ibv_qp_attr rtrAttr = {};
  rtrAttr.qp_state = IBV_QPS_RTR;
  rtrAttr.path_mtu = peer.mtu;
  rtrAttr.dest_qp_num = 0; // overridden per-QP by business card
  rtrAttr.rq_psn = 0;
  rtrAttr.max_dest_rd_atomic = kMaxRdAtomic;
  rtrAttr.min_rnr_timer = kMinRnrTimer;
  rtrAttr.ah_attr.is_global = 1;
  rtrAttr.ah_attr.grh.dgid.global.subnet_prefix = peer.subnetPrefix;
  rtrAttr.ah_attr.grh.dgid.global.interface_id = peer.interfaceId;
  rtrAttr.ah_attr.grh.flow_label = 0;
  rtrAttr.ah_attr.grh.sgid_index = kGidIndex;
  rtrAttr.ah_attr.grh.hop_limit = kHopLimit;
  rtrAttr.ah_attr.grh.traffic_class = 0;
  rtrAttr.ah_attr.sl = 0;
  rtrAttr.ah_attr.src_path_bits = 0;
  rtrAttr.ah_attr.port_num = peer.port;

  auto rtrResult = vqp_->modifyVirtualQp(
      &rtrAttr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER,
      peer.busCard);
  if (!rtrResult) {
    throw std::runtime_error(
        "Failed to transition QP to RTR: " + rtrResult.error().errStr);
  }

  // Transition QP: RTR → RTS
  ibv_qp_attr rtsAttr = {};
  rtsAttr.qp_state = IBV_QPS_RTS;
  rtsAttr.timeout = kTimeout;
  rtsAttr.retry_cnt = kRetryCnt;
  rtsAttr.rnr_retry = kRnrRetryCnt;
  rtsAttr.sq_psn = 0;
  rtsAttr.max_rd_atomic = kMaxRdAtomic;

  auto rtsResult = vqp_->modifyVirtualQp(
      &rtsAttr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (!rtsResult) {
    throw std::runtime_error(
        "Failed to transition QP to RTS: " + rtsResult.error().errStr);
  }
}

std::string StagedRdmaTransportBase::serializeConnInfo(
    const StagingRendezvousInfo& localStaging) {
  auto busCard = vqp_->getVirtualQpBusinessCard();

  auto maybeGid = device_->queryGid(kPortNum, kGidIndex);
  if (!maybeGid) {
    throw std::runtime_error("Failed to query GID: " + maybeGid.error().errStr);
  }
  auto& gid = *maybeGid;

  auto maybeMtu = device_->queryPort(kPortNum);
  if (!maybeMtu) {
    throw std::runtime_error(
        "Failed to query port: " + maybeMtu.error().errStr);
  }

  return serializeConnectionInfo(
      busCard,
      gid.global.subnet_prefix,
      gid.global.interface_id,
      kPortNum,
      maybeMtu->active_mtu,
      localStaging);
}

// --- StagedRdmaServerTransport ---

StagedRdmaServerTransport::~StagedRdmaServerTransport() = default;

std::string StagedRdmaServerTransport::setupLocalTransport() {
  initIbResources();

  // Allocate recvReady flag (CPU-pinned, cache-line aligned).
  // Pre-initialized to kRecvReadyValue so the first send() proceeds
  // immediately without waiting for a client signal.
  readyToSendFlag_.reset(new (std::align_val_t{64})
                             std::atomic<uint64_t>{kRecvReadyValue});

  // Register recvReady flag for RDMA access (client writes to it)
  auto maybeFlagMr = pd_->regMr(
      readyToSendFlag_.get(),
      sizeof(std::atomic<uint64_t>),
      static_cast<ibv_access_flags>(
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
  if (!maybeFlagMr) {
    throw std::runtime_error(
        "Failed to register recvReady flag MR: " + maybeFlagMr.error().errStr);
  }
  recvReadyServerMr_.emplace(std::move(*maybeFlagMr));

  // Build staging info with recvReady for the peer
  StagingRendezvousInfo localStaging;
  localStaging.stagingBuf = {
      .addr = reinterpret_cast<uintptr_t>(stagingBuf_->data()),
      .rkey = stagingBuf_->rkey(),
      .size = stagingBuf_->size(),
  };
  localStaging.recvReady = StagingRendezvousInfo::BufferInfo{
      .addr = reinterpret_cast<uintptr_t>(readyToSendFlag_.get()),
      .rkey = recvReadyServerMr_->mr()->rkey,
      .size = sizeof(uint64_t),
  };

  return serializeConnInfo(localStaging);
}

void StagedRdmaServerTransport::connectRemoteTransport(
    const std::string& peerConnInfo) {
  connectQp(peerConnInfo);
}

folly::SemiFuture<commResult_t> StagedRdmaServerTransport::send(
    const ScatterGatherDescriptor& src) {
  CHECK_THROW(evb_, std::runtime_error);

  size_t totalBytes = src.totalBytes();
  size_t numChunks =
      (totalBytes + config_.stagingBufSize - 1) / config_.stagingBufSize;

  auto [promise, sf] = folly::makePromiseContract<commResult_t>();
  evb_->runInEventBaseThread(
      [this, src, numChunks, totalBytes, p = std::move(promise)]() mutable {
        try {
          ensureCudaStream();
          int32_t deviceId = getDeviceId();
          auto deadline =
              std::chrono::steady_clock::now() + config_.chunkTimeout;

          // SGCursor for scatter/gather — tracks position across entries
          size_t sgEntryIdx = 0;
          size_t sgEntryOffset = 0;

          for (size_t chunk = 0; chunk < numChunks; chunk++) {
            // 1. Wait for client recvReady signal
            while (readyToSendFlag_->load(std::memory_order_acquire) == 0) {
              if (std::chrono::steady_clock::now() >= deadline) {
                p.setValue(commTimeout);
                return;
              }
            }
            readyToSendFlag_->store(0, std::memory_order_release);

            // 2. D2D copy src→staging
            size_t offset = chunk * config_.stagingBufSize;
            size_t chunkSize =
                std::min(config_.stagingBufSize, totalBytes - offset);

            if (src.entries.size() == 1) {
              // Contiguous path
              CUDA_CHECK(cudaMemcpyAsync(
                  stagingBuf_->data(),
                  static_cast<const uint8_t*>(src.entries[0].ptr) + offset,
                  chunkSize,
                  cudaMemcpyDefault,
                  stream_));
            } else {
              // Gather path: copy from non-contiguous GPU regions into staging
              size_t stagingOffset = 0;
              while (stagingOffset < chunkSize) {
                auto& entry = src.entries[sgEntryIdx];
                size_t remainInEntry = entry.size - sgEntryOffset;
                size_t remainInChunk = chunkSize - stagingOffset;
                size_t copySize = std::min(remainInEntry, remainInChunk);
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<uint8_t*>(stagingBuf_->data()) + stagingOffset,
                    static_cast<const uint8_t*>(entry.ptr) + sgEntryOffset,
                    copySize,
                    cudaMemcpyDefault,
                    stream_));
                stagingOffset += copySize;
                sgEntryOffset += copySize;
                if (sgEntryOffset >= entry.size) {
                  sgEntryIdx++;
                  sgEntryOffset = 0;
                }
              }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            // 3. Post RDMA_WRITE_WITH_IMM
            ibverbx::IbvVirtualSendWr sendWr = {};
            sendWr.wrId = chunk;
            sendWr.localAddr = stagingBuf_->data();
            sendWr.length = static_cast<uint32_t>(chunkSize);
            sendWr.remoteAddr = peerStaging_.stagingBuf.addr;
            sendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            sendWr.sendFlags = IBV_SEND_SIGNALED;
            sendWr.immData = static_cast<uint32_t>(chunk);
            sendWr.deviceKeys[deviceId] = ibverbx::MemoryRegionKeys{
                stagingBuf_->lkey(), peerStaging_.stagingBuf.rkey};

            auto postResult = vqp_->postSend(sendWr);
            if (!postResult) {
              p.setValue(commInternalError);
              return;
            }

            // Drain send completion to free QP slot
            while (true) {
              auto maybeWcs = vcq_->pollCq();
              if (!maybeWcs) {
                p.setValue(commInternalError);
                return;
              }
              for (auto& wc : *maybeWcs) {
                if (wc.status != IBV_WC_SUCCESS) {
                  p.setValue(commInternalError);
                  return;
                }
              }
              if (!maybeWcs->empty()) {
                break;
              }
            }

            deadline = std::chrono::steady_clock::now() + config_.chunkTimeout;
          }
          p.setValue(commSuccess);
        } catch (const std::exception& e) {
          XLOGF(ERR, "StagedRdmaServerTransport::send() failed: {}", e.what());
          p.setValue(commInternalError);
        }
      });
  return std::move(sf); // NOLINT(performance-move-const-arg)
}

// --- StagedRdmaClientTransport ---

StagedRdmaClientTransport::~StagedRdmaClientTransport() = default;

void StagedRdmaClientTransport::cancelPendingRecv() {
  recvCancelled_.store(true, std::memory_order_release);
}

std::string StagedRdmaClientTransport::setupLocalTransport() {
  initIbResources();

  // Build staging info without recvReady (only server has it)
  StagingRendezvousInfo localStaging;
  localStaging.stagingBuf = {
      .addr = reinterpret_cast<uintptr_t>(stagingBuf_->data()),
      .rkey = stagingBuf_->rkey(),
      .size = stagingBuf_->size(),
  };

  return serializeConnInfo(localStaging);
}

void StagedRdmaClientTransport::connectRemoteTransport(
    const std::string& peerConnInfo) {
  connectQp(peerConnInfo);

  // Register source MR for &kRecvReadyValue — used to RDMA_WRITE the
  // recvReady acknowledgement back to the server's readyToSendFlag_.
  auto maybeSrcMr = pd_->regMr(
      const_cast<uint64_t*>(&kRecvReadyValue),
      sizeof(uint64_t),
      static_cast<ibv_access_flags>(IBV_ACCESS_LOCAL_WRITE));
  if (!maybeSrcMr) {
    throw std::runtime_error(
        "Failed to register recvReady client MR: " + maybeSrcMr.error().errStr);
  }
  recvReadyClientMr_.emplace(std::move(*maybeSrcMr));

  // Post initial dummy recv to trigger initializeDqplbReceiver on multi-QP
  int32_t deviceId = getDeviceId();
  ibverbx::IbvVirtualRecvWr recvWr = {};
  recvWr.wrId = 0;
  recvWr.localAddr = nullptr;
  recvWr.length = 0;
  recvWr.deviceKeys[deviceId] = ibverbx::MemoryRegionKeys{0, 0};

  auto postResult = vqp_->postRecv(recvWr);
  if (!postResult) {
    throw std::runtime_error(
        "Failed to post initial recv: " + postResult.error().errStr);
  }
}

folly::SemiFuture<commResult_t> StagedRdmaClientTransport::recv(
    const ScatterGatherDescriptor& dst) {
  CHECK_THROW(evb_, std::runtime_error);
  recvCancelled_.store(false, std::memory_order_release);

  size_t totalBytes = dst.totalBytes();
  // Use peer's staging buffer size (server's) to compute chunk count
  size_t numChunks = (totalBytes + peerStaging_.stagingBuf.size - 1) /
      peerStaging_.stagingBuf.size;

  auto [promise, sf] = folly::makePromiseContract<commResult_t>();
  evb_->runInEventBaseThread(
      [this, dst, numChunks, totalBytes, p = std::move(promise)]() mutable {
        try {
          ensureCudaStream();
          int32_t deviceId = getDeviceId();
          auto deadline =
              std::chrono::steady_clock::now() + config_.chunkTimeout;

          // SGCursor for scatter/gather — tracks position across entries
          size_t sgEntryIdx = 0;
          size_t sgEntryOffset = 0;

          for (size_t chunk = 0; chunk < numChunks; chunk++) {
            // 1. Poll CQ for RECV_RDMA_WITH_IMM (data arrived)
            bool readyToRecv = false;
            while (!readyToRecv) {
              auto maybeWcs = vcq_->pollCq();
              if (!maybeWcs) {
                p.setValue(commInternalError);
                return;
              }
              for (auto& wc : *maybeWcs) {
                if (wc.status != IBV_WC_SUCCESS) {
                  p.setValue(commInternalError);
                  return;
                }
                if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                  readyToRecv = true;
                  break;
                }
              }
              if (!readyToRecv &&
                  recvCancelled_.load(std::memory_order_acquire)) {
                p.setValue(commUserAbort);
                return;
              }
              if (!readyToRecv &&
                  std::chrono::steady_clock::now() >= deadline) {
                p.setValue(commTimeout);
                return;
              }
            }

            // 2. Replenish recv WR for next chunk
            {
              ibverbx::IbvVirtualRecvWr recvWr = {};
              recvWr.wrId = chunk + 1;
              recvWr.localAddr = nullptr;
              recvWr.length = 0;
              recvWr.deviceKeys[deviceId] = ibverbx::MemoryRegionKeys{0, 0};

              auto postResult = vqp_->postRecv(recvWr);
              if (!postResult) {
                p.setValue(commInternalError);
                return;
              }
            }

            // 3. D2D copy staging→dst
            size_t offset = chunk * peerStaging_.stagingBuf.size;
            size_t chunkSize =
                std::min(peerStaging_.stagingBuf.size, totalBytes - offset);

            if (dst.entries.size() == 1) {
              // Contiguous path
              CUDA_CHECK(cudaMemcpyAsync(
                  static_cast<uint8_t*>(dst.entries[0].ptr) + offset,
                  stagingBuf_->data(),
                  chunkSize,
                  cudaMemcpyDefault,
                  stream_));
            } else {
              // Scatter path: copy from staging to non-contiguous GPU regions
              size_t stagingOffset = 0;
              while (stagingOffset < chunkSize) {
                auto& entry = dst.entries[sgEntryIdx];
                size_t remainInEntry = entry.size - sgEntryOffset;
                size_t remainInChunk = chunkSize - stagingOffset;
                size_t copySize = std::min(remainInEntry, remainInChunk);
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<uint8_t*>(entry.ptr) + sgEntryOffset,
                    static_cast<const uint8_t*>(stagingBuf_->data()) +
                        stagingOffset,
                    copySize,
                    cudaMemcpyDefault,
                    stream_));
                stagingOffset += copySize;
                sgEntryOffset += copySize;
                if (sgEntryOffset >= entry.size) {
                  sgEntryIdx++;
                  sgEntryOffset = 0;
                }
              }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            // 4. Signal server: staging buffer consumed (always, including last
            // chunk, to ensure next transfer can start safely)
            {
              ibverbx::IbvVirtualSendWr flagWr = {};
              flagWr.wrId = numChunks + chunk;
              flagWr.localAddr = const_cast<uint64_t*>(&kRecvReadyValue);
              flagWr.length = sizeof(uint64_t);
              flagWr.remoteAddr = peerStaging_.recvReady->addr;
              flagWr.opcode = IBV_WR_RDMA_WRITE;
              flagWr.sendFlags = IBV_SEND_SIGNALED;
              flagWr.deviceKeys[deviceId] = ibverbx::MemoryRegionKeys{
                  recvReadyClientMr_->mr()->lkey, peerStaging_.recvReady->rkey};

              auto postResult = vqp_->postSend(flagWr);
              if (!postResult) {
                p.setValue(commInternalError);
                return;
              }
            }

            deadline = std::chrono::steady_clock::now() + config_.chunkTimeout;
          }
          p.setValue(commSuccess);
        } catch (const std::exception& e) {
          XLOGF(ERR, "StagedRdmaClientTransport::recv() failed: {}", e.what());
          p.setValue(commInternalError);
        }
      });
  return std::move(sf); // NOLINT(performance-move-const-arg)
}

} // namespace torch::comms
