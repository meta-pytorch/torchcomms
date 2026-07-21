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

// QP capacity
constexpr int kTotalQps = 1; // 1 for H100; increase for GB200
constexpr int kMaxMsgCntPerQp = 4; // Single-buffer protocol headroom
constexpr int kMaxSge = 1; // One scatter-gather entry per WR

// QP transport — hardcoded defaults for params without NCCL cvars
constexpr uint8_t kRnrRetryCnt = 7; // RNR retries (7 = max, no cvar)
constexpr uint8_t kMinRnrTimer = 12; // RNR NAK timer
constexpr uint8_t kMaxRdAtomic = 1; // Outstanding RDMA read/atomic
constexpr uint8_t kHopLimit = 255; // GRH hop limit
constexpr int kDefaultGidIndex = 3; // RoCEv2 GID index fallback

// Read QP transport config from NCCL cvars. Centralizes all cvar reads so QP
// setup call sites don't scatter IB policy. Ensures cvars are initialized
// first so this can be called from the constructor's member initializer list,
// before the constructor body runs.
torch::comms::QpTransportConfig makeQpTransportConfig() {
  initEnvironment();
  return torch::comms::QpTransportConfig{
      .timeout = static_cast<uint8_t>(NCCL_IB_TIMEOUT),
      .retryCnt = static_cast<uint8_t>(NCCL_IB_RETRY_CNT),
      .gidIndex = static_cast<uint8_t>(
          NCCL_IB_GID_INDEX >= 0 ? NCCL_IB_GID_INDEX : kDefaultGidIndex),
      .trafficClass = static_cast<uint8_t>(NCCL_IB_TC),
      .sl = static_cast<uint8_t>(NCCL_IB_SL),
      .pkeyIndex = static_cast<uint16_t>(NCCL_IB_PKEY),
  };
}

// Value the client writes to the server's readyToSendFlag_ via RDMA_WRITE
// to signal that it has finished copying data out of its staging buffer.
static uint64_t kRecvReadyValue = 1;

// Serialize local connection info (business card + GID + port + MTU + staging)
// to JSON. Staging info allows the peer to populate peerStaging_ during
// connectQp() for subsequent one-sided RDMA operations.
std::string serializeConnectionInfo(
    uint32_t qpNum,
    uint64_t subnetPrefix,
    uint64_t interfaceId,
    uint8_t port,
    ibv_mtu mtu,
    const torch::comms::StagingRendezvousInfo& staging) {
  folly::dynamic obj = folly::dynamic::object;
  obj["qpNum"] = static_cast<int64_t>(qpNum);
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
  uint32_t qpNum{};
  uint64_t subnetPrefix{};
  uint64_t interfaceId{};
  uint8_t port{};
  ibv_mtu mtu{};
  torch::comms::StagingRendezvousInfo staging;
};

ConnectionInfo deserializeConnectionInfo(const std::string& json) {
  auto obj = folly::parseJson(json);

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
      .qpNum = static_cast<uint32_t>(obj["qpNum"].asInt()),
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
      (void)cudaFree(buf_);
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
      (void)cudaFree(buf_);
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
      (void)cudaFree(buf_);
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
        (void)cudaFree(buf_);
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
    : cudaDev_(cudaDev),
      config_(config),
      evb_(evb),
      qpConfig_(makeQpTransportConfig()) {}

StagedRdmaTransportBase::~StagedRdmaTransportBase() {
  if (stream_) {
    // Sync to ensure pending cudaMemcpyAsync completes before staging
    // buffer is freed. Don't destroy — stream is shared (process lifetime).
    (void)cudaStreamSynchronize(stream_);
  }
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

  // 3. Create CQ
  int cqe = 2 * kTotalQps * kMaxMsgCntPerQp;
  auto maybeCq = device_->createCq(cqe, nullptr, nullptr, 0);
  if (!maybeCq) {
    throw std::runtime_error("Failed to create CQ: " + maybeCq.error().errStr);
  }
  cq_.emplace(std::move(*maybeCq));

  // 4. Create RC QP wired to the CQ
  ibv_qp_init_attr initAttr = {};
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.sq_sig_all = 0;
  initAttr.send_cq = cq_->cq();
  initAttr.recv_cq = cq_->cq();
  initAttr.cap.max_send_wr = kMaxMsgCntPerQp;
  initAttr.cap.max_recv_wr = kMaxMsgCntPerQp;
  initAttr.cap.max_send_sge = kMaxSge;
  initAttr.cap.max_recv_sge = kMaxSge;

  auto maybeQp = pd_->createQp(&initAttr);
  if (!maybeQp) {
    throw std::runtime_error("Failed to create QP: " + maybeQp.error().errStr);
  }
  qp_.emplace(std::move(*maybeQp));

  // 5. Transition QP to INIT
  ibv_qp_attr initQpAttr = {};
  initQpAttr.qp_state = IBV_QPS_INIT;
  initQpAttr.qp_access_flags = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  initQpAttr.pkey_index = qpConfig_.pkeyIndex;
  initQpAttr.port_num = kPortNum;

  auto initResult = qp_->modifyQp(
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
    CUDA_CHECK(cudaSetDevice(cudaDev_));
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
  rtrAttr.dest_qp_num = peer.qpNum;
  rtrAttr.rq_psn = 0;
  rtrAttr.max_dest_rd_atomic = kMaxRdAtomic;
  rtrAttr.min_rnr_timer = kMinRnrTimer;
  rtrAttr.ah_attr.is_global = 1;
  rtrAttr.ah_attr.grh.dgid.global.subnet_prefix = peer.subnetPrefix;
  rtrAttr.ah_attr.grh.dgid.global.interface_id = peer.interfaceId;
  rtrAttr.ah_attr.grh.flow_label = 0;
  rtrAttr.ah_attr.grh.sgid_index = qpConfig_.gidIndex;
  rtrAttr.ah_attr.grh.hop_limit = kHopLimit;
  rtrAttr.ah_attr.grh.traffic_class = qpConfig_.trafficClass;
  rtrAttr.ah_attr.sl = qpConfig_.sl;
  rtrAttr.ah_attr.src_path_bits = 0;
  rtrAttr.ah_attr.port_num = peer.port;

  auto rtrResult = qp_->modifyQp(
      &rtrAttr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (!rtrResult) {
    throw std::runtime_error(
        "Failed to transition QP to RTR: " + rtrResult.error().errStr);
  }

  // Transition QP: RTR → RTS
  ibv_qp_attr rtsAttr = {};
  rtsAttr.qp_state = IBV_QPS_RTS;
  rtsAttr.timeout = qpConfig_.timeout;
  rtsAttr.retry_cnt = qpConfig_.retryCnt;
  rtsAttr.rnr_retry = kRnrRetryCnt;
  rtsAttr.sq_psn = 0;
  rtsAttr.max_rd_atomic = kMaxRdAtomic;

  auto rtsResult = qp_->modifyQp(
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
  uint32_t qpNum = qp_->getQpNum();

  auto maybeGid = device_->queryGid(kPortNum, qpConfig_.gidIndex);
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
      qpNum,
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
            ibv_sge sge = {};
            sge.addr = reinterpret_cast<uint64_t>(stagingBuf_->data());
            sge.length = static_cast<uint32_t>(chunkSize);
            sge.lkey = stagingBuf_->lkey();

            ibv_send_wr sendWr = {};
            sendWr.wr_id = chunk;
            sendWr.sg_list = &sge;
            sendWr.num_sge = 1;
            sendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            sendWr.send_flags = IBV_SEND_SIGNALED;
            sendWr.imm_data = static_cast<uint32_t>(chunk);
            sendWr.wr.rdma.remote_addr = peerStaging_.stagingBuf.addr;
            sendWr.wr.rdma.rkey = peerStaging_.stagingBuf.rkey;

            auto postResult = qp_->postSend(&sendWr, nullptr);
            if (!postResult) {
              p.setValue(commInternalError);
              return;
            }

            // Drain send completion to free QP slot
            while (true) {
              auto maybeWcs = cq_->pollCq(kMaxMsgCntPerQp);
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

  // Pre-post a recv WR to absorb the server's first RDMA_WRITE_WITH_IMM.
  ibv_recv_wr recvWr = {};
  recvWr.wr_id = 0;
  recvWr.sg_list = nullptr;
  recvWr.num_sge = 0;

  auto postResult = qp_->postRecv(&recvWr, nullptr);
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
          auto deadline =
              std::chrono::steady_clock::now() + config_.chunkTimeout;

          // SGCursor for scatter/gather — tracks position across entries
          size_t sgEntryIdx = 0;
          size_t sgEntryOffset = 0;

          for (size_t chunk = 0; chunk < numChunks; chunk++) {
            // 1. Poll CQ for RECV_RDMA_WITH_IMM (data arrived)
            bool readyToRecv = false;
            while (!readyToRecv) {
              auto maybeWcs = cq_->pollCq(kMaxMsgCntPerQp);
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
              ibv_recv_wr recvWr = {};
              recvWr.wr_id = chunk + 1;
              recvWr.sg_list = nullptr;
              recvWr.num_sge = 0;

              auto postResult = qp_->postRecv(&recvWr, nullptr);
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
              ibv_sge sge = {};
              sge.addr = reinterpret_cast<uint64_t>(
                  const_cast<uint64_t*>(&kRecvReadyValue));
              sge.length = sizeof(uint64_t);
              sge.lkey = recvReadyClientMr_->mr()->lkey;

              ibv_send_wr flagWr = {};
              flagWr.wr_id = numChunks + chunk;
              flagWr.sg_list = &sge;
              flagWr.num_sge = 1;
              flagWr.opcode = IBV_WR_RDMA_WRITE;
              flagWr.send_flags = IBV_SEND_SIGNALED;
              flagWr.wr.rdma.remote_addr = peerStaging_.recvReady->addr;
              flagWr.wr.rdma.rkey = peerStaging_.recvReady->rkey;

              auto postResult = qp_->postSend(&flagWr, nullptr);
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
