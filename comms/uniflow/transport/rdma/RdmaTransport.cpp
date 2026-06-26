// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaTransport.h"
#include "comms/uniflow/core/NumaUtils.h"
#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/rdma/RdmaRegistrationHandle.h"
#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <random>
#include <stdexcept>

namespace uniflow {

// ---------------------------------------------------------------------------
// RdmaTransportInfo serialization
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kSlabIdxInvalid{0xFFFFFFFF};

// RAII guard that releases a DmaBuff's resources on scope exit. Closing is
// only attempted when an fd was actually exported (fd >= 0).
class DmaBuffCloseGuard {
 public:
  DmaBuffCloseGuard(DeviceAdapter& adapter, DmaBuff& buff) noexcept
      : adapter_{adapter}, buff_{buff} {}

  ~DmaBuffCloseGuard() {
    if (buff_.fd >= 0) {
      (void)adapter_.closeDmaBuff(buff_);
    }
  }

  DmaBuffCloseGuard(const DmaBuffCloseGuard&) = delete;
  DmaBuffCloseGuard& operator=(const DmaBuffCloseGuard&) = delete;
  DmaBuffCloseGuard(DmaBuffCloseGuard&&) = delete;
  DmaBuffCloseGuard& operator=(DmaBuffCloseGuard&&) = delete;

 private:
  DeviceAdapter& adapter_;
  DmaBuff& buff_;
};

struct __attribute__((packed)) RdmaTopologyInfo {
  uint8_t version{kRdmaVersion};
  uint32_t numQps{1};

  static Result<RdmaTopologyInfo> deserialize(std::span<const uint8_t> data) {
    if (data.size() != sizeof(RdmaTopologyInfo)) {
      return Err(ErrCode::InvalidArgument, "TransportInfo payload mismatch");
    }

    RdmaTopologyInfo info;
    std::memcpy(&info, data.data(), sizeof(RdmaTopologyInfo));
    return info;
  }
};

const char* getNicName(const NicResources& nic) {
  if (nic.ctx && nic.ctx->device) {
    return nic.ctx->device->name;
  }
  return "(unknown)";
}

uint64_t qpMapKey(size_t nicIdx, uint32_t qpNum) {
  return (static_cast<uint64_t>(nicIdx) << 32) | qpNum;
}

Result<std::vector<RdmaSlab>> allocateSlab(RdmaSlabPool* pool, size_t slabNum) {
  auto slabResult = pool->acquire(static_cast<uint32_t>(slabNum));
  if (slabResult) {
    return slabResult;
  }
  return Err(ErrCode::ResourceExhausted, "no resource to allocate slab");
}

} // namespace

RdmaTransport::SendRecvTransfer::SendRecvTransfer(
    IOType opType,
    Segment::Span data)
    : opType(opType), data(std::move(data)) {}
RdmaTransport::SendRecvTransfer::~SendRecvTransfer() = default;

size_t RdmaTransportInfo::RegisteredBuffer::serialize(uint8_t* data) const {
  size_t offset = 0;
  std::memcpy(data + offset, &addr, sizeof(addr));
  offset += sizeof(addr);
  std::memcpy(data + offset, &length, sizeof(length));
  offset += sizeof(length);
  if (!rkeys.empty()) {
    size_t len = rkeys.size() * sizeof(uint32_t);
    std::memcpy(data + offset, rkeys.data(), len);
    offset += len;
  }
  return offset;
}

size_t RdmaTransportInfo::RegisteredBuffer::deserialize(
    const uint8_t* data,
    uint8_t numNics) {
  size_t offset = 0;
  std::memcpy(&addr, data + offset, sizeof(addr));
  offset += sizeof(addr);
  std::memcpy(&length, data + offset, sizeof(length));
  offset += sizeof(length);
  if (numNics > 0) {
    size_t len = numNics * sizeof(uint32_t);
    rkeys.resize(numNics);
    std::memcpy(rkeys.data(), data + offset, len);
    offset += len;
  }
  return offset;
}

void RdmaTransportInfo::RegisteredBuffer::reset() {
  addr = 0;
  length = 0;
  rkeys.clear();
}

TransportInfo RdmaTransportInfo::serialize() const {
  constexpr size_t headerSize = sizeof(RdmaTransportInfo::Header);
  const size_t totalSize = headerSize + nicInfos.size() * sizeof(NicInfo) +
      qpInfos.size() * sizeof(QpInfo) + ctrl.size() + slab.size();
  TransportInfo data(totalSize);

  size_t offset = 0;
  std::memcpy(data.data() + offset, &header, headerSize);
  offset += headerSize;

  if (!nicInfos.empty()) {
    size_t len = nicInfos.size() * sizeof(NicInfo);
    std::memcpy(data.data() + offset, nicInfos.data(), len);
    offset += len;
  }

  if (!qpInfos.empty()) {
    size_t len = qpInfos.size() * sizeof(QpInfo);
    std::memcpy(data.data() + offset, qpInfos.data(), len);
    offset += len;
  }

  if (header.numNics > 0) {
    offset += ctrl.serialize(data.data() + offset);
    offset += slab.serialize(data.data() + offset);
  }
  return data;
}

Result<RdmaTransportInfo> RdmaTransportInfo::deserialize(
    std::span<const uint8_t> data) {
  constexpr size_t headerSize = sizeof(RdmaTransportInfo::Header);

  if (data.size() < headerSize) {
    return Err(ErrCode::InvalidArgument, "TransportInfo too small for header");
  }

  RdmaTransportInfo info;
  std::memcpy(&info.header, data.data(), headerSize);
  const auto& header = info.header;

  if (header.version != kRdmaVersion) {
    return Err(
        ErrCode::InvalidArgument, "Unsupported RdmaTransportInfo version");
  }

  const size_t expectedSize = headerSize + header.numNics * sizeof(NicInfo) +
      header.numQps * sizeof(QpInfo) +
      RegisteredBuffer::expectedSize(header.numNics) * 2;
  if (data.size() < expectedSize) {
    return Err(
        ErrCode::InvalidArgument,
        fmt::format(
            "TransportInfo too small for payload, expected {} bytes, got {} bytes",
            expectedSize,
            data.size()));
  }

  size_t offset = headerSize;

  if (header.numNics > 0) {
    info.nicInfos.resize(header.numNics);
    std::memcpy(
        info.nicInfos.data(),
        data.data() + offset,
        header.numNics * sizeof(NicInfo));
    offset += header.numNics * sizeof(NicInfo);
  }

  if (header.numQps > 0) {
    info.qpInfos.resize(header.numQps);
    std::memcpy(
        info.qpInfos.data(),
        data.data() + offset,
        header.numQps * sizeof(QpInfo));
    offset += header.numQps * sizeof(QpInfo);
  }

  if (header.numNics > 0) {
    offset += info.ctrl.deserialize(data.data() + offset, header.numNics);
    offset += info.slab.deserialize(data.data() + offset, header.numNics);
  }
  return info;
}

void RdmaTransportInfo::reset() {
  header = {};
  nicInfos.clear();
  qpInfos.clear();
  ctrl.reset();
  slab.reset();
}

// ---------------------------------------------------------------------------
// RdmaTransport
// ---------------------------------------------------------------------------

RdmaTransport::RdmaTransport(
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    EventBase* evb,
    std::shared_ptr<std::vector<NicResources>> nics,
    uint64_t domainId,
    RdmaTransportConfig config,
    std::shared_ptr<RdmaSlabPool> slabPool)
    : ibvApi_(std::move(ibvApi)),
      cudaApi_(std::move(cudaApi)),
      cudaDriverApi_(std::move(cudaDriverApi)),
      evb_(evb),
      nicsHandle_(std::move(nics)),
      config_(config),
      domainId_(domainId),
      slabPool_(std::move(slabPool)) {
  CHECK_THROW_EXCEPTION(
      nicsHandle_ && !nicsHandle_->empty(), std::invalid_argument);
  CHECK_THROW_EXCEPTION(nicsHandle_->size() <= 255, std::invalid_argument);
  CHECK_THROW_EXCEPTION(
      config_.numQps > 0 && config_.numQps <= 255, std::invalid_argument);

  deviceAdapter_ = createDeviceAdapter(cudaApi_);

  nics_ = std::span<NicResources>(nicsHandle_->data(), nicsHandle_->size());

  name_ = "rdma";
  for (const auto& nic : nics_) {
    name_ += "_";
    name_ += nic.ctx->device->name;
  }
}

RdmaTransport::~RdmaTransport() {
  shutdown();
}

TransportInfo RdmaTransport::bind() {
  if (state_ == TransportState::Initialized) {
    return info_.serialize();
  }

  info_.reset();
  const uint32_t numQps = config_.numQps;
  const uint32_t numNics = nics_.size();
  numPendingCqe_.resize(nics_.size());
  slotPendingCqe_.resize(config_.pipelineDepth);

  // Allocate and register ctrl buffer for send/recv pipeline flow control.
  // Layout: [CTS: D entries][Notify: D * Q entries], all atomic<uint32_t>.
  // CTS and Notify are separate regions so bidirectional send/recv is safe.
  const size_t ctrlSize = ctrlBufferSize();
  info_.ctrl.length = static_cast<uint32_t>(ctrlSize);
  auto ctrlAllocResult = deviceAdapter_->pinnedHostAlloc(ctrlSize);
  if (!ctrlAllocResult) {
    UNIFLOW_LOG_ERROR("bind: failed to allocate ctrl buffer");
    state_ = TransportState::Error;
    return {};
  }
  ctrlBuffer_ = ctrlAllocResult.value();
  info_.ctrl.addr = reinterpret_cast<uint64_t>(ctrlBuffer_);
  // Initialize all entries to kSlabIdxInvalid.
  auto* ctrlEntries = static_cast<uint32_t*>(ctrlBuffer_);
  const size_t numEntries = ctrlSize / sizeof(uint32_t);
  for (size_t i = 0; i < numEntries; ++i) {
    std::atomic_ref<uint32_t>(ctrlEntries[i])
        .store(kSlabIdxInvalid, std::memory_order_relaxed);
  }

  cqs_.reserve(numNics);
  uint32_t qpsPerNic = (numQps + numNics - 1) / numNics;
  for (uint8_t n = 0; n < numNics; ++n) {
    auto cqResult = ibvApi_->createCq(
        nics_[n].ctx, config_.maxWr * qpsPerNic, nullptr, nullptr, 0);
    if (cqResult.hasError()) {
      UNIFLOW_LOG_ERROR(
          "bind: failed to create CQ for NIC {}", getNicName(nics_[n]));
      shutdown();
      state_ = TransportState::Error;
      return {};
    }
    cqs_.push_back(cqResult.value());
  }

  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<uint32_t> dist(0, 0x00FFFFFF);

  qps_.reserve(numQps);
  psns_.reserve(numQps);
  numWrsPerQp_.resize(numQps, 0);

  info_.header.version = kRdmaVersion;
  info_.header.numQps = numQps;
  info_.header.numNics = numNics;
  info_.header.domainId = domainId_;
  info_.qpInfos.reserve(numQps);
  info_.nicInfos.reserve(numNics);

  for (uint32_t i = 0; i < numQps; ++i) {
    uint32_t nicIdx = i % numNics;

    ibv_qp_init_attr initAttr{};
    initAttr.send_cq = cqs_[nicIdx];
    initAttr.recv_cq = cqs_[nicIdx];
    initAttr.qp_type = IBV_QPT_RC;
    initAttr.sq_sig_all = 0;
    initAttr.cap.max_send_wr = config_.maxWr;
    initAttr.cap.max_recv_wr = config_.maxWr;
    initAttr.cap.max_send_sge = config_.maxSge;
    initAttr.cap.max_recv_sge = config_.maxSge;
    initAttr.cap.max_inline_data = config_.maxInlineData;

    auto qpResult = ibvApi_->createQp(nics_[nicIdx].pd, &initAttr);
    if (qpResult.hasError()) {
      UNIFLOW_LOG_ERROR(
          "bind: failed to create QP {} on NIC {}",
          i,
          getNicName(nics_[nicIdx]));
      shutdown();
      state_ = TransportState::Error;
      return {};
    }
    ibv_qp* qp = qpResult.value();

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = config_.pkeyIndex;
    attr.port_num = nics_[nicIdx].portNum;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_LOCAL_WRITE;

    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    auto modifyStatus = ibvApi_->modifyQp(qp, &attr, mask);
    if (modifyStatus.hasError()) {
      UNIFLOW_LOG_ERROR(
          "bind: failed to transition QP {} to INIT on NIC {}",
          i,
          getNicName(nics_[nicIdx]));
      ibvApi_->destroyQp(qp);
      shutdown();
      state_ = TransportState::Error;
      return {};
    }

    uint32_t psn = dist(rng);
    qps_.push_back(qp);
    qpNumToIdx_[qpMapKey(nicIdx, qp->qp_num)] = i;
    psns_.push_back(psn);
    info_.qpInfos.push_back({.qpNum = qp->qp_num, .psn = psn});
  }

  for (const auto& nic : nics_) {
    info_.nicInfos.push_back(
        {.lid = nic.lid,
         .linkLayer = static_cast<uint8_t>(nic.linkLayer),
         .mtu = static_cast<uint8_t>(nic.mtu),
         .gid = nic.gid});
    UNIFLOW_LOG_INFO("Bind NIC {} successfully", getNicName(nic));
  }

  constexpr int kCtrlAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  for (const auto& nic : nics_) {
    auto mrResult = ibvApi_->regMr(nic.pd, ctrlBuffer_, ctrlSize, kCtrlAccess);
    if (!mrResult) {
      UNIFLOW_LOG_ERROR(
          "RdmaTransport::bind: ctrl buffer ibv_reg_mr failed: {}",
          mrResult.error().toString());
      shutdown();
      state_ = TransportState::Error;
      return {};
    }
    auto* mr = mrResult.value();
    ctrlMrs_.push_back(mr);
    info_.ctrl.rkeys.push_back(mr->rkey);
  }

  if (slabPool_) {
    info_.header.slabSize = static_cast<uint32_t>(slabPool_->slabSize());
    info_.slab.addr = slabPool_->slabAddr(0);
    info_.slab.length =
        info_.header.slabSize * static_cast<uint32_t>(slabPool_->numSlabs());
    info_.slab.rkeys.resize(numNics);
    for (uint8_t n = 0; n < numNics; ++n) {
      info_.slab.rkeys[n] = slabPool_->slabRkey(n);
    }
  } else {
    info_.header.slabSize = 0;
    info_.slab.addr = 0;
    info_.slab.length = 0;
    info_.slab.rkeys.resize(numNics, 0);
  }

  state_ = TransportState::Initialized;
  return info_.serialize();
}

Status RdmaTransport::connect(std::span<const uint8_t> remoteInfo) {
  if (qps_.size() != config_.numQps) {
    return Err(ErrCode::NotConnected, "bind() must be called before connect()");
  }

  auto result = RdmaTransportInfo::deserialize(remoteInfo);
  if (result.hasError()) {
    UNIFLOW_LOG_ERROR(
        "connect: failed to deserialize remote info: {}",
        result.error().message());
    state_ = TransportState::Error;
    return std::move(result).error();
  }

  auto& remote = result.value();
  remoteDomainId_ = remote.header.domainId;
  remoteNumNics_ = remote.header.numNics;

  if (remote.header.numQps != config_.numQps) {
    UNIFLOW_LOG_ERROR(
        "connect: QP count mismatch (local={}, remote={})",
        config_.numQps,
        remote.header.numQps);
    state_ = TransportState::Error;
    return Err(ErrCode::InvalidArgument, "QP count mismatch");
  }
  if (remote.header.slabSize != info_.header.slabSize) {
    UNIFLOW_LOG_WARN(
        "connect: slab size mismatch (local={}, remote={})",
        info_.header.slabSize,
        remote.header.slabSize);
    remote.header.slabSize = info_.header.slabSize =
        std::min(remote.header.slabSize, info_.header.slabSize);
  }

  const uint32_t numNics = static_cast<uint32_t>(nics_.size());
  const uint32_t remoteNumNics = static_cast<uint32_t>(remote.nicInfos.size());

  for (uint32_t i = 0; i < config_.numQps; ++i) {
    uint32_t localNicIdx = i % numNics;
    uint32_t remoteNicIdx = i % remoteNumNics;
    const auto& remoteNic = remote.nicInfos[remoteNicIdx];
    const auto& localNic = nics_[localNicIdx];

    ibv_mtu negotiatedMtu = static_cast<ibv_mtu>(
        std::min(static_cast<uint8_t>(localNic.mtu), remoteNic.mtu));

    ibv_qp_attr rtrAttr{};
    rtrAttr.qp_state = IBV_QPS_RTR;
    rtrAttr.path_mtu = negotiatedMtu;
    rtrAttr.dest_qp_num = remote.qpInfos[i].qpNum;
    rtrAttr.rq_psn = remote.qpInfos[i].psn;
    rtrAttr.max_dest_rd_atomic = config_.maxRdAtomic;
    rtrAttr.min_rnr_timer = 12;

    if (remoteNic.linkLayer == IBV_LINK_LAYER_ETHERNET) {
      rtrAttr.ah_attr.is_global = 1;
      rtrAttr.ah_attr.grh.dgid = remoteNic.gid;
      rtrAttr.ah_attr.grh.sgid_index = config_.gidIndex;
      rtrAttr.ah_attr.grh.hop_limit = 255;
      rtrAttr.ah_attr.grh.flow_label = 0;
      rtrAttr.ah_attr.grh.traffic_class = config_.trafficClass;
    } else {
      rtrAttr.ah_attr.is_global = 0;
      rtrAttr.ah_attr.dlid = remoteNic.lid;
    }
    rtrAttr.ah_attr.sl = 0;
    rtrAttr.ah_attr.src_path_bits = 0;
    rtrAttr.ah_attr.port_num = localNic.portNum;

    int rtrMask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    auto rtrStatus = ibvApi_->modifyQp(qps_[i], &rtrAttr, rtrMask);
    if (rtrStatus.hasError()) {
      UNIFLOW_LOG_ERROR("connect: failed to transition QP {} to RTR", i);
      state_ = TransportState::Error;
      return Err(ErrCode::ConnectionFailed, "Failed to transition QP to RTR");
    }

    ibv_qp_attr rtsAttr{};
    rtsAttr.qp_state = IBV_QPS_RTS;
    rtsAttr.sq_psn = psns_[i];
    rtsAttr.timeout = config_.timeout;
    rtsAttr.retry_cnt = config_.retryCnt;
    rtsAttr.rnr_retry = 7;
    rtsAttr.max_rd_atomic = config_.maxRdAtomic;

    int rtsMask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

    auto rtsStatus = ibvApi_->modifyQp(qps_[i], &rtsAttr, rtsMask);
    if (rtsStatus.hasError()) {
      UNIFLOW_LOG_ERROR("connect: failed to transition QP {} to RTS", i);
      state_ = TransportState::Error;
      return Err(ErrCode::ConnectionFailed, "Failed to transition QP to RTS");
    }
  }

  remoteCtrlBuffer_ = std::move(remote.ctrl);
  remoteSlabBuffer_ = std::move(remote.slab);

  state_ = TransportState::Connected;
  UNIFLOW_LOG_INFO(
      "connect: {} QPs connected (localDomain={:#x}, remoteDomain={:#x})",
      config_.numQps,
      domainId_,
      remoteDomainId_);
  return Ok();
}

std::future<Status> RdmaTransport::put(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  return rdmaPutGetTransfer(requests, IBV_WR_RDMA_WRITE, options);
}

std::future<Status> RdmaTransport::get(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  return rdmaPutGetTransfer(requests, IBV_WR_RDMA_READ, options);
}

// ---------------------------------------------------------------------------
// Request preprocessing (caller thread)
// ---------------------------------------------------------------------------

Status RdmaTransport::preprocessRequest(
    const TransferRequest& req,
    RdmaRegistrationHandle const** localHandle,
    RdmaRemoteRegistrationHandle const** remoteHandle) const {
  if (req.local.size() != req.remote.size()) {
    return Err(
        ErrCode::InvalidArgument,
        "RDMA transfer: local and remote sizes must match");
  }

  *localHandle = nullptr;
  *remoteHandle = nullptr;

  for (const auto& h : req.local.handles_) {
    if (h->transportType() == TransportType::RDMA) {
      auto* rh = dynamic_cast<const RdmaRegistrationHandle*>(h.get());
      if (rh && rh->domainId() == domainId_) {
        *localHandle = rh;
        break;
      }
    }
  }

  for (const auto& h : req.remote.handles_) {
    auto* rh = dynamic_cast<const RdmaRemoteRegistrationHandle*>(h.get());
    if (rh && rh->domainId() == remoteDomainId_) {
      *remoteHandle = rh;
      break;
    }
  }

  if (*localHandle == nullptr || *remoteHandle == nullptr) {
    return Err(
        ErrCode::InvalidArgument,
        "RDMA transfer: no matching registration handle");
  }

  if ((*localHandle)->numMrs() != nics_.size()) {
    return Err(
        ErrCode::InvalidArgument,
        "RDMA transfer: local handle NIC count mismatch (expected " +
            std::to_string(nics_.size()) + ", got " +
            std::to_string((*localHandle)->numMrs()) + ")");
  }

  if ((*remoteHandle)->numMrs() != remoteNumNics_) {
    return Err(
        ErrCode::InvalidArgument,
        "RDMA transfer: remote handle NIC count mismatch (expected " +
            std::to_string(remoteNumNics_) + ", got " +
            std::to_string((*remoteHandle)->numMrs()) + ")");
  }

  return Ok();
}

Result<std::unique_ptr<std::vector<RdmaTransport::SendWr>>>
RdmaTransport::buildSendWrs(
    std::span<const TransferRequest> requests,
    ibv_wr_opcode opcode) {
  auto wrs = std::make_unique<std::vector<SendWr>>();

  size_t totalChunks = 0;
  const size_t kChunkSize = static_cast<size_t>(config_.chunkSize);
  for (const auto& req : requests) {
    const size_t reqSize = req.local.size();
    if (reqSize > 0) {
      totalChunks += (reqSize + kChunkSize - 1) / kChunkSize;
    }
  }
  wrs->reserve(totalChunks);

  for (const auto& req : requests) {
    const RdmaRegistrationHandle* localHandle = nullptr;
    const RdmaRemoteRegistrationHandle* remoteHandle = nullptr;
    CHECK_EXPR(preprocessRequest(req, &localHandle, &remoteHandle));

    const size_t reqSize = req.local.size();
    if (reqSize == 0) {
      continue;
    }
    // preprocessRequest guarantees both handles are non-null; guard here to
    // document the invariant and satisfy nullable-dereference analysis.
    if (localHandle == nullptr || remoteHandle == nullptr) {
      return Err(
          ErrCode::InvalidArgument, "RDMA transfer: null registration handle");
    }
    const uint64_t localWireBase = localHandle->toWireAddr(req.local.data());
    const uint64_t remoteWireBase = remoteHandle->toWireAddr(req.remote.data());

    // Split into fixed-size chunks. Each chunk becomes one SendWr.
    size_t offset = 0;
    while (offset < reqSize) {
      size_t len = std::min(reqSize - offset, kChunkSize);
      auto& sendWr = wrs->emplace_back();

      sendWr.sge.addr = localWireBase + offset;
      sendWr.sge.length = static_cast<uint32_t>(len);
      sendWr.wr.num_sge = 1;
      sendWr.wr.opcode = opcode;
      sendWr.wr.send_flags = 0; // Signaled only on the last WR per QP.
      sendWr.wr.wr.rdma.remote_addr = remoteWireBase + offset;
      sendWr.localHandle = localHandle;
      sendWr.remoteHandle = remoteHandle;

      offset += len;
    }
  }

  for (auto& wr : *wrs) {
    wr.wr.sg_list = &wr.sge;
  }

  return wrs;
}

// ---------------------------------------------------------------------------
// QP posting (EventBase thread)
// ---------------------------------------------------------------------------

uint32_t RdmaTransport::postSend(
    uint32_t qpIdx,
    ibv_send_wr* head,
    uint32_t count,
    uint32_t taskId,
    Task& task,
    std::optional<uint16_t> slot) {
  ibv_send_wr* badWr = nullptr;
  auto st = ibvApi_->postSend(qps_[qpIdx], head, &badWr);
  size_t nicIdx = qpIdx % nics_.size();
  if (st) {
    // All WRs posted successfully.
    ++numPendingCqe_[nicIdx];
    numWrsPerQp_[qpIdx] += count;
    if (slot) {
      ++slotPendingCqe_[slot.value()];
    }
    task.posted(count);
    return count;
  } else {
    UNIFLOW_LOG_ERROR(
        "postSend: failed on NIC {} QP {} taskId={}: {}",
        getNicName(nics_[nicIdx]),
        qpIdx,
        taskId,
        st.error().message());
  }

  // postSend failed. Count consumed WRs (everything before badWr).
  // RC QPs process WRs in order, so all WRs before badWr were accepted
  // by the HCA but are unsignaled — no CQE will arrive for them.
  uint32_t consumed = 0;
  for (auto* w = head; w != nullptr && w != badWr; w = w->next) {
    ++consumed;
  }

  if (consumed > 0) {
    // Post a zero-length signaled "flush" WR to reclaim the consumed slots.
    //
    // Why this works:
    //   - RC guarantees in-order completion.
    //   - The flush WR will complete after all consumed unsignaled WRs.
    //   - Its CQE carries wr_id with flushCount encoded in the layout that
    //     pollCompletions expects for this task type:
    //       non-copy-op: lower 32 bits = flushCount
    //       copy-op:     lower 32 bits = (flushCount << 16) | slot
    //   - pollCompletions decrements numWrsPerQp_ by flushCount.
    //
    // Counter invariant (numWrsPerQp_):
    //   += flushCount [here]  /  -= flushCount [pollCompletions]
    // Counter invariant (slotPendingCqe_, copy-op only):
    //   += 1 [here]  /  -= 1 [pollCompletions]
    uint32_t flushCount = consumed + 1;
    uint32_t wrInfo =
        slot.has_value() ? (flushCount << 16 | slot.value()) : flushCount;

    ibv_sge flushSge{};
    ibv_send_wr flushWr{};
    flushWr.wr_id = (static_cast<uint64_t>(taskId) << 32) | wrInfo;
    flushWr.next = nullptr;
    flushWr.sg_list = &flushSge;
    flushWr.num_sge = 0;
    flushWr.opcode = IBV_WR_SEND;
    flushWr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

    ibv_send_wr* flushBadWr = nullptr;
    if (ibvApi_->postSend(qps_[qpIdx], &flushWr, &flushBadWr)) {
      UNIFLOW_LOG_WARN(
          "postSend: partial failure on NIC {} QP {} taskId={}, consumed={}, "
          "flush WR posted successfully",
          getNicName(nics_[nicIdx]),
          qpIdx,
          taskId,
          consumed);
      ++numPendingCqe_[nicIdx];
      numWrsPerQp_[qpIdx] += flushCount;
      if (slot) {
        ++slotPendingCqe_[slot.value()];
      }
      task.posted(flushCount);
    } else {
      UNIFLOW_LOG_ERROR(
          "postSend: flush WR also failed on NIC {} QP {} taskId={}, "
          "consumed={} WRs leaked",
          getNicName(nics_[nicIdx]),
          qpIdx,
          taskId,
          consumed);
      state_ = TransportState::Error;
    }
  }

  task.recordCompletion(st.error());
  return 0; // Signal to caller that posting failed.
}

uint32_t RdmaTransport::getQpAvail(
    std::vector<uint32_t>& qpAvail,
    int bufNuma) {
  const uint32_t kMaxWr = config_.maxWr;
  const uint32_t numNics = static_cast<uint32_t>(nics_.size());
  if (numNics == 0) {
    return 0;
  }

  // For host memory with a known NUMA node, only NICs on that node are eligible
  // (so put/get avoids cross-socket DMA); cross-socket QPs are capped to zero
  // capacity. Fall back to all NICs if none are NUMA-local, so a transfer can
  // never stall.
  bool restrictToNuma = false;
  if (bufNuma >= 0) {
    for (size_t q = 0; q < qpAvail.size(); ++q) {
      if (nics_[q % numNics].numaNode == bufNuma) {
        restrictToNuma = true;
        break;
      }
    }
    if (!restrictToNuma) {
      UNIFLOW_LOG_DEBUG(
          "no NUMA-local NIC for bufNuma={}, falling back to all NICs",
          bufNuma);
    }
  }

  uint32_t totalAvail = 0;
  for (size_t q = 0; q < qpAvail.size(); ++q) {
    if (restrictToNuma && nics_[q % numNics].numaNode != bufNuma) {
      qpAvail[q] = 0; // cross-socket QP — not eligible for this buffer
      continue;
    }
    qpAvail[q] = kMaxWr - numWrsPerQp_[q];
    totalAvail += qpAvail[q];
  }
  return totalAvail;
}

uint32_t RdmaTransport::assignToQps(
    const uint32_t remaining,
    const uint32_t totalAvail,
    std::vector<uint32_t>& qpAvail,
    std::vector<uint32_t>& qpAssigned) {
  // Weighted distribution: assign chunks to each QP proportional
  // to its available capacity. Ensures QPs with more room get more work,
  // avoiding head-of-line blocking on a full QP.
  uint32_t totalAssigned = 0;
  for (size_t q = 0; q < qpAvail.size() && totalAssigned < remaining; ++q) {
    if (qpAvail[q] == 0) {
      continue;
    }
    uint32_t share = std::max(1u, qpAvail[q] * remaining / totalAvail);
    share = std::min(share, remaining - totalAssigned);
    share = std::min(share, qpAvail[q]);
    qpAssigned[q] = share;
    totalAssigned += share;
  }

  // Fill any capacity left by integer truncation in the weighted pass.
  const uint32_t targetAssigned = std::min(remaining, totalAvail);
  while (totalAssigned < targetAssigned) {
    bool progressed = false;
    for (size_t q = 0; q < qpAvail.size() && totalAssigned < targetAssigned;
         ++q) {
      if (qpAssigned[q] >= qpAvail[q]) {
        continue;
      }
      ++qpAssigned[q];
      ++totalAssigned;
      progressed = true;
    }
    if (!progressed) {
      break;
    }
  }
  return totalAssigned;
}

Result<uint32_t> RdmaTransport::spray(
    std::vector<SendWr>& wrs,
    size_t& idx,
    uint32_t taskId,
    Task& task) {
  if (idx >= wrs.size()) {
    return 0;
  }
  const uint32_t numQps = static_cast<uint32_t>(qps_.size());

  const uint32_t remaining = static_cast<uint32_t>(wrs.size() - idx);

  // NUMA policy is derived from the first WR's local handle. In practice each
  // spray batch comes from a single TransferRequest (one local buffer), so
  // this is correct. If buildSendWrs ever batches WRs from multiple buffers
  // with different NUMA nodes into one vector, this should be split into
  // per-NUMA-policy runs to avoid posting later WRs to the wrong NICs. But in
  // reality, we will bind the process into one numa node. In addition, even if
  // Wrs in spray cross different numa nodes, it will be gated by depth of the
  // QP and the next iteration will choose the correct numa node.
  const int bufNuma = wrs[idx].localHandle != nullptr
      ? wrs[idx].localHandle->hostBufferNumaNode()
      : -1;
  std::vector<uint32_t> qpAvail(numQps);
  uint32_t totalAvail = getQpAvail(qpAvail, bufNuma);

  if (totalAvail == 0) {
    return 0; // All eligible QPs full — caller should poll and retry.
  }

  // Weighted distribution: assign chunks to each QP proportional
  // to its available capacity. Ensures QPs with more room get more work,
  // avoiding head-of-line blocking on a full QP.
  std::vector<uint32_t> qpAssigned(numQps, 0);
  assignToQps(remaining, totalAvail, qpAvail, qpAssigned);

  // Post assigned chunks to each QP.
  uint32_t totalPosted = 0;
  for (uint32_t q = 0; q < numQps && idx < wrs.size(); ++q) {
    if (qpAssigned[q] == 0) {
      continue;
    }

    // Build a chained WR list for this QP.
    // All WRs are unsignaled except the last, which carries the count
    // in its wr_id lower 32 bits for pollCompletions to decrement.
    ibv_send_wr* head = &wrs[idx].wr;
    ibv_send_wr* prev = nullptr;
    for (uint32_t c = 0; c < qpAssigned[q] && idx < wrs.size(); ++c, ++idx) {
      auto& sendWr = wrs[idx];
      uint32_t mrIdx = q % sendWr.localHandle->numMrs();
      uint32_t remoteMrIdx = q % sendWr.remoteHandle->numMrs();

      sendWr.wr.wr_id = static_cast<uint64_t>(taskId) << 32;
      sendWr.sge.lkey = sendWr.localHandle->lkey(mrIdx);
      sendWr.wr.wr.rdma.rkey = sendWr.remoteHandle->rkey(remoteMrIdx);

      if (prev) {
        prev->next = &sendWr.wr;
      }
      prev = &sendWr.wr;
    }

    // Signal the last WR and encode count for completion tracking.
    prev->next = nullptr;
    prev->wr_id = (static_cast<uint64_t>(taskId) << 32) | qpAssigned[q];
    prev->send_flags = IBV_SEND_SIGNALED;

    uint32_t posted = postSend(q, head, qpAssigned[q], taskId, task);
    if (posted == 0) {
      // postSend failed — task is already errored and postFinished.
      // Return error so caller stops posting and lets poll chain drain.
      return Err(ErrCode::DriverError, "postSend failed");
    }
    totalPosted += posted;
  }

  return Result<uint32_t>(totalPosted);
}

// ---------------------------------------------------------------------------
// Unified IO processing loop (EventBase thread)
// ---------------------------------------------------------------------------

void RdmaTransport::ioLooper() noexcept {
  // Phase 1 — Poll: drain all CQs (round-robin).
  pollCompletions();

  // Phase 2 — Post: walk pendingTransfers_ front-to-back.
  bool tryNext = true;
  while (!pendingTransfers_.empty() && tryNext) {
    auto& entry = pendingTransfers_.front();
    if (entry.putGetTransfer) {
      tryNext = putGetIoProcess(*entry.putGetTransfer);
    } else {
      tryNext = sendRecvIoProcess(*entry.sendRecvTransfer);
    }
  }

  // Phase 3 — Fulfill: walk pendingCompletions_ front-to-back for in-order
  // promise delivery.
  while (!pendingCompletions_.empty()) {
    auto& entry = pendingCompletions_.front();

    if (entry.task->isDone()) {
      entry.task->fulfill();
    } else if (state_ != TransportState::Connected) {
      entry.task->recordCompletion(
          Err(ErrCode::TransportError, "RDMA transport is broken"));
      entry.task->fulfill();
    }

    // Remove when fully drained (all CQEs received), or when the transport
    // is broken (no more CQEs will ever arrive).
    if (entry.task->isFullyDrained() || state_ != TransportState::Connected) {
      inflightTasks_.erase(entry.taskId);
      pendingCompletions_.pop_front();
      continue;
    }

    break;
  }

  // Phase 4 — Yield & reschedule.
  if (!pendingTransfers_.empty() || !pendingCompletions_.empty()) {
    evb_->dispatch([this]() noexcept { ioLooper(); });
  } else {
    ioLooperScheduled_ = false;
  }
}

bool RdmaTransport::putGetIoProcess(PutGetTransfer& entry) noexcept {
  if (state_ != TransportState::Connected) {
    entry.task->recordCompletion(
        Err(ErrCode::TransportError, "RDMA transport is broken"));
    pendingCompletions_.push_back({entry.taskId, std::move(entry.task)});
    pendingTransfers_.pop_front();
    return true;
  }

  if (entry.idx >= entry.sendWrs->size()) {
    entry.task->postFinished();
    pendingCompletions_.push_back({entry.taskId, std::move(entry.task)});
    pendingTransfers_.pop_front();
    return true;
  }

  auto postResult = spray(*entry.sendWrs, entry.idx, entry.taskId, *entry.task);
  if (postResult.hasError()) {
    // postSend failed on a QP. Task is already errored and
    // postFinished. Don't post more — just let the poll chain
    // drain CQEs from earlier successful QPs and the flush WR.
    UNIFLOW_LOG_ERROR("ioLooper: taskId={} spray failed", entry.taskId);
    pendingCompletions_.push_back({entry.taskId, std::move(entry.task)});
    pendingTransfers_.pop_front();
    return true;
  }

  if (entry.idx >= entry.sendWrs->size()) {
    UNIFLOW_LOG_DEBUG(
        "ioLooper: taskId={} all {} chunks posted",
        entry.taskId,
        entry.sendWrs->size());
    entry.task->postFinished();
    pendingCompletions_.push_back({entry.taskId, std::move(entry.task)});
    pendingTransfers_.pop_front();
    return true;
  } else {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Transfer dispatch (caller thread → EventBase thread)
// ---------------------------------------------------------------------------

std::future<Status> RdmaTransport::rdmaPutGetTransfer(
    std::span<const TransferRequest> requests,
    ibv_wr_opcode opcode,
    const RequestOptions& /*options*/) {
  if (state_ != TransportState::Connected) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "RDMA transfer: not connected"));
  }
  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  auto wrsResult = buildSendWrs(requests, opcode);
  if (wrsResult.hasError()) {
    UNIFLOW_LOG_ERROR(
        "rdmaPutGetTransfer: buildSendWrs failed: {}",
        wrsResult.error().message());
    return make_ready_future<Status>(std::move(wrsResult).error());
  }
  auto reqWrs = std::move(wrsResult).value();
  assert(!reqWrs->empty());
  UNIFLOW_LOG_DEBUG(
      "rdmaPutGetTransfer: {} requests, {} chunks, opcode={}",
      requests.size(),
      reqWrs->size(),
      static_cast<int>(opcode));

  IOType type = opcode == IBV_WR_RDMA_WRITE ? IOType::Put : IOType::Get;
  auto task = std::make_shared<Task>(type);
  auto future = task->get_future();
  auto transfer = std::make_unique<PutGetTransfer>();
  transfer->sendWrs = std::move(reqWrs);
  transfer->task = task;

  evb_->dispatch([this,
                  task = std::move(task),
                  transfer = std::move(transfer)]() mutable noexcept {
    uint32_t taskId = nextTaskId_++;
    UNIFLOW_LOG_DEBUG(
        "rdmaPutGetTransfer: taskId={} dispatched to EventBase", taskId);
    inflightTasks_[taskId] = std::move(task);
    transfer->taskId = taskId;
    pendingTransfers_.push_back(
        {.putGetTransfer = std::move(transfer), .sendRecvTransfer = nullptr});

    if (!ioLooperScheduled_) {
      ioLooperScheduled_ = true;
      evb_->dispatch([this]() noexcept { ioLooper(); });
    }
  });

  return future;
}

std::future<Status> RdmaTransport::rdmaSendRecvTransfer(
    IOType ioType,
    Segment::Span data,
    const RequestOptions& options) {
  if (data.size() == 0) {
    return make_ready_future(Ok());
  }

  auto slabSize = info_.header.slabSize;
  if (!slabPool_ || slabSize == 0) {
    return make_ready_future<Status>(
        Err(ErrCode::ResourceExhausted, "no slab pool"));
  }

  if (data.memType() == MemoryType::VRAM && !options.stream.has_value()) {
    return make_ready_future<Status>(
        Err(ErrCode::InvalidArgument,
            "VRAM transfer requires an explicit CUDA stream"));
  }

  auto task = std::make_shared<Task>(ioType);
  auto future = task->get_future();
  auto transfer = std::make_unique<SendRecvTransfer>(ioType, std::move(data));
  if (options.stream.has_value()) {
    transfer->stream = static_cast<cudaStream_t>(options.stream.value());
  }
  transfer->copyEngine.emplace(
      transfer->data.memType(),
      transfer->data.deviceId(),
      transfer->stream,
      cudaApi_.get(),
      cudaDriverApi_.get());
  transfer->totalSteps = (transfer->data.size() + slabSize - 1) / slabSize;
  const uint16_t depth = static_cast<uint16_t>(std::min(
      {static_cast<uint64_t>(config_.pipelineDepth),
       transfer->totalSteps,
       slabPool_->numSlabs()}));

  auto slabResult = allocateSlab(slabPool_.get(), depth);
  if (!slabResult) {
    return make_ready_future<Status>(std::move(slabResult).error());
  }
  transfer->localSlabs = std::move(slabResult.value());
  transfer->task = task;

  evb_->dispatch([this,
                  task = std::move(task),
                  transfer = std::move(transfer)]() mutable noexcept {
    uint32_t taskId = nextTaskId_++;
    transfer->taskId = taskId;
    inflightTasks_[taskId] = std::move(task);
    UNIFLOW_LOG_DEBUG(
        "{}: taskId={} totalSteps={} depth={} slabSize={} dispatched to EventBase",
        transfer->opType == IOType::Send ? "send" : "recv",
        transfer->taskId,
        transfer->totalSteps,
        transfer->localSlabs.size(),
        info_.slabSize);

    pendingTransfers_.push_back(
        {.putGetTransfer = nullptr, .sendRecvTransfer = std::move(transfer)});

    if (!ioLooperScheduled_) {
      ioLooperScheduled_ = true;
      evb_->dispatch([this]() noexcept { ioLooper(); });
    }
  });

  return future;
}

// ---------------------------------------------------------------------------
// Completion polling (EventBase thread)
// ---------------------------------------------------------------------------

Status RdmaTransport::pollCompletions() {
  constexpr int kNumBatch = 16;

  // Cap total CQEs per call to ensure fairness with other transports
  // sharing the same EventBase.
  const uint64_t maxCqes = config_.maxWr * qps_.size();
  uint64_t totalDrained = 0;
  Status res = Ok();

  // Round-robin: poll one batch from each CQ per round, repeat until
  // CQs are empty or the cap is reached.
  while (totalDrained < maxCqes) {
    uint64_t before = totalDrained;
    for (size_t i = 0; i < cqs_.size() && totalDrained < maxCqes; ++i) {
      if (numPendingCqe_[i] <= 0) {
        continue;
      }

      ibv_wc wcs[kNumBatch];
      auto pollResult = ibvApi_->pollCq(cqs_[i], kNumBatch, wcs);
      if (pollResult.hasError()) {
        UNIFLOW_LOG_ERROR(
            "pollCompletions: pollCq failed on CQ {}: {}",
            i,
            pollResult.error().message());
        state_ = TransportState::Error;
        for (auto& [taskId, task] : inflightTasks_) {
          task->recordCompletion(
              Err(ErrCode::TransportError, "RDMA transport is broken"));
        }
        return pollResult.error();
      }

      int n = pollResult.value();
      totalDrained += static_cast<uint64_t>(n);
      for (const auto& wc : std::span(wcs).subspan(0, n)) {
        uint32_t taskId = static_cast<uint32_t>(wc.wr_id >> 32);
        uint32_t wrInfo = static_cast<uint32_t>(wc.wr_id & 0xffffffff);

        if (auto it = inflightTasks_.find(taskId); it != inflightTasks_.end()) {
          bool isCopyOp = it->second->type() == IOType::Send ||
              it->second->type() == IOType::Recv;
          uint32_t numWrs = isCopyOp ? (wrInfo >> 16) : wrInfo;

          if (numWrs > 0) {
            --numPendingCqe_[i];
          }

          if (auto qp = qpNumToIdx_.find(qpMapKey(i, wc.qp_num));
              qp != qpNumToIdx_.end()) {
            numWrsPerQp_[qp->second] -= numWrs;
          }

          if (isCopyOp) {
            uint32_t slot = wrInfo & 0xffff;
            --slotPendingCqe_[slot];
          }

          if (wc.status != IBV_WC_SUCCESS) {
            auto msg = fmt::format(
                "RDMA WR failed: wr_id={} taskId={} wrInfo={} "
                "qp_num={} status={}",
                wc.wr_id,
                taskId,
                wrInfo,
                wc.qp_num,
                static_cast<int>(wc.status));
            UNIFLOW_LOG_ERROR(msg);
            res = Err(ErrCode::DriverError, std::move(msg));
            it->second->recordCompletion(res.error());
          }

          it->second->recordCompletion(numWrs);

        } else {
          auto msg = fmt::format(
              "pollCompletions: cannot find task for taskId={} wrInfo={}",
              taskId,
              wrInfo);
          UNIFLOW_LOG_ERROR(msg);
          res = Err(ErrCode::TransportError, std::move(msg));
          state_ = TransportState::Error;
          for (auto& [_, task] : inflightTasks_) {
            task->recordCompletion(res.error());
          }
          return res;
        }
      }
    }
    if (totalDrained == before) {
      break;
    }
  }
  return res;
}

std::future<Status> RdmaTransport::send(
    RegisteredSegment::Span src,
    const RequestOptions& options) {
  std::promise<Status> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

std::future<Status> RdmaTransport::send(
    Segment::Span src,
    const RequestOptions& options) {
  return rdmaSendRecvTransfer(IOType::Send, src, options);
}

std::future<Status> RdmaTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  std::promise<Status> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

std::future<Status> RdmaTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  return rdmaSendRecvTransfer(IOType::Recv, dst, options);
}

// ---------------------------------------------------------------------------
// Unified IO processing loop (EventBase thread)
// ---------------------------------------------------------------------------

bool RdmaTransport::sendRecvIoProcess(SendRecvTransfer& entry) noexcept {
  if (state_ != TransportState::Connected) {
    entry.task->recordCompletion(
        Err(ErrCode::TransportError, "RDMA transport is broken"));
    pendingCompletions_.push_back({entry.taskId, std::move(entry.task)});
    pendingTransfers_.pop_front();
    return true;
  }

  size_t before = pendingTransfers_.size();
  Status res = Ok();
  if (entry.opType == IOType::Send) {
    res = sendProgress(entry);
  } else {
    res = recvProgress(entry);
  }

  if (res.hasError()) {
    entry.task->recordCompletion(std::move(res).error());
    pendingCompletions_.push_back({entry.taskId, std::move(entry.task)});
    pendingTransfers_.pop_front();
    return true;
  }

  return pendingTransfers_.size() < before;
}

// ---------------------------------------------------------------------------
// Send pipeline
// ---------------------------------------------------------------------------

Status RdmaTransport::sendProgress(SendRecvTransfer& transfer) {
  const uint32_t depth = static_cast<uint32_t>(transfer.localSlabs.size());
  const size_t slabSize = info_.header.slabSize;

  if (transfer.done < transfer.totalSteps) {
    sendCopyProgress(transfer, depth, slabSize);
    CHECK_EXPR(sendTransmitProgress(transfer, depth, slabSize));
    CHECK_EXPR(pollCompletions());
  }

  // Advance done cursor for completed slots.
  while (transfer.done < transfer.notified) {
    uint32_t doneSlot = transfer.done % depth;
    if (slotPendingCqe_[doneSlot] == 0) {
      ++transfer.done;
    } else {
      break;
    }
  }

  if (transfer.done >= transfer.totalSteps) {
    UNIFLOW_LOG_DEBUG("sendProgress: taskId={} completed", transfer.taskId);
    transfer.task->postFinished();
    pendingCompletions_.push_back({transfer.taskId, std::move(transfer.task)});
    pendingTransfers_.pop_front();
  }
  return Ok();
}

void RdmaTransport::sendCopyProgress(
    SendRecvTransfer& transfer,
    uint32_t depth,
    size_t slabSize) {
  if (transfer.copied >= transfer.totalSteps ||
      transfer.copied >= transfer.done + depth) {
    return;
  }

  uint32_t slot = transfer.copied % depth;
  size_t offset = transfer.copied * slabSize;
  size_t len = std::min(slabSize, transfer.data.size() - offset);
  const void* src = transfer.ptr(offset);

  transfer.copyEngine->copyToSlab(transfer.localSlabs[slot], src, len);

  ++transfer.copied;
}

Status RdmaTransport::sendTransmitProgress(
    SendRecvTransfer& transfer,
    uint32_t depth,
    size_t slabSize) {
  if (transfer.notified >= transfer.copied) {
    return Ok();
  }

  uint32_t slot = transfer.notified % depth;
  auto& slab = transfer.localSlabs[slot];

  if (!transfer.copyEngine->isCopyDone(slab)) {
    return Ok();
  }

  uint32_t remoteSlabIdx = ctrlCts(slot).load(std::memory_order_acquire);
  if (remoteSlabIdx == kSlabIdxInvalid) {
    return Ok();
  }

  size_t offset = transfer.notified * slabSize;
  size_t len = std::min(slabSize, transfer.data.size() - offset);
  uint16_t remoteSlab = static_cast<uint16_t>(remoteSlabIdx);

  auto res =
      postSlabTransfer(transfer, slab, remoteSlab, len, slot, transfer.taskId);
  if (res.hasError()) {
    return std::move(res).error();
  } else if (res.value() == 0) {
    return Ok();
  }

  transfer.copyEngine->resetState(slab);
  ctrlCts(slot).store(kSlabIdxInvalid, std::memory_order_release);
  ++transfer.notified;
  return Ok();
}

Result<size_t> RdmaTransport::postSlabTransfer(
    SendRecvTransfer& transfer,
    RdmaSlab& localSlab,
    uint16_t remoteSlab,
    size_t len,
    uint32_t slot,
    uint32_t taskId) {
  const uint32_t numQps = config_.numQps;
  const size_t chunkSize = config_.chunkSize;
  const uint32_t numChunks =
      static_cast<uint32_t>((len + chunkSize - 1) / chunkSize);
  const uint32_t numWrs = numChunks + numQps;

  std::vector<uint32_t> qpAvail(numQps);
  uint32_t totalAvail = getQpAvail(qpAvail);

  if (totalAvail < numWrs) {
    return 0;
  }

  std::vector<uint32_t> qpAssigned(numQps, 0);
  uint32_t totalAssigned = assignToQps(numWrs, totalAvail, qpAvail, qpAssigned);

  if (totalAssigned < numWrs) {
    return 0;
  }

  uint64_t localBase = localSlab.addr();
  uint64_t remoteBase = remoteSlabBuffer_.addr +
      static_cast<size_t>(remoteSlab) * info_.header.slabSize;

  size_t chunkIdx = 0;
  for (uint32_t q = 0; q < numQps; ++q) {
    if (qpAssigned[q] <= 1) {
      continue;
    }
    uint32_t localNicIdx = q % nics_.size();
    uint32_t remoteNicIdx = q % remoteSlabBuffer_.rkeys.size();
    struct WrPair {
      ibv_send_wr wr{};
      ibv_sge sge{};
    };
    const uint32_t dataWrCount = qpAssigned[q] - 1;
    std::vector<WrPair> wrs(qpAssigned[q]);
    uint32_t postedLen = 0;
    uint32_t postedWrCount = 0;
    for (; postedWrCount < dataWrCount; ++postedWrCount, ++chunkIdx) {
      size_t off = chunkIdx * chunkSize;
      if (off >= len) {
        break;
      }
      size_t length = std::min(chunkSize, len - off);
      auto& wr = wrs[postedWrCount];

      wr.sge.addr = localBase + off;
      wr.sge.length = static_cast<uint32_t>(length);
      wr.sge.lkey = slabPool_->slabLkey(localNicIdx);

      wr.wr.sg_list = &wr.sge;
      wr.wr.num_sge = 1;
      wr.wr.opcode = IBV_WR_RDMA_WRITE;
      wr.wr.send_flags = 0;
      wr.wr.wr.rdma.remote_addr = remoteBase + off;
      wr.wr.wr.rdma.rkey = remoteSlabBuffer_.rkeys[remoteNicIdx];
      wr.wr.next = &wrs[postedWrCount + 1].wr;
      postedLen += static_cast<uint32_t>(length);
    }

    auto& notifyWr = wrs[postedWrCount++];
    notifyWr.sge.addr = reinterpret_cast<uint64_t>(&postedLen);
    notifyWr.sge.length = sizeof(uint32_t);
    notifyWr.sge.lkey = 0;

    notifyWr.wr.sg_list = &notifyWr.sge;
    notifyWr.wr.num_sge = 1;
    notifyWr.wr.opcode = IBV_WR_RDMA_WRITE;
    notifyWr.wr.send_flags = IBV_SEND_INLINE;
    notifyWr.wr.wr.rdma.remote_addr = remoteCtrlNotifyAddr(slot, q);
    notifyWr.wr.wr.rdma.rkey = remoteCtrlBuffer_.rkeys[remoteNicIdx];
    notifyWr.wr.next = nullptr;

    notifyWr.wr.wr_id = (static_cast<uint64_t>(taskId) << 32) |
        (static_cast<uint32_t>(postedWrCount) << 16) | (slot & 0xFFFF);
    notifyWr.wr.send_flags |= IBV_SEND_SIGNALED;

    uint32_t posted = postSend(
        q, &wrs.front().wr, postedWrCount, taskId, *transfer.task, slot);
    if (posted == 0) {
      state_ = TransportState::Error;
      return Err(ErrCode::DriverError, "postSend failed");
    }
  }

  return len;
}

// ---------------------------------------------------------------------------
// Recv pipeline
// ---------------------------------------------------------------------------

Status RdmaTransport::recvProgress(SendRecvTransfer& transfer) {
  const uint32_t depth = static_cast<uint32_t>(transfer.localSlabs.size());
  const size_t slabSize = info_.header.slabSize;

  // post initial CTS entries so sender knows which remote slabs to write to.
  while (transfer.initStep <
         std::min(static_cast<uint64_t>(depth), transfer.totalSteps)) {
    auto res = postCts(
        transfer.initStep,
        transfer.localSlabs[transfer.initStep].index(),
        transfer.taskId,
        *transfer.task);
    if (res.hasError()) {
      return std::move(res).error();
    }
    // No wr space to post cts
    if (!res.value()) {
      return Ok();
    }
    ++transfer.initStep;
  }

  if (transfer.done < transfer.totalSteps) {
    recvNotifyProgress(transfer, depth, slabSize);
    recvCopyProgress(transfer, depth, slabSize);
    CHECK_EXPR(recvDoneProgress(transfer, depth));
    CHECK_EXPR(pollCompletions());
  }

  if (transfer.done >= transfer.totalSteps) {
    UNIFLOW_LOG_DEBUG("recvProgress: taskId={} completed", transfer.taskId);
    transfer.task->postFinished();
    pendingCompletions_.push_back({transfer.taskId, std::move(transfer.task)});
    pendingTransfers_.pop_front();
  }
  return Ok();
}

void RdmaTransport::recvNotifyProgress(
    SendRecvTransfer& transfer,
    uint32_t depth,
    size_t slabSize) {
  if (transfer.notified >= transfer.totalSteps ||
      transfer.notified >= transfer.done + depth) {
    return;
  }

  const uint32_t numQps = config_.numQps;
  uint32_t slot = transfer.notified % depth;
  uint32_t length = 0;
  for (uint32_t q = 0; q < numQps; ++q) {
    uint32_t notify = ctrlNotify(slot, q).load(std::memory_order_acquire);
    if (notify != kSlabIdxInvalid) {
      length += notify;
    }
  }

  size_t offset = transfer.notified * slabSize;
  if (length < std::min(slabSize, transfer.data.size() - offset)) {
    return;
  }

  for (uint32_t q = 0; q < numQps; ++q) {
    ctrlNotify(slot, q).store(kSlabIdxInvalid, std::memory_order_release);
  }
  ++transfer.notified;
}

void RdmaTransport::recvCopyProgress(
    SendRecvTransfer& transfer,
    uint32_t depth,
    size_t slabSize) {
  if (transfer.copied >= transfer.notified) {
    return;
  }

  uint32_t slot = transfer.copied % depth;
  auto& slab = transfer.localSlabs[slot];
  size_t offset = transfer.copied * slabSize;
  size_t len = std::min(slabSize, transfer.data.size() - offset);
  void* dst = transfer.ptr(offset);

  transfer.copyEngine->copyFromSlab(dst, slab, len);

  ++transfer.copied;
}

Status RdmaTransport::recvDoneProgress(
    SendRecvTransfer& transfer,
    uint32_t depth) {
  if (transfer.done >= transfer.copied) {
    return Ok();
  }

  uint32_t slot = transfer.done % depth;

  // Wait for the previous CTS CQE before recycling this slot. This also
  // guarantees all CTS CQEs are drained by completion: the last D steps
  // skip postCts (nextStep >= totalSteps) but still hit this check.
  if (slotPendingCqe_[slot] != 0) {
    return Ok();
  }

  auto& slab = transfer.localSlabs[slot];

  if (!transfer.copyEngine->isCopyDone(slab)) {
    return Ok();
  }

  uint64_t nextStep = transfer.done + depth;
  if (nextStep < transfer.totalSteps) {
    auto res = postCts(slot, slab.index(), transfer.taskId, *transfer.task);
    if (res.hasError()) {
      return std::move(res).error();
    }
    // No wr space to post cts
    if (!res.value()) {
      return Ok();
    }
  }

  transfer.copyEngine->resetState(slab);

  ++transfer.done;
  return Ok();
}

Result<bool> RdmaTransport::postCts(
    uint32_t slot,
    uint16_t slabIdx,
    uint32_t taskId,
    Task& task) {
  uint32_t qpIdx = 0;
  for (; qpIdx < config_.numQps; ++qpIdx) {
    if (numWrsPerQp_[qpIdx] < config_.maxWr) {
      break;
    }
  }

  if (qpIdx == config_.numQps) {
    return false;
  }
  uint32_t nicIdx = qpIdx % nics_.size();

  uint32_t ctsPayload = slabIdx;
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(&ctsPayload);
  sge.length = sizeof(uint32_t);
  sge.lkey = 0;

  ibv_send_wr wr{};
  wr.wr_id =
      (static_cast<uint64_t>(taskId) << 32) | (1u << 16) | (slot & 0xFFFF);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.wr.rdma.remote_addr = remoteCtrlCtsAddr(slot);
  wr.wr.rdma.rkey = remoteCtrlBuffer_.rkeys[nicIdx];
  wr.next = nullptr;

  if (postSend(qpIdx, &wr, 1, taskId, task, slot) == 0) {
    return Err(ErrCode::TransportError, "postCts failed");
  }
  return true;
}

// ---------------------------------------------------------------------------
// Ctrl buffer helpers
// ---------------------------------------------------------------------------

size_t RdmaTransport::ctrlBufferSize() const {
  constexpr size_t align = alignof(uint64_t);
  const uint32_t D = config_.pipelineDepth;
  const uint32_t Q = config_.numQps;
  const size_t raw = D * (1 + Q) * sizeof(uint32_t);
  return (raw + align - 1) & ~(align - 1);
}

std::atomic_ref<uint32_t> RdmaTransport::ctrlCts(uint16_t slotIdx) {
  auto* p = static_cast<uint32_t*>(ctrlBuffer_) + slotIdx;
  return std::atomic_ref<uint32_t>(*p);
}

std::atomic_ref<uint32_t> RdmaTransport::ctrlNotify(
    uint16_t slotIdx,
    uint32_t qpIdx) {
  const uint32_t D = config_.pipelineDepth;
  const uint32_t Q = config_.numQps;
  auto* p = static_cast<uint32_t*>(ctrlBuffer_) + D + slotIdx * Q + qpIdx;
  return std::atomic_ref<uint32_t>(*p);
}

uint64_t RdmaTransport::remoteCtrlCtsAddr(uint16_t slotIdx) const {
  return remoteCtrlBuffer_.addr + slotIdx * sizeof(uint32_t);
}

uint64_t RdmaTransport::remoteCtrlNotifyAddr(uint16_t slotIdx, uint32_t qpIdx)
    const {
  const uint32_t D = config_.pipelineDepth;
  const uint32_t Q = config_.numQps;
  return remoteCtrlBuffer_.addr + (D + slotIdx * Q + qpIdx) * sizeof(uint32_t);
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

void RdmaTransport::shutdown() {
  if (shutdown_.exchange(true)) {
    return;
  }

  // Make sure that there is no tasks in the evb loop thread related to this
  // transport instance
  if (!evb_->isLoopRunning()) {
    state_ = TransportState::Disconnected;
    UNIFLOW_LOG_WARN("shutdown: event loop already stopped, skipping drain");
  } else {
    UNIFLOW_LOG_INFO("shutdown: draining inflight tasks");
    std::mutex m;
    std::condition_variable cv;
    bool done = false;

    // Make sure that there is no tasks in the evb loop thread related to this
    // transport instance
    evb_->dispatch([this, &m, &cv, &done]() noexcept {
      auto drain = [this, &m, &cv, &done](auto& self) -> void {
        if (pendingTransfers_.empty() && pendingCompletions_.empty() &&
            inflightTasks_.empty()) {
          state_ = TransportState::Disconnected;
          {
            std::lock_guard<std::mutex> lock(m);
            done = true;
          }
          cv.notify_one();
        } else {
          evb_->dispatch([self = std::move(self)]() noexcept { self(self); });
        }
      };
      drain(drain);
    });

    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&done] { return done; });
  }

  UNIFLOW_LOG_INFO("shutdown: cleanup complete");

  for (auto* qp : qps_) {
    if (qp) {
      ibv_qp_attr attr{};
      attr.qp_state = IBV_QPS_ERR;
      ibvApi_->modifyQp(qp, &attr, IBV_QP_STATE);
      ibvApi_->destroyQp(qp);
    }
  }
  qps_.clear();
  psns_.clear();

  for (auto* cq : cqs_) {
    if (cq) {
      ibvApi_->destroyCq(cq);
    }
  }
  cqs_.clear();

  for (auto* mr : ctrlMrs_) {
    ibvApi_->deregMr(mr);
  }
  ctrlMrs_.clear();

  if (ctrlBuffer_) {
    deviceAdapter_->pinnedHostFree(ctrlBuffer_);
    ctrlBuffer_ = nullptr;
  }

  UNIFLOW_LOG_INFO("shutdown: complete");
}

// ---------------------------------------------------------------------------
// RdmaTransportFactory
// ---------------------------------------------------------------------------

Status RdmaTransportFactory::supported(std::shared_ptr<IbvApi> ibvApi) {
  if (!ibvApi) {
    ibvApi = std::make_shared<IbvApi>();
  }

  CHECK_EXPR(ibvApi->init());

  int numDevices = 0;
  auto devListResult = ibvApi->getDeviceList(&numDevices);
  CHECK_RETURN(devListResult);
  ibv_device** deviceList = devListResult.value();
  struct DevListGuard {
    std::shared_ptr<IbvApi> api;
    ibv_device** list;
    ~DevListGuard() {
      api->freeDeviceList(list);
    }
  } devListGuard{ibvApi, deviceList};

  if (numDevices == 0) {
    UNIFLOW_LOG_INFO("RDMA not supported: no IB devices found");
    return Err(ErrCode::ResourceExhausted, "No RDMA devices found");
  }

  for (int i = 0; i < numDevices; ++i) {
    bool found = false;
    auto ctxResult = ibvApi->openDevice(deviceList[i]);
    CHECK_RETURN(ctxResult);
    ibv_context* ctx = ctxResult.value();

    ibv_device_attr devAttr{};
    if (ibvApi->queryDevice(ctx, &devAttr)) {
      for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
        ibv_port_attr portAttr{};
        auto portStatus = ibvApi->queryPort(ctx, port, &portAttr);
        if (!portStatus.hasError() && portAttr.state == IBV_PORT_ACTIVE) {
          found = true;
          break;
        }
      }
    }

    ibvApi->closeDevice(ctx);
    if (!found) {
      return Err(
          ErrCode::ResourceExhausted,
          fmt::format(
              "No active RDMA ports found for {}", deviceList[i]->name));
    }
  }

  return Ok();
}

RdmaTransportFactory::RdmaTransportFactory(
    const std::vector<std::string>& deviceNames,
    EventBase* evb,
    RdmaTransportConfig config,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    std::shared_ptr<CudaApi> cudaApi,
    std::optional<uint8_t> portNum)
    : TransportFactory(TransportType::RDMA),
      ibvApi_(std::move(ibvApi)),
      cudaDriverApi_(std::move(cudaDriverApi)),
      cudaApi_(std::move(cudaApi)),
      evb_(evb),
      nicsHandle_(std::make_shared<std::vector<NicResources>>()),
      config_(config) {
  assert(evb_ != nullptr);
  if (deviceNames.empty()) {
    throw std::runtime_error("No device names provided");
  }
  if (!ibvApi_) {
    ibvApi_ = std::make_shared<IbvApi>();
  }
  if (!cudaDriverApi_) {
    cudaDriverApi_ = std::make_shared<CudaDriverApi>();
  }
  if (!cudaApi_) {
    cudaApi_ = std::make_shared<CudaApi>();
  }

  deviceAdapter_ = createDeviceAdapter(cudaApi_, cudaDriverApi_);

  // Generate a random domain id to identify handles from this factory.
  std::mt19937_64 rng{std::random_device{}()};
  domainId_ = rng();

  int numDevices = 0;
  auto devListResult = ibvApi_->getDeviceList(&numDevices);
  if (devListResult.hasError() || numDevices == 0) {
    throw std::runtime_error("No RDMA devices found");
  }
  ibv_device** deviceList = devListResult.value();
  auto devListDeleter = [this](ibv_device** p) {
    if (p) {
      ibvApi_->freeDeviceList(p);
    }
  };
  std::unique_ptr<ibv_device*[], decltype(devListDeleter)> devListGuard(
      deviceList, devListDeleter);

  nicsHandle_->reserve(deviceNames.size());
  for (const auto& deviceName : deviceNames) {
    ibv_device* targetDevice = nullptr;
    for (int i = 0; i < numDevices; ++i) {
      auto nameResult = ibvApi_->getDeviceName(deviceList[i]);
      if (nameResult.hasValue() && deviceName == nameResult.value()) {
        targetDevice = deviceList[i];
        break;
      }
    }

    if (!targetDevice) {
      throw std::runtime_error("RDMA device not found: " + deviceName);
    }

    int numaNode = sharedTopology().nicNumaNode(deviceName);
    nicsHandle_->emplace_back(
        targetDevice, ibvApi_, numaNode, config_.gidIndex, portNum);
  }

  if (config_.slabPoolConfig.slabNum > 0) {
    slabPool_ = std::make_shared<RdmaSlabPool>(
        config_.slabPoolConfig, cudaApi_, ibvApi_, nicsHandle_);
  }
}

Result<std::unique_ptr<RegistrationHandle>>
RdmaTransportFactory::registerSegment(Segment& segment) {
  if (nicsHandle_->empty()) {
    return Err(ErrCode::InvalidArgument, "No NICs available for registration");
  }

  int access =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  // For VRAM, ask the DeviceAdapter to export a DMA-BUF fd for GPU Direct
  // RDMA (NIC reads/writes accelerator memory directly). For DRAM, fall
  // back to standard ibv_reg_mr.
  // RAII guard ensures closeDmaBuff() runs on every exit path.
  DmaBuff dmaBuff;
  DmaBuffCloseGuard closeGuard{*deviceAdapter_, dmaBuff};
  uint64_t registrationBase = 0;

  if (segment.memType() == MemoryType::VRAM) {
    auto dmaBufSupported =
        deviceAdapter_->isDmaBuffSupported(segment.deviceId());
    CHECK_RETURN(dmaBufSupported);
    if (dmaBufSupported.value()) {
      auto exportRes = deviceAdapter_->exportDmaBuff(
          segment.deviceId(), segment.mutable_data(), segment.len());
      if (exportRes.hasValue()) {
        dmaBuff = std::move(exportRes).value();
        registrationBase = dmaBuff.registrationBase;
      } else if (deviceAdapter_->allowsRegMrFallback()) {
        UNIFLOW_LOG_WARN(
            "registerSegment: exportDmaBuff failed for VRAM segment "
            "(device={}, ptr=0x{:x}, len={}): {}. Falling back to ibv_reg_mr; "
            "GPU Direct RDMA may be disabled for this segment.",
            segment.deviceId(),
            reinterpret_cast<uint64_t>(segment.mutable_data()),
            segment.len(),
            exportRes.error().toString());
      } else {
        UNIFLOW_LOG_ERROR(
            "registerSegment: exportDmaBuff failed for VRAM segment "
            "(device={}, ptr=0x{:x}, len={}): {}. Backend does not allow "
            "ibv_reg_mr fallback for accelerator memory (a plain registration "
            "would be invalid, e.g. out-of-pool or wrong-device pointer); "
            "failing registration.",
            segment.deviceId(),
            reinterpret_cast<uint64_t>(segment.mutable_data()),
            segment.len(),
            exportRes.error().toString());
        return std::move(exportRes).error();
      }
    } else if (!deviceAdapter_->allowsRegMrFallback()) {
      // DMA-BUF is the only valid registration path for this backend's
      // accelerator memory; a plain ibv_reg_mr would mis-register it.
      UNIFLOW_LOG_ERROR(
          "registerSegment: DMA-BUF unsupported for VRAM segment "
          "(device={}, ptr=0x{:x}, len={}) and backend does not allow "
          "ibv_reg_mr fallback; failing registration.",
          segment.deviceId(),
          reinterpret_cast<uint64_t>(segment.mutable_data()),
          segment.len());
      return Err(
          ErrCode::DriverError,
          "registerSegment: DMA-BUF unsupported and ibv_reg_mr fallback not "
          "permitted for accelerator memory on device " +
              std::to_string(segment.deviceId()));
    }
  }
  const bool useRegMrFallback =
      segment.memType() == MemoryType::VRAM && dmaBuff.fd < 0;
  if (useRegMrFallback) {
    dmaBufFallbackCount_.fetch_add(1, std::memory_order_relaxed);
  }

  // Register with every NIC's protection domain so the region is usable
  // across all QPs regardless of which NIC they belong to.
  std::vector<ibv_mr*> mrs;
  mrs.reserve(nicsHandle_->size());
  for (const auto& nic : *nicsHandle_) {
    Result<ibv_mr*> mrResult = Err(ErrCode::NotImplemented);
    if (nic.dmaBufSupported && dmaBuff.fd >= 0) {
      // Accelerator memory: register via DMA-BUF for GPU Direct RDMA.
      mrResult = ibvApi_->regDmabufMr(
          nic.pd,
          dmaBuff.offset,
          dmaBuff.len,
          dmaBuff.iova,
          dmaBuff.fd,
          access);
    } else {
      // standard registration.
      mrResult =
          ibvApi_->regMr(nic.pd, segment.mutable_data(), segment.len(), access);
    }
    if (mrResult.hasError()) {
      for (auto* mr : mrs) {
        ibvApi_->deregMr(mr);
      }
      return std::move(mrResult).error();
    }
    mrs.push_back(mrResult.value());
  }

  // For pinned (non-ODP) MRs, ibv_reg_mr has faulted the pages, so the host
  // buffer's NUMA node is now stable; capture it for NUMA-aware QP scheduling.
  const int numaNode = segment.memType() == MemoryType::DRAM
      ? detectHostNumaNode(segment.mutable_data())
      : -1;
  return std::make_unique<RdmaRegistrationHandle>(
      std::move(mrs),
      ibvApi_,
      domainId_,
      registrationBase,
      deviceAdapter_,
      segment.deviceId(),
      numaNode);
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
RdmaTransportFactory::importSegment(
    [[maybe_unused]] size_t segmentLength,
    std::span<const uint8_t> payload) {
  if (payload.size() < RdmaRegistrationHandle::kPayloadHeaderSize) {
    return Err(
        ErrCode::InvalidArgument, "RDMA importSegment: payload too small");
  }

  RdmaRegistrationHandle::Header header;
  std::memcpy(&header, payload.data(), sizeof(header));

  if (header.numMrs == 0) {
    return Err(ErrCode::InvalidArgument, "RDMA importSegment: numMrs is zero");
  }

  size_t expectedSize = RdmaRegistrationHandle::kPayloadHeaderSize +
      header.numMrs * sizeof(uint32_t);
  if (payload.size() != expectedSize) {
    return Err(
        ErrCode::InvalidArgument,
        "RDMA importSegment: expected " + std::to_string(expectedSize) +
            " bytes, got " + std::to_string(payload.size()));
  }

  // Deserialize per-NIC rkeys.
  std::vector<uint32_t> rkeys(header.numMrs);
  size_t offset = RdmaRegistrationHandle::kPayloadHeaderSize;
  std::memcpy(
      rkeys.data(), payload.data() + offset, sizeof(uint32_t) * header.numMrs);

  uint64_t domainId = header.domainId;
  return std::make_unique<RdmaRemoteRegistrationHandle>(
      std::move(rkeys), domainId, header.registrationBase, deviceAdapter_);
}

Result<std::unique_ptr<Transport>> RdmaTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  CHECK_EXPR(canConnect(peerTopology));
  auto peerTopo = RdmaTopologyInfo::deserialize(peerTopology).value();
  auto config = config_;
  config.numQps = std::min(peerTopo.numQps, config.numQps);

  return std::make_unique<RdmaTransport>(
      ibvApi_,
      cudaApi_,
      cudaDriverApi_,
      evb_,
      nicsHandle_,
      domainId_,
      config,
      slabPool_);
}

// TODO: get ai_zone_name from fbwhoami / serfwhoami or develop a plugin for
// customized cluster info
std::vector<uint8_t> RdmaTransportFactory::getTopology() {
  std::vector<uint8_t> data(sizeof(RdmaTopologyInfo));
  RdmaTopologyInfo info{kRdmaVersion, config_.numQps};
  std::memcpy(data.data(), &info, sizeof(info));
  return data;
}

Status RdmaTransportFactory::canConnect(std::span<const uint8_t> peerTopology) {
  auto info = RdmaTopologyInfo::deserialize(peerTopology);
  CHECK_RETURN(info);
  if (info->version != kRdmaVersion) {
    return Err(ErrCode::TopologyDisconnect, "Invalid topology version");
  }
  if (info->numQps == 0) {
    return Err(ErrCode::TopologyDisconnect, "Invalid topology numQps");
  }
  return Ok();
}

} // namespace uniflow
