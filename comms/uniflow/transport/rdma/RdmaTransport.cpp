// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaTransport.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/rdma/RdmaRegistrationHandle.h"

#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

namespace uniflow {

// ---------------------------------------------------------------------------
// RdmaTransportInfo serialization
// ---------------------------------------------------------------------------

namespace {

struct RdmaTopologyInfo {
  uint8_t version{kRdmaVersion};
};

const char* getNicName(const NicResources& nic) {
  if (nic.ctx && nic.ctx->device) {
    return nic.ctx->device->name;
  }
  return "(unknown)";
}

} // namespace

TransportInfo RdmaTransportInfo::serialize() const {
  constexpr size_t headerSize = sizeof(RdmaTransportInfo::Header);
  const size_t totalSize = headerSize + nicInfos.size() * sizeof(NicInfo) +
      qpInfos.size() * sizeof(QpInfo);
  TransportInfo data(totalSize);

  size_t offset = 0;
  std::memcpy(data.data() + offset, &header, headerSize);
  offset += headerSize;

  if (!nicInfos.empty()) {
    std::memcpy(
        data.data() + offset,
        nicInfos.data(),
        nicInfos.size() * sizeof(NicInfo));
    offset += nicInfos.size() * sizeof(NicInfo);
  }

  if (!qpInfos.empty()) {
    std::memcpy(
        data.data() + offset, qpInfos.data(), qpInfos.size() * sizeof(QpInfo));
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
      header.numQps * sizeof(QpInfo);
  if (data.size() < expectedSize) {
    return Err(ErrCode::InvalidArgument, "TransportInfo too small for payload");
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
  }

  return info;
}

void RdmaTransportInfo::reset() {
  header = {};
  nicInfos.clear();
  qpInfos.clear();
}

// ---------------------------------------------------------------------------
// RdmaTransport
// ---------------------------------------------------------------------------

RdmaTransport::RdmaTransport(
    std::shared_ptr<IbvApi> ibvApi,
    EventBase* evb,
    std::vector<NicResources> nics,
    uint64_t domainId,
    RdmaTransportConfig config)
    : ibvApi_(std::move(ibvApi)),
      evb_(evb),
      nics_(std::move(nics)),
      config_(config),
      domainId_(domainId) {
  CHECK_THROW_EXCEPTION(!nics_.empty(), std::invalid_argument);
  CHECK_THROW_EXCEPTION(config_.numQps <= 255, std::invalid_argument);
  numPendingCqe_.resize(nics_.size());

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
  const uint32_t numNics = static_cast<uint32_t>(nics_.size());
  const uint32_t numQps = config_.numQps;

  cqs_.reserve(numNics);
  uint32_t qpsPerNic = (numQps + numNics - 1) / numNics;
  for (uint32_t n = 0; n < numNics; ++n) {
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

  info_.header.version = 1;
  info_.header.numQps = static_cast<uint8_t>(numQps);
  info_.header.numNics = static_cast<uint8_t>(numNics);
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
    qpNumToIdx_[qp->qp_num] = i;
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
    rtrAttr.max_dest_rd_atomic = 1;
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
    rtsAttr.max_rd_atomic = 1;

    int rtsMask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

    auto rtsStatus = ibvApi_->modifyQp(qps_[i], &rtsAttr, rtsMask);
    if (rtsStatus.hasError()) {
      UNIFLOW_LOG_ERROR("connect: failed to transition QP {} to RTS", i);
      state_ = TransportState::Error;
      return Err(ErrCode::ConnectionFailed, "Failed to transition QP to RTS");
    }
  }

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
  return rdmaTransfer(requests, IBV_WR_RDMA_WRITE, options);
}

std::future<Status> RdmaTransport::get(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  return rdmaTransfer(requests, IBV_WR_RDMA_READ, options);
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

    // Split into fixed-size chunks. Each chunk becomes one SendWr.
    size_t offset = 0;
    while (offset < reqSize) {
      size_t len = std::min(reqSize - offset, kChunkSize);
      auto& sendWr = wrs->emplace_back();

      sendWr.sge.addr = reinterpret_cast<uint64_t>(req.local.data()) + offset;
      sendWr.sge.length = static_cast<uint32_t>(len);
      sendWr.wr.num_sge = 1;
      sendWr.wr.opcode = opcode;
      sendWr.wr.send_flags = 0; // Signaled only on the last WR per QP.
      sendWr.wr.wr.rdma.remote_addr =
          reinterpret_cast<uint64_t>(req.remote.data()) + offset;
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
    std::shared_ptr<Task>& task) {
  ibv_send_wr* badWr = nullptr;
  auto st = ibvApi_->postSend(qps_[qpIdx], head, &badWr);
  size_t nicIdx = qpIdx % nics_.size();
  if (st) {
    // All WRs posted successfully.
    ++numPendingCqe_[nicIdx];
    numWrsPerQp_[qpIdx] += count;
    task->posted(count);
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
    //   - Its CQE carries wr_id encoding (consumed + 1) in the lower 32 bits.
    //   - pollCompletions decrements numWrsPerQp_ by that count.
    //
    // Counter invariant:
    //   numWrsPerQp_ += (consumed + 1)   [here]
    //   numWrsPerQp_ -= (consumed + 1)   [in pollCompletions when flush CQE
    //   arrives] Net effect: 0. No counter leak.
    uint32_t flushCount = consumed + 1;

    ibv_sge flushSge{};
    ibv_send_wr flushWr{};
    flushWr.wr_id = (static_cast<uint64_t>(taskId) << 32) | flushCount;
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
      task->posted(flushCount);
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

  task->complete(st.error());
  return 0; // Signal to caller that posting failed.
}

Result<uint32_t> RdmaTransport::spray(
    std::vector<SendWr>& wrs,
    size_t& idx,
    uint32_t taskId,
    std::shared_ptr<Task>& task) {
  const uint32_t numQps = static_cast<uint32_t>(qps_.size());
  const uint32_t remaining = static_cast<uint32_t>(wrs.size() - idx);

  // Calculate available capacity per QP.
  uint32_t totalAvail = 0;
  const uint32_t kMaxWr = config_.maxWr;
  std::vector<uint32_t> qpAvail(numQps);
  for (uint32_t q = 0; q < numQps; ++q) {
    qpAvail[q] = kMaxWr - numWrsPerQp_[q];
    totalAvail += qpAvail[q];
  }

  if (totalAvail == 0) {
    return 0; // All QPs full — caller should poll and retry.
  }

  // Weighted distribution: assign chunks to each QP proportional
  // to its available capacity. Ensures QPs with more room get more work,
  // avoiding head-of-line blocking on a full QP.
  std::vector<uint32_t> qpAssigned(numQps, 0);
  uint32_t totalAssigned = 0;
  for (uint32_t q = 0; q < numQps && totalAssigned < remaining; ++q) {
    if (qpAvail[q] == 0) {
      continue;
    }
    uint32_t share = std::max(1u, qpAvail[q] * remaining / totalAvail);
    share = std::min(share, remaining - totalAssigned);
    share = std::min(share, qpAvail[q]);
    qpAssigned[q] = share;
    totalAssigned += share;
  }

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
// Transfer dispatch (caller thread → EventBase thread)
// ---------------------------------------------------------------------------

std::future<Status> RdmaTransport::rdmaTransfer(
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

  // Build SendWrs on the caller thread (validates handles, splits chunks).
  auto wrsResult = buildSendWrs(requests, opcode);
  if (wrsResult.hasError()) {
    UNIFLOW_LOG_ERROR(
        "rdmaTransfer: buildSendWrs failed: {}", wrsResult.error().message());
    return make_ready_future<Status>(std::move(wrsResult).error());
  }
  auto reqWrs = std::move(wrsResult).value();
  assert(!reqWrs->empty());
  UNIFLOW_LOG_DEBUG(
      "rdmaTransfer: {} requests, {} chunks, opcode={}",
      requests.size(),
      reqWrs->size(),
      static_cast<int>(opcode));

  auto task = std::make_shared<Task>();
  auto future = task->get_future();

  // Dispatch posting + polling to the EventBase thread.
  // All state mutations (numWrsPerQp_, inflightTasks_, nextTaskId_)
  // happen exclusively on this thread — no locks needed.
  evb_->dispatch([this,
                  task = std::move(task),
                  reqWrs = std::move(reqWrs)]() mutable noexcept {
    size_t idx = 0;
    uint32_t taskId = nextTaskId_++;
    UNIFLOW_LOG_DEBUG(
        "rdmaTransfer: taskId={} dispatched to EventBase", taskId);
    inflightTasks_[taskId] = task;
    // Note: put and get does not require the post order, but we still use a
    // queue and guarantee the post order, since send & recv requires post-order
    // and it will be helpful for the `flush` implementation in the future.
    pendingRequests_.emplace(taskId, std::move(reqWrs));

    // Self-re-dispatching post loop:
    //   0. If the head of pendingRequests_ is not the current task, re-dispatch
    //   1. Distribute and post WRs across QPs (weighted by capacity)
    //   2. If all posted → mark postFinished, start poll chain
    //   3. If QPs full or partial post → poll to free slots, re-dispatch
    auto postLoop = [this, idx, taskId, task = std::move(task)](
                        auto& self) mutable noexcept -> void {
      auto& [currentId, reqWrs] = pendingRequests_.front();
      if (currentId != taskId) {
        // Previous task is not completed yet. Self dispatch to retry.
        evb_->dispatch(
            [self = std::move(self)]() mutable noexcept { self(self); });
        return;
      }

      if (state_ != TransportState::Connected) {
        UNIFLOW_LOG_ERROR(
            "rdmaTransfer: taskId={} aborted, transport not connected", taskId);
        pendingRequests_.pop();
        pollCompletions(taskId, false);
        return;
      }

      if (idx >= reqWrs->size()) {
        task->postFinished();
      } else {
        auto postResult = spray(*reqWrs, idx, taskId, task);
        if (postResult.hasError()) {
          // postSend failed on a QP. Task is already errored and
          // postFinished. Don't post more — just let the poll chain
          // drain CQEs from earlier successful QPs and the flush WR.
          UNIFLOW_LOG_ERROR("rdmaTransfer: taskId={} spray failed", taskId);
        } else if (idx >= reqWrs->size()) {
          // All chunks posted successfully.
          UNIFLOW_LOG_DEBUG(
              "rdmaTransfer: taskId={} all {} chunks posted",
              taskId,
              reqWrs->size());
          task->postFinished();
        } else {
          // Not all chunks posted (QPs full or partial capacity).
          // Poll to free slots and re-dispatch to retry remaining.
          UNIFLOW_LOG_WARN(
              "rdmaTransfer: taskId={} QPs full, posted {}/{} chunks, retrying",
              taskId,
              idx,
              reqWrs->size());
          pollCompletions(taskId, false);
          evb_->dispatch(
              [self = std::move(self)]() mutable noexcept { self(self); });
          return;
        }
      }

      // requests is posted, remove it from the queue.
      pendingRequests_.pop();

      // Start the poll chain to drain completions — but only if the
      // task still has outstanding WRs. If it's already complete (e.g.,
      // postSend failed with consumed=0), skip the unnecessary poll.
      if (!task->isComplete()) {
        UNIFLOW_LOG_DEBUG(
            "rdmaTransfer: taskId={} starting poll chain", taskId);
        evb_->dispatch(
            [this, taskId]() noexcept { pollCompletions(taskId, true); });
      } else {
        UNIFLOW_LOG_DEBUG(
            "rdmaTransfer: taskId={} completed immediately", taskId);
        inflightTasks_.erase(taskId);
      }
    };
    postLoop(postLoop);
  });

  return future;
}

// ---------------------------------------------------------------------------
// Completion polling (EventBase thread)
// ---------------------------------------------------------------------------

void RdmaTransport::pollCompletions(uint32_t id, bool retry) {
  constexpr int kNumBatch = 16;

  auto cleanup = [this, id]() {
    if (auto it = inflightTasks_.find(id); it != inflightTasks_.end()) {
      it->second->complete(
          Err(ErrCode::TransportError, "RDMA transport is broken"));
      inflightTasks_.erase(it);
    }
  };

  // If transport is broken, fail the task immediately.
  if (state_ != TransportState::Connected) {
    UNIFLOW_LOG_ERROR(
        "pollCompletions: taskId={} aborted, transport not connected", id);
    return cleanup();
  }

  for (size_t i = 0; i < cqs_.size(); ++i) {
    int expected = numPendingCqe_[i];
    // Drain CQ in batches.
    ibv_wc wcs[kNumBatch];
    while (expected > 0) {
      auto pollResult = ibvApi_->pollCq(cqs_[i], kNumBatch, wcs);
      if (pollResult.hasError()) {
        UNIFLOW_LOG_ERROR(
            "pollCompletions: pollCq failed on CQ {}: {}",
            i,
            pollResult.error().message());
        state_ = TransportState::Error;
        return cleanup();
      }

      int n = pollResult.value();
      for (const auto& wc : std::span(wcs).subspan(0, n)) {
        uint32_t taskId = static_cast<uint32_t>(wc.wr_id >> 32);
        uint32_t numWrs = static_cast<uint32_t>(wc.wr_id & 0xffffffff);

        // Decrement expected CQE count only for signaled WRs.
        if (numWrs > 0) {
          --numPendingCqe_[i];
        }

        // Decrement per-QP counter by the encoded WR count.
        // This count includes unsignaled WRs + the signaled WR itself.
        if (auto qp = qpNumToIdx_.find(wc.qp_num); qp != qpNumToIdx_.end()) {
          numWrsPerQp_[qp->second] -= numWrs;
        }

        if (auto it = inflightTasks_.find(taskId); it != inflightTasks_.end()) {
          if (wc.status != IBV_WC_SUCCESS) {
            UNIFLOW_LOG_ERROR(
                "pollCompletions: WC error wr_id={} taskId={} numWrs={} "
                "qp_num={} status={}",
                wc.wr_id,
                taskId,
                numWrs,
                wc.qp_num,
                wc.status);
            it->second->complete(
                Err(ErrCode::DriverError,
                    "RDMA WR failed: wr_id=" + std::to_string(wc.wr_id) +
                        " taskId=" + std::to_string(taskId) +
                        " numWrs=" + std::to_string(numWrs) +
                        " status=" + std::to_string(wc.status)));
          }

          // decrement counter
          it->second->complete(numWrs);
        }
      }

      if (n < kNumBatch) {
        break; // CQ drained for now.
      }

      expected -= kNumBatch;
    }
  }

  // Re-dispatch poll chain if this task is still in-flight.
  if (auto it = inflightTasks_.find(id); it != inflightTasks_.end()) {
    if (it->second->isComplete()) {
      inflightTasks_.erase(it);
    } else if (retry) {
      evb_->dispatch([this, id]() noexcept { pollCompletions(id, true); });
    }
  }
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
  std::promise<Status> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

std::future<Result<size_t>> RdmaTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  std::promise<Result<size_t>> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

std::future<Result<size_t>> RdmaTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  std::promise<Result<size_t>> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

void RdmaTransport::shutdown() {
  if (shutdown_.exchange(true)) {
    return;
  }

  // Make sure that there is no tasks in the evb loop thread related to this
  // transport instance
  if (!evb_->isLoopRunning()) {
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
        if (inflightTasks_.empty() && pendingRequests_.empty()) {
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

  state_ = TransportState::Disconnected;
  UNIFLOW_LOG_INFO("shutdown: complete");
}

// ---------------------------------------------------------------------------
// RdmaTransportFactory
// ---------------------------------------------------------------------------

uint8_t RdmaTransportFactory::findActivePort(ibv_context* ctx) {
  ibv_device_attr devAttr{};
  auto status = ibvApi_->queryDevice(ctx, &devAttr);
  if (status.hasError()) {
    return 0;
  }

  for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
    ibv_port_attr portAttr{};
    auto portStatus = ibvApi_->queryPort(ctx, port, &portAttr);
    if (portStatus.hasError()) {
      continue;
    }
    if (portAttr.state == IBV_PORT_ACTIVE) {
      return port;
    }
  }
  return 0;
}

RdmaTransportFactory::RdmaTransportFactory(
    const std::vector<std::string>& deviceNames,
    EventBase* evb,
    RdmaTransportConfig config,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    std::optional<uint8_t> portNum)
    : TransportFactory(TransportType::RDMA),
      ibvApi_(std::move(ibvApi)),
      cudaDriverApi_(std::move(cudaDriverApi)),
      evb_(evb),
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

  // Generate a random domain id to identify handles from this factory.
  std::mt19937_64 rng{std::random_device{}()};
  domainId_ = rng();

  pageSize_ = sysconf(_SC_PAGESIZE);

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

  // TODO: Replace manual cleanup with EXIT_SCOPE macro (core/Utils.h).
  auto cleanupNics = [this]() {
    for (auto& nic : nics_) {
      if (nic.pd) {
        ibvApi_->deallocPd(nic.pd);
      }
      if (nic.ctx) {
        ibvApi_->closeDevice(nic.ctx);
      }
    }
    nics_.clear();
  };

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
      cleanupNics();
      throw std::runtime_error("RDMA device not found: " + deviceName);
    }

    auto ctxResult = ibvApi_->openDevice(targetDevice);
    if (ctxResult.hasError()) {
      cleanupNics();
      throw std::runtime_error("Failed to open RDMA device: " + deviceName);
    }

    NicResources nic;
    nic.ctx = ctxResult.value();

    // Discover port: use specified portNum, or find first active port
    uint8_t port = portNum.value_or(0);
    if (port == 0) {
      port = findActivePort(nic.ctx);
      if (port == 0) {
        ibvApi_->closeDevice(nic.ctx);
        cleanupNics();
        throw std::runtime_error("No active port found on: " + deviceName);
      }
    }
    nic.portNum = port;

    auto pdResult = ibvApi_->allocPd(nic.ctx);
    if (pdResult.hasError()) {
      ibvApi_->closeDevice(nic.ctx);
      cleanupNics();
      throw std::runtime_error("Failed to allocate PD on: " + deviceName);
    }
    nic.pd = pdResult.value();

    // Probe kernel DMA-BUF support on this NIC's PD.
    auto dmaBufSupported = ibvApi_->isDmaBufSupported(nic.pd);
    CHECK_THROW_ERROR(dmaBufSupported);
    nic.dmaBufSupported = dmaBufSupported.value();

    ibv_port_attr portAttr{};
    auto portStatus = ibvApi_->queryPort(nic.ctx, port, &portAttr);
    if (portStatus.hasError()) {
      ibvApi_->deallocPd(nic.pd);
      ibvApi_->closeDevice(nic.ctx);
      cleanupNics();
      throw std::runtime_error("Failed to query port on: " + deviceName);
    }
    nic.lid = portAttr.lid;
    nic.mtu = portAttr.active_mtu;
    nic.linkLayer = portAttr.link_layer;

    auto gidStatus =
        ibvApi_->queryGid(nic.ctx, port, config_.gidIndex, &nic.gid);
    if (gidStatus.hasError()) {
      ibvApi_->deallocPd(nic.pd);
      ibvApi_->closeDevice(nic.ctx);
      cleanupNics();
      throw std::runtime_error("Failed to query GID on: " + deviceName);
    }

    nics_.push_back(nic);
  }
}

RdmaTransportFactory::~RdmaTransportFactory() {
  for (auto& nic : nics_) {
    if (nic.pd) {
      ibvApi_->deallocPd(nic.pd);
    }
    if (nic.ctx) {
      ibvApi_->closeDevice(nic.ctx);
    }
  }
}

Result<std::unique_ptr<RegistrationHandle>>
RdmaTransportFactory::registerSegment(Segment& segment) {
  if (nics_.empty()) {
    return Err(ErrCode::InvalidArgument, "No NICs available for registration");
  }

  int access =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  // For VRAM, get DMA-BUF fd for GPU Direct RDMA (NIC reads/writes GPU
  // memory directly). For DRAM, use standard ibv_reg_mr.
  // RAII guard ensures the fd is closed when we leave this scope.
  struct FdGuard {
    int fd = -1;
    ~FdGuard() {
      if (fd >= 0) {
        ::close(fd);
      }
    }
  } fdGuard;

  // For cuMemGetHandleForAddressRange, the address must be page-aligned.
  // Align down to page boundary and track the offset for regDmabufMr.
  uint64_t dmaBufOffset = 0;
  uintptr_t addr = reinterpret_cast<uintptr_t>(segment.mutable_data());

  if (segment.memType() == MemoryType::VRAM) {
    auto dmaBufSupported =
        cudaDriverApi_->isDmaBufSupported(segment.deviceId());
    CHECK_RETURN(dmaBufSupported);
    if (dmaBufSupported.value()) {
      const uintptr_t alignedAddr = addr & ~(pageSize_ - 1);
      dmaBufOffset = addr - alignedAddr;
      const size_t dmaBufLen =
          (segment.len() + dmaBufOffset + pageSize_ - 1) & ~(pageSize_ - 1);

      int flags = 0;
      // TODO: set CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE if data direct
      // link is available.
      if (!cudaDriverApi_->cuMemGetHandleForAddressRange(
              &fdGuard.fd,
              static_cast<CUdeviceptr>(alignedAddr),
              dmaBufLen,
              CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
              flags)) {
        // TODO: WARNING LOG without exit, fallback to ibv_reg_mr.
      }
    }
  }

  // Register with every NIC's protection domain so the region is usable
  // across all QPs regardless of which NIC they belong to.
  std::vector<ibv_mr*> mrs;
  mrs.reserve(nics_.size());
  for (const auto& nic : nics_) {
    Result<ibv_mr*> mrResult = Err(ErrCode::NotImplemented);
    if (nic.dmaBufSupported && fdGuard.fd >= 0) {
      // GPU memory: register via DMA-BUF for GPU Direct RDMA.
      mrResult = ibvApi_->regDmabufMr(
          nic.pd,
          dmaBufOffset,
          segment.len(),
          reinterpret_cast<uint64_t>(addr),
          fdGuard.fd,
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

  return std::make_unique<RdmaRegistrationHandle>(
      std::move(mrs), ibvApi_, domainId_);
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
      std::move(rkeys), domainId);
}

Result<std::unique_ptr<Transport>> RdmaTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  CHECK_EXPR(canConnect(peerTopology));
  return std::make_unique<RdmaTransport>(
      ibvApi_, evb_, nics_, domainId_, config_);
}

// TODO: get ai_zone_name from fbwhoami / serfwhoami or develop a plugin for
// customized cluster info
std::vector<uint8_t> RdmaTransportFactory::getTopology() {
  std::vector<uint8_t> data(sizeof(RdmaTopologyInfo));
  RdmaTopologyInfo info;
  std::memcpy(data.data(), &info, sizeof(info));
  return data;
}

Status RdmaTransportFactory::canConnect(std::span<const uint8_t> peerTopology) {
  if (peerTopology.size() != sizeof(RdmaTopologyInfo)) {
    return Err(ErrCode::InvalidArgument, "Invalid topology data");
  }
  RdmaTopologyInfo info;
  std::memcpy(&info, peerTopology.data(), sizeof(info));
  if (info.version != kRdmaVersion) {
    return Err(ErrCode::TopologyDisconnect, "Invalid topology version");
  }
  return Ok();
}

} // namespace uniflow
