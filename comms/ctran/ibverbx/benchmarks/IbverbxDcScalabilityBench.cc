// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * DC vs RC RDMA Scalability Benchmark
 *
 * Compares DC vs RC RDMA write performance across two dimensions:
 *
 * 1. Peer count sweep (N=2..2048): Fixed total input (128 MB) divided
 *    across N-1 receivers, so per-peer message size = 128MB / N.
 *    Tests how transport scales with connection count.
 *
 * 2. Message size sweep (N=2048 fixed): Vary per-peer message size from
 *    64B to 1MB with 2047 receivers. Tests throughput vs message size at
 *    high fan-out.
 *
 * DC variants: 2, 4, 8, 16 DCIs to measure DCI parallelism impact.
 *
 * Setup: device 0 = sender (rank 0), device 1 = N-1 receivers.
 * Sender posts N-1 RDMA writes per iteration.
 *
 * Design for fair comparison:
 * - Shared CQ per device side (no serial per-endpoint CQ polling)
 * - DCI with large SQ depth (8192) for fair batching at high N
 * - Same batch formula for both DC and RC
 */

#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gflags/gflags.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/benchmarks/IbverbxDcBenchUtils.h"
#include "comms/ctran/ibverbx/tests/dc_utils.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

DEFINE_bool(raw_only, true, "Print only RAW CSV results, suppress folly table");

using namespace ibverbx;

namespace {

// Tuning constants
constexpr int kSharedCqDepth = 8192;
constexpr int kDciSqDepth = 8192;
constexpr int kRcQpSqDepth = 1024;
constexpr size_t kTotalInputSize = 128 * 1024 * 1024; // 128 MB
constexpr int kWarmupIters = 10;
constexpr int kMsgSweepNumPeers = 2048;

// Global flags
bool g_rdmaAvailable = false;
bool g_rdmaChecked = false;
bool g_dcAvailable = false;
bool g_dcChecked = false;
std::vector<int> g_dcCapableDevices;

bool checkRdmaAvailable() {
  if (g_rdmaChecked) {
    return g_rdmaAvailable;
  }
  g_rdmaChecked = true;

  ncclCvarInit();
  if (!ibverbx::ibvInit()) {
    XLOG(WARNING) << "Failed to initialize ibverbs";
    return false;
  }

  auto devices =
      ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  if (!devices || devices->empty()) {
    XLOG(WARNING) << "No RDMA devices found";
    return false;
  }

  if (devices->size() < 2) {
    XLOG(WARNING) << "Need at least 2 RDMA devices, found " << devices->size();
    return false;
  }

  g_rdmaAvailable = true;
  return true;
}

bool checkDcAvailable() {
  if (g_dcChecked) {
    return g_dcAvailable;
  }
  g_dcChecked = true;

  if (!checkRdmaAvailable()) {
    return false;
  }

  ncclCvarInit();
  auto devices =
      ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  if (!devices || devices->empty()) {
    return false;
  }

  std::vector<int> dcCapableDevices;
  for (size_t i = 0; i < devices->size(); ++i) {
    DcEndPoint testEndpoint;
    auto initResult = testEndpoint.init(static_cast<int>(i));
    if (!initResult) {
      XLOGF(
          WARNING,
          "DC check: device {} init failed: {}",
          i,
          initResult.error().errStr);
      continue;
    }
    auto dcResult = testEndpoint.initDc();
    if (!dcResult) {
      XLOGF(
          WARNING,
          "DC check: device {} initDc failed: {}",
          i,
          dcResult.error().errStr);
      continue;
    }
    dcCapableDevices.push_back(static_cast<int>(i));
  }

  if (dcCapableDevices.size() < 2) {
    XLOGF(
        WARNING,
        "Need at least 2 DC-capable devices, found {}",
        dcCapableDevices.size());
    return false;
  }

  g_dcCapableDevices = std::move(dcCapableDevices);
  g_dcAvailable = true;
  return true;
}

// ---------------------------------------------------------------------------
// Shared Resources — one per device side (sender / receiver)
// ---------------------------------------------------------------------------
struct SharedResources {
  std::unique_ptr<ibverbx::IbvDevice> device;
  std::unique_ptr<ibverbx::IbvPd> pd;
  std::unique_ptr<ibverbx::IbvCq> cq;
  ibverbx::ibv_gid gid{};

  bool init(int deviceIndex) {
    ncclCvarInit();
    if (!ibverbx::ibvInit()) {
      return false;
    }
    auto devices =
        ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
    if (!devices || deviceIndex >= static_cast<int>(devices->size())) {
      return false;
    }
    device = std::make_unique<ibverbx::IbvDevice>(
        std::move(devices->at(deviceIndex)));

    auto pdResult = device->allocPd();
    if (!pdResult) {
      return false;
    }
    pd = std::make_unique<ibverbx::IbvPd>(std::move(*pdResult));

    auto cqResult = device->createCq(kSharedCqDepth, nullptr, nullptr, 0);
    if (!cqResult) {
      return false;
    }
    cq = std::make_unique<ibverbx::IbvCq>(std::move(*cqResult));

    auto gidResult = device->queryGid(kPortNum, kGidIndex);
    if (!gidResult) {
      return false;
    }
    gid = *gidResult;

    return true;
  }

  void reset() {
    cq.reset();
    pd.reset();
    device.reset();
  }
};

// ---------------------------------------------------------------------------
// Helper: create DCI with configurable SQ depth
// ---------------------------------------------------------------------------
folly::Expected<ibverbx::IbvQp, ibverbx::Error>
createDCILargeSq(ibverbx::IbvPd& pd, ibverbx::IbvCq& cq, int sqDepth) {
  ibverbx::ibv_qp_init_attr_ex initAttr{};
  ibverbx::mlx5dv_qp_init_attr dvInitAttr{};
  memset(&initAttr, 0, sizeof(initAttr));
  memset(&dvInitAttr, 0, sizeof(dvInitAttr));

  initAttr.qp_type = ibverbx::IBV_QPT_DRIVER;
  initAttr.send_cq = cq.cq();
  initAttr.recv_cq = cq.cq();
  initAttr.comp_mask = ibverbx::IBV_QP_INIT_ATTR_PD;
  initAttr.cap.max_send_wr = sqDepth;
  initAttr.cap.max_send_sge = 1;
  initAttr.comp_mask |= ibverbx::IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  initAttr.send_ops_flags = ibverbx::IBV_QP_EX_WITH_RDMA_WRITE;

  dvInitAttr.comp_mask = ibverbx::MLX5DV_QP_INIT_ATTR_MASK_DC;
  dvInitAttr.dc_init_attr.dc_type = ibverbx::MLX5DV_DCTYPE_DCI;

  return pd.createDcQp(&initAttr, &dvInitAttr);
}

// ---------------------------------------------------------------------------
// Helper: create DCI with configurable SQ depth and streams enabled
// ---------------------------------------------------------------------------
folly::Expected<ibverbx::IbvQp, ibverbx::Error> createDCILargeSqWithStreams(
    ibverbx::IbvPd& pd,
    ibverbx::IbvCq& cq,
    int sqDepth,
    uint8_t logNumConcurrent,
    uint8_t logNumErrored = 0) {
  ibverbx::ibv_qp_init_attr_ex initAttr{};
  ibverbx::mlx5dv_qp_init_attr dvInitAttr{};

  initAttr.qp_type = ibverbx::IBV_QPT_DRIVER;
  initAttr.send_cq = cq.cq();
  initAttr.recv_cq = cq.cq();
  initAttr.comp_mask = ibverbx::IBV_QP_INIT_ATTR_PD;
  initAttr.cap.max_send_wr = sqDepth;
  initAttr.cap.max_send_sge = 1;
  initAttr.comp_mask |= ibverbx::IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  initAttr.send_ops_flags = ibverbx::IBV_QP_EX_WITH_RDMA_WRITE;

  dvInitAttr.comp_mask = ibverbx::MLX5DV_QP_INIT_ATTR_MASK_DC |
      ibverbx::MLX5DV_QP_INIT_ATTR_MASK_DCI_STREAMS;
  dvInitAttr.dc_init_attr.dc_type = ibverbx::MLX5DV_DCTYPE_DCI;
  dvInitAttr.dc_init_attr.dci_streams.log_num_concurent = logNumConcurrent;
  dvInitAttr.dc_init_attr.dci_streams.log_num_errored = logNumErrored;

  return pd.createDcQp(&initAttr, &dvInitAttr);
}

// ---------------------------------------------------------------------------
// Helper: busy-spin poll on raw CQ
// ---------------------------------------------------------------------------
bool pollCqBusySpin(
    ibverbx::ibv_cq* rawCq,
    int expected,
    int timeoutMs = 5000) {
  int completed = 0;
  ibverbx::ibv_wc wc{};
  auto start = std::chrono::steady_clock::now();
  while (completed < expected) {
    int n = rawCq->context->ops.poll_cq(rawCq, 1, &wc);
    if (n < 0) {
      XLOGF(ERR, "CQ poll error: returned {}", n);
      return false;
    }
    if (n == 1) {
      if (wc.status != ibverbx::IBV_WC_SUCCESS) {
        XLOGF(
            ERR,
            "WC error: status={}, opcode={}, vendor_err={}",
            wc.status,
            wc.opcode,
            wc.vendor_err);
        return false;
      }
      completed++;
    } else {
      auto elapsed = std::chrono::steady_clock::now() - start;
      if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
              .count() > timeoutMs) {
        XLOGF(
            ERR,
            "CQ busy-spin timeout: got {}/{} completions",
            completed,
            expected);
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Helper: post DC RDMA write via extended QP API
// ---------------------------------------------------------------------------
int postDcRdmaWrite(
    ibverbx::ibv_qp_ex* exQp,
    mlx5dv_qp_ex* dvQp,
    ibverbx::IbvAh& ah,
    const DcBusinessCard& target,
    ibverbx::ibv_sge& sge,
    uint64_t wrId) {
  ibverbx::ibvSymbols.ibv_internal_wr_start(exQp);
  exQp->wr_id = wrId;
  exQp->wr_flags = ibverbx::IBV_SEND_SIGNALED;
  ibverbx::ibvSymbols.ibv_internal_wr_rdma_write(
      exQp, target.rkey, target.remoteAddr);
  ibverbx::ibvSymbols.ibv_internal_wr_set_sge_list(exQp, 1, &sge);
  ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr(
      dvQp, ah.ah(), target.dctNum, DC_KEY);
  return ibverbx::ibvSymbols.ibv_internal_wr_complete(exQp);
}

// ---------------------------------------------------------------------------
// Helper: post DC RDMA write with stream_id via extended QP API
// ---------------------------------------------------------------------------
int postDcRdmaWriteStream(
    ibverbx::ibv_qp_ex* exQp,
    mlx5dv_qp_ex* dvQp,
    ibverbx::IbvAh& ah,
    const DcBusinessCard& target,
    ibverbx::ibv_sge& sge,
    uint64_t wrId,
    uint16_t streamId) {
  ibverbx::ibvSymbols.ibv_internal_wr_start(exQp);
  exQp->wr_id = wrId;
  exQp->wr_flags = ibverbx::IBV_SEND_SIGNALED;
  ibverbx::ibvSymbols.ibv_internal_wr_rdma_write(
      exQp, target.rkey, target.remoteAddr);
  ibverbx::ibvSymbols.ibv_internal_wr_set_sge_list(exQp, 1, &sge);
  ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr_stream(
      dvQp, ah.ah(), target.dctNum, DC_KEY, streamId);
  return ibverbx::ibvSymbols.ibv_internal_wr_complete(exQp);
}

// ---------------------------------------------------------------------------
// Helper: post RC RDMA write via ibv_post_send
// ---------------------------------------------------------------------------
int postRcRdmaWrite(
    ibverbx::IbvQp& qp,
    const RcBusinessCard& target,
    ibverbx::ibv_sge& sge,
    uint64_t wrId) {
  ibverbx::ibv_send_wr wr{};
  wr.wr_id = wrId;
  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = ibverbx::IBV_WR_RDMA_WRITE;
  wr.send_flags = ibverbx::IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = target.remoteAddr;
  wr.wr.rdma.rkey = target.rkey;

  ibverbx::ibv_send_wr* badWr = nullptr;
  auto result = qp.postSend(&wr, badWr);
  return result.hasError() ? -1 : 0;
}

// ---------------------------------------------------------------------------
// Helper: connect RC QP pair (INIT -> RTR -> RTS for both sides)
// ---------------------------------------------------------------------------
bool connectRcQpPair(
    ibverbx::IbvQp& senderQp,
    ibverbx::IbvQp& receiverQp,
    const ibverbx::ibv_gid& senderGid,
    const ibverbx::ibv_gid& receiverGid) {
  // Sender INIT
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = kPortNum;
    attr.qp_access_flags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
        ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_PKEY_INDEX |
        ibverbx::IBV_QP_PORT | ibverbx::IBV_QP_ACCESS_FLAGS;
    if (senderQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  // Receiver INIT
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = kPortNum;
    attr.qp_access_flags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
        ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_PKEY_INDEX |
        ibverbx::IBV_QP_PORT | ibverbx::IBV_QP_ACCESS_FLAGS;
    if (receiverQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  // Sender RTR
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_RTR;
    attr.path_mtu = ibverbx::IBV_MTU_4096;
    attr.dest_qp_num = receiverQp.qp()->qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = receiverGid;
    attr.ah_attr.grh.sgid_index = kGidIndex;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = kPortNum;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_AV |
        ibverbx::IBV_QP_PATH_MTU | ibverbx::IBV_QP_DEST_QPN |
        ibverbx::IBV_QP_RQ_PSN | ibverbx::IBV_QP_MAX_DEST_RD_ATOMIC |
        ibverbx::IBV_QP_MIN_RNR_TIMER;
    if (senderQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  // Receiver RTR
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_RTR;
    attr.path_mtu = ibverbx::IBV_MTU_4096;
    attr.dest_qp_num = senderQp.qp()->qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = senderGid;
    attr.ah_attr.grh.sgid_index = kGidIndex;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = kPortNum;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_AV |
        ibverbx::IBV_QP_PATH_MTU | ibverbx::IBV_QP_DEST_QPN |
        ibverbx::IBV_QP_RQ_PSN | ibverbx::IBV_QP_MAX_DEST_RD_ATOMIC |
        ibverbx::IBV_QP_MIN_RNR_TIMER;
    if (receiverQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  // Sender RTS
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_SQ_PSN |
        ibverbx::IBV_QP_TIMEOUT | ibverbx::IBV_QP_RETRY_CNT |
        ibverbx::IBV_QP_RNR_RETRY | ibverbx::IBV_QP_MAX_QP_RD_ATOMIC;
    if (senderQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  // Receiver RTS
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_SQ_PSN |
        ibverbx::IBV_QP_TIMEOUT | ibverbx::IBV_QP_RETRY_CNT |
        ibverbx::IBV_QP_RNR_RETRY | ibverbx::IBV_QP_MAX_QP_RD_ATOMIC;
    if (receiverQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Raw results collection
// ---------------------------------------------------------------------------
struct RawResult {
  std::string name;
  int numPeers;
  double latencyUs;
  double postUs;
  double pollUs;
  double bwGbps;
  int batch;
  size_t msgBytes;
  int numDcis;
};
std::vector<RawResult> g_rawResults;
std::mutex g_rawResultsMutex;

void recordRawResult(
    const char* name,
    int numPeers,
    double latencyUs,
    double postUs,
    double pollUs,
    double bwGbps,
    int batch,
    size_t msgBytes,
    int numDcis) {
  std::lock_guard<std::mutex> lock(g_rawResultsMutex);
  for (auto& r : g_rawResults) {
    if (r.name == name && r.numPeers == numPeers && r.msgBytes == msgBytes) {
      r = {
          name,
          numPeers,
          latencyUs,
          postUs,
          pollUs,
          bwGbps,
          batch,
          msgBytes,
          numDcis};
      return;
    }
  }
  g_rawResults.push_back(
      {name,
       numPeers,
       latencyUs,
       postUs,
       pollUs,
       bwGbps,
       batch,
       msgBytes,
       numDcis});
}

void printAllRawResults() {
  fprintf(
      stderr,
      "RAW,benchmark,N,latency_us,post_us,poll_us,bw_gbps,batch,msg_bytes,num_dcis\n");
  for (const auto& r : g_rawResults) {
    fprintf(
        stderr,
        "RAW,%s,%d,%.2f,%.2f,%.2f,%.6f,%d,%zu,%d\n",
        r.name.c_str(),
        r.numPeers,
        r.latencyUs,
        r.postUs,
        r.pollUs,
        r.bwGbps,
        r.batch,
        r.msgBytes,
        r.numDcis);
  }
}

// ---------------------------------------------------------------------------
// DC Scalability Benchmark
// ---------------------------------------------------------------------------
static void dcScalabilityCore(
    uint32_t iters,
    size_t numPeers,
    size_t msgSize,
    folly::UserCounters& counters,
    const char* benchName) {
  if (!checkDcAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  int numReceivers = static_cast<int>(numPeers) - 1;
  if (numReceivers < 1) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Setup shared resources on two devices
  SharedResources sender, receiver;
  if (!sender.init(g_dcCapableDevices[0]) ||
      !receiver.init(g_dcCapableDevices[1])) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Create SRQ on receiver side
  auto srqResult = createSRQ(*receiver.pd, 1024);
  if (!srqResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto srq = std::make_unique<ibverbx::IbvSrq>(std::move(*srqResult));

  // Create DCI with large SQ on sender side
  auto dciResult = createDCILargeSq(*sender.pd, *sender.cq, kDciSqDepth);
  if (!dciResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto dci = std::make_unique<ibverbx::IbvQp>(std::move(*dciResult));

  // Get extended QP interface
  auto* exQp = ibverbx::ibvSymbols.ibv_internal_qp_to_qp_ex(dci->qp());
  auto* dvQp = ibverbx::ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(exQp);
  if (!exQp || !dvQp) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Transition DCI to RTS
  auto dciTrans = transitionDCIToRts(*dci, kPortNum, ibverbx::IBV_MTU_4096);
  if (!dciTrans) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Create DCT on receiver side
  auto dctResult = createDCT(*receiver.pd, *receiver.cq, *srq);
  if (!dctResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto dct = std::make_unique<ibverbx::IbvQp>(std::move(*dctResult));

  // Transition DCT to RTR
  auto dctTrans = transitionDCTToRtr(*dct, kPortNum, ibverbx::IBV_MTU_4096);
  if (!dctTrans) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Allocate sender buffer and MR
  std::vector<uint8_t> senderBuf(msgSize, 0xAA);
  auto accessFlags = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);
  auto senderMrResult =
      sender.pd->regMr(senderBuf.data(), senderBuf.size(), accessFlags);
  if (!senderMrResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto senderMr = std::make_unique<ibverbx::IbvMr>(std::move(*senderMrResult));

  ibverbx::ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(senderBuf.data());
  sge.length = static_cast<uint32_t>(msgSize);
  sge.lkey = senderMr->mr()->lkey;

  // Create N-1 receiver buffers, MRs, business cards, and AHs
  std::vector<std::vector<uint8_t>> recvBufs(numReceivers);
  std::vector<std::unique_ptr<ibverbx::IbvMr>> recvMrs;
  std::vector<DcBusinessCard> recvCards;
  std::vector<std::unique_ptr<ibverbx::IbvAh>> ahs;

  for (int i = 0; i < numReceivers; ++i) {
    recvBufs[i].resize(msgSize, 0x00);
    auto mrResult =
        receiver.pd->regMr(recvBufs[i].data(), recvBufs[i].size(), accessFlags);
    if (!mrResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    recvMrs.push_back(std::make_unique<ibverbx::IbvMr>(std::move(*mrResult)));

    DcBusinessCard card{};
    card.mtu = 5;
    card.dctNum = dct->qp()->qp_num;
    card.port = kPortNum;
    card.subnetPrefix = receiver.gid.global.subnet_prefix;
    card.interfaceId = receiver.gid.global.interface_id;
    card.rank = i;
    card.remoteAddr = reinterpret_cast<uint64_t>(recvBufs[i].data());
    card.rkey = recvMrs.back()->mr()->rkey;
    recvCards.push_back(card);

    auto ahResult = createAddressHandle(*sender.pd, card);
    if (!ahResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    ahs.push_back(std::make_unique<ibverbx::IbvAh>(std::move(*ahResult)));
  }

  // Batch size: cap by both SQ depth and CQ depth
  int batch = std::max(1, std::min(kDciSqDepth, kSharedCqDepth / numReceivers));

  // Warmup
  for (int w = 0; w < kWarmupIters; ++w) {
    int rem = numReceivers;
    int idx = 0;
    while (rem > 0) {
      int b = std::min(rem, batch);
      for (int j = 0; j < b; ++j) {
        if (postDcRdmaWrite(exQp, dvQp, *ahs[idx], recvCards[idx], sge, idx) !=
            0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      if (!pollCqBusySpin(sender.cq->cq(), b)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      rem -= b;
    }
  }

  // Timed iterations
  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds totalPostNs{0};
  std::chrono::nanoseconds totalPollNs{0};

  for (uint32_t iter = 0; iter < iters; ++iter) {
    int remaining = numReceivers;
    int idx = 0;
    while (remaining > 0) {
      int thisBatch = std::min(remaining, batch);

      auto postStart = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < thisBatch; ++j) {
        if (postDcRdmaWrite(exQp, dvQp, *ahs[idx], recvCards[idx], sge, idx) !=
            0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      auto postEnd = std::chrono::high_resolution_clock::now();
      totalPostNs += (postEnd - postStart);

      auto pollStart = std::chrono::high_resolution_clock::now();
      if (!pollCqBusySpin(sender.cq->cq(), thisBatch)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      auto pollEnd = std::chrono::high_resolution_clock::now();
      totalPollNs += (pollEnd - pollStart);

      remaining -= thisBatch;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count() /
      1000.0;

  double avgLatencyUs = elapsedUs / iters;
  double avgPostUs = totalPostNs.count() / 1000.0 / iters;
  double avgPollUs = totalPollNs.count() / 1000.0 / iters;
  double bandwidthGBps =
      (msgSize * numReceivers * iters / 1e9) / (elapsedUs / 1e6);

  counters["latency_us"] =
      folly::UserMetric(avgLatencyUs, folly::UserMetric::Type::METRIC);
  counters["post_us"] =
      folly::UserMetric(avgPostUs, folly::UserMetric::Type::METRIC);
  counters["poll_us"] =
      folly::UserMetric(avgPollUs, folly::UserMetric::Type::METRIC);
  counters["bw_gbps"] =
      folly::UserMetric(bandwidthGBps, folly::UserMetric::Type::METRIC);
  counters["batch"] = folly::UserMetric(batch, folly::UserMetric::Type::METRIC);
  counters["N"] = folly::UserMetric(numPeers, folly::UserMetric::Type::METRIC);
  recordRawResult(
      benchName,
      static_cast<int>(numPeers),
      avgLatencyUs,
      avgPostUs,
      avgPollUs,
      bandwidthGBps,
      batch,
      msgSize,
      1);
}

// Wrapper: peer count sweep (fixed total input, msg size = 128MB / N)
static void
dcScalability(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  dcScalabilityCore(
      iters, numPeers, kTotalInputSize / numPeers, counters, "dcScalability");
}

// Wrapper: message size sweep (fixed N=2048, varying per-peer msg size)
static void
dcMsgSweep(uint32_t iters, size_t msgSize, folly::UserCounters& counters) {
  dcScalabilityCore(iters, kMsgSweepNumPeers, msgSize, counters, "dcMsgSweep");
}

// ---------------------------------------------------------------------------
// DC Multi-DCI Scalability Benchmark
//
// Same as dcScalability but with multiple DCIs and DCTs to test whether
// NIC-level QP parallelism recovers performance.
// ---------------------------------------------------------------------------
static void dcMultiDciCore(
    uint32_t iters,
    size_t numPeers,
    size_t msgSize,
    int numDcis,
    folly::UserCounters& counters,
    const char* benchName,
    uint8_t logNumConcurrentStreams = 0) {
  if (!checkDcAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  bool useStreams = logNumConcurrentStreams > 0;
  if (useStreams &&
      !ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr_stream) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  int numStreamsPerDci = useStreams ? (1 << logNumConcurrentStreams) : 0;

  int numReceivers = static_cast<int>(numPeers) - 1;
  if (numReceivers < 1) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Cap DCIs at numReceivers (can't have more DCIs than targets)
  numDcis = std::min(numDcis, numReceivers);

  // Setup shared resources on two devices
  SharedResources sender, receiver;
  if (!sender.init(g_dcCapableDevices[0]) ||
      !receiver.init(g_dcCapableDevices[1])) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Create SRQ on receiver side (shared by all DCTs)
  auto srqResult = createSRQ(*receiver.pd, 1024);
  if (!srqResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto srq = std::make_unique<ibverbx::IbvSrq>(std::move(*srqResult));

  auto accessFlags = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);

  // Create numDcis DCIs on sender side
  struct DciState {
    std::unique_ptr<ibverbx::IbvQp> qp;
    ibverbx::ibv_qp_ex* exQp = nullptr;
    mlx5dv_qp_ex* dvQp = nullptr;
  };
  std::vector<DciState> dcis(numDcis);
  for (int d = 0; d < numDcis; ++d) {
    auto dciResult = useStreams
        ? createDCILargeSqWithStreams(
              *sender.pd, *sender.cq, kDciSqDepth, logNumConcurrentStreams)
        : createDCILargeSq(*sender.pd, *sender.cq, kDciSqDepth);
    if (!dciResult) {
      if (useStreams) {
        // Streams not supported on this HW
        counters["skipped"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      } else {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      }
      return;
    }
    dcis[d].qp = std::make_unique<ibverbx::IbvQp>(std::move(*dciResult));
    dcis[d].exQp =
        ibverbx::ibvSymbols.ibv_internal_qp_to_qp_ex(dcis[d].qp->qp());
    dcis[d].dvQp =
        ibverbx::ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(dcis[d].exQp);
    if (!dcis[d].exQp || !dcis[d].dvQp) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    auto dciTrans =
        transitionDCIToRts(*dcis[d].qp, kPortNum, ibverbx::IBV_MTU_4096);
    if (!dciTrans) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
  }

  // Create numDcis DCTs on receiver side (sharing SRQ)
  std::vector<std::unique_ptr<ibverbx::IbvQp>> dcts;
  for (int d = 0; d < numDcis; ++d) {
    auto dctResult = createDCT(*receiver.pd, *receiver.cq, *srq);
    if (!dctResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    dcts.push_back(std::make_unique<ibverbx::IbvQp>(std::move(*dctResult)));
    auto dctTrans =
        transitionDCTToRtr(*dcts.back(), kPortNum, ibverbx::IBV_MTU_4096);
    if (!dctTrans) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
  }

  // Allocate sender buffer and MR (shared across all DCIs)
  std::vector<uint8_t> senderBuf(msgSize, 0xAA);
  auto senderMrResult =
      sender.pd->regMr(senderBuf.data(), senderBuf.size(), accessFlags);
  if (!senderMrResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto senderMr = std::make_unique<ibverbx::IbvMr>(std::move(*senderMrResult));

  ibverbx::ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(senderBuf.data());
  sge.length = static_cast<uint32_t>(msgSize);
  sge.lkey = senderMr->mr()->lkey;

  // Create N-1 receiver buffers, MRs, business cards, and AHs
  // Receivers are round-robin'd across DCTs
  std::vector<std::vector<uint8_t>> recvBufs(numReceivers);
  std::vector<std::unique_ptr<ibverbx::IbvMr>> recvMrs;
  std::vector<DcBusinessCard> recvCards;
  std::vector<std::unique_ptr<ibverbx::IbvAh>> ahs;

  for (int i = 0; i < numReceivers; ++i) {
    recvBufs[i].resize(msgSize, 0x00);
    auto mrResult =
        receiver.pd->regMr(recvBufs[i].data(), recvBufs[i].size(), accessFlags);
    if (!mrResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    recvMrs.push_back(std::make_unique<ibverbx::IbvMr>(std::move(*mrResult)));

    DcBusinessCard card{};
    card.mtu = 5;
    card.dctNum = dcts[i % numDcis]->qp()->qp_num; // round-robin DCT
    card.port = kPortNum;
    card.subnetPrefix = receiver.gid.global.subnet_prefix;
    card.interfaceId = receiver.gid.global.interface_id;
    card.rank = i;
    card.remoteAddr = reinterpret_cast<uint64_t>(recvBufs[i].data());
    card.rkey = recvMrs.back()->mr()->rkey;
    recvCards.push_back(card);

    auto ahResult = createAddressHandle(*sender.pd, card);
    if (!ahResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    ahs.push_back(std::make_unique<ibverbx::IbvAh>(std::move(*ahResult)));
  }

  // Batch size: cap by CQ depth
  int batch = std::max(1, std::min(kDciSqDepth, kSharedCqDepth / numReceivers));

  // Warmup
  for (int w = 0; w < kWarmupIters; ++w) {
    int rem = numReceivers;
    int idx = 0;
    while (rem > 0) {
      int b = std::min(rem, batch);
      for (int j = 0; j < b; ++j) {
        int d = idx % numDcis;
        int ret = useStreams
            ? postDcRdmaWriteStream(
                  dcis[d].exQp,
                  dcis[d].dvQp,
                  *ahs[idx],
                  recvCards[idx],
                  sge,
                  idx,
                  static_cast<uint16_t>(idx % numStreamsPerDci))
            : postDcRdmaWrite(
                  dcis[d].exQp,
                  dcis[d].dvQp,
                  *ahs[idx],
                  recvCards[idx],
                  sge,
                  idx);
        if (ret != 0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      if (!pollCqBusySpin(sender.cq->cq(), b)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      rem -= b;
    }
  }

  // Timed iterations
  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds totalPostNs{0};
  std::chrono::nanoseconds totalPollNs{0};

  for (uint32_t iter = 0; iter < iters; ++iter) {
    int remaining = numReceivers;
    int idx = 0;
    while (remaining > 0) {
      int thisBatch = std::min(remaining, batch);

      auto postStart = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < thisBatch; ++j) {
        int d = idx % numDcis;
        int ret = useStreams
            ? postDcRdmaWriteStream(
                  dcis[d].exQp,
                  dcis[d].dvQp,
                  *ahs[idx],
                  recvCards[idx],
                  sge,
                  idx,
                  static_cast<uint16_t>(idx % numStreamsPerDci))
            : postDcRdmaWrite(
                  dcis[d].exQp,
                  dcis[d].dvQp,
                  *ahs[idx],
                  recvCards[idx],
                  sge,
                  idx);
        if (ret != 0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      auto postEnd = std::chrono::high_resolution_clock::now();
      totalPostNs += (postEnd - postStart);

      auto pollStart = std::chrono::high_resolution_clock::now();
      if (!pollCqBusySpin(sender.cq->cq(), thisBatch)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      auto pollEnd = std::chrono::high_resolution_clock::now();
      totalPollNs += (pollEnd - pollStart);

      remaining -= thisBatch;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count() /
      1000.0;

  double avgLatencyUs = elapsedUs / iters;
  double avgPostUs = totalPostNs.count() / 1000.0 / iters;
  double avgPollUs = totalPollNs.count() / 1000.0 / iters;
  double bandwidthGBps =
      (msgSize * numReceivers * iters / 1e9) / (elapsedUs / 1e6);

  counters["latency_us"] =
      folly::UserMetric(avgLatencyUs, folly::UserMetric::Type::METRIC);
  counters["post_us"] =
      folly::UserMetric(avgPostUs, folly::UserMetric::Type::METRIC);
  counters["poll_us"] =
      folly::UserMetric(avgPollUs, folly::UserMetric::Type::METRIC);
  counters["bw_gbps"] =
      folly::UserMetric(bandwidthGBps, folly::UserMetric::Type::METRIC);
  counters["batch"] = folly::UserMetric(batch, folly::UserMetric::Type::METRIC);
  counters["N"] = folly::UserMetric(numPeers, folly::UserMetric::Type::METRIC);
  counters["num_dcis"] =
      folly::UserMetric(numDcis, folly::UserMetric::Type::METRIC);

  recordRawResult(
      benchName,
      static_cast<int>(numPeers),
      avgLatencyUs,
      avgPostUs,
      avgPollUs,
      bandwidthGBps,
      batch,
      msgSize,
      numDcis);
}

// Wrappers: peer count sweep with specific DCI counts
static void
dcMultiDci2(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  dcMultiDciCore(
      iters, numPeers, kTotalInputSize / numPeers, 2, counters, "dcMultiDci_2");
}
static void
dcMultiDci4(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  dcMultiDciCore(
      iters, numPeers, kTotalInputSize / numPeers, 4, counters, "dcMultiDci_4");
}
static void
dcMultiDci8(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  dcMultiDciCore(
      iters, numPeers, kTotalInputSize / numPeers, 8, counters, "dcMultiDci_8");
}
static void
dcMultiDci16(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  dcMultiDciCore(
      iters,
      numPeers,
      kTotalInputSize / numPeers,
      16,
      counters,
      "dcMultiDci_16");
}

// Wrappers: multi-DCI with streams (4 concurrent streams per DCI)
static void dcMultiDci4Streams4(
    uint32_t iters,
    size_t numPeers,
    folly::UserCounters& counters) {
  dcMultiDciCore(
      iters,
      numPeers,
      kTotalInputSize / numPeers,
      4,
      counters,
      "dcMultiDci4_streams4",
      2); // log2(4) = 2
}

// Wrapper: message size sweep with 4 DCIs at N=2048
static void dcMultiDci4MsgSweep(
    uint32_t iters,
    size_t msgSize,
    folly::UserCounters& counters) {
  dcMultiDciCore(
      iters, kMsgSweepNumPeers, msgSize, 4, counters, "dcMultiDci4MsgSweep");
}

// ---------------------------------------------------------------------------
// DC Streams Scalability Benchmark
//
// Uses a single DCI with DCI Streams enabled. Each receiver is assigned a
// unique stream_id, so the NIC can process writes to different targets
// concurrently within the single DCI. Compares whether stream-level
// concurrency provides similar benefits to having multiple DCIs.
// ---------------------------------------------------------------------------
static void dcStreamsScalabilityCore(
    uint32_t iters,
    size_t numPeers,
    size_t msgSize,
    uint8_t logNumConcurrent,
    folly::UserCounters& counters,
    const char* benchName) {
  if (!checkDcAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  if (!ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr_stream) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  int numReceivers = static_cast<int>(numPeers) - 1;
  if (numReceivers < 1) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Setup shared resources on two devices
  SharedResources sender, receiver;
  if (!sender.init(g_dcCapableDevices[0]) ||
      !receiver.init(g_dcCapableDevices[1])) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Create SRQ on receiver side
  auto srqResult = createSRQ(*receiver.pd, 1024);
  if (!srqResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto srq = std::make_unique<ibverbx::IbvSrq>(std::move(*srqResult));

  // Create DCI with streams enabled
  auto dciResult = createDCILargeSqWithStreams(
      *sender.pd, *sender.cq, kDciSqDepth, logNumConcurrent);
  if (!dciResult) {
    // DCI Streams not supported on this HW — skip
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto dci = std::make_unique<ibverbx::IbvQp>(std::move(*dciResult));

  // Get extended QP interface
  auto* exQp = ibverbx::ibvSymbols.ibv_internal_qp_to_qp_ex(dci->qp());
  auto* dvQp = ibverbx::ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(exQp);
  if (!exQp || !dvQp) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Transition DCI to RTS
  auto dciTrans = transitionDCIToRts(*dci, kPortNum, ibverbx::IBV_MTU_4096);
  if (!dciTrans) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Create DCT on receiver side
  auto dctResult = createDCT(*receiver.pd, *receiver.cq, *srq);
  if (!dctResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto dct = std::make_unique<ibverbx::IbvQp>(std::move(*dctResult));

  // Transition DCT to RTR
  auto dctTrans = transitionDCTToRtr(*dct, kPortNum, ibverbx::IBV_MTU_4096);
  if (!dctTrans) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Allocate sender buffer and MR
  std::vector<uint8_t> senderBuf(msgSize, 0xAA);
  auto accessFlags = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);
  auto senderMrResult =
      sender.pd->regMr(senderBuf.data(), senderBuf.size(), accessFlags);
  if (!senderMrResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto senderMr = std::make_unique<ibverbx::IbvMr>(std::move(*senderMrResult));

  ibverbx::ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(senderBuf.data());
  sge.length = static_cast<uint32_t>(msgSize);
  sge.lkey = senderMr->mr()->lkey;

  // Create N-1 receiver buffers, MRs, business cards, and AHs
  std::vector<std::vector<uint8_t>> recvBufs(numReceivers);
  std::vector<std::unique_ptr<ibverbx::IbvMr>> recvMrs;
  std::vector<DcBusinessCard> recvCards;
  std::vector<std::unique_ptr<ibverbx::IbvAh>> ahs;

  for (int i = 0; i < numReceivers; ++i) {
    recvBufs[i].resize(msgSize, 0x00);
    auto mrResult =
        receiver.pd->regMr(recvBufs[i].data(), recvBufs[i].size(), accessFlags);
    if (!mrResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    recvMrs.push_back(std::make_unique<ibverbx::IbvMr>(std::move(*mrResult)));

    DcBusinessCard card{};
    card.mtu = 5;
    card.dctNum = dct->qp()->qp_num;
    card.port = kPortNum;
    card.subnetPrefix = receiver.gid.global.subnet_prefix;
    card.interfaceId = receiver.gid.global.interface_id;
    card.rank = i;
    card.remoteAddr = reinterpret_cast<uint64_t>(recvBufs[i].data());
    card.rkey = recvMrs.back()->mr()->rkey;
    recvCards.push_back(card);

    auto ahResult = createAddressHandle(*sender.pd, card);
    if (!ahResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    ahs.push_back(std::make_unique<ibverbx::IbvAh>(std::move(*ahResult)));
  }

  // Number of concurrent streams
  int numStreams = 1 << logNumConcurrent;

  // Batch size: cap by SQ depth and CQ depth
  int batch = std::max(1, std::min(kDciSqDepth, kSharedCqDepth / numReceivers));

  // Warmup
  for (int w = 0; w < kWarmupIters; ++w) {
    int rem = numReceivers;
    int idx = 0;
    while (rem > 0) {
      int b = std::min(rem, batch);
      for (int j = 0; j < b; ++j) {
        uint16_t streamId = static_cast<uint16_t>(idx % numStreams);
        if (postDcRdmaWriteStream(
                exQp, dvQp, *ahs[idx], recvCards[idx], sge, idx, streamId) !=
            0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      if (!pollCqBusySpin(sender.cq->cq(), b)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      rem -= b;
    }
  }

  // Timed iterations
  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds totalPostNs{0};
  std::chrono::nanoseconds totalPollNs{0};

  for (uint32_t iter = 0; iter < iters; ++iter) {
    int remaining = numReceivers;
    int idx = 0;
    while (remaining > 0) {
      int thisBatch = std::min(remaining, batch);

      auto postStart = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < thisBatch; ++j) {
        uint16_t streamId = static_cast<uint16_t>(idx % numStreams);
        if (postDcRdmaWriteStream(
                exQp, dvQp, *ahs[idx], recvCards[idx], sge, idx, streamId) !=
            0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      auto postEnd = std::chrono::high_resolution_clock::now();
      totalPostNs += (postEnd - postStart);

      auto pollStart = std::chrono::high_resolution_clock::now();
      if (!pollCqBusySpin(sender.cq->cq(), thisBatch)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      auto pollEnd = std::chrono::high_resolution_clock::now();
      totalPollNs += (pollEnd - pollStart);

      remaining -= thisBatch;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count() /
      1000.0;

  double avgLatencyUs = elapsedUs / iters;
  double avgPostUs = totalPostNs.count() / 1000.0 / iters;
  double avgPollUs = totalPollNs.count() / 1000.0 / iters;
  double bandwidthGBps =
      (msgSize * numReceivers * iters / 1e9) / (elapsedUs / 1e6);

  counters["latency_us"] =
      folly::UserMetric(avgLatencyUs, folly::UserMetric::Type::METRIC);
  counters["post_us"] =
      folly::UserMetric(avgPostUs, folly::UserMetric::Type::METRIC);
  counters["poll_us"] =
      folly::UserMetric(avgPollUs, folly::UserMetric::Type::METRIC);
  counters["bw_gbps"] =
      folly::UserMetric(bandwidthGBps, folly::UserMetric::Type::METRIC);
  counters["batch"] = folly::UserMetric(batch, folly::UserMetric::Type::METRIC);
  counters["N"] = folly::UserMetric(numPeers, folly::UserMetric::Type::METRIC);
  counters["num_streams"] =
      folly::UserMetric(numStreams, folly::UserMetric::Type::METRIC);
  recordRawResult(
      benchName,
      static_cast<int>(numPeers),
      avgLatencyUs,
      avgPostUs,
      avgPollUs,
      bandwidthGBps,
      batch,
      msgSize,
      1); // 1 DCI with streams
}

// Wrappers: DC with streams, varying concurrent stream counts
static void
dcStreams4(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  dcStreamsScalabilityCore(
      iters,
      numPeers,
      kTotalInputSize / numPeers,
      2, // log2(4) = 2
      counters,
      "dcStreams_4");
}
static void
dcStreams16(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  dcStreamsScalabilityCore(
      iters,
      numPeers,
      kTotalInputSize / numPeers,
      4, // log2(16) = 4
      counters,
      "dcStreams_16");
}

// ---------------------------------------------------------------------------
// RC Scalability Benchmark
// ---------------------------------------------------------------------------
static void rcScalabilityCore(
    uint32_t iters,
    size_t numPeers,
    size_t msgSize,
    folly::UserCounters& counters,
    const char* benchName) {
  if (!checkDcAvailable() && !checkRdmaAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  int numConnections = static_cast<int>(numPeers) - 1;
  if (numConnections < 1) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Use same devices as DC for fair comparison
  int senderDevIdx = g_dcCapableDevices.size() >= 2 ? g_dcCapableDevices[0] : 0;
  int receiverDevIdx =
      g_dcCapableDevices.size() >= 2 ? g_dcCapableDevices[1] : 1;

  // Setup shared resources on two devices
  SharedResources sender, receiver;
  if (!sender.init(senderDevIdx) || !receiver.init(receiverDevIdx)) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  auto accessFlags = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);

  // Allocate sender buffer and MR
  std::vector<uint8_t> senderBuf(msgSize, 0xAA);
  auto senderMrResult =
      sender.pd->regMr(senderBuf.data(), senderBuf.size(), accessFlags);
  if (!senderMrResult) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }
  auto senderMr = std::make_unique<ibverbx::IbvMr>(std::move(*senderMrResult));

  ibverbx::ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(senderBuf.data());
  sge.length = static_cast<uint32_t>(msgSize);
  sge.lkey = senderMr->mr()->lkey;

  // Create N-1 receiver buffers and MRs
  std::vector<std::vector<uint8_t>> recvBufs(numConnections);
  std::vector<std::unique_ptr<ibverbx::IbvMr>> recvMrs;

  for (int i = 0; i < numConnections; ++i) {
    recvBufs[i].resize(msgSize, 0x00);
    auto mrResult =
        receiver.pd->regMr(recvBufs[i].data(), recvBufs[i].size(), accessFlags);
    if (!mrResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    recvMrs.push_back(std::make_unique<ibverbx::IbvMr>(std::move(*mrResult)));
  }

  // Create N-1 QP pairs on shared resources
  std::vector<std::unique_ptr<ibverbx::IbvQp>> senderQps;
  std::vector<std::unique_ptr<ibverbx::IbvQp>> receiverQps;
  std::vector<RcBusinessCard> rcCards;

  for (int i = 0; i < numConnections; ++i) {
    // Sender QP on shared PD/CQ
    ibverbx::ibv_qp_init_attr sInitAttr{};
    sInitAttr.send_cq = sender.cq->cq();
    sInitAttr.recv_cq = sender.cq->cq();
    sInitAttr.qp_type = ibverbx::IBV_QPT_RC;
    sInitAttr.cap.max_send_wr = kRcQpSqDepth;
    sInitAttr.cap.max_recv_wr = 1024;
    sInitAttr.cap.max_send_sge = 1;
    sInitAttr.cap.max_recv_sge = 1;
    auto sqpResult = sender.pd->createQp(&sInitAttr);
    if (!sqpResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    senderQps.push_back(
        std::make_unique<ibverbx::IbvQp>(std::move(*sqpResult)));

    // Receiver QP on shared PD/CQ
    ibverbx::ibv_qp_init_attr rInitAttr{};
    rInitAttr.send_cq = receiver.cq->cq();
    rInitAttr.recv_cq = receiver.cq->cq();
    rInitAttr.qp_type = ibverbx::IBV_QPT_RC;
    rInitAttr.cap.max_send_wr = kRcQpSqDepth;
    rInitAttr.cap.max_recv_wr = 1024;
    rInitAttr.cap.max_send_sge = 1;
    rInitAttr.cap.max_recv_sge = 1;
    auto rqpResult = receiver.pd->createQp(&rInitAttr);
    if (!rqpResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    receiverQps.push_back(
        std::make_unique<ibverbx::IbvQp>(std::move(*rqpResult)));

    // Connect the pair
    if (!connectRcQpPair(
            *senderQps.back(), *receiverQps.back(), sender.gid, receiver.gid)) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }

    // Build business card for this receiver
    RcBusinessCard card{};
    card.qpNum = receiverQps.back()->qp()->qp_num;
    card.port = kPortNum;
    card.subnetPrefix = receiver.gid.global.subnet_prefix;
    card.interfaceId = receiver.gid.global.interface_id;
    card.remoteAddr = reinterpret_cast<uint64_t>(recvBufs[i].data());
    card.rkey = recvMrs[i]->mr()->rkey;
    card.psn = 0;
    rcCards.push_back(card);
  }

  // Batch size: same formula structure as DC
  int batch =
      std::max(1, std::min(kRcQpSqDepth, kSharedCqDepth / numConnections));

  // Warmup
  for (int w = 0; w < kWarmupIters; ++w) {
    int rem = numConnections;
    int idx = 0;
    while (rem > 0) {
      int b = std::min(rem, batch);
      for (int j = 0; j < b; ++j) {
        if (postRcRdmaWrite(*senderQps[idx], rcCards[idx], sge, idx) != 0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      if (!pollCqBusySpin(sender.cq->cq(), b)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      rem -= b;
    }
  }

  // Timed iterations
  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds totalPostNs{0};
  std::chrono::nanoseconds totalPollNs{0};

  for (uint32_t iter = 0; iter < iters; ++iter) {
    int remaining = numConnections;
    int idx = 0;
    while (remaining > 0) {
      int thisBatch = std::min(remaining, batch);

      auto postStart = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < thisBatch; ++j) {
        if (postRcRdmaWrite(*senderQps[idx], rcCards[idx], sge, idx) != 0) {
          counters["error"] =
              folly::UserMetric(1, folly::UserMetric::Type::METRIC);
          return;
        }
        idx++;
      }
      auto postEnd = std::chrono::high_resolution_clock::now();
      totalPostNs += (postEnd - postStart);

      auto pollStart = std::chrono::high_resolution_clock::now();
      if (!pollCqBusySpin(sender.cq->cq(), thisBatch)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
      auto pollEnd = std::chrono::high_resolution_clock::now();
      totalPollNs += (pollEnd - pollStart);

      remaining -= thisBatch;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count() /
      1000.0;

  double avgLatencyUs = elapsedUs / iters;
  double avgPostUs = totalPostNs.count() / 1000.0 / iters;
  double avgPollUs = totalPollNs.count() / 1000.0 / iters;
  double bandwidthGBps =
      (msgSize * numConnections * iters / 1e9) / (elapsedUs / 1e6);

  counters["latency_us"] =
      folly::UserMetric(avgLatencyUs, folly::UserMetric::Type::METRIC);
  counters["post_us"] =
      folly::UserMetric(avgPostUs, folly::UserMetric::Type::METRIC);
  counters["poll_us"] =
      folly::UserMetric(avgPollUs, folly::UserMetric::Type::METRIC);
  counters["bw_gbps"] =
      folly::UserMetric(bandwidthGBps, folly::UserMetric::Type::METRIC);
  counters["batch"] = folly::UserMetric(batch, folly::UserMetric::Type::METRIC);
  counters["N"] = folly::UserMetric(numPeers, folly::UserMetric::Type::METRIC);
  recordRawResult(
      benchName,
      static_cast<int>(numPeers),
      avgLatencyUs,
      avgPostUs,
      avgPollUs,
      bandwidthGBps,
      batch,
      msgSize,
      0);
}

// Wrapper: peer count sweep (fixed total input, msg size = 128MB / N)
static void
rcScalability(uint32_t iters, size_t numPeers, folly::UserCounters& counters) {
  rcScalabilityCore(
      iters, numPeers, kTotalInputSize / numPeers, counters, "rcScalability");
}

// Wrapper: message size sweep (fixed N=2048, varying per-peer msg size)
static void
rcMsgSweep(uint32_t iters, size_t msgSize, folly::UserCounters& counters) {
  rcScalabilityCore(iters, kMsgSweepNumPeers, msgSize, counters, "rcMsgSweep");
}

} // namespace

// ---------------------------------------------------------------------------
// Benchmark Registration
// ---------------------------------------------------------------------------

#define REGISTER_SCALABILITY_BENCH(func, name, numPeers) \
  BENCHMARK_MULTI_PARAM_COUNTERS(func, name, numPeers)

// DC Scalability: varying peer count
REGISTER_SCALABILITY_BENCH(dcScalability, N2, 2);
REGISTER_SCALABILITY_BENCH(dcScalability, N4, 4);
REGISTER_SCALABILITY_BENCH(dcScalability, N8, 8);
REGISTER_SCALABILITY_BENCH(dcScalability, N16, 16);
REGISTER_SCALABILITY_BENCH(dcScalability, N32, 32);
REGISTER_SCALABILITY_BENCH(dcScalability, N64, 64);
REGISTER_SCALABILITY_BENCH(dcScalability, N128, 128);
REGISTER_SCALABILITY_BENCH(dcScalability, N256, 256);
REGISTER_SCALABILITY_BENCH(dcScalability, N512, 512);
REGISTER_SCALABILITY_BENCH(dcScalability, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcScalability, N2048, 2048);

BENCHMARK_DRAW_LINE();

// DC Multi-DCI (2 DCIs): test if even 2 DCIs helps
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N2, 2);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N4, 4);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N8, 8);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N16, 16);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N32, 32);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N64, 64);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N128, 128);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N256, 256);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N512, 512);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcMultiDci2, N2048, 2048);

BENCHMARK_DRAW_LINE();

// DC Multi-DCI (4 DCIs): test intermediate parallelism
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N2, 2);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N4, 4);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N8, 8);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N16, 16);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N32, 32);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N64, 64);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N128, 128);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N256, 256);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N512, 512);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcMultiDci4, N2048, 2048);

BENCHMARK_DRAW_LINE();

// DC Multi-DCI (8 DCIs)
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N2, 2);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N4, 4);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N8, 8);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N16, 16);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N32, 32);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N64, 64);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N128, 128);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N256, 256);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N512, 512);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcMultiDci8, N2048, 2048);

BENCHMARK_DRAW_LINE();

// DC Multi-DCI (16 DCIs)
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N2, 2);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N4, 4);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N8, 8);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N16, 16);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N32, 32);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N64, 64);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N128, 128);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N256, 256);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N512, 512);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcMultiDci16, N2048, 2048);

BENCHMARK_DRAW_LINE();

// DC Streams (4 concurrent streams, 1 DCI)
REGISTER_SCALABILITY_BENCH(dcStreams4, N2, 2);
REGISTER_SCALABILITY_BENCH(dcStreams4, N4, 4);
REGISTER_SCALABILITY_BENCH(dcStreams4, N8, 8);
REGISTER_SCALABILITY_BENCH(dcStreams4, N16, 16);
REGISTER_SCALABILITY_BENCH(dcStreams4, N32, 32);
REGISTER_SCALABILITY_BENCH(dcStreams4, N64, 64);
REGISTER_SCALABILITY_BENCH(dcStreams4, N128, 128);
REGISTER_SCALABILITY_BENCH(dcStreams4, N256, 256);
REGISTER_SCALABILITY_BENCH(dcStreams4, N512, 512);
REGISTER_SCALABILITY_BENCH(dcStreams4, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcStreams4, N2048, 2048);

BENCHMARK_DRAW_LINE();

// DC Streams (16 concurrent streams, 1 DCI)
REGISTER_SCALABILITY_BENCH(dcStreams16, N2, 2);
REGISTER_SCALABILITY_BENCH(dcStreams16, N4, 4);
REGISTER_SCALABILITY_BENCH(dcStreams16, N8, 8);
REGISTER_SCALABILITY_BENCH(dcStreams16, N16, 16);
REGISTER_SCALABILITY_BENCH(dcStreams16, N32, 32);
REGISTER_SCALABILITY_BENCH(dcStreams16, N64, 64);
REGISTER_SCALABILITY_BENCH(dcStreams16, N128, 128);
REGISTER_SCALABILITY_BENCH(dcStreams16, N256, 256);
REGISTER_SCALABILITY_BENCH(dcStreams16, N512, 512);
REGISTER_SCALABILITY_BENCH(dcStreams16, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcStreams16, N2048, 2048);

BENCHMARK_DRAW_LINE();

// DC Multi-DCI (4 DCIs) + Streams (4 concurrent streams per DCI)
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N2, 2);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N4, 4);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N8, 8);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N16, 16);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N32, 32);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N64, 64);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N128, 128);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N256, 256);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N512, 512);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N1024, 1024);
REGISTER_SCALABILITY_BENCH(dcMultiDci4Streams4, N2048, 2048);

BENCHMARK_DRAW_LINE();

// RC Scalability: same peer counts
REGISTER_SCALABILITY_BENCH(rcScalability, N2, 2);
REGISTER_SCALABILITY_BENCH(rcScalability, N4, 4);
REGISTER_SCALABILITY_BENCH(rcScalability, N8, 8);
REGISTER_SCALABILITY_BENCH(rcScalability, N16, 16);
REGISTER_SCALABILITY_BENCH(rcScalability, N32, 32);
REGISTER_SCALABILITY_BENCH(rcScalability, N64, 64);
REGISTER_SCALABILITY_BENCH(rcScalability, N128, 128);
REGISTER_SCALABILITY_BENCH(rcScalability, N256, 256);
REGISTER_SCALABILITY_BENCH(rcScalability, N512, 512);
REGISTER_SCALABILITY_BENCH(rcScalability, N1024, 1024);
REGISTER_SCALABILITY_BENCH(rcScalability, N2048, 2048);

BENCHMARK_DRAW_LINE();

// ---------------------------------------------------------------------------
// Message Size Sweep at N=2048
// ---------------------------------------------------------------------------

// DC (1 DCI) message size sweep
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 64B, 64);
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 256B, 256);
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 1KB, 1024);
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 4KB, 4096);
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 16KB, 16384);
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 64KB, 65536);
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 256KB, 262144);
REGISTER_SCALABILITY_BENCH(dcMsgSweep, 1MB, 1048576);

BENCHMARK_DRAW_LINE();

// DC Multi-DCI (4 DCIs) message size sweep
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 64B, 64);
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 256B, 256);
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 1KB, 1024);
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 4KB, 4096);
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 16KB, 16384);
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 64KB, 65536);
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 256KB, 262144);
REGISTER_SCALABILITY_BENCH(dcMultiDci4MsgSweep, 1MB, 1048576);

BENCHMARK_DRAW_LINE();

// RC message size sweep
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 64B, 64);
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 256B, 256);
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 1KB, 1024);
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 4KB, 4096);
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 16KB, 16384);
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 64KB, 65536);
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 256KB, 262144);
REGISTER_SCALABILITY_BENCH(rcMsgSweep, 1MB, 1048576);

BENCHMARK_DRAW_LINE();

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  XLOG(INFO) << "DC vs RC RDMA Scalability Benchmark";
  XLOG(INFO) << "====================================";
  XLOG(INFO) << "Total input size: " << kTotalInputSize << " bytes (128 MB)";
  XLOG(INFO) << "Shared CQ depth: " << kSharedCqDepth;
  XLOG(INFO) << "DCI SQ depth: " << kDciSqDepth;
  XLOG(INFO) << "RC QP SQ depth: " << kRcQpSqDepth;

  if (!checkRdmaAvailable()) {
    XLOG(ERR) << "RDMA not available - benchmarks will be skipped";
  } else {
    XLOG(INFO) << "RDMA devices available";
  }

  // Suppress folly benchmark table when --raw_only is set
  int savedStdout = -1;
  if (FLAGS_raw_only) {
    savedStdout = dup(STDOUT_FILENO);
    int devNull = open("/dev/null", O_WRONLY);
    dup2(devNull, STDOUT_FILENO);
    close(devNull);
  }

  folly::runBenchmarks();

  if (FLAGS_raw_only && savedStdout >= 0) {
    dup2(savedStdout, STDOUT_FILENO);
    close(savedStdout);
  }

  printAllRawResults();

  return 0;
}
