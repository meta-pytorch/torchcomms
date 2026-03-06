// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Two-Rank All-to-All DC vs RC Scalability Benchmark
 *
 * Uses MPI with exactly 2 ranks on the same node, each using a different NIC.
 * Rank 0 represents a single rank with configurable DCIs + DCTs.
 * Rank 1 represents N-1 simulated ranks, each with its own DCI + shared DCTs.
 * Both ranks simultaneously post writes to each other (natural MPI overlap),
 * creating bidirectional traffic that mimics a real all-to-all pattern.
 *
 * Measures from rank 0's perspective: rank 0's time to complete its N-1
 * outbound writes while also receiving N-1 inbound writes from rank 1.
 *
 * Variants:
 * - DcAllToAll: 1 DCI + 1 DCT per rank (baseline)
 * - DcAllToAllMultiDci: 4 DCIs on rank 0, 4 DCTs per rank
 * - RcAllToAll: RC transport for comparison
 */

#include <chrono>
#include <cstring>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/tests/dc_utils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ibverbx::createAddressHandle;
using ibverbx::createDCT;
using ibverbx::createSRQ;
using ibverbx::DC_KEY;
using ibverbx::DcBusinessCard;
using ibverbx::kGidIndex;
using ibverbx::kPortNum;
using ibverbx::transitionDCIToRts;
using ibverbx::transitionDCTToRtr;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace {

constexpr size_t kTotalInputSize = 128 * 1024 * 1024; // 128 MB
constexpr int kWarmupIters = 10;
constexpr int kBenchIters = 1000;
constexpr int kCqDepth = 8192;
constexpr int kDciSqDepth = 8192;
constexpr int kRcQpSqDepth = 1024;
constexpr int kPollTimeoutMs = 30000;
constexpr int kMultiDciCount = 4;
constexpr size_t kPeerCounts[] =
    {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

// Create DCI with configurable SQ depth
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

// Busy-spin CQ poll with timeout
bool pollCqBusySpin(
    ibverbx::ibv_cq* rawCq,
    int expected,
    int timeoutMs = 30000) {
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

// Post DC RDMA write
int postDcRdmaWrite(
    ibverbx::ibv_qp_ex* exQp,
    mlx5dv_qp_ex* dvQp,
    ibverbx::IbvAh& ah,
    uint32_t dctNum,
    uint32_t rkey,
    uint64_t remoteAddr,
    ibverbx::ibv_sge& sge,
    uint64_t wrId) {
  ibverbx::ibvSymbols.ibv_internal_wr_start(exQp);
  exQp->wr_id = wrId;
  exQp->wr_flags = ibverbx::IBV_SEND_SIGNALED;
  ibverbx::ibvSymbols.ibv_internal_wr_rdma_write(exQp, rkey, remoteAddr);
  ibverbx::ibvSymbols.ibv_internal_wr_set_sge_list(exQp, 1, &sge);
  ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr(
      dvQp, ah.ah(), dctNum, DC_KEY);
  return ibverbx::ibvSymbols.ibv_internal_wr_complete(exQp);
}

// Transition an RC QP through INIT -> RTR -> RTS
bool connectRcQp(
    ibverbx::IbvQp& localQp,
    const ibverbx::ibv_gid& remoteGid,
    uint32_t remoteQpNum) {
  // INIT
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = kPortNum;
    attr.qp_access_flags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
        ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_PKEY_INDEX |
        ibverbx::IBV_QP_PORT | ibverbx::IBV_QP_ACCESS_FLAGS;
    if (localQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  // RTR
  {
    ibverbx::ibv_qp_attr attr{};
    attr.qp_state = ibverbx::IBV_QPS_RTR;
    attr.path_mtu = ibverbx::IBV_MTU_4096;
    attr.dest_qp_num = remoteQpNum;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remoteGid;
    attr.ah_attr.grh.sgid_index = kGidIndex;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.port_num = kPortNum;
    int mask = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_AV |
        ibverbx::IBV_QP_PATH_MTU | ibverbx::IBV_QP_DEST_QPN |
        ibverbx::IBV_QP_RQ_PSN | ibverbx::IBV_QP_MAX_DEST_RD_ATOMIC |
        ibverbx::IBV_QP_MIN_RNR_TIMER;
    if (localQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  // RTS
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
    if (localQp.modifyQp(&attr, mask).hasError()) {
      return false;
    }
  }
  return true;
}

// Exchange card for MPI (DC) - supports multiple DCTs
struct DcExchangeCard {
  static constexpr int kMaxDcts = 8;
  uint32_t dctNums[kMaxDcts];
  int numDcts;
  uint64_t subnetPrefix;
  uint64_t interfaceId;
  uint64_t recvBufAddr;
  uint32_t rkey;
};

struct RcExchangeCard {
  uint32_t qpNum;
  uint64_t subnetPrefix;
  uint64_t interfaceId;
  uint64_t recvBufAddr;
  uint32_t rkey;
};

} // namespace

class AllToAllBenchFixture : public meta::comms::MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();
    ASSERT_TRUE(ibverbx::ibvInit());
  }

  // Parameterized DC all-to-all: numDcisRank0 DCIs on rank 0,
  // numDcts DCTs per rank (capped at numPeers).
  void runDcAllToAll(int numDcisRank0, int numDcts, const char* benchName);
};

void AllToAllBenchFixture::runDcAllToAll(
    int numDcisRank0,
    int numDcts,
    const char* benchName) {
  ASSERT_EQ(numRanks, 2) << "This benchmark requires exactly 2 MPI ranks";
  ASSERT_LE(numDcts, DcExchangeCard::kMaxDcts);

  auto devices =
      ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  ASSERT_TRUE(devices);
  ASSERT_GT(devices->size(), static_cast<size_t>(localRank));

  auto& device = devices->at(localRank);
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  auto cq = device.createCq(kCqDepth, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);

  auto access = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);

  if (globalRank == 0) {
    fprintf(
        stderr,
        "RAW,benchmark,N,latency_us,bw_gbps,msg_bytes,num_dcis,num_dcts\n");
  }

  for (size_t N : kPeerCounts) {
    int numPeers = static_cast<int>(N) - 1;
    size_t msgSize = kTotalInputSize / N;

    // Rank 0: numDcisRank0 DCIs. Rank 1: N-1 DCIs (one per simulated rank).
    int numDcis = (globalRank == 0) ? numDcisRank0 : numPeers;

    // Create SRQ + DCTs (both ranks, capped at numPeers)
    auto srqResult = createSRQ(*pd, 1024);
    ASSERT_TRUE(srqResult);
    auto srq = std::make_unique<ibverbx::IbvSrq>(std::move(*srqResult));

    int dctsUsed = std::min(numDcts, numPeers);
    std::vector<std::unique_ptr<ibverbx::IbvQp>> dcts;
    for (int t = 0; t < dctsUsed; ++t) {
      auto dctResult = createDCT(*pd, *cq, *srq);
      ASSERT_TRUE(dctResult);
      auto dctQp = std::make_unique<ibverbx::IbvQp>(std::move(*dctResult));
      ASSERT_TRUE(transitionDCTToRtr(*dctQp, kPortNum, ibverbx::IBV_MTU_4096));
      dcts.push_back(std::move(dctQp));
    }

    // Create DCIs
    struct DciState {
      std::unique_ptr<ibverbx::IbvQp> qp;
      ibverbx::ibv_qp_ex* exQp = nullptr;
      mlx5dv_qp_ex* dvQp = nullptr;
    };
    std::vector<DciState> dcis(numDcis);
    for (int d = 0; d < numDcis; ++d) {
      auto dciResult = createDCILargeSq(*pd, *cq, kDciSqDepth);
      ASSERT_TRUE(dciResult);
      dcis[d].qp = std::make_unique<ibverbx::IbvQp>(std::move(*dciResult));
      dcis[d].exQp =
          ibverbx::ibvSymbols.ibv_internal_qp_to_qp_ex(dcis[d].qp->qp());
      dcis[d].dvQp = ibverbx::ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(
          dcis[d].exQp);
      ASSERT_NE(dcis[d].exQp, nullptr);
      ASSERT_NE(dcis[d].dvQp, nullptr);
      ASSERT_TRUE(
          transitionDCIToRts(*dcis[d].qp, kPortNum, ibverbx::IBV_MTU_4096));
    }

    // Allocate buffers
    std::vector<uint8_t> sendBuf(msgSize, 0xAA);
    auto sendMr = pd->regMr(sendBuf.data(), msgSize, access);
    ASSERT_TRUE(sendMr);

    std::vector<uint8_t> recvBuf(numPeers * msgSize, 0x00);
    auto recvMr = pd->regMr(recvBuf.data(), numPeers * msgSize, access);
    ASSERT_TRUE(recvMr);

    // Exchange cards via MPI
    DcExchangeCard myCard{};
    myCard.numDcts = dctsUsed;
    for (int t = 0; t < dctsUsed; ++t) {
      myCard.dctNums[t] = dcts[t]->qp()->qp_num;
    }
    myCard.subnetPrefix = gid->global.subnet_prefix;
    myCard.interfaceId = gid->global.interface_id;
    myCard.recvBufAddr = reinterpret_cast<uint64_t>(recvBuf.data());
    myCard.rkey = recvMr->mr()->rkey;

    DcExchangeCard remoteCard{};
    MPI_CHECK(MPI_Sendrecv(
        &myCard,
        sizeof(DcExchangeCard),
        MPI_BYTE,
        1 - globalRank,
        0,
        &remoteCard,
        sizeof(DcExchangeCard),
        MPI_BYTE,
        1 - globalRank,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    // Create AH to remote (AH depends on GID, not DCT number)
    DcBusinessCard targetCard{};
    targetCard.mtu = 5;
    targetCard.dctNum = remoteCard.dctNums[0];
    targetCard.port = kPortNum;
    targetCard.subnetPrefix = remoteCard.subnetPrefix;
    targetCard.interfaceId = remoteCard.interfaceId;
    auto ahResult = createAddressHandle(*pd, targetCard);
    ASSERT_TRUE(ahResult);
    auto ah = std::make_unique<ibverbx::IbvAh>(std::move(*ahResult));

    ibverbx::ibv_sge sendSge{};
    sendSge.addr = reinterpret_cast<uint64_t>(sendBuf.data());
    sendSge.length = static_cast<uint32_t>(msgSize);
    sendSge.lkey = sendMr->mr()->lkey;

    // Each batch posts thisBatch writes then polls thisBatch completions,
    // so CQ only needs to hold thisBatch entries at a time. Per-DCI SQ
    // usage is ceil(thisBatch/numDcis). Limit by both.
    int batch = std::min(numDcis * kDciSqDepth, kCqDepth);

    // Post all writes for one iteration
    auto postAllWrites = [&]() -> bool {
      int remaining = numPeers;
      int idx = 0;
      while (remaining > 0) {
        int thisBatch = std::min(remaining, batch);
        for (int j = 0; j < thisBatch; ++j) {
          int d = (globalRank == 0) ? (idx % numDcis) : idx;
          uint32_t targetDctNum = remoteCard.dctNums[idx % remoteCard.numDcts];
          uint64_t targetAddr =
              remoteCard.recvBufAddr + static_cast<uint64_t>(idx) * msgSize;
          if (postDcRdmaWrite(
                  dcis[d].exQp,
                  dcis[d].dvQp,
                  *ah,
                  targetDctNum,
                  remoteCard.rkey,
                  targetAddr,
                  sendSge,
                  idx) != 0) {
            return false;
          }
          idx++;
        }
        if (!pollCqBusySpin(cq->cq(), thisBatch, kPollTimeoutMs)) {
          return false;
        }
        remaining -= thisBatch;
      }
      return true;
    };

    // Warmup
    for (int w = 0; w < kWarmupIters; ++w) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      ASSERT_TRUE(postAllWrites()) << "DC warmup failed at iter " << w;
    }

    // Timed phase
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < kBenchIters; ++iter) {
      ASSERT_TRUE(postAllWrites()) << "DC timed failed at iter " << iter;
    }
    auto end = std::chrono::high_resolution_clock::now();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      double elapsedUs =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          1000.0;
      double latencyUs = elapsedUs / kBenchIters;
      double bwGbps =
          (static_cast<double>(msgSize) * numPeers * kBenchIters / 1e9) /
          (elapsedUs / 1e6);
      fprintf(
          stderr,
          "RAW,%s,%zu,%.2f,%.6f,%zu,%d,%d\n",
          benchName,
          N,
          latencyUs,
          bwGbps,
          msgSize,
          numDcis,
          dctsUsed);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } // end N loop
}

TEST_F(AllToAllBenchFixture, DcAllToAll) {
  runDcAllToAll(1, 1, "dcAllToAll");
}

TEST_F(AllToAllBenchFixture, DcAllToAllMultiDci) {
  runDcAllToAll(kMultiDciCount, kMultiDciCount, "dcAllToAllMultiDci");
}

TEST_F(AllToAllBenchFixture, RcAllToAll) {
  ASSERT_EQ(numRanks, 2) << "This benchmark requires exactly 2 MPI ranks";

  auto devices =
      ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  ASSERT_TRUE(devices);
  ASSERT_GT(devices->size(), static_cast<size_t>(localRank));

  auto& device = devices->at(localRank);
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  auto cq = device.createCq(kCqDepth, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);

  auto access = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);

  if (globalRank == 0) {
    fprintf(stderr, "RAW,benchmark,N,latency_us,bw_gbps,msg_bytes,num_qps\n");
  }

  for (size_t N : kPeerCounts) {
    int numPeers = static_cast<int>(N) - 1;
    size_t msgSize = kTotalInputSize / N;

    // Each rank creates numPeers sender QPs + numPeers receiver QPs
    std::vector<std::unique_ptr<ibverbx::IbvQp>> senderQps;
    std::vector<std::unique_ptr<ibverbx::IbvQp>> receiverQps;

    for (int i = 0; i < numPeers; ++i) {
      // Sender QP
      ibverbx::ibv_qp_init_attr sInit{};
      sInit.send_cq = cq->cq();
      sInit.recv_cq = cq->cq();
      sInit.qp_type = ibverbx::IBV_QPT_RC;
      sInit.cap.max_send_wr = kRcQpSqDepth;
      sInit.cap.max_recv_wr = 1024;
      sInit.cap.max_send_sge = 1;
      sInit.cap.max_recv_sge = 1;
      auto sqp = pd->createQp(&sInit);
      ASSERT_TRUE(sqp);
      senderQps.push_back(std::make_unique<ibverbx::IbvQp>(std::move(*sqp)));

      // Receiver QP
      ibverbx::ibv_qp_init_attr rInit{};
      rInit.send_cq = cq->cq();
      rInit.recv_cq = cq->cq();
      rInit.qp_type = ibverbx::IBV_QPT_RC;
      rInit.cap.max_send_wr = kRcQpSqDepth;
      rInit.cap.max_recv_wr = 1024;
      rInit.cap.max_send_sge = 1;
      rInit.cap.max_recv_sge = 1;
      auto rqp = pd->createQp(&rInit);
      ASSERT_TRUE(rqp);
      receiverQps.push_back(std::make_unique<ibverbx::IbvQp>(std::move(*rqp)));
    }

    // Allocate buffers
    std::vector<uint8_t> sendBuf(msgSize, 0xAA);
    auto sendMr = pd->regMr(sendBuf.data(), msgSize, access);
    ASSERT_TRUE(sendMr);

    std::vector<uint8_t> recvBuf(numPeers * msgSize, 0x00);
    auto recvMr = pd->regMr(recvBuf.data(), numPeers * msgSize, access);
    ASSERT_TRUE(recvMr);

    // Exchange cards: each rank sends numPeers sender QP numbers and
    // numPeers receiver QP numbers + recv buffer info
    std::vector<RcExchangeCard> mySenderCards(numPeers);
    std::vector<RcExchangeCard> myReceiverCards(numPeers);
    for (int i = 0; i < numPeers; ++i) {
      mySenderCards[i] = {
          .qpNum = senderQps[i]->qp()->qp_num,
          .subnetPrefix = gid->global.subnet_prefix,
          .interfaceId = gid->global.interface_id,
          .recvBufAddr = 0, // senders don't receive
          .rkey = 0,
      };
      myReceiverCards[i] = {
          .qpNum = receiverQps[i]->qp()->qp_num,
          .subnetPrefix = gid->global.subnet_prefix,
          .interfaceId = gid->global.interface_id,
          .recvBufAddr = reinterpret_cast<uint64_t>(recvBuf.data()) +
              static_cast<uint64_t>(i) * msgSize,
          .rkey = recvMr->mr()->rkey,
      };
    }

    std::vector<RcExchangeCard> remoteSenderCards(numPeers);
    std::vector<RcExchangeCard> remoteReceiverCards(numPeers);

    // Exchange sender cards
    MPI_CHECK(MPI_Sendrecv(
        mySenderCards.data(),
        numPeers * sizeof(RcExchangeCard),
        MPI_BYTE,
        1 - globalRank,
        0,
        remoteSenderCards.data(),
        numPeers * sizeof(RcExchangeCard),
        MPI_BYTE,
        1 - globalRank,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    // Exchange receiver cards
    MPI_CHECK(MPI_Sendrecv(
        myReceiverCards.data(),
        numPeers * sizeof(RcExchangeCard),
        MPI_BYTE,
        1 - globalRank,
        1,
        remoteReceiverCards.data(),
        numPeers * sizeof(RcExchangeCard),
        MPI_BYTE,
        1 - globalRank,
        1,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    // Connect QPs:
    // My sender QP[i] connects to remote receiver QP[i]
    // My receiver QP[i] connects to remote sender QP[i]
    for (int i = 0; i < numPeers; ++i) {
      ibverbx::ibv_gid remoteGid{};
      remoteGid.global.subnet_prefix = remoteReceiverCards[i].subnetPrefix;
      remoteGid.global.interface_id = remoteReceiverCards[i].interfaceId;
      ASSERT_TRUE(
          connectRcQp(*senderQps[i], remoteGid, remoteReceiverCards[i].qpNum));

      ibverbx::ibv_gid remoteGid2{};
      remoteGid2.global.subnet_prefix = remoteSenderCards[i].subnetPrefix;
      remoteGid2.global.interface_id = remoteSenderCards[i].interfaceId;
      ASSERT_TRUE(
          connectRcQp(*receiverQps[i], remoteGid2, remoteSenderCards[i].qpNum));
    }

    ibverbx::ibv_sge sendSge{};
    sendSge.addr = reinterpret_cast<uint64_t>(sendBuf.data());
    sendSge.length = static_cast<uint32_t>(msgSize);
    sendSge.lkey = sendMr->mr()->lkey;

    // Each QP gets exactly 1 write per batch, so per-QP SQ depth is
    // trivially satisfied. CQ only holds thisBatch entries at a time.
    int batch = kCqDepth;

    // Post all writes for one iteration
    auto postAllWrites = [&]() -> bool {
      int remaining = numPeers;
      int idx = 0;
      while (remaining > 0) {
        int thisBatch = std::min(remaining, batch);
        for (int j = 0; j < thisBatch; ++j) {
          ibverbx::ibv_send_wr wr{};
          wr.wr_id = static_cast<uint64_t>(idx);
          wr.next = nullptr;
          wr.sg_list = &sendSge;
          wr.num_sge = 1;
          wr.opcode = ibverbx::IBV_WR_RDMA_WRITE;
          wr.send_flags = ibverbx::IBV_SEND_SIGNALED;
          wr.wr.rdma.remote_addr = remoteReceiverCards[idx].recvBufAddr;
          wr.wr.rdma.rkey = remoteReceiverCards[idx].rkey;

          ibverbx::ibv_send_wr* badWr = nullptr;
          auto postResult = senderQps[idx]->postSend(&wr, badWr);
          if (postResult.hasError()) {
            return false;
          }
          idx++;
        }
        if (!pollCqBusySpin(cq->cq(), thisBatch, kPollTimeoutMs)) {
          return false;
        }
        remaining -= thisBatch;
      }
      return true;
    };

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Warmup
    for (int w = 0; w < kWarmupIters; ++w) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      ASSERT_TRUE(postAllWrites()) << "RC warmup failed at iter " << w;
    }

    // Timed phase
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < kBenchIters; ++iter) {
      ASSERT_TRUE(postAllWrites()) << "RC timed failed at iter " << iter;
    }
    auto end = std::chrono::high_resolution_clock::now();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      double elapsedUs =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          1000.0;
      double latencyUs = elapsedUs / kBenchIters;
      double bwGbps =
          (static_cast<double>(msgSize) * numPeers * kBenchIters / 1e9) /
          (elapsedUs / 1e6);
      fprintf(
          stderr,
          "RAW,rcAllToAll,%zu,%.2f,%.6f,%zu,%d\n",
          N,
          latencyUs,
          bwGbps,
          msgSize,
          numPeers);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } // end N loop
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
