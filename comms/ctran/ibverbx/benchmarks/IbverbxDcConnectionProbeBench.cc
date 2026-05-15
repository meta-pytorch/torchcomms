// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * DC Connection Persistence Probe
 *
 * Uses MPI with 16 ranks across 2 nodes (8 ranks/node).
 * Each rank writes to exactly 2 remote peers on different NICs with two
 * patterns to determine whether the DCI caches connections or only holds
 * them via SQ look-ahead:
 *
 *   sorted:      AAAA...BBBB (1 DCT switch)
 *   alternating: ABABABAB... (N DCT switches)
 *
 * If DCI caches connections (cache >= 2):
 *   both patterns perform similarly after warmup
 * If DCI only holds connection via SQ look-ahead (no cache):
 *   alternating degrades proportionally to switch count
 */

#include <chrono>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/benchmarks/IbverbxDcBenchUtils.h"
#include "comms/ctran/ibverbx/tests/dc_utils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ibverbx::createAddressHandle;
using ibverbx::createDCILargeSq;
using ibverbx::createDCILargeSqWithStreams;
using ibverbx::createDCT;
using ibverbx::createSRQ;
using ibverbx::DC_KEY;
using ibverbx::DcBusinessCard;
using ibverbx::kGidIndex;
using ibverbx::kPortNum;
using ibverbx::pollCqBusySpin;
using ibverbx::transitionDCIToRts;
using ibverbx::transitionDCTToRtr;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace {

constexpr int kWarmupIters = 10;
constexpr int kBenchIters = 100;
constexpr int kCqDepth = 8192;
constexpr int kDciSqDepth = 8192;
constexpr int kPollTimeoutMs = 30000;
constexpr int kPpn = 8;

// MPI exchange card
struct ExchangeCard {
  uint32_t dctNum;
  uint64_t subnetPrefix;
  uint64_t interfaceId;
  uint64_t recvBufAddr;
  uint32_t rkey;
  int nicIdx;
};

// Compute the slot index for 'sender' in 'receiver's recv buffer.
int peerSlot(int sender, int receiver) {
  return sender < receiver ? sender : sender - 1;
}

} // namespace

class DcConnectionProbeBench : public meta::comms::MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();
    ASSERT_TRUE(ibverbx::ibvInit());
  }

  void runProbe(int writesPerPeer, size_t chunkSize, bool useStreams = false);
};

void DcConnectionProbeBench::runProbe(
    int writesPerPeer,
    size_t chunkSize,
    bool useStreams) {
  ASSERT_EQ(numRanks, 2 * kPpn)
      << "This benchmark requires exactly " << 2 * kPpn << " MPI ranks";

  int sendDeviceIdx = localRank % 8;
  int recvDeviceIdx = localRank % 8;
  int totalWrites = 2 * writesPerPeer;
  int numPeers = numRanks - 1;

  auto devices =
      ibverbx::IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  ASSERT_TRUE(devices);
  ASSERT_GT(
      devices->size(),
      static_cast<size_t>(std::max(sendDeviceIdx, recvDeviceIdx)));

  auto access = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);

  // --- Send side ---
  auto& sendDevice = devices->at(sendDeviceIdx);
  auto sendPd = sendDevice.allocPd();
  ASSERT_TRUE(sendPd);
  auto sendCq = sendDevice.createCq(kCqDepth, nullptr, nullptr, 0);
  ASSERT_TRUE(sendCq);

  auto dciResult = useStreams
      ? createDCILargeSqWithStreams(*sendPd, *sendCq, kDciSqDepth, 1)
      : createDCILargeSq(*sendPd, *sendCq, kDciSqDepth);
  if (!dciResult) {
    if (globalRank == 0) {
      fprintf(
          stderr,
          "SKIP: DCI creation failed (streams=%s): %s\n",
          useStreams ? "true" : "false",
          dciResult.error().errStr.c_str());
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    GTEST_SKIP() << "DCI creation failed"
                 << (useStreams ? " (streams not supported)" : "");
  }
  auto dci = std::make_unique<ibverbx::IbvQp>(std::move(*dciResult));
  auto* exQp = ibverbx::ibvSymbols.ibv_internal_qp_to_qp_ex(dci->qp());
  auto* dvQp = ibverbx::ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(exQp);
  ASSERT_NE(exQp, nullptr);
  ASSERT_NE(dvQp, nullptr);
  ASSERT_TRUE(transitionDCIToRts(*dci, kPortNum, ibverbx::IBV_MTU_4096));

  std::vector<uint8_t> sendBuf(chunkSize, 0xAA);
  auto sendMr = sendPd->regMr(sendBuf.data(), chunkSize, access);
  ASSERT_TRUE(sendMr);

  // --- Recv side ---
  auto& recvDevice = devices->at(recvDeviceIdx);
  auto recvPd = recvDevice.allocPd();
  ASSERT_TRUE(recvPd);
  auto recvCq = recvDevice.createCq(kCqDepth, nullptr, nullptr, 0);
  ASSERT_TRUE(recvCq);
  auto recvGid = recvDevice.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(recvGid);

  auto srqResult = createSRQ(*recvPd, 1024);
  ASSERT_TRUE(srqResult);
  auto srq = std::make_unique<ibverbx::IbvSrq>(std::move(*srqResult));

  auto dctResult = createDCT(*recvPd, *recvCq, *srq);
  ASSERT_TRUE(dctResult);
  auto dct = std::make_unique<ibverbx::IbvQp>(std::move(*dctResult));
  ASSERT_TRUE(transitionDCTToRtr(*dct, kPortNum, ibverbx::IBV_MTU_4096));

  // Recv buffer sized for all peers at their global slot
  size_t recvBufSize = numPeers * writesPerPeer * chunkSize;
  std::vector<uint8_t> recvBuf(recvBufSize, 0x00);
  auto recvMr = recvPd->regMr(recvBuf.data(), recvBufSize, access);
  ASSERT_TRUE(recvMr);

  // Exchange cards
  ExchangeCard myCard{};
  myCard.dctNum = dct->qp()->qp_num;
  myCard.subnetPrefix = recvGid->global.subnet_prefix;
  myCard.interfaceId = recvGid->global.interface_id;
  myCard.recvBufAddr = reinterpret_cast<uint64_t>(recvBuf.data());
  myCard.rkey = recvMr->mr()->rkey;
  myCard.nicIdx = recvDeviceIdx;

  std::vector<ExchangeCard> allCards(numRanks);
  MPI_CHECK(MPI_Allgather(
      &myCard,
      sizeof(ExchangeCard),
      MPI_BYTE,
      allCards.data(),
      sizeof(ExchangeCard),
      MPI_BYTE,
      MPI_COMM_WORLD));

  // Pick 2 remote peers on different recv NICs
  int peerA = -1, peerB = -1;
  for (int r = 0; r < numRanks; ++r) {
    if (r == globalRank) {
      continue;
    }
    if (peerA == -1) {
      peerA = r;
    } else if (allCards[r].nicIdx != allCards[peerA].nicIdx && peerB == -1) {
      peerB = r;
      break;
    }
  }
  ASSERT_NE(peerA, -1);
  ASSERT_NE(peerB, -1);

  // Create AHs (different NICs = different GIDs = different AHs)
  auto makeAh = [&](int r) {
    DcBusinessCard card{};
    card.mtu = 5;
    card.dctNum = allCards[r].dctNum;
    card.port = kPortNum;
    card.subnetPrefix = allCards[r].subnetPrefix;
    card.interfaceId = allCards[r].interfaceId;
    return createAddressHandle(*sendPd, card);
  };
  auto ahA = std::make_unique<ibverbx::IbvAh>(std::move(*makeAh(peerA)));
  auto ahB = std::make_unique<ibverbx::IbvAh>(std::move(*makeAh(peerB)));

  ibverbx::ibv_sge sendSge{};
  sendSge.addr = reinterpret_cast<uint64_t>(sendBuf.data());
  sendSge.length = static_cast<uint32_t>(chunkSize);
  sendSge.lkey = sendMr->mr()->lkey;

  int batch = std::min(kDciSqDepth, kCqDepth);

  // Write target for the schedule
  struct WriteTarget {
    ibverbx::IbvAh* ah;
    uint32_t dctNum;
    uint32_t rkey;
    uint64_t addr;
  };

  // Build write schedule
  int peers[2] = {peerA, peerB};
  ibverbx::IbvAh* ahs[2] = {ahA.get(), ahB.get()};

  auto buildSchedule = [&](bool sorted) {
    std::vector<WriteTarget> sched;
    sched.reserve(totalWrites);
    if (sorted) {
      for (int p = 0; p < 2; ++p) {
        for (int w = 0; w < writesPerPeer; ++w) {
          int slot = peerSlot(globalRank, peers[p]);
          sched.push_back(
              {ahs[p],
               allCards[peers[p]].dctNum,
               allCards[peers[p]].rkey,
               allCards[peers[p]].recvBufAddr +
                   static_cast<uint64_t>(slot) * writesPerPeer * chunkSize +
                   static_cast<uint64_t>(w) * chunkSize});
        }
      }
    } else {
      for (int w = 0; w < writesPerPeer; ++w) {
        for (int p = 0; p < 2; ++p) {
          int slot = peerSlot(globalRank, peers[p]);
          sched.push_back(
              {ahs[p],
               allCards[peers[p]].dctNum,
               allCards[peers[p]].rkey,
               allCards[peers[p]].recvBufAddr +
                   static_cast<uint64_t>(slot) * writesPerPeer * chunkSize +
                   static_cast<uint64_t>(w) * chunkSize});
        }
      }
    }
    return sched;
  };

  // Run a pattern and report results
  auto runPattern = [&](const std::vector<WriteTarget>& sched,
                        const char* label) {
    auto postAll = [&]() -> bool {
      int remaining = totalWrites;
      int idx = 0;
      while (remaining > 0) {
        int thisBatch = std::min(remaining, batch);
        ibverbx::ibvSymbols.ibv_internal_wr_start(exQp);
        for (int j = 0; j < thisBatch; ++j) {
          auto& t = sched[idx];
          exQp->wr_id = static_cast<uint64_t>(idx);
          exQp->wr_flags = ibverbx::IBV_SEND_SIGNALED;
          ibverbx::ibvSymbols.ibv_internal_wr_rdma_write(exQp, t.rkey, t.addr);
          ibverbx::ibvSymbols.ibv_internal_wr_set_sge_list(exQp, 1, &sendSge);
          ibverbx::ibvSymbols.mlx5dv_internal_wr_set_dc_addr(
              dvQp, t.ah->ah(), t.dctNum, DC_KEY);
          idx++;
        }
        if (ibverbx::ibvSymbols.ibv_internal_wr_complete(exQp) != 0) {
          return false;
        }
        if (!pollCqBusySpin(sendCq->cq(), thisBatch, kPollTimeoutMs)) {
          return false;
        }
        remaining -= thisBatch;
      }
      return true;
    };

    for (int w = 0; w < kWarmupIters; ++w) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      ASSERT_TRUE(postAll()) << label << " warmup failed at iter " << w;
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < kBenchIters; ++iter) {
      ASSERT_TRUE(postAll()) << label << " timed failed at iter " << iter;
    }
    auto end = std::chrono::high_resolution_clock::now();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      double elapsedUs =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          1000.0;
      double latencyUs = elapsedUs / kBenchIters;
      size_t totalBytes = chunkSize * totalWrites;
      double bwGbps = (static_cast<double>(totalBytes) * kBenchIters / 1e9) /
          (elapsedUs / 1e6);
      fprintf(
          stderr,
          "RAW,%s,%d,%.2f,%.6f,%zu,%d\n",
          label,
          writesPerPeer,
          latencyUs,
          bwGbps,
          chunkSize,
          totalWrites);
    }
  };

  auto sortedSched = buildSchedule(true);
  auto altSched = buildSchedule(false);

  runPattern(sortedSched, "dcProbe_sorted");
  runPattern(altSched, "dcProbe_alternating");
}

TEST_F(DcConnectionProbeBench, DcConnectionProbe) {
  if (globalRank == 0) {
    fprintf(
        stderr,
        "RAW,benchmark,writes_per_peer,latency_us,bw_gbps,chunk_bytes,total_writes\n");
  }
  // 1MB per peer, varying chunk count (= writes per peer)
  // More writes = more DCT switches in alternating mode
  for (int wpp : {16, 64, 256, 1024}) {
    size_t chunk = 1024 * 1024 / wpp;
    runProbe(wpp, chunk);
  }
}

TEST_F(DcConnectionProbeBench, DcConnectionProbeWithStreams) {
  if (globalRank == 0) {
    fprintf(
        stderr,
        "RAW,benchmark,writes_per_peer,latency_us,bw_gbps,chunk_bytes,total_writes\n");
  }
  for (int wpp : {16, 64, 256, 1024}) {
    size_t chunk = 1024 * 1024 / wpp;
    runProbe(wpp, chunk, /*useStreams=*/true);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
