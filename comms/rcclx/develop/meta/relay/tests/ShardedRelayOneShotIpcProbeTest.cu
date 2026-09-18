// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Feasibility probe for a one-shot IPC small-message collective.
 *
 * The sharded-relay small-message paths are launch-bound: below 576 KB the
 * relay time is flat, so the only unit of cost is a launch. Every collective
 * except reduce-scatter is now above 1x there, and reduce-scatter cannot get
 * there with ncclSend/ncclRecv: measured, one ncclGroup of P2P ops costs ~0.038
 * ms while NCCL's ENTIRE fused reduce_scatter kernel costs ~0.035 ms, so even
 * deleting our trailing reduce leaves us behind.
 *
 * The only way out is a single kernel that moves the data AND reduces it, with
 * no group machinery at all. RCCL ships exactly that
 * (ncclSymRun_ReduceScatter_LL) but it is unreachable here:
 * comm->symmetricSupport requires ncclCuMemEnable(), and ncclIsCuMemSupported()
 * returns 0 unconditionally on AMD, so the symmetric window registration that
 * supplies peer pointers can never be enabled.
 *
 * This probe answers, with numbers rather than assumptions, whether the manual
 * route works:
 *
 *   1. Does hipIpcGetMemHandle / hipIpcOpenMemHandle work across the 8 ranks
 *      (separate processes) on this node, and what does the setup cost?
 *   2. Can a kernel STORE into a peer's IPC-mapped buffer and have the peer
 * read it correctly, with a flag-based handshake and system-scope fences?
 *   3. Does ncclCommRegister accept a user buffer here, and what does it cost?
 *      (If user buffers can be registered we could read peer sendbuffs directly
 *      and skip the staging store entirely.)
 *   4. What is the end-to-end latency of a one-shot reduce-scatter against
 *      ncclReduceScatter on the same 2 ranks, over the sizes that are below 1x?
 *
 * Everything here is measurement scaffolding. The numbers, not this file, are
 * the deliverable.
 */

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "nccl.h"

#define HIPCHECK_TEST(cmd)                                          \
  do {                                                              \
    hipError_t error = cmd;                                         \
    if (error != hipSuccess) {                                      \
      FAIL() << "HIP error: " << hipGetErrorString(error) << " at " \
             << __FILE__ << ":" << __LINE__;                        \
    }                                                               \
  } while (0)

#define HIPCHECK_SOFT(cmd, okFlag) \
  do {                             \
    hipError_t error = cmd;        \
    if (error != hipSuccess) {     \
      okFlag = false;              \
    }                              \
  } while (0)

#define NCCLCHECK_TEST(cmd)                                            \
  do {                                                                 \
    ncclResult_t result = cmd;                                         \
    if (result != ncclSuccess) {                                       \
      FAIL() << "NCCL error: " << ncclGetErrorString(result) << " at " \
             << __FILE__ << ":" << __LINE__;                           \
    }                                                                  \
  } while (0)

namespace {

// Largest per-rank output the probe stages, in elements. The staging region is
// nRanks slots of this, allocated once.
constexpr size_t kMaxSlotElems = 2 * 1024 * 1024; // 8 MiB of float
constexpr int kMaxRanks = 8;
// One flag per (source slot, block). Blocks handshake independently, so no
// global barrier and no co-residency requirement.
constexpr int kMaxBlocks = 32;
constexpr int kThreadsPerBlock = 256;

struct OneShotIpcMem {
  // Peer-visible: [nRanks slots of kMaxSlotElems floats][flags]
  float* staging{nullptr};
  uint32_t* flags{nullptr};
};

// Device view: for each rank, where its staging and flags live in OUR address
// space (self entry is the local allocation, peers are IPC-mapped).
struct PeerTable {
  float* staging[kMaxRanks];
  uint32_t* flags[kMaxRanks];
};

__device__ __forceinline__ bool epochReached(uint32_t got, uint32_t want) {
  // Wraparound-safe "got >= want", same trick RCCL's barrier uses.
  return (got - want) <= (uint32_t(-1) >> 1);
}

/**
 * One-shot 2-rank reduce-scatter.
 *
 * sendBuff holds 2*rc elements. Rank m must produce
 *   out[i] = (sendBuff[m*rc + i] + peerSendBuff[m*rc + i]) / divisor
 *
 * Step 1: store my foreign block (block 1-m) into the PEER's staging slot m.
 * Step 2: fence, then flag the peer's slot-m flag for this block.
 * Step 3: spin on MY slot-(1-m) flag for this block.
 * Step 4: out = own block m + my staging slot (1-m).
 *
 * All four steps in one launch. Each block handshakes only with the peer's
 * block of the same index, so progress does not require co-residency.
 */
__global__ void oneShotReduceScatter2(
    const float* __restrict__ sendBuff,
    float* __restrict__ out,
    PeerTable table,
    int myRank,
    int peerRank,
    int mySlot, // index the peer files my data under == my active index
    int peerSlot,
    size_t rc,
    uint32_t epoch,
    int divisor) {
  const size_t chunk = (rc + gridDim.x - 1) / gridDim.x;
  const size_t begin = chunk * blockIdx.x;
  const size_t end = (begin + chunk < rc) ? (begin + chunk) : rc;

  // Step 1: push my foreign block into the peer's staging slot.
  float* dst =
      table.staging[peerRank] + static_cast<size_t>(mySlot) * kMaxSlotElems;
  const float* src = sendBuff + static_cast<size_t>(peerSlot) * rc;
  for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    dst[i] = src[i];
  }

  // Step 2: make those stores visible, then raise the peer's flag.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    __atomic_store_n(
        &table.flags[peerRank][mySlot * kMaxBlocks + blockIdx.x],
        epoch,
        __ATOMIC_RELEASE);
  }

  // Step 3: wait for the peer's matching block.
  if (threadIdx.x == 0) {
    volatile uint32_t* mine =
        &table.flags[myRank][peerSlot * kMaxBlocks + blockIdx.x];
    while (!epochReached(
        __atomic_load_n(const_cast<uint32_t*>(mine), __ATOMIC_ACQUIRE),
        epoch)) {
    }
  }
  __syncthreads();

  // Step 4: reduce own contribution with what the peer staged.
  const float* staged =
      table.staging[myRank] + static_cast<size_t>(peerSlot) * kMaxSlotElems;
  const float* own = sendBuff + static_cast<size_t>(mySlot) * rc;
  const float scale = 1.0f / static_cast<float>(divisor);
  for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    float v = own[i] + staged[i];
    out[i] = (divisor > 1) ? (v * scale) : v;
  }
}

double nowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

} // namespace

class OneShotIpcProbeTest : public ::testing::Test {
 public:
  OneShotIpcProbeTest() = default;

  void SetUp() override {
    int localSize;
    std::tie(this->localRank, this->globalRank, this->numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    bool isServer = (this->globalRank == 0);
    if (checkTcpStoreEnv()) {
      server = createTcpStore(isServer);
    } else if (isServer) {
      server = createTcpStore(true);
    }
    this->comm = createNcclComm(
        this->globalRank,
        this->numRanks,
        this->localRank,
        false,
        nullptr,
        server.get());
    HIPCHECK_TEST(hipStreamCreate(&stream));
  }

  void TearDown() override {
    HIPCHECK_TEST(hipStreamDestroy(this->stream));
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(this->globalRank, server.get());
    }
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    server.reset();
  }

  template <typename T>
  void allGatherFixed(std::vector<T>& data) {
    NCCLCHECK_TEST(bootstrapAllGather(comm->bootstrap, data.data(), sizeof(T)));
  }

  ncclComm_t comm{nullptr};
  hipStream_t stream{};
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  std::unique_ptr<c10d::TCPStore> server{nullptr};
};

/**
 * Q1 + Q2: does cross-process IPC work here, what does setup cost, and can a
 * kernel store into a peer's mapped buffer correctly?
 */
TEST_F(OneShotIpcProbeTest, IpcSetupCostAndPeerStore) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }

  const size_t stagingBytes = kMaxRanks * kMaxSlotElems * sizeof(float);
  const size_t flagBytes = kMaxRanks * kMaxBlocks * sizeof(uint32_t);

  // IPC requires plain hipMalloc; the relay's ScratchBufferCache uses
  // cudaMallocAsync (mempool-backed), which CANNOT be exported. That is the
  // first concrete constraint this probe establishes.
  void* base = nullptr;
  HIPCHECK_TEST(hipMalloc(&base, stagingBytes + flagBytes));
  HIPCHECK_TEST(hipMemset(base, 0, stagingBytes + flagBytes));

  OneShotIpcMem local;
  local.staging = static_cast<float*>(base);
  local.flags =
      reinterpret_cast<uint32_t*>(static_cast<char*>(base) + stagingBytes);

  // --- hipIpcGetMemHandle ---
  double t0 = nowMs();
  hipIpcMemHandle_t myHandle{};
  bool exportOk = true;
  HIPCHECK_SOFT(hipIpcGetMemHandle(&myHandle, base), exportOk);
  double tExport = nowMs() - t0;
  ASSERT_TRUE(exportOk) << "hipIpcGetMemHandle failed -- one-shot IPC is not "
                        << "available on this platform";

  // --- exchange handles over the existing bootstrap ---
  std::vector<hipIpcMemHandle_t> handles(this->numRanks);
  handles[this->globalRank] = myHandle;
  t0 = nowMs();
  allGatherFixed(handles);
  double tExchange = nowMs() - t0;

  // --- hipIpcOpenMemHandle for each peer ---
  PeerTable table{};
  t0 = nowMs();
  int opened = 0;
  bool openOk = true;
  for (int r = 0; r < this->numRanks; r++) {
    if (r == this->globalRank) {
      table.staging[r] = local.staging;
      table.flags[r] = local.flags;
      continue;
    }
    void* peerBase = nullptr;
    hipError_t e = hipIpcOpenMemHandle(
        &peerBase, handles[r], hipIpcMemLazyEnablePeerAccess);
    if (e != hipSuccess) {
      openOk = false;
      break;
    }
    opened++;
    table.staging[r] = static_cast<float*>(peerBase);
    table.flags[r] = reinterpret_cast<uint32_t*>(
        static_cast<char*>(peerBase) + stagingBytes);
  }
  double tOpen = nowMs() - t0;
  ASSERT_TRUE(openOk) << "hipIpcOpenMemHandle failed after " << opened
                      << " peers";

  if (this->globalRank == 0) {
    std::cout << "\n[probe] IPC setup, one-time, 8 ranks:\n"
              << "  hipIpcGetMemHandle       " << std::fixed
              << std::setprecision(3) << tExport << " ms\n"
              << "  bootstrapAllGather(8x64B) " << tExchange << " ms\n"
              << "  hipIpcOpenMemHandle x7    " << tOpen << " ms\n"
              << "  staging+flags allocated   "
              << (stagingBytes + flagBytes) / 1024 << " KiB\n";
  }

  // --- Q2: correctness of a peer store + flag handshake ---
  // Ranks 0 and 1 act as the 2-rank active pair; everyone else idles.
  const bool active = this->globalRank < 2;
  if (active) {
    const size_t rc = 4096;
    const int mySlot = this->globalRank;
    const int peerSlot = 1 - this->globalRank;
    const int peerRank = 1 - this->globalRank;

    float* sendBuff = nullptr;
    float* out = nullptr;
    HIPCHECK_TEST(hipMalloc(&sendBuff, 2 * rc * sizeof(float)));
    HIPCHECK_TEST(hipMalloc(&out, rc * sizeof(float)));

    // Position-dependent fill so a block/offset permutation cannot pass: block
    // b of rank r holds r*1000 + b*100 + (i % 97).
    std::vector<float> host(2 * rc);
    for (int b = 0; b < 2; b++) {
      for (size_t i = 0; i < rc; i++) {
        host[b * rc + i] = static_cast<float>(this->globalRank) * 1000.0f +
            static_cast<float>(b) * 100.0f + static_cast<float>(i % 97);
      }
    }
    HIPCHECK_TEST(hipMemcpy(
        sendBuff, host.data(), 2 * rc * sizeof(float), hipMemcpyHostToDevice));

    const int blocks = 8;
    hipLaunchKernelGGL(
        oneShotReduceScatter2,
        dim3(blocks),
        dim3(kThreadsPerBlock),
        0,
        stream,
        sendBuff,
        out,
        table,
        this->globalRank,
        peerRank,
        mySlot,
        peerSlot,
        rc,
        /*epoch=*/1u,
        /*divisor=*/1);
    HIPCHECK_TEST(hipStreamSynchronize(stream));

    std::vector<float> got(rc);
    HIPCHECK_TEST(
        hipMemcpy(got.data(), out, rc * sizeof(float), hipMemcpyDeviceToHost));

    // out[i] = own block[mySlot] + peer block[mySlot]
    //        = (me*1000 + mySlot*100 + i%97) + (peer*1000 + mySlot*100 + i%97)
    int mismatches = 0;
    for (size_t i = 0; i < rc; i++) {
      const float expected = static_cast<float>(this->globalRank) * 1000.0f +
          static_cast<float>(mySlot) * 100.0f + static_cast<float>(i % 97) +
          static_cast<float>(peerRank) * 1000.0f +
          static_cast<float>(mySlot) * 100.0f + static_cast<float>(i % 97);
      if (got[i] != expected) {
        if (mismatches < 4) {
          std::cout << "[probe] R" << this->globalRank << " mismatch at " << i
                    << ": got " << got[i] << " want " << expected << "\n";
        }
        mismatches++;
      }
    }
    EXPECT_EQ(mismatches, 0)
        << "one-shot peer store + handshake produced wrong results on rank "
        << this->globalRank;

    HIPCHECK_TEST(hipFree(sendBuff));
    HIPCHECK_TEST(hipFree(out));
  }

  // --- Q4: latency against ncclReduceScatter over the sub-1x sizes ---
  {
    // One 2-rank sub-comm for the whole sweep. ncclCommSplit is collective over
    // the parent, so every rank calls it; ranks >= 2 get NOCOLOR and a null
    // comm back.
    ncclComm_t sub = nullptr;
    const int color = (this->globalRank < 2) ? 0 : NCCL_SPLIT_NOCOLOR;
    NCCLCHECK_TEST(
        ncclCommSplit(this->comm, color, this->globalRank, &sub, nullptr));

    hipEvent_t evStart, evStop;
    HIPCHECK_TEST(hipEventCreate(&evStart));
    HIPCHECK_TEST(hipEventCreate(&evStop));

    const std::vector<size_t> inputBytes = {
        4u << 10,
        9u << 10,
        18u << 10,
        36u << 10,
        72u << 10,
        144u << 10,
        288u << 10,
        576u << 10,
        1152u << 10,
        2304u << 10,
        4608u << 10,
        9216u << 10,
        13824u << 10};

    if (this->globalRank == 0) {
      std::cout << "\n[probe] 2-rank reduce-scatter, float, median of 10 reps"
                << " (ms)\n"
                << "    input    oneshot       nccl   kernel speedup\n";
    }

    const bool act = this->globalRank < 2;
    for (size_t ib : inputBytes) {
      const size_t rc = ib / 2 / sizeof(float);
      if (rc > kMaxSlotElems) {
        if (this->globalRank == 0) {
          std::cout << std::setw(8) << (ib >> 10) << "K   (over staging cap)\n";
        }
        continue;
      }
      float* sendBuff = nullptr;
      float* out = nullptr;
      HIPCHECK_TEST(hipMalloc(&sendBuff, 2 * rc * sizeof(float)));
      HIPCHECK_TEST(hipMalloc(&out, rc * sizeof(float)));
      HIPCHECK_TEST(hipMemset(sendBuff, 1, 2 * rc * sizeof(float)));

      const int mySlot = this->globalRank;
      const int peerSlot = 1 - this->globalRank;
      const int peerRank = 1 - this->globalRank;
      uint32_t epoch = 16;

      auto launchOneShot = [&]() {
        hipLaunchKernelGGL(
            oneShotReduceScatter2,
            dim3(kMaxBlocks),
            dim3(kThreadsPerBlock),
            0,
            stream,
            sendBuff,
            out,
            table,
            this->globalRank,
            peerRank,
            mySlot,
            peerSlot,
            rc,
            epoch++,
            1);
      };

      // Streamed regime: 20 back-to-back launches / 20. Successive calls
      // pipeline, so this isolates the KERNEL cost and strips the per-call
      // launch overhead. That is deliberate -- it is the number that says
      // whether a one-shot kernel is intrinsically cheaper than NCCL's fused
      // one, and where the crossover sits. The per-call regime a caller
      // actually pays is measured by the checked-in single-group sweep, so it
      // is not duplicated here.
      double oneShotMs = 0.0;
      if (act) {
        for (int it = 0; it < 20; it++) {
          launchOneShot();
        }
        HIPCHECK_TEST(hipStreamSynchronize(stream));
        // A silently-failed launch makes the timing read ~0, so check.
        HIPCHECK_TEST(hipGetLastError());
        std::vector<double> reps;
        for (int rep = 0; rep < 10; rep++) {
          HIPCHECK_TEST(hipEventRecord(evStart, stream));
          for (int it = 0; it < 20; it++) {
            launchOneShot();
          }
          HIPCHECK_TEST(hipEventRecord(evStop, stream));
          HIPCHECK_TEST(hipEventSynchronize(evStop));
          float ms = 0.f;
          HIPCHECK_TEST(hipEventElapsedTime(&ms, evStart, evStop));
          reps.push_back(static_cast<double>(ms) / 20.0);
        }
        std::sort(reps.begin(), reps.end());
        oneShotMs = reps[reps.size() / 2];
      }

      double ncclMs = 0.0;
      if (sub != nullptr) {
        for (int it = 0; it < 20; it++) {
          NCCLCHECK_TEST(ncclReduceScatter(
              sendBuff, out, rc, ncclFloat, ncclSum, sub, stream));
        }
        HIPCHECK_TEST(hipStreamSynchronize(stream));
        std::vector<double> reps;
        for (int rep = 0; rep < 10; rep++) {
          HIPCHECK_TEST(hipEventRecord(evStart, stream));
          for (int it = 0; it < 20; it++) {
            NCCLCHECK_TEST(ncclReduceScatter(
                sendBuff, out, rc, ncclFloat, ncclSum, sub, stream));
          }
          HIPCHECK_TEST(hipEventRecord(evStop, stream));
          HIPCHECK_TEST(hipEventSynchronize(evStop));
          float ms = 0.f;
          HIPCHECK_TEST(hipEventElapsedTime(&ms, evStart, evStop));
          reps.push_back(static_cast<double>(ms) / 20.0);
        }
        std::sort(reps.begin(), reps.end());
        ncclMs = reps[reps.size() / 2];
      }

      if (this->globalRank == 0) {
        std::cout << std::setw(8) << (ib >> 10) << "K " << std::fixed
                  << std::setprecision(4) << std::setw(10) << oneShotMs << " "
                  << std::setw(10) << ncclMs << "   " << std::setprecision(2)
                  << (oneShotMs > 0 ? ncclMs / oneShotMs : 0.0) << "x\n";
      }

      HIPCHECK_TEST(hipFree(sendBuff));
      HIPCHECK_TEST(hipFree(out));
    }

    HIPCHECK_TEST(hipEventDestroy(evStart));
    HIPCHECK_TEST(hipEventDestroy(evStop));
    if (sub != nullptr) {
      ncclCommDestroy(sub);
    }
  }

  // --- Q3: does ncclCommRegister work here, and what does it cost? ---
  {
    void* ub = nullptr;
    HIPCHECK_TEST(hipMalloc(&ub, 1 << 20));
    void* handle = nullptr;
    double t = nowMs();
    ncclResult_t rr = ncclCommRegister(this->comm, ub, 1 << 20, &handle);
    double tReg = nowMs() - t;
    if (this->globalRank == 0) {
      std::cout << "\n[probe] ncclCommRegister(1 MiB): "
                << ncclGetErrorString(rr) << ", " << std::setprecision(3)
                << tReg << " ms, handle=" << (handle ? "non-null" : "null")
                << "\n";
    }
    if (rr == ncclSuccess && handle != nullptr) {
      t = nowMs();
      ncclResult_t dr = ncclCommDeregister(this->comm, handle);
      if (this->globalRank == 0) {
        std::cout << "[probe] ncclCommDeregister: " << ncclGetErrorString(dr)
                  << ", " << (nowMs() - t) << " ms\n";
      }
    }
    HIPCHECK_TEST(hipFree(ub));
  }

  for (int r = 0; r < this->numRanks; r++) {
    if (r != this->globalRank && table.staging[r] != nullptr) {
      hipIpcCloseMemHandle(table.staging[r]);
    }
  }
  HIPCHECK_TEST(hipFree(base));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
