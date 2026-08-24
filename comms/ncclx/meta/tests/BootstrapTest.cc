// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <future>

#include <gtest/gtest.h>

#include "bootstrap.h"
#include "meta/NcclxConfig.h"
#include "meta/ctran-integration/BootstrapCleanup.h"

namespace {

using namespace std::chrono_literals;

enum class RecvStage {
  rootMagic,
  rootType,
  rootMessageSize,
  rootPayload,
  ringMagic,
  ringType,
  ringMessageSize,
  done,
  invalid,
};

struct RecvFaultState {
  bool shouldInject(std::size_t requested) const {
    return stage == RecvStage::ringMessageSize && remaining == 0 &&
        requested == sizeof(int);
  }

  void record(std::size_t requested, ssize_t received) {
    if (received <= 0 || stage == RecvStage::done ||
        stage == RecvStage::invalid) {
      return;
    }

    if (remaining == 0) {
      switch (stage) {
        case RecvStage::rootMagic:
        case RecvStage::ringMagic:
          remaining = sizeof(uint64_t);
          break;
        case RecvStage::rootType:
        case RecvStage::rootMessageSize:
        case RecvStage::ringType:
        case RecvStage::ringMessageSize:
          remaining = sizeof(int);
          break;
        case RecvStage::rootPayload:
          remaining = requested;
          break;
        case RecvStage::done:
        case RecvStage::invalid:
          return;
      }
    }

    if (requested != remaining ||
        static_cast<std::size_t>(received) > remaining) {
      stage = RecvStage::invalid;
      return;
    }

    remaining -= received;
    if (remaining == 0) {
      stage = static_cast<RecvStage>(static_cast<int>(stage) + 1);
    }
  }

  RecvStage stage{RecvStage::rootMagic};
  std::size_t remaining{0};
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
thread_local RecvFaultState* activeRecvFault = nullptr;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::atomic<int> injectedFaults{0};

struct NetCloseState {
  int sendCalls{0};
  int receiveCalls{0};
  int listenCalls{0};

  bool operator==(const NetCloseState& other) const {
    return sendCalls == other.sendCalls && receiveCalls == other.receiveCalls &&
        listenCalls == other.listenCalls;
  }
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
NetCloseState netCloseState;

ncclResult_t closeSend(void*) {
  ++netCloseState.sendCalls;
  return ncclSuccess;
}

ncclResult_t failCloseSend(void*) {
  ++netCloseState.sendCalls;
  return ncclSystemError;
}

ncclResult_t closeReceive(void*) {
  ++netCloseState.receiveCalls;
  return ncclSuccess;
}

ncclResult_t closeListen(void*) {
  ++netCloseState.listenCalls;
  return ncclSuccess;
}

bootstrapState* allocateBootstrapState(ncclNet_t* net, bool ringUsesOobNet) {
  auto* state =
      static_cast<bootstrapState*>(std::calloc(1, sizeof(bootstrapState)));
  if (state != nullptr) {
    state->net = net;
    state->ringUsesOobNet = ringUsesOobNet;
  }
  return state;
}

class ScopedRecvFault {
 public:
  explicit ScopedRecvFault(RecvFaultState& state) {
    activeRecvFault = &state;
  }

  ~ScopedRecvFault() {
    activeRecvFault = nullptr;
  }

  ScopedRecvFault(const ScopedRecvFault&) = delete;
  ScopedRecvFault& operator=(const ScopedRecvFault&) = delete;
};

} // namespace

extern "C" ssize_t
recv(int socket, void* buffer, std::size_t length, int flags) {
  if (activeRecvFault != nullptr && activeRecvFault->shouldInject(length)) {
    activeRecvFault->stage = RecvStage::done;
    injectedFaults.fetch_add(1, std::memory_order_relaxed);
    return 0;
  }

  const ssize_t result = static_cast<ssize_t>(
      syscall(SYS_recvfrom, socket, buffer, length, flags, nullptr, nullptr));
  if (activeRecvFault != nullptr) {
    activeRecvFault->record(length, result);
  }
  return result;
}

namespace {

TEST(BootstrapCleanupTest, OobNetFailureStillClosesRemainingEndpoints) {
  netCloseState = {};
  ncclNet_t net{};
  net.closeSend = failCloseSend;
  net.closeRecv = closeReceive;
  net.closeListen = closeListen;

  auto* state = allocateBootstrapState(&net, true);
  if (state == nullptr) {
    FAIL() << "Failed to allocate bootstrap state";
  }

  EXPECT_EQ(bootstrapAbort(state), ncclSystemError);
  const NetCloseState expected{1, 1, 1};
  EXPECT_EQ(netCloseState, expected);
}

TEST(BootstrapCleanupTest, NormalCloseUsesStoredOobNetTransport) {
  netCloseState = {};
  ncclNet_t net{};
  net.closeSend = closeSend;
  net.closeRecv = closeReceive;
  net.closeListen = closeListen;

  auto* state = allocateBootstrapState(&net, true);
  if (state == nullptr) {
    FAIL() << "Failed to allocate bootstrap state";
  }

  EXPECT_EQ(bootstrapClose(state), ncclSuccess);
  const NetCloseState expected{1, 1, 1};
  EXPECT_EQ(netCloseState, expected);
}

TEST(BootstrapCleanupTest, StoredSocketTransportDoesNotUseNetCleanup) {
  netCloseState = {};
  ncclNet_t net{};
  net.closeSend = closeSend;
  net.closeRecv = closeReceive;
  net.closeListen = closeListen;

  auto* state = allocateBootstrapState(&net, false);
  if (state == nullptr) {
    FAIL() << "Failed to allocate bootstrap state";
  }

  EXPECT_EQ(bootstrapAbort(state), ncclSuccess);
  const NetCloseState expected{};
  EXPECT_EQ(netCloseState, expected);
}

TEST(BootstrapCleanupTest, RingFailureCleanupIsIdempotent) {
  netCloseState = {};
  ncclNet_t net{};
  net.closeSend = closeSend;
  net.closeRecv = closeReceive;
  net.closeListen = closeListen;

  ncclComm comm{};
  comm.bootstrap = allocateBootstrapState(&net, true);
  if (comm.bootstrap == nullptr) {
    FAIL() << "Failed to allocate bootstrap state";
  }
  auto* proxySocket =
      static_cast<ncclSocket*>(std::calloc(1, sizeof(ncclSocket)));
  if (proxySocket == nullptr) {
    std::free(comm.bootstrap);
    FAIL() << "Failed to allocate proxy socket";
  }

  ncclx::abortBootstrapAfterRingAllInfoFailure(&comm, proxySocket);
  EXPECT_EQ(comm.bootstrap, nullptr);
  EXPECT_EQ(proxySocket, nullptr);
  const NetCloseState expected{1, 1, 1};
  EXPECT_EQ(netCloseState, expected);

  ncclx::abortBootstrapAfterRingAllInfoFailure(&comm, proxySocket);
  EXPECT_EQ(netCloseState, expected);
}

TEST(BootstrapInitTest, RingAllGatherFailureClosesPeerSockets) {
  ASSERT_EQ(bootstrapNetInit(), ncclSuccess);

  ncclBootstrapHandle handle{};
  ASSERT_EQ(bootstrapGetUniqueId(&handle, nullptr), ncclSuccess);

  std::array<ncclComm, 2> comms{};
  std::array<ncclx::Config, 2> configs{};
  std::array<uint32_t, 2> abortFlags{};
  for (int rank = 0; rank < 2; ++rank) {
    comms[rank].rank = rank;
    comms[rank].nRanks = 2;
    comms[rank].cudaDev = 0;
    comms[rank].nvmlDev = 0;
    comms[rank].abortFlag = &abortFlags[rank];
    comms[rank].config.ncclxConfig = &configs[rank];
  }

  injectedFaults.store(0, std::memory_order_relaxed);
  auto runRank = [&](int rank) {
    RecvFaultState faultState;
    if (rank == 0) {
      ScopedRecvFault fault(faultState);
      return bootstrapInit(1, &handle, &comms[rank], nullptr);
    }
    return bootstrapInit(1, &handle, &comms[rank], nullptr);
  };

  auto rank0 = std::async(std::launch::async, runRank, 0);
  auto rank1 = std::async(std::launch::async, runRank, 1);

  const bool rank0Ready = rank0.wait_for(10s) == std::future_status::ready;
  const bool rank1Ready = rank1.wait_for(10s) == std::future_status::ready;
  if (!rank0Ready || !rank1Ready) {
    __atomic_store_n(&abortFlags[0], 1, __ATOMIC_RELEASE);
    __atomic_store_n(&abortFlags[1], 1, __ATOMIC_RELEASE);
  }

  ASSERT_EQ(rank0.wait_for(5s), std::future_status::ready);
  ASSERT_EQ(rank1.wait_for(5s), std::future_status::ready);
  EXPECT_TRUE(rank0Ready);
  EXPECT_TRUE(rank1Ready);
  EXPECT_EQ(rank0.get(), ncclRemoteError);
  EXPECT_NE(rank1.get(), ncclSuccess);
  EXPECT_EQ(injectedFaults.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(comms[0].bootstrap, nullptr);
  EXPECT_EQ(comms[1].bootstrap, nullptr);
}

} // namespace
