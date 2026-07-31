// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>

#include <gtest/gtest.h>

#include "bootstrap.h"
#include "meta/NcclxConfig.h"

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

TEST(BootstrapInitTest, DISABLED_RingAllGatherFailureClosesPeerSockets) {
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
