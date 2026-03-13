// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/pipes/window/HostWindow.h"

using comms::pipes::DeviceWindow;
using comms::pipes::WindowConfig;
using ctran::CtranWin;

class CtranWinDeviceEnvironment : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();
    setenv("NCCL_CTRAN_USE_PIPES", "1", 1);
    setenv("NCCL_DEBUG", "INFO", 1);
  }
};

class CtranWinDeviceTest : public ctran::CtranDistTestFixture {
 public:
  static constexpr size_t kWinSize = 4096;

  // Helper: create a comm, allocate+exchange a window, return both.
  std::pair<std::unique_ptr<CtranComm>, CtranWin*> makeCommAndWindow() {
    auto comm = makeCtranComm();
    CtranWin* win = nullptr;
    void* baseptr = nullptr;
    auto res = ctran::ctranWinAllocate(kWinSize, comm.get(), &baseptr, &win);
    EXPECT_EQ(res, commSuccess);
    EXPECT_NE(win, nullptr);
    EXPECT_NE(baseptr, nullptr);
    return {std::move(comm), win};
  }
};

// Verify get_device_win() returns a correctly populated CtranWinDevice.
TEST_F(CtranWinDeviceTest, GetDeviceWin) {
  auto [comm, win] = makeCommAndWindow();
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  WindowConfig config{.peerSignalCount = win->signalSize};
  DeviceWindow devWin{};
  ASSERT_EQ(win->get_device_win(&devWin, config), commSuccess);

  // Verify rank and n_ranks match the communicator's values
  EXPECT_EQ(devWin.rank(), globalRank);
  EXPECT_EQ(devWin.n_ranks(), numRanks);

  // Verify peer count is consistent with n_ranks
  EXPECT_EQ(devWin.num_peers(), numRanks - 1);

  // NVL and IBGDA peer sets overlap: IBGDA is the universal transport
  // covering all non-self peers, while NVL is additionally available for
  // NVLink-connected peers. Verify each independently.
  EXPECT_EQ(devWin.num_nvl_peers(), numRanks - 1);
  EXPECT_EQ(devWin.num_ibgda_peers(), numRanks - 1);

  ctran::ctranWinFree(win);
}

// Verify get_device_win() returns the same result on repeated calls.
TEST_F(CtranWinDeviceTest, GetDeviceWinIdempotent) {
  auto [comm, win] = makeCommAndWindow();
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  WindowConfig config{.peerSignalCount = win->signalSize};
  DeviceWindow devWin1{};
  DeviceWindow devWin2{};
  ASSERT_EQ(win->get_device_win(&devWin1, config), commSuccess);
  ASSERT_EQ(win->get_device_win(&devWin2, config), commSuccess);

  EXPECT_EQ(devWin1.rank(), devWin2.rank());
  EXPECT_EQ(devWin1.n_ranks(), devWin2.n_ranks());
  EXPECT_EQ(devWin1.num_nvl_peers(), devWin2.num_nvl_peers());
  EXPECT_EQ(devWin1.num_ibgda_peers(), devWin2.num_ibgda_peers());

  ctran::ctranWinFree(win);
}

// Verify get_device_win() fails when MultiPeerTransport is not initialized.
TEST_F(CtranWinDeviceTest, GetDeviceWinNoTransportFails) {
  auto [comm, win] = makeCommAndWindow();

  // Simulate no-pipes by nulling out the transport
  comm->multiPeerTransport_.reset();

  // get_device_win should fail since multiPeerTransport_ is null
  WindowConfig config{.peerSignalCount = win->signalSize};
  DeviceWindow devWin{};
  EXPECT_NE(win->get_device_win(&devWin, config), commSuccess);

  ctran::ctranWinFree(win);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranWinDeviceEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
