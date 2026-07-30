// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/nccl/TorchCommNCCLBootstrap.hpp"
#include "comms/torchcomms/nccl/tests/unit/cpp/mocks/CudaMock.hpp"
#include "comms/torchcomms/nccl/tests/unit/cpp/mocks/NcclMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms::test {

constexpr std::chrono::seconds kTimeout{60};

class TorchCommNCCLBootstrapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    store_ = c10::make_intrusive<c10d::HashStore>();

    // Use a CPU device so the constructor does not need a real GPU; all CUDA
    // calls are mocked.
    device_ = at::Device(at::DeviceType::CPU, 0);

    nccl_mock_ = std::make_shared<NiceMock<NcclMock>>();
    cuda_mock_ = std::make_shared<NiceMock<CudaMock>>();
    cuda_mock_->setupDefaultBehaviors();
  }

  void TearDown() override {
    unsetenv("TORCHCOMM_RANK");
    unsetenv("TORCHCOMM_SIZE");
    unsetenv("MASTER_ADDR");
    unsetenv("MASTER_PORT");
  }

  void setupRankAndSize(int rank, int size) {
    setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
    setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);
  }

  std::unique_ptr<TorchCommNCCLBootstrap> createBootstrap(
      c10::intrusive_ptr<c10d::Store> store = nullptr) {
    if (!store) {
      store = store_;
    }
    return std::make_unique<TorchCommNCCLBootstrap>(
        store, device_, nccl_mock_, cuda_mock_, kTimeout);
  }

  static ncclUniqueId makeUniqueId(int fill) {
    ncclUniqueId id{};
    // NOLINTNEXTLINE(facebook-hte-BadMemset)
    memset(&id, fill, sizeof(id));
    return id;
  }

  c10::intrusive_ptr<c10d::Store> store_;
  at::Device device_{at::DeviceType::CPU, 0};
  std::shared_ptr<NiceMock<NcclMock>> nccl_mock_;
  std::shared_ptr<NiceMock<CudaMock>> cuda_mock_;
};

// The bedrock contract: the store key is a *pure function of the comm name*.
// Feed it the same name and you always get the same key back; feed it two
// different names and the keys never alias. No hidden per-call or per-rank
// state (the old process-wide counter) is allowed to leak in -- killing that
// leak is the entire reason this change exists.
TEST_F(
    TorchCommNCCLBootstrapTest,
    GetNCCLStoreKey_SameName_IsDeterministicAndNameScoped) {
  EXPECT_EQ(TorchCommNCCLBootstrap::getNCCLStoreKeyPrefix(), "nccl_storekey_");

  const std::string key_a1 = TorchCommNCCLBootstrap::getNCCLStoreKey("tp");
  const std::string key_a2 = TorchCommNCCLBootstrap::getNCCLStoreKey("tp");
  EXPECT_EQ(key_a1, key_a2);
  EXPECT_EQ(key_a1, "nccl_storekey_tp");

  EXPECT_NE(
      TorchCommNCCLBootstrap::getNCCLStoreKey("tp"),
      TorchCommNCCLBootstrap::getNCCLStoreKey("dp"));
}

// Fail-fast guardrail: with no TORCHCOMM_RANK / TORCHCOMM_SIZE in the
// environment, the bootstrap has no idea which rank it is or how big the world
// is. It must refuse to construct rather than silently invent defaults and wire
// up a bogus communicator.
TEST_F(
    TorchCommNCCLBootstrapTest,
    Constructor_EnvironmentVariablesMissing_Throws) {
  EXPECT_THROW(createBootstrap(), std::runtime_error);
}

// Rank 0 plays the publisher in the rendezvous: it mints the NCCL unique ID
// and drops it in the store under nccl_storekey_<name> for everyone else to
// pick up. This exercises the publish half -- the right bytes must land under
// the key derived from the comm's name.
TEST_F(
    TorchCommNCCLBootstrapTest,
    CreateNcclComm_Rank0_StoresUniqueIdUnderNameKey) {
  setupRankAndSize(0, 2);
  auto bootstrap = createBootstrap();

  const ncclUniqueId expected_id = makeUniqueId(0x42);

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ncclComm_t comm = bootstrap->createNcclComm("mycomm");
  EXPECT_NE(comm, nullptr);

  auto stored = store_->get("nccl_storekey_mycomm");
  ASSERT_EQ(stored.size(), sizeof(ncclUniqueId));
  ncclUniqueId stored_id{};
  memcpy(&stored_id, stored.data(), sizeof(stored_id));
  EXPECT_EQ(memcmp(&stored_id, &expected_id, sizeof(ncclUniqueId)), 0);
}

// The mirror image: a non-zero rank is a subscriber. Rank 0 has already
// published, so this rank must locate and read the unique ID back from that
// same name-derived key. This exercises the read half of the rendezvous.
TEST_F(
    TorchCommNCCLBootstrapTest,
    CreateNcclComm_NonRank0_ReadsUniqueIdFromNameKey) {
  setupRankAndSize(1, 2);

  // Pre-populate the store as if rank 0 had already published under the
  // name-scoped key.
  const ncclUniqueId expected_id = makeUniqueId(0x42);
  std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
  memcpy(id_vec.data(), &expected_id, sizeof(expected_id));
  store_->set("nccl_storekey_mycomm", id_vec);

  auto bootstrap = createBootstrap();
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 1, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  EXPECT_NE(bootstrap->createNcclComm("mycomm"), nullptr);
}

// The headline test -- a direct reenactment of the bug this change fixes.
// Picture two ranks that lived different lives: rank 0 joined a couple of
// earlier members-only groups, rank 1 sat those out. Later they meet in a
// shared comm. The store key for that shared comm must not remember any of that
// history -- both ranks derive it purely from the name and rendezvous cleanly.
// Under the old process-wide counter their keys drifted apart and the bootstrap
// deadlocked; keyed on the name, it just works.
TEST_F(
    TorchCommNCCLBootstrapTest,
    CreateNcclComm_RanksWithDivergentCommHistory_AgreeOnKey) {
  const ncclUniqueId id = makeUniqueId(0x42);

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(id), Return(ncclSuccess)));
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, _, _, _, _))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  // Rank 0 participates in two earlier comms and the later shared "pp" comm.
  setupRankAndSize(0, 2);
  auto rank0 = createBootstrap();
  EXPECT_NO_THROW(rank0->createNcclComm("dp"));
  EXPECT_NO_THROW(rank0->createNcclComm("tp"));
  EXPECT_NO_THROW(rank0->createNcclComm("pp"));

  // Rank 1 skipped the earlier comms and only joins "pp"; it must read the key
  // rank 0 wrote for "pp", which is name-scoped and independent of comm count.
  setupRankAndSize(1, 2);
  auto rank1 = createBootstrap();
  EXPECT_NO_THROW(rank1->createNcclComm("pp"));

  auto stored = store_->get("nccl_storekey_pp");
  EXPECT_EQ(stored.size(), sizeof(ncclUniqueId));
}

// Sad-path hygiene: if NCCL can't even mint a unique ID, the bootstrap has
// nothing to broadcast. It must abort with a clear, wrapped error instead of
// marching on and publishing garbage to the store.
TEST_F(TorchCommNCCLBootstrapTest, CreateNcclComm_GetUniqueIdFailure_Throws) {
  setupRankAndSize(0, 2);
  auto bootstrap = createBootstrap();

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(Return(ncclInvalidArgument));
  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInvalidArgument))
      .WillOnce(Return("Invalid argument"));

  EXPECT_THROW(
      {
        try {
          bootstrap->createNcclComm("mycomm");
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("Failed to get NCCL unique ID") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// Sad-path hygiene, one step later: the ID exchange succeeds but the actual
// communicator init fails. That failure has to surface (wrapped) to the caller
// rather than handing back a half-built comm that detonates mid-collective.
TEST_F(
    TorchCommNCCLBootstrapTest,
    CreateNcclComm_InitRankConfigFailure_Throws) {
  setupRankAndSize(0, 2);
  auto bootstrap = createBootstrap();

  const ncclUniqueId expected_id = makeUniqueId(0x42);

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(Return(ncclInvalidArgument));
  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInvalidArgument))
      .WillOnce(Return("Invalid argument"));

  EXPECT_THROW(
      {
        try {
          bootstrap->createNcclComm("mycomm");
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Failed to initialize NCCL communicator") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// Corruption guard: a non-zero rank reads the unique ID straight out of the
// store. If what comes back isn't a full ncclUniqueId (truncated or garbage),
// the bootstrap must reject it outright rather than memcpy nonsense into an
// ncclUniqueId and hang the whole job on a malformed handshake.
TEST_F(TorchCommNCCLBootstrapTest, CreateNcclComm_InvalidStoreDataSize_Throws) {
  setupRankAndSize(1, 2);

  // Store data of the wrong size under the name-scoped key.
  std::vector<uint8_t> invalid_vec(10);
  store_->set("nccl_storekey_mycomm", invalid_vec);

  auto bootstrap = createBootstrap();

  EXPECT_THROW(
      {
        try {
          bootstrap->createNcclComm("mycomm");
        } catch (const std::runtime_error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find("Invalid NCCL unique ID size") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

} // namespace torch::comms::test
