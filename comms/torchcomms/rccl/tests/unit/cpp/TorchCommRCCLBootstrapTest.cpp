// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/rccl/TorchCommRCCLBootstrap.hpp"
#include "comms/torchcomms/rccl/tests/unit/cpp/mocks/HipMock.hpp"
#include "comms/torchcomms/rccl/tests/unit/cpp/mocks/RcclMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms::test {

constexpr std::chrono::seconds kTimeout{60};

class TorchCommRCCLBootstrapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create hash store for communication
    store_ = c10::make_intrusive<c10d::HashStore>();

    // Set up device - use CPU device because we're mocking HIP
    device_ = at::Device(at::DeviceType::CPU, 0);

    // Create fresh mocks for each test
    rccl_mock_ = std::make_shared<NiceMock<RcclMock>>();
    hip_mock_ = std::make_shared<NiceMock<HipMock>>();

    // Reset the static counter to a known state
    // We'll access it through the public interface
    TorchCommRCCLBootstrap::getRCCLStoreKey(); // This increments counter
    initial_counter_ = TorchCommRCCLBootstrap::getRCCLStoreKeyCounter();
  }

  void TearDown() override {
    // Clean up environment variables
    unsetenv("TORCHCOMM_RANK");
    unsetenv("TORCHCOMM_SIZE");
    unsetenv("MASTER_ADDR");
    unsetenv("MASTER_PORT");
  }

  void setupRankAndSize(int rank, int size) {
    setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
    setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);
  }

  std::unique_ptr<TorchCommRCCLBootstrap> createBootstrap(
      c10::intrusive_ptr<c10d::Store> store = nullptr) {
    if (!store) {
      store = store_;
    }
    return std::make_unique<TorchCommRCCLBootstrap>(
        store, device_, rccl_mock_, hip_mock_, kTimeout);
  }

  c10::intrusive_ptr<c10d::Store> store_;
  at::Device device_{at::DeviceType::CPU, 0};
  std::shared_ptr<NiceMock<RcclMock>> rccl_mock_;
  std::shared_ptr<NiceMock<HipMock>> hip_mock_;
  int initial_counter_{-1};
};

TEST_F(TorchCommRCCLBootstrapTest, StaticMethodsStoreKeyGeneration) {
  // Test that store key generation works correctly with counter
  std::string prefix = TorchCommRCCLBootstrap::getRCCLStoreKeyPrefix();
  EXPECT_EQ(prefix, "rccl_storekey_");

  int counter_before = TorchCommRCCLBootstrap::getRCCLStoreKeyCounter();
  std::string key1 = TorchCommRCCLBootstrap::getRCCLStoreKey();
  int counter_after = TorchCommRCCLBootstrap::getRCCLStoreKeyCounter();

  EXPECT_EQ(counter_after, counter_before + 1);
  EXPECT_EQ(key1, prefix + std::to_string(counter_before));

  // Test that subsequent calls increment the counter
  std::string key2 = TorchCommRCCLBootstrap::getRCCLStoreKey();
  int final_counter = TorchCommRCCLBootstrap::getRCCLStoreKeyCounter();

  EXPECT_EQ(final_counter, counter_after + 1);
  EXPECT_EQ(key2, prefix + std::to_string(counter_after));
  EXPECT_NE(key1, key2);
}

TEST_F(TorchCommRCCLBootstrapTest, GetRankAndSizeFromEnvironment) {
  setupRankAndSize(1, 4);

  auto bootstrap = createBootstrap();

  // Set up store with unique ID (as if rank 0 already stored it)
  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));
  std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
  memcpy(id_vec.data(), &expected_id, sizeof(expected_id));

  std::string store_key = TorchCommRCCLBootstrap::getRCCLStoreKeyPrefix() +
      std::to_string(TorchCommRCCLBootstrap::getRCCLStoreKeyCounter());
  store_->set(store_key, id_vec);

  // Set up mock expectations
  EXPECT_CALL(*rccl_mock_, commInitRankConfig(_, 4, _, 1, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  EXPECT_NO_THROW(bootstrap->createNcclComm("test_comm"));
}

TEST_F(TorchCommRCCLBootstrapTest, GetRankAndSizeEnvironmentVariablesMissing) {
  // Don't set environment variables
  EXPECT_THROW(createBootstrap(), std::runtime_error);
}

TEST_F(TorchCommRCCLBootstrapTest, ExchangeUniqueIdRank0) {
  setupRankAndSize(0, 2);

  auto bootstrap = createBootstrap();

  // Rank 0 should generate unique ID and store it
  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*rccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  EXPECT_CALL(*rccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ncclComm_t comm = bootstrap->createNcclComm("test_comm");
  EXPECT_NE(comm, nullptr);

  // Verify the unique ID was stored
  std::string store_key = TorchCommRCCLBootstrap::getRCCLStoreKeyPrefix() +
      std::to_string(initial_counter_);
  auto stored_vec = store_->get(store_key);
  ncclUniqueId stored_id;
  memcpy(&stored_id, stored_vec.data(), sizeof(stored_id));

  EXPECT_EQ(memcmp(&stored_id, &expected_id, sizeof(ncclUniqueId)), 0);
}

TEST_F(TorchCommRCCLBootstrapTest, ExchangeUniqueIdNonRank0) {
  setupRankAndSize(1, 2);

  // Pre-populate store with unique ID (as if rank 0 already stored it)
  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));
  std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
  memcpy(id_vec.data(), &expected_id, sizeof(expected_id));

  std::string store_key = TorchCommRCCLBootstrap::getRCCLStoreKeyPrefix() +
      std::to_string(TorchCommRCCLBootstrap::getRCCLStoreKeyCounter());
  store_->set(store_key, id_vec);

  auto bootstrap = createBootstrap();

  EXPECT_CALL(*rccl_mock_, commInitRankConfig(_, 2, _, 1, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ncclComm_t comm = bootstrap->createNcclComm("test_comm");
  EXPECT_NE(comm, nullptr);
}

TEST_F(TorchCommRCCLBootstrapTest, CreateNcclCommGetUniqueIdFailure) {
  setupRankAndSize(0, 2);

  auto bootstrap = createBootstrap();

  // Simulate getUniqueId failure
  EXPECT_CALL(*rccl_mock_, getUniqueId(_))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_CALL(*rccl_mock_, getErrorString(ncclInvalidArgument))
      .WillOnce(Return("Invalid argument"));

  EXPECT_THROW(
      {
        try {
          bootstrap->createNcclComm("test_comm");
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Failed to get NCCL unique ID") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommRCCLBootstrapTest, CreateNcclCommInitRankConfigFailure) {
  setupRankAndSize(0, 2);

  auto bootstrap = createBootstrap();

  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*rccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  // Simulate commInitRankConfig failure
  EXPECT_CALL(*rccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_CALL(*rccl_mock_, getErrorString(ncclInvalidArgument))
      .WillOnce(Return("Invalid argument"));

  EXPECT_THROW(
      {
        try {
          bootstrap->createNcclComm("test_comm");
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Failed to initialize NCCL communicator") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommRCCLBootstrapTest, ExchangeUniqueIdInvalidStoreData) {
  setupRankAndSize(1, 2);

  // Store invalid data (wrong size)
  std::vector<uint8_t> invalid_vec(
      10); // Wrong size, should be sizeof(ncclUniqueId)
  std::string store_key = TorchCommRCCLBootstrap::getRCCLStoreKeyPrefix() +
      std::to_string(TorchCommRCCLBootstrap::getRCCLStoreKeyCounter());
  store_->set(store_key, invalid_vec);

  auto bootstrap = createBootstrap();

  EXPECT_THROW(
      {
        try {
          bootstrap->createNcclComm("test_comm");
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Invalid NCCL unique ID size") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommRCCLBootstrapTest, CreateNcclCommWithNullStore) {
  setupRankAndSize(0, 2);
  setenv("MASTER_ADDR", "localhost", 1);
  setenv("MASTER_PORT", "0", 1);

  // Create bootstrap with null store to test TCPStore creation
  auto bootstrap = std::make_unique<TorchCommRCCLBootstrap>(
      nullptr, device_, rccl_mock_, hip_mock_, kTimeout);

  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*rccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  EXPECT_CALL(*rccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  EXPECT_CALL(*rccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclSuccess));

  // This should create an internal TCPStore and succeed
  EXPECT_NO_THROW(bootstrap->createNcclComm("test_comm"));
}

TEST_F(TorchCommRCCLBootstrapTest, CleanupTCPStoreBarrierFailure) {
  setupRankAndSize(0, 2);
  setenv("MASTER_ADDR", "localhost", 1);
  setenv("MASTER_PORT", "0", 1);

  // Create bootstrap with null store to trigger internal store creation
  auto bootstrap = std::make_unique<TorchCommRCCLBootstrap>(
      nullptr, device_, rccl_mock_, hip_mock_, kTimeout);

  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*rccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  EXPECT_CALL(*rccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  // Simulate allReduce failure during cleanup
  EXPECT_CALL(*rccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_CALL(*rccl_mock_, getErrorString(ncclInvalidArgument))
      .WillOnce(Return("Invalid argument"));

  // Should still succeed despite barrier failure (it just logs the error)
  EXPECT_NO_THROW(bootstrap->createNcclComm("test_comm"));
}

} // namespace torch::comms::test
