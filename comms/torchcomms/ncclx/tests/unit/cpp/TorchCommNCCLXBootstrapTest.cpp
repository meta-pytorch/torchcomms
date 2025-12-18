// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXBootstrap.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CudaMock.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/NcclxMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch {
namespace comms {
namespace test {

constexpr std::chrono::seconds kTimeout{60};

class TorchCommNCCLXBootstrapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create hash store for communication
    store_ = c10::make_intrusive<c10d::HashStore>();
    StoreManager::get().injectMockStore(store_);

    // Set up device - use CPU device because we're mocking cuda
    device_ = at::Device(at::DeviceType::CPU, 0);

    // Create fresh mocks for each test
    nccl_mock_ = std::make_shared<NiceMock<NcclxMock>>();
    cuda_mock_ = std::make_shared<NiceMock<CudaMock>>();
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

  std::unique_ptr<TorchCommNCCLXBootstrap> createBootstrap(
      c10::intrusive_ptr<c10d::Store> store = nullptr) {
    if (store) {
      StoreManager::get().injectMockStore(store);
    }
    return std::make_unique<TorchCommNCCLXBootstrap>(
        device_, nccl_mock_, cuda_mock_, kTimeout);
  }

  c10::intrusive_ptr<c10d::Store> store_;
  at::Device device_{at::DeviceType::CPU, 0};
  std::shared_ptr<NiceMock<NcclxMock>> nccl_mock_;
  std::shared_ptr<NiceMock<CudaMock>> cuda_mock_;
};

TEST_F(TorchCommNCCLXBootstrapTest, GetRankAndSizeFromEnvironment) {
  const std::string name = "test_comm";
  setupRankAndSize(1, 4);

  auto bootstrap = createBootstrap();

  // Set up store with unique ID (as if rank 0 already stored it)
  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));
  std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
  memcpy(id_vec.data(), &expected_id, sizeof(expected_id));

  StoreManager::get()
      .getStore(TorchCommNCCLX::kBackendName, name, kTimeout)
      ->set(name, id_vec);

  // Set up mock expectations
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 4, _, 1, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  EXPECT_NO_THROW(bootstrap->createNcclComm(name));
}

TEST_F(TorchCommNCCLXBootstrapTest, GetRankAndSizeEnvironmentVariablesMissing) {
  // Don't set environment variables
  EXPECT_THROW(createBootstrap(), std::runtime_error);
}

TEST_F(TorchCommNCCLXBootstrapTest, ExchangeUniqueIdRank0) {
  const std::string name = "test_comm";
  setupRankAndSize(0, 2);

  auto bootstrap = createBootstrap();

  // Rank 0 should generate unique ID and store it
  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ncclComm_t comm = bootstrap->createNcclComm(name);
  EXPECT_NE(comm, nullptr);

  // Verify the unique ID was stored
  auto stored_vec = StoreManager::get()
                        .getStore(TorchCommNCCLX::kBackendName, name, kTimeout)
                        ->get(name);
  ncclUniqueId stored_id;
  memcpy(&stored_id, stored_vec.data(), sizeof(stored_id));

  EXPECT_EQ(memcmp(&stored_id, &expected_id, sizeof(ncclUniqueId)), 0);
}

TEST_F(TorchCommNCCLXBootstrapTest, ExchangeUniqueIdNonRank0) {
  const std::string name = "test_comm";
  setupRankAndSize(1, 2);

  // Pre-populate store with unique ID (as if rank 0 already stored it)
  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));
  std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
  memcpy(id_vec.data(), &expected_id, sizeof(expected_id));

  StoreManager::get()
      .getStore(TorchCommNCCLX::kBackendName, name, kTimeout)
      ->set(name, id_vec);

  auto bootstrap = createBootstrap();

  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 1, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ncclComm_t comm = bootstrap->createNcclComm(name);
  EXPECT_NE(comm, nullptr);
}

TEST_F(TorchCommNCCLXBootstrapTest, CreateNcclCommGetUniqueIdFailure) {
  setupRankAndSize(0, 2);

  auto bootstrap = createBootstrap();

  // Simulate getUniqueId failure
  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInvalidArgument))
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

TEST_F(TorchCommNCCLXBootstrapTest, CreateNcclCommInitRankConfigFailure) {
  setupRankAndSize(0, 2);

  auto bootstrap = createBootstrap();

  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  // Simulate commInitRankConfig failure
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInvalidArgument))
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

TEST_F(TorchCommNCCLXBootstrapTest, ExchangeUniqueIdInvalidStoreData) {
  const std::string name = "test_comm";
  setupRankAndSize(1, 2);

  // Store invalid data (wrong size)
  std::vector<uint8_t> invalid_vec(
      10); // Wrong size, should be sizeof(ncclUniqueId)
  StoreManager::get()
      .getStore(TorchCommNCCLX::kBackendName, name, kTimeout)
      ->set(name, invalid_vec);

  auto bootstrap = createBootstrap();

  EXPECT_THROW(
      {
        try {
          bootstrap->createNcclComm(name);
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

TEST_F(TorchCommNCCLXBootstrapTest, CreateNcclCommWithNullStore) {
  setupRankAndSize(0, 2);
  setenv("MASTER_ADDR", "localhost", 1);
  setenv("MASTER_PORT", "0", 1);

  // Create bootstrap with null store to test TCPStore creation
  auto bootstrap = createBootstrap();

  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  EXPECT_CALL(*nccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclSuccess));

  // This should create an internal TCPStore and succeed
  EXPECT_NO_THROW(bootstrap->createNcclComm("test_comm"));
}

TEST_F(TorchCommNCCLXBootstrapTest, CleanupTCPStoreBarrierFailure) {
  setupRankAndSize(0, 2);
  setenv("MASTER_ADDR", "localhost", 1);
  setenv("MASTER_PORT", "0", 1);

  // Create bootstrap with null store to trigger internal store creation
  auto bootstrap = createBootstrap();

  ncclUniqueId expected_id{};
  memset(&expected_id, 0x42, sizeof(expected_id));

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  // Simulate allReduce failure during cleanup
  EXPECT_CALL(*nccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInvalidArgument))
      .WillOnce(Return("Invalid argument"));

  // Should still succeed despite barrier failure (it just logs the error)
  EXPECT_NO_THROW(bootstrap->createNcclComm("test_comm"));
}

} // namespace test
} // namespace comms
} // namespace torch
