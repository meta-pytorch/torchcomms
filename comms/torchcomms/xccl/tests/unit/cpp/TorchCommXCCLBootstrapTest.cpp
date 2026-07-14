#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp>

#include "comms/torchcomms/xccl/TorchCommXCCLBootstrap.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XcclMock.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XpuMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms::test {

constexpr std::chrono::seconds kTimeout{60};

class TorchCommXCCLBootstrapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    store_ = c10::make_intrusive<c10d::HashStore>();
    device_ = at::Device(at::DeviceType::CPU, 0);
    xccl_mock_ = std::make_shared<NiceMock<XcclMock>>();
    xpu_mock_ = std::make_shared<NiceMock<XpuMock>>();

    // Reset the static counter to a known state
    TorchCommXCCLBootstrap::getXCCLStoreKey(); // This increments counter
    initial_counter_ = TorchCommXCCLBootstrap::getXCCLStoreKeyCounter();
  }

  void TearDown() override {
    unsetenv("TORCHCOMM_RANK");
    unsetenv("TORCHCOMM_SIZE");
    unsetenv("MASTER_ADDR");
    unsetenv("MASTER_PORT");
    unsetenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD");
  }

  void setupRankAndSize(int rank, int size) {
    setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
    setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);
  }

  std::unique_ptr<TorchCommXCCLBootstrap> createBootstrap(
      c10::intrusive_ptr<c10d::Store> store = nullptr) {
    if (!store) {
      store = store_;
    }
    return std::make_unique<TorchCommXCCLBootstrap>(
        store, *device_, xccl_mock_, xpu_mock_, kTimeout);
  }

  c10::intrusive_ptr<c10d::Store> store_;
  std::optional<at::Device> device_;
  std::shared_ptr<NiceMock<XcclMock>> xccl_mock_;
  std::shared_ptr<NiceMock<XpuMock>> xpu_mock_;
  int initial_counter_{-1};
  CommOptions default_options_;
};

TEST_F(TorchCommXCCLBootstrapTest, StaticMethodsStoreKeyGeneration) {
  std::string prefix = TorchCommXCCLBootstrap::getXCCLStoreKeyPrefix();
  EXPECT_EQ(prefix, "xccl_storekey_");

  int counter_before = TorchCommXCCLBootstrap::getXCCLStoreKeyCounter();
  std::string key1 = TorchCommXCCLBootstrap::getXCCLStoreKey();
  int counter_after = TorchCommXCCLBootstrap::getXCCLStoreKeyCounter();

  EXPECT_EQ(counter_after, counter_before + 1);
  EXPECT_EQ(key1, prefix + std::to_string(counter_before));

  // Test that subsequent calls increment the counter
  std::string key2 = TorchCommXCCLBootstrap::getXCCLStoreKey();
  int final_counter = TorchCommXCCLBootstrap::getXCCLStoreKeyCounter();

  EXPECT_EQ(final_counter, counter_after + 1);
  EXPECT_EQ(key2, prefix + std::to_string(counter_after));
  EXPECT_NE(key1, key2);
}

TEST_F(TorchCommXCCLBootstrapTest, GetRankAndSizeFromEnvironment) {
  setupRankAndSize(1, 4);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  setenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD", "c10d", 1);

  auto bootstrap = createBootstrap();

  // Set up store with unique ID (as if rank 0 already stored it)
  onecclUniqueId expected_id{};
  std::fill(
      reinterpret_cast<uint8_t*>(&expected_id),
      reinterpret_cast<uint8_t*>(&expected_id) + sizeof(onecclUniqueId),
      0x42);
  std::vector<uint8_t> id_vec(sizeof(onecclUniqueId));
  memcpy(id_vec.data(), &expected_id, sizeof(expected_id));

  std::string store_key = TorchCommXCCLBootstrap::getXCCLStoreKeyPrefix() +
      std::to_string(TorchCommXCCLBootstrap::getXCCLStoreKeyCounter());
  store_->set(store_key, id_vec);

  EXPECT_CALL(*xccl_mock_, commInitRankConfig(_, 4, _, 1, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<onecclComm_t>(0x3000)),
          Return(onecclSuccess)));

  EXPECT_NO_THROW(bootstrap->createXcclComm("test_comm", default_options_));
}

TEST_F(TorchCommXCCLBootstrapTest, GetRankAndSizeEnvironmentVariablesMissing) {
  // Don't set environment variables — query_ranksize should throw
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  EXPECT_THROW(createBootstrap(), std::runtime_error);
}

TEST_F(TorchCommXCCLBootstrapTest, ExchangeIdViaC10dStore) {
  setupRankAndSize(1, 2); // Rank 1 (receiver)
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  setenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD", "c10d", 1);

  // Set up mock unique ID behavior for rank 0
  onecclUniqueId fake_id;
  std::fill(
      reinterpret_cast<uint8_t*>(&fake_id),
      reinterpret_cast<uint8_t*>(&fake_id) + sizeof(onecclUniqueId),
      0xAA);

  // Pre-seed the c10d store since we are Rank 1 and need to receive the ID
  std::vector<uint8_t> id_bytes(
      reinterpret_cast<uint8_t*>(&fake_id),
      reinterpret_cast<uint8_t*>(&fake_id) + sizeof(onecclUniqueId));
  std::string expected_key = TorchCommXCCLBootstrap::getXCCLStoreKeyPrefix() +
      std::to_string(TorchCommXCCLBootstrap::getXCCLStoreKeyCounter());
  store_->set(expected_key, id_bytes);

  onecclComm_t fake_comm = reinterpret_cast<onecclComm_t>(0x1234);
  EXPECT_CALL(*xccl_mock_, commInitRankConfig(_, 2, _, 1, _))
      .WillOnce(DoAll(SetArgPointee<0>(fake_comm), Return(onecclSuccess)));

  auto bootstrap = std::make_unique<TorchCommXCCLBootstrap>(
      store_, *device_, xccl_mock_, xpu_mock_, default_options_.timeout);

  onecclComm_t comm;
  EXPECT_NO_THROW(
      comm = bootstrap->createXcclComm("test_comm", default_options_));
  EXPECT_EQ(comm, fake_comm);
}

TEST_F(TorchCommXCCLBootstrapTest, ExchangeIdViaC10dStoreRank0) {
  setupRankAndSize(0, 2); // Rank 0 (sender / ID generator)
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  setenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD", "c10d", 1);

  // Mock getUniqueId to populate a fake unique ID on rank 0
  onecclUniqueId fake_id;
  std::fill(
      reinterpret_cast<uint8_t*>(&fake_id),
      reinterpret_cast<uint8_t*>(&fake_id) + sizeof(onecclUniqueId),
      0xBB);

  EXPECT_CALL(*xccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_id), Return(onecclSuccess)));

  onecclComm_t fake_comm = reinterpret_cast<onecclComm_t>(0x5678);
  EXPECT_CALL(*xccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(SetArgPointee<0>(fake_comm), Return(onecclSuccess)));

  auto bootstrap = std::make_unique<TorchCommXCCLBootstrap>(
      store_, *device_, xccl_mock_, xpu_mock_, default_options_.timeout);

  onecclComm_t comm;
  EXPECT_NO_THROW(
      comm = bootstrap->createXcclComm("test_comm", default_options_));
  EXPECT_EQ(comm, fake_comm);

  // Verify rank 0 stored the unique ID in the c10d store
  std::string expected_key = TorchCommXCCLBootstrap::getXCCLStoreKeyPrefix() +
      std::to_string(TorchCommXCCLBootstrap::getXCCLStoreKeyCounter() - 1);
  auto stored_bytes = store_->get(expected_key);
  EXPECT_EQ(stored_bytes.size(), sizeof(onecclUniqueId));

  onecclUniqueId stored_id;
  std::memcpy(&stored_id, stored_bytes.data(), sizeof(onecclUniqueId));
  EXPECT_EQ(std::memcmp(&stored_id, &fake_id, sizeof(onecclUniqueId)), 0);
}

TEST_F(TorchCommXCCLBootstrapTest, CreateXcclCommGetUniqueIdFailure) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  setenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD", "c10d", 1);

  auto bootstrap = createBootstrap();

  // Simulate getUniqueId failure
  EXPECT_CALL(*xccl_mock_, getUniqueId(_))
      .WillOnce(Return(onecclInternalError));

  EXPECT_CALL(*xccl_mock_, getErrorString(onecclInternalError))
      .WillOnce(Return("Internal error"));

  EXPECT_THROW(
      {
        try {
          bootstrap->createXcclComm("test_comm", default_options_);
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Failed to get XCCL unique ID") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommXCCLBootstrapTest, CreateXcclCommInitRankConfigFailure) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  setenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD", "c10d", 1);

  auto bootstrap = createBootstrap();

  onecclUniqueId expected_id{};
  std::fill(
      reinterpret_cast<uint8_t*>(&expected_id),
      reinterpret_cast<uint8_t*>(&expected_id) + sizeof(onecclUniqueId),
      0x42);

  EXPECT_CALL(*xccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(onecclSuccess)));

  // Simulate commInitRankConfig failure
  EXPECT_CALL(*xccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(Return(onecclInternalError));

  EXPECT_CALL(*xccl_mock_, getErrorString(onecclInternalError))
      .WillOnce(Return("Internal error"));

  EXPECT_THROW(
      {
        try {
          bootstrap->createXcclComm("test_comm", default_options_);
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Failed to initialize XCCL communicator") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommXCCLBootstrapTest, ExchangeUniqueIdInvalidStoreData) {
  setupRankAndSize(1, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  setenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD", "c10d", 1);

  // Store invalid data (wrong size)
  std::vector<uint8_t> invalid_vec(10);
  std::string store_key = TorchCommXCCLBootstrap::getXCCLStoreKeyPrefix() +
      std::to_string(TorchCommXCCLBootstrap::getXCCLStoreKeyCounter());
  store_->set(store_key, invalid_vec);

  auto bootstrap = createBootstrap();

  EXPECT_THROW(
      {
        try {
          bootstrap->createXcclComm("test_comm", default_options_);
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("Invalid XCCL unique ID size") !=
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

} // namespace torch::comms::test
