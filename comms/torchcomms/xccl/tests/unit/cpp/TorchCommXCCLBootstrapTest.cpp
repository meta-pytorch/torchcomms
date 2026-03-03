#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "TorchCommXCCLTestBase.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCLBootstrap.hpp"

namespace torch::comms::test {

class TorchCommXCCLBootstrapTest : public TorchCommXCCLTest {};

TEST_F(TorchCommXCCLBootstrapTest, ExchangeIdViaC10dStore) {
  setupRankAndSize(1, 2); // Rank 1 (receiver)
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  
  setenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD", "c10d", 1);

  // Set up mock unique ID behavior for rank 0
  onecclUniqueId fake_id;
  std::fill(reinterpret_cast<uint8_t*>(&fake_id), reinterpret_cast<uint8_t*>(&fake_id) + sizeof(onecclUniqueId), 0xAA);
  
  // Pre-seed the c10d store since we are Rank 1 and need to receive the ID
  std::vector<uint8_t> id_bytes(reinterpret_cast<uint8_t*>(&fake_id), reinterpret_cast<uint8_t*>(&fake_id) + sizeof(onecclUniqueId));
  std::string expected_key = TorchCommXCCLBootstrap::getXCCLStoreKeyPrefix() + std::to_string(TorchCommXCCLBootstrap::getXCCLStoreKeyCounter());
  store_->set(expected_key, id_bytes);

  onecclComm_t fake_comm = reinterpret_cast<onecclComm_t>(0x1234);
  EXPECT_CALL(*xccl_mock_, commInitRankConfig(_, 2, _, 1, _))
      .WillOnce(DoAll(SetArgPointee<0>(fake_comm), Return(onecclSuccess)));

  auto bootstrap = std::make_unique<TorchCommXCCLBootstrap>(
      store_, *device_, xccl_mock_, xpu_mock_, default_options_.timeout);

  onecclComm_t comm;
  EXPECT_NO_THROW(comm = bootstrap->createXcclComm("test_comm", default_options_));
  EXPECT_EQ(comm, fake_comm);
  
  unsetenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD");
}

} // namespace torch::comms::test
