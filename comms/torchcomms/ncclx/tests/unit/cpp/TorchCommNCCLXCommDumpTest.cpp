// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/ncclx/tests/unit/cpp/TorchCommNCCLXTestBase.hpp"

namespace torch::comms::test {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class TorchCommNCCLXCommDumpTest : public TorchCommNCCLXTest {};

// Diagnostic test: verify mock commDump works via direct call
TEST_F(TorchCommNCCLXCommDumpTest, DirectMockCommDump) {
  EXPECT_CALL(*nccl_mock_, commDump(_, _))
      .WillOnce(
          [](ncclComm_t, std::unordered_map<std::string, std::string>& map) {
            map["key"] = "val";
            return ncclSuccess;
          });
  std::unordered_map<std::string, std::string> map;
  auto result = nccl_mock_->commDump(nullptr, map);
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_EQ(map["key"], "val");
}

// Diagnostic test: verify mock commDump works via base class pointer
TEST_F(TorchCommNCCLXCommDumpTest, VirtualDispatchCommDump) {
  EXPECT_CALL(*nccl_mock_, commDump(_, _))
      .WillOnce(
          [](ncclComm_t, std::unordered_map<std::string, std::string>& map) {
            map["key"] = "val";
            return ncclSuccess;
          });
  std::shared_ptr<NcclxApi> base_ptr = nccl_mock_;
  std::unordered_map<std::string, std::string> map;
  auto result = base_ptr->commDump(nullptr, map);
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_EQ(map["key"], "val");
}

TEST_F(TorchCommNCCLXCommDumpTest, CommDumpSuccess) {
  cuda_mock_->setupDefaultBehaviors();
  auto torchcomm = createMockedTorchComm();

  // Bootstrap expectations: getUniqueId + commInitRankConfig
  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(ncclUniqueId{}), Return(ncclSuccess)));
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  setupCCAExpectations(1, 2, 1);

  torchcomm->init(*device_, "test_comm", default_options_);

  setupNormalDestruction(*torchcomm);

  // Set up commDump mock AFTER init to avoid interfering with bootstrap mocks
  EXPECT_CALL(*nccl_mock_, commDump(_, _))
      .WillOnce(
          [](ncclComm_t, std::unordered_map<std::string, std::string>& map) {
            map["commHash"] = "\"abc123\"";
            map["rank"] = "0";
            map["nRanks"] = "2";
            return ncclSuccess;
          });

  auto result = torchcomm->comm_dump();

  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result["commHash"], "\"abc123\"");
  EXPECT_EQ(result["rank"], "0");
  EXPECT_EQ(result["nRanks"], "2");

  torchcomm->finalize();
}

TEST_F(TorchCommNCCLXCommDumpTest, CommDumpFailure) {
  cuda_mock_->setupDefaultBehaviors();
  auto torchcomm = createMockedTorchComm();

  // Bootstrap expectations: getUniqueId + commInitRankConfig
  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(ncclUniqueId{}), Return(ncclSuccess)));
  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  setupCCAExpectations(1, 1, 1);

  // After NCCLXException from commDump, the comm will be aborted
  EXPECT_CALL(*nccl_mock_, commAbort(_)).WillOnce(Return(ncclSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).Times(0);

  torchcomm->init(*device_, "test_comm", default_options_);

  // NCCLX_CHECK calls getErrorString when commDump fails
  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInternalError))
      .WillRepeatedly(Return("internal error"));

  // Set up commDump mock AFTER init to avoid interfering with bootstrap mocks
  EXPECT_CALL(*nccl_mock_, commDump(_, _)).WillOnce(Return(ncclInternalError));

  EXPECT_THROW(torchcomm->comm_dump(), NCCLXException);
}

} // namespace torch::comms::test
