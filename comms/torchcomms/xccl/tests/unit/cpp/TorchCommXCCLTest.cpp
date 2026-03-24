#include <gtest/gtest.h>
#include "TorchCommXCCLTestBase.hpp"

namespace torch::comms::test {

// ============================================================================
// 1. INITIALIZATION TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, TestOptionsEnvironmentVariables) {
  setOptionsEnvironmentVariables(false, 1); // false abort, 1 second

  CommOptions options1;
  EXPECT_EQ(options1.abort_process_on_timeout_or_error, false);
  EXPECT_EQ(options1.timeout, std::chrono::milliseconds(1000));

  setOptionsEnvironmentVariables(true, 2); // true abort, 2 seconds
  CommOptions options2;
  EXPECT_EQ(options2.abort_process_on_timeout_or_error, true);
  EXPECT_EQ(options2.timeout, std::chrono::milliseconds(2000));
}

TEST_F(TorchCommXCCLTest, SplitCommunicatorSuccess) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 4);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  EXPECT_CALL(*xccl_mock_, commSplit(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<3>(reinterpret_cast<onecclComm_t>(0x2000)),
          Return(onecclSuccess)));

  // Need commUserRank and commCount for the new communicator's init
  EXPECT_CALL(
      *xccl_mock_, commUserRank(reinterpret_cast<onecclComm_t>(0x2000), _))
      .WillOnce(DoAll(SetArgPointee<1>(0), Return(onecclSuccess)));
  EXPECT_CALL(*xccl_mock_, commCount(reinterpret_cast<onecclComm_t>(0x2000), _))
      .WillOnce(DoAll(SetArgPointee<1>(2), Return(onecclSuccess)));

  auto split_comm = comm->split({0, 1}, "split_comm");
  EXPECT_NE(split_comm, nullptr);

  comm->finalize();
}

TEST_F(TorchCommXCCLTest, UniqueCommDesc) {
  xpu_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 4); // rank 0, size 4

  ON_CALL(*xccl_mock_, getUniqueId(_))
      .WillByDefault(
          DoAll(SetArgPointee<0>(onecclUniqueId{}), Return(onecclSuccess)));

  ON_CALL(*xccl_mock_, commInitRankConfig(_, _, _, 0, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<onecclComm_t>(0x1000)),
          Return(onecclSuccess)));

  auto comm = createMockedTorchComm();

  EXPECT_CALL(*xccl_mock_, commDestroy(reinterpret_cast<onecclComm_t>(0x1000)))
      .WillOnce(Return(onecclSuccess));

  comm->init(*device_, "test_comm", default_options_);

  comm->finalize();
}

// ============================================================================
// 2. ALL_REDUCE TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, AllReduce_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));

  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, AllReduceSingleRankPreMulSumScalesTensor) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 1);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({2, 2});

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_reduce(
      tensor, ReduceOp::make_nccl_premul_sum(3.0), true, AllReduceOptions());
  work->wait();

  EXPECT_TRUE(tensor.eq(3).all().item<bool>());
  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

// ============================================================================
// 3. BROADCAST TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, Broadcast_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({5, 5});

  EXPECT_CALL(*xccl_mock_, broadcast(_, _, _, _, 0, _, _))
      .WillOnce(Return(onecclSuccess));

  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->broadcast(tensor, 0, true, BroadcastOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

// ============================================================================
// 4. POINT-TO-POINT AND OTHER COLLECTIVE TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, Send_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({4, 4});

  EXPECT_CALL(*xccl_mock_, send(_, tensor.numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->send(tensor, 1, true, SendOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, Recv_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({4, 4});

  EXPECT_CALL(*xccl_mock_, recv(_, tensor.numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->recv(tensor, 1, true, RecvOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, SendAndRecvRejectSelfRank) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  auto tensor = createTestTensor({4, 4});

  EXPECT_THROW(
      {
        try {
          comm->send(tensor, 0, true, SendOptions());
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInvalidUsage);
          throw;
        }
      },
      XCCLException);

  EXPECT_THROW(
      {
        try {
          comm->recv(tensor, 0, true, RecvOptions());
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInvalidUsage);
          throw;
        }
      },
      XCCLException);

  comm->finalize();
}

TEST_F(TorchCommXCCLTest, BatchOpIssue_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto send_tensor = createTestTensor({4, 4});
  auto recv_tensor = createTestTensor({4, 4});
  std::vector<BatchSendRecv::P2POp> ops = {
      {BatchSendRecv::P2POp::OpType::SEND, send_tensor, 1},
      {BatchSendRecv::P2POp::OpType::RECV, recv_tensor, 1}};

  EXPECT_CALL(*xccl_mock_, groupStart()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, send(_, send_tensor.numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, recv(_, recv_tensor.numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, groupEnd()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->batch_op_issue(ops, true, BatchP2POptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, Reduce_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({4, 4});

  EXPECT_CALL(*xccl_mock_, reduce(_, _, tensor.numel(), _, _, 0, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->reduce(tensor, 0, ReduceOp::SUM, true, ReduceOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, AllGatherSingle_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto input = createTestTensor({4, 4});
  auto output = createTestTensor({8, 4});

  EXPECT_CALL(*xccl_mock_, allGather(_, _, input.numel(), _, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work =
      comm->all_gather_single(output, input, true, AllGatherSingleOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, AllGather_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto input = createTestTensor({4, 4});
  std::vector<at::Tensor> outputs = {
      createTestTensor({4, 4}), createTestTensor({4, 4})};

  EXPECT_CALL(*xccl_mock_, allGather(_, _, input.numel(), _, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_gather(outputs, input, true, AllGatherOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, AllGatherV_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto input = createTestTensor({4, 4});
  std::vector<at::Tensor> outputs = {
      createTestTensor({4, 4}), createTestTensor({4, 4})};

  EXPECT_CALL(*xccl_mock_, groupStart()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, broadcast(_, _, input.numel(), _, 0, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, broadcast(_, _, input.numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, groupEnd()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_gather_v(outputs, input, true, AllGatherOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, ReduceScatterSingle_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto input = createTestTensor({8, 4});
  auto output = createTestTensor({4, 4});

  EXPECT_CALL(*xccl_mock_, reduceScatter(_, _, output.numel(), _, _, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->reduce_scatter_single(
      output, input, ReduceOp::SUM, true, ReduceScatterSingleOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, ReduceScatter_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto output = createTestTensor({4, 4});
  std::vector<at::Tensor> inputs = {
      createTestTensor({4, 4}), createTestTensor({4, 4})};

  EXPECT_CALL(*xccl_mock_, reduceScatter(_, _, output.numel(), _, _, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->reduce_scatter(
      output, inputs, ReduceOp::SUM, true, ReduceScatterOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, ReduceScatterV_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto output = createTestTensor({4, 4});
  std::vector<at::Tensor> inputs = {
      createTestTensor({4, 4}), createTestTensor({4, 4})};

  EXPECT_CALL(*xccl_mock_, groupStart()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, reduce(_, _, output.numel(), _, _, 0, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, reduce(_, _, output.numel(), _, _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, groupEnd()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->reduce_scatter_v(
      output, inputs, ReduceOp::SUM, true, ReduceScatterOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, AllToAllSingle_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto input = createTestTensor({2, 4});
  auto output = createTestTensor({2, 4});

  EXPECT_CALL(*xccl_mock_, allToAll(_, _, input.numel() / 2, _, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work =
      comm->all_to_all_single(output, input, true, AllToAllSingleOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(
    TorchCommXCCLTest,
    AllToAllVSingle_UsesMemcpyForSelfAndSendRecvForPeers) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto input = createTestTensor({2, 2});
  auto output = createTestTensor({2, 2});
  std::vector<uint64_t> split_sizes = {1, 1};

  EXPECT_CALL(*xccl_mock_, groupStart()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, memcpyAsync(_, _, 2 * input.element_size(), _))
      .WillOnce(Return(XPU_SUCCESS));
  EXPECT_CALL(*xccl_mock_, send(_, 2, _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, recv(_, 2, _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, groupEnd()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_to_all_v_single(
      output, input, split_sizes, split_sizes, true, AllToAllvSingleOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, AllToAll_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  std::vector<at::Tensor> inputs = {
      createTestTensor({2, 2}), createTestTensor({2, 2})};
  std::vector<at::Tensor> outputs = {
      createTestTensor({2, 2}), createTestTensor({2, 2})};

  EXPECT_CALL(*xccl_mock_, groupStart()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(
      *xpu_mock_,
      memcpyAsync(
          outputs[0].data_ptr(),
          inputs[0].data_ptr(),
          inputs[0].numel() * inputs[0].element_size(),
          _))
      .WillOnce(Return(XPU_SUCCESS));
  EXPECT_CALL(*xccl_mock_, send(_, inputs[1].numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, recv(_, outputs[1].numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, groupEnd()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_to_all(outputs, inputs, true, AllToAllOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, Barrier_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, 1, onecclFloat32, onecclSum, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->barrier(true, BarrierOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, ScatterRoot_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto output = createTestTensor({2, 2});
  std::vector<at::Tensor> input_tensors = {
      createTestTensor({2, 2}), createTestTensor({2, 2})};

  EXPECT_CALL(*xccl_mock_, groupStart()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, send(_, input_tensors[1].numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(
      *xpu_mock_,
      memcpyAsync(
          output.data_ptr(),
          input_tensors[0].data_ptr(),
          input_tensors[0].numel() * input_tensors[0].element_size(),
          _))
      .WillOnce(Return(XPU_SUCCESS));
  EXPECT_CALL(*xccl_mock_, groupEnd()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->scatter(output, input_tensors, 0, true, ScatterOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, GatherRoot_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto input = createTestTensor({2, 2});
  std::vector<at::Tensor> output_tensors = {
      createTestTensor({2, 2}), createTestTensor({2, 2})};

  EXPECT_CALL(*xccl_mock_, groupStart()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xccl_mock_, recv(_, output_tensors[1].numel(), _, 1, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(
      *xpu_mock_,
      memcpyAsync(
          output_tensors[0].data_ptr(),
          input.data_ptr(),
          input.numel() * input.element_size(),
          _))
      .WillOnce(Return(XPU_SUCCESS));
  EXPECT_CALL(*xccl_mock_, groupEnd()).WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->gather(output_tensors, input, 0, true, GatherOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

// ============================================================================
// 5. SYNC OP TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, AllReduce_SyncOp_Success) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  // Sync ops don't record a dependency event, so only 2 eventRecord calls
  // (start + end) instead of the 3 that setupEventsForWork expects.
  EXPECT_CALL(*xpu_mock_, eventCreateWithFlags(_, _))
      .WillOnce(Return(XPU_SUCCESS))
      .WillOnce(Return(XPU_SUCCESS));
  EXPECT_CALL(*xpu_mock_, eventRecord(_, _))
      .Times(2)
      .WillRepeatedly(Return(XPU_SUCCESS));

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));

  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work =
      comm->all_reduce(tensor, ReduceOp::SUM, false, AllReduceOptions());
  work->wait();

  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

// ============================================================================
// 6. AVG AND PREMUL_SUM REDUCTION TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, AllReduceAvgDividesByCommSize) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  // Tensor of all 4.0 values; AVG with comm_size=2 should give 2.0
  auto tensor = createTestTensor({2, 2});
  tensor.fill_(4.0);

  // AVG is converted to SUM, so the mock sees onecclSum
  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, onecclSum, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_reduce(tensor, ReduceOp::AVG, true, AllReduceOptions());
  work->wait();

  // After all_reduce with AVG, tensor should be divided by comm_size (2)
  EXPECT_TRUE(tensor.eq(2.0).all().item<bool>());
  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchCommXCCLTest, AllReduceMultiRankPreMulSumScalesTensor) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({2, 2});

  // PREMUL_SUM is converted to SUM after pre-multiplying the tensor
  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, onecclSum, _, _))
      .WillOnce(Return(onecclSuccess));
  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_reduce(
      tensor, ReduceOp::make_nccl_premul_sum(5.0), true, AllReduceOptions());
  work->wait();

  // After pre-multiplication: 1.0 * 5.0 = 5.0
  EXPECT_TRUE(tensor.eq(5.0).all().item<bool>());
  comm->finalize();
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
}

// ============================================================================
// 7. FAILURE AND ERROR TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, AllReduceFailureThrows) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclInternalError));

  EXPECT_THROW(
      {
        try {
          comm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInternalError);
          throw;
        }
      },
      XCCLException);

  comm->finalize();
}

TEST_F(TorchCommXCCLTest, BroadcastFailureThrows) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  auto tensor = createTestTensor({5, 5});

  EXPECT_CALL(*xccl_mock_, broadcast(_, _, _, _, 0, _, _))
      .WillOnce(Return(onecclInternalError));

  EXPECT_THROW(
      {
        try {
          comm->broadcast(tensor, 0, true, BroadcastOptions());
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInternalError);
          throw;
        }
      },
      XCCLException);

  comm->finalize();
}

TEST_F(TorchCommXCCLTest, SendFailureThrows) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  auto tensor = createTestTensor({4, 4});

  EXPECT_CALL(*xccl_mock_, send(_, _, _, 1, _, _))
      .WillOnce(Return(onecclInternalError));

  EXPECT_THROW(
      {
        try {
          comm->send(tensor, 1, true, SendOptions());
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInternalError);
          throw;
        }
      },
      XCCLException);

  comm->finalize();
}

TEST_F(TorchCommXCCLTest, RecvFailureThrows) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  auto tensor = createTestTensor({4, 4});

  EXPECT_CALL(*xccl_mock_, recv(_, _, _, 1, _, _))
      .WillOnce(Return(onecclInternalError));

  EXPECT_THROW(
      {
        try {
          comm->recv(tensor, 1, true, RecvOptions());
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInternalError);
          throw;
        }
      },
      XCCLException);

  comm->finalize();
}

TEST_F(TorchCommXCCLTest, ReduceFailureThrows) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  auto tensor = createTestTensor({4, 4});

  EXPECT_CALL(*xccl_mock_, reduce(_, _, tensor.numel(), _, _, 0, _, _))
      .WillOnce(Return(onecclInternalError));

  EXPECT_THROW(
      {
        try {
          comm->reduce(tensor, 0, ReduceOp::SUM, true, ReduceOptions());
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInternalError);
          throw;
        }
      },
      XCCLException);

  comm->finalize();
}

// ============================================================================
// 8. TIMEOUT / ABORT TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, WorkTimeoutDetectedByFinalize) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  // Use a very short timeout so the test completes quickly
  default_options_.abort_process_on_timeout_or_error = false;
  default_options_.timeout = std::chrono::milliseconds(50);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));

  // Start event completes, end event never completes -> timeout
  setupWorkToTimeout();

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());

  // Sleep long enough for the timeout to trigger
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  EXPECT_THROW(comm->finalize(), std::runtime_error);
}

TEST_F(TorchCommXCCLTest, WorkErrorDetectedByFinalize) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  default_options_.abort_process_on_timeout_or_error = false;
  default_options_.timeout = std::chrono::milliseconds(2000);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));

  // Start event succeeds, end event returns error
  setupWorkToError();

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());

  EXPECT_THROW(comm->finalize(), XCCLException);
}

TEST_F(TorchCommXCCLTest, AsyncErrorAbortsCommunicator) {
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  setupRankAndSize(0, 2);

  // Set the abort on timeout/error option so watchdog can abort
  default_options_.abort_process_on_timeout_or_error = false;
  default_options_.timeout = std::chrono::milliseconds(200);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({10, 10});

  // Make collective succeed initially
  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));

  // A subsequent API entrypoint should surface the communicator async error.
  EXPECT_CALL(*xccl_mock_, commGetAsyncError(_, _))
      .WillOnce(
          DoAll(SetArgPointee<1>(onecclInternalError), Return(onecclSuccess)))
      .WillRepeatedly(Return(onecclNotImplemented));

  EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());
  work->wait();

  comm->commStateForTests() =
      TorchCommXCCLTest::TestTorchCommXCCL::CommState::ERROR;
  EXPECT_THROW(
      {
        try {
          comm->finalize();
        } catch (const XCCLException& e) {
          EXPECT_EQ(e.getResult(), onecclInternalError);
          throw;
        }
      },
      XCCLException);
}

} // namespace torch::comms::test
