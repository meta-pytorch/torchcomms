// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "SendRecvTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchWork.hpp"

std::unique_ptr<TorchCommTestWrapper> SendRecvTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void SendRecvTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_type_ = wrapper_->getDevice().type();
}

void SendRecvTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous send/recv with work object
void SendRecvTest::testSyncSendRecv(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync send/recv with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  auto params = createSendRecvParams(count, dtype);
  // Alternate send/recv order based on rank to avoid deadlock
  // Even ranks send first, then receive
  // Odd ranks receive first, then send
  c10::intrusive_ptr<torch::comms::TorchWork> send_work;
  c10::intrusive_ptr<torch::comms::TorchWork> recv_work;

  if (rank_ % 2 == 0) {
    // Even ranks: send first, then receive
    send_work = torchcomm_->send(params.send_tensor, params.send_rank, false);
    recv_work = torchcomm_->recv(params.recv_tensor, params.recv_rank, false);
  } else {
    // Odd ranks: receive first, then send
    recv_work = torchcomm_->recv(params.recv_tensor, params.recv_rank, false);
    send_work = torchcomm_->send(params.send_tensor, params.send_rank, false);
  }
  send_work->wait();
  recv_work->wait();

  // Verify the results
  verifyResults(params.recv_tensor, params.recv_rank);
}

// Test function for synchronous send/recv without work object
void SendRecvTest::testSyncSendRecvNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync send/recv without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  auto params = createSendRecvParams(count, dtype);
  // Alternate send/recv order based on rank to avoid deadlock
  // Even ranks send first, then receive
  // Odd ranks receive first, then send
  if (rank_ % 2 == 0) {
    // Even ranks: send first, then receive
    torchcomm_->send(params.send_tensor, params.send_rank, false);
    torchcomm_->recv(params.recv_tensor, params.recv_rank, false);
  } else {
    // Odd ranks: receive first, then send
    torchcomm_->recv(params.recv_tensor, params.recv_rank, false);
    torchcomm_->send(params.send_tensor, params.send_rank, false);
  }

  // Verify the results
  verifyResults(params.recv_tensor, params.recv_rank);
}

// Test function for asynchronous send/recv with wait
void SendRecvTest::testAsyncSendRecv(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async send/recv with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  auto params = createSendRecvParams(count, dtype);

  // Alternate send/recv order based on rank to avoid deadlock
  // Even ranks send first, then receive
  // Odd ranks receive first, then send
  c10::intrusive_ptr<torch::comms::TorchWork> send_work;
  c10::intrusive_ptr<torch::comms::TorchWork> recv_work;

  if (rank_ % 2 == 0) {
    // Even ranks: send first, then receive
    send_work = torchcomm_->send(params.send_tensor, params.send_rank, true);
    recv_work = torchcomm_->recv(params.recv_tensor, params.recv_rank, true);
  } else {
    // Odd ranks: receive first, then send
    recv_work = torchcomm_->recv(params.recv_tensor, params.recv_rank, true);
    send_work = torchcomm_->send(params.send_tensor, params.send_rank, true);
  }

  // Wait for the operations to complete
  // For async operations, we can wait in any order
  send_work->wait();
  recv_work->wait();

  // Verify the results
  verifyResults(params.recv_tensor, params.recv_rank);
}

// Test function for asynchronous send/recv with early reset
void SendRecvTest::testAsyncSendRecvEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async send/recv with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  auto params = createSendRecvParams(count, dtype);

  // Alternate send/recv order based on rank to avoid deadlock
  // Even ranks send first, then receive
  // Odd ranks receive first, then send
  c10::intrusive_ptr<torch::comms::TorchWork> send_work;
  c10::intrusive_ptr<torch::comms::TorchWork> recv_work;

  if (rank_ % 2 == 0) {
    // Even ranks: send first, then receive
    send_work = torchcomm_->send(params.send_tensor, params.send_rank, true);
    recv_work = torchcomm_->recv(params.recv_tensor, params.recv_rank, true);
  } else {
    // Odd ranks: receive first, then send
    recv_work = torchcomm_->recv(params.recv_tensor, params.recv_rank, true);
    send_work = torchcomm_->send(params.send_tensor, params.send_rank, true);
  }

  // Wait for the operations to complete before resetting
  send_work->wait();
  recv_work->wait();

  // Reset the work objects
  send_work.reset();
  recv_work.reset();

  // Verify the results
  verifyResults(params.recv_tensor, params.recv_rank);
}

// Test function for asynchronous send/recv with input deleted after enqueue
void SendRecvTest::testSendRecvInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async send/recv with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create recv tensor that persists throughout the test
  at::Tensor recv_tensor = createRecvTensor(count, dtype);

  // Create recv parameters
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  // Create work objects to hold the async operations
  c10::intrusive_ptr<torch::comms::TorchWork> send_work;
  c10::intrusive_ptr<torch::comms::TorchWork> recv_work;

  {
    // Create send tensor in a limited scope
    at::Tensor send_tensor = createSendTensor(count, dtype);
    int send_rank = (rank_ + 1) % num_ranks_;

    // Alternate send/recv order based on rank to avoid deadlock
    if (rank_ % 2 == 0) {
      // Even ranks: send first, then receive
      send_work = torchcomm_->send(send_tensor, send_rank, true);
      recv_work = torchcomm_->recv(recv_tensor, recv_rank, true);
    } else {
      // Odd ranks: receive first, then send
      recv_work = torchcomm_->recv(recv_tensor, recv_rank, true);
      send_work = torchcomm_->send(send_tensor, send_rank, true);
    }

    // Send tensor goes out of scope here and gets deleted
  }

  // Wait for the operations to complete even though send tensor is deleted
  send_work->wait();
  recv_work->wait();

  // Verify the results
  verifyResults(recv_tensor, recv_rank);
}

// Helper function to create send tensor
at::Tensor SendRecvTest::createSendTensor(int count, at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor send_tensor;
  if (dtype == at::kFloat || dtype == at::kBFloat16) {
    send_tensor = at::ones({count}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    send_tensor = at::ones({count}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    send_tensor =
        at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
  }
  return send_tensor;
}

// Helper function to create receive tensor
at::Tensor SendRecvTest::createRecvTensor(int count, at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to verify results
void SendRecvTest::verifyResults(const at::Tensor& recv_tensor, int recv_rank) {
  std::string description =
      "recv rank " + std::to_string(recv_rank) + " tensor";
  verifyTensorEquality(recv_tensor.cpu(), recv_rank + 1, description);
}

// CUDA Graph test function for send/recv
void SendRecvTest::testGraphSendRecv(int count, at::ScalarType dtype) {
  // Skip CUDA Graph tests when running on CPU
  if (isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph tests are not supported on CPU";
  }

  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph send/recv with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create send/recv parameters AFTER setting non-default stream but BEFORE
  // graph capture
  auto params = createSendRecvParams(count, dtype);
  at::Tensor original_recv_tensor = params.recv_tensor.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the send/recv operations in the graph
  graph.capture_begin();

  // Call send/recv without keeping the work objects
  // Alternate send/recv order based on rank to avoid deadlock
  if (rank_ % 2 == 0) {
    // Even ranks: send first, then receive
    torchcomm_->send(params.send_tensor, params.send_rank, false);
    torchcomm_->recv(params.recv_tensor, params.recv_rank, false);
  } else {
    // Odd ranks: receive first, then send
    torchcomm_->recv(params.recv_tensor, params.recv_rank, false);
    torchcomm_->send(params.send_tensor, params.send_rank, false);
  }

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before each replay
    params.recv_tensor.copy_(original_recv_tensor);

    graph.replay();

    // Verify the results after each replay
    verifyResults(params.recv_tensor, params.recv_rank);
  }
}

// CUDA Graph test function for send/recv with input deleted after graph
// creation
void SendRecvTest::testGraphSendRecvInputDeleted(
    int count,
    at::ScalarType dtype) {
  // Skip CUDA Graph tests when running on CPU
  if (isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph tests are not supported on CPU";
  }

  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph send/recv with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create recv tensor that persists throughout the test
  at::Tensor recv_tensor = createRecvTensor(count, dtype);
  at::Tensor original_recv_tensor = recv_tensor.clone();

  // Create recv parameters
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create send tensor in a limited scope
    at::Tensor send_tensor = createSendTensor(count, dtype);
    int send_rank = (rank_ + 1) % num_ranks_;

    // Capture the send/recv operations in the graph
    graph.capture_begin();

    // Call send/recv without keeping the work objects
    // Alternate send/recv order based on rank to avoid deadlock
    if (rank_ % 2 == 0) {
      // Even ranks: send first, then receive
      torchcomm_->send(send_tensor, send_rank, false);
      torchcomm_->recv(recv_tensor, recv_rank, false);
    } else {
      // Odd ranks: receive first, then send
      torchcomm_->recv(recv_tensor, recv_rank, false);
      torchcomm_->send(send_tensor, send_rank, false);
    }

    graph.capture_end();

    // Send tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though send tensor is
  // deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before each replay
    recv_tensor.copy_(original_recv_tensor);

    graph.replay();

    // Verify the results after each replay
    verifyResults(recv_tensor, recv_rank);
  }
}
