// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "SendRecvTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous send/recv with work object
template <typename Fixture>
void SendRecvTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  at::Tensor send_tensor = createSendTensor(count, dtype);
  at::Tensor recv_tensor = createRecvTensor(count, dtype);
  auto original_recv_tensor = recv_tensor.clone();

  auto execute = [&]() {
    if (rank_ % 2 == 0) {
      auto send_work = torchcomm_->send(send_tensor, send_rank, false);
      auto recv_work = torchcomm_->recv(recv_tensor, recv_rank, false);
      send_work->wait();
      recv_work->wait();
    } else {
      auto recv_work = torchcomm_->recv(recv_tensor, recv_rank, false);
      auto send_work = torchcomm_->send(send_tensor, send_rank, false);
      recv_work->wait();
      send_work->wait();
    }
  };
  auto reset = [&]() { recv_tensor.copy_(original_recv_tensor); };
  auto verify = [&]() { verifyResults(recv_tensor, recv_rank); };
  run(execute, reset, verify);
}

// Test function for synchronous send/recv without work object
template <typename Fixture>
void SendRecvTest<Fixture>::testSyncNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  at::Tensor send_tensor = createSendTensor(count, dtype);
  at::Tensor recv_tensor = createRecvTensor(count, dtype);
  auto original_recv_tensor = recv_tensor.clone();

  auto execute = [&]() {
    if (rank_ % 2 == 0) {
      torchcomm_->send(send_tensor, send_rank, false);
      torchcomm_->recv(recv_tensor, recv_rank, false);
    } else {
      torchcomm_->recv(recv_tensor, recv_rank, false);
      torchcomm_->send(send_tensor, send_rank, false);
    }
  };
  auto reset = [&]() { recv_tensor.copy_(original_recv_tensor); };
  auto verify = [&]() { verifyResults(recv_tensor, recv_rank); };
  run(execute, reset, verify);
}

// Test function for asynchronous send/recv with wait
template <typename Fixture>
void SendRecvTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  at::Tensor send_tensor = createSendTensor(count, dtype);
  at::Tensor recv_tensor = createRecvTensor(count, dtype);
  auto original_recv_tensor = recv_tensor.clone();

  auto execute = [&]() {
    if (rank_ % 2 == 0) {
      auto send_work = torchcomm_->send(send_tensor, send_rank, true);
      auto recv_work = torchcomm_->recv(recv_tensor, recv_rank, true);
      send_work->wait();
      recv_work->wait();
    } else {
      auto recv_work = torchcomm_->recv(recv_tensor, recv_rank, true);
      auto send_work = torchcomm_->send(send_tensor, send_rank, true);
      recv_work->wait();
      send_work->wait();
    }
  };
  auto reset = [&]() { recv_tensor.copy_(original_recv_tensor); };
  auto verify = [&]() { verifyResults(recv_tensor, recv_rank); };
  run(execute, reset, verify);
}

// Test function for asynchronous send/recv with early reset
template <typename Fixture>
void SendRecvTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  at::Tensor send_tensor = createSendTensor(count, dtype);
  at::Tensor recv_tensor = createRecvTensor(count, dtype);
  auto original_recv_tensor = recv_tensor.clone();

  auto execute = [&]() {
    if (rank_ % 2 == 0) {
      auto send_work = torchcomm_->send(send_tensor, send_rank, true);
      auto recv_work = torchcomm_->recv(recv_tensor, recv_rank, true);
      send_work->wait();
      recv_work->wait();
      send_work.reset();
      recv_work.reset();
    } else {
      auto recv_work = torchcomm_->recv(recv_tensor, recv_rank, true);
      auto send_work = torchcomm_->send(send_tensor, send_rank, true);
      recv_work->wait();
      send_work->wait();
      recv_work.reset();
      send_work.reset();
    }
  };
  auto reset = [&]() { recv_tensor.copy_(original_recv_tensor); };
  auto verify = [&]() { verifyResults(recv_tensor, recv_rank); };
  run(execute, reset, verify);
}

// Test function for asynchronous send/recv with input deleted after enqueue
template <typename Fixture>
void SendRecvTest<Fixture>::testInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  auto send_tensor =
      std::make_shared<at::Tensor>(createSendTensor(count, dtype));
  at::Tensor recv_tensor = createRecvTensor(count, dtype);
  auto original_recv_tensor = recv_tensor.clone();

  auto execute = [&]() {
    if (rank_ % 2 == 0) {
      torchcomm_->send(*send_tensor, send_rank, false);
      torchcomm_->recv(recv_tensor, recv_rank, false);
    } else {
      torchcomm_->recv(recv_tensor, recv_rank, false);
      torchcomm_->send(*send_tensor, send_rank, false);
    }
  };
  auto reset = [&]() { recv_tensor.copy_(original_recv_tensor); };
  auto verify = [&]() { verifyResults(recv_tensor, recv_rank); };
  auto cleanup = [&]() { send_tensor.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create send tensor
template <typename Fixture>
at::Tensor SendRecvTest<Fixture>::createSendTensor(
    int count,
    at::ScalarType dtype) {
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
template <typename Fixture>
at::Tensor SendRecvTest<Fixture>::createRecvTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to verify results
template <typename Fixture>
void SendRecvTest<Fixture>::verifyResults(
    const at::Tensor& recv_tensor,
    int recv_rank) {
  std::string description =
      "recv rank " + std::to_string(recv_rank) + " tensor";
  verifyTensorEquality(recv_tensor.cpu(), recv_rank + 1, description);
}

template class SendRecvTest<EagerTestFixture<SendRecvParams>>;
template class SendRecvTest<GraphTestFixture<SendRecvParams, 1>>;
template class SendRecvTest<GraphTestFixture<SendRecvParams, 2>>;
