// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BatchSendRecvTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "TorchCommTestHelpers.h"

// Test function for synchronous batch SendRecv operations
template <typename Fixture>
void BatchSendRecvTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  std::vector<at::Tensor> send_tensors = createSendTensors(count, dtype);
  std::vector<at::Tensor> recv_tensors = createRecvTensors(count, dtype);

  std::vector<at::Tensor> original_recv_tensors;
  original_recv_tensors.reserve(recv_tensors.size());
  for (const auto& t : recv_tensors) {
    original_recv_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto batch_op = torchcomm_->batch_op_create();
    for (size_t i = 0; i < send_tensors.size(); ++i) {
      batch_op.send(send_tensors[i], send_rank);
    }
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      batch_op.recv(recv_tensors[i], recv_rank);
    }
    auto work = batch_op.issue(false);
    work->wait();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      recv_tensors[i].copy_(original_recv_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(recv_tensors, recv_rank); };
  run(execute, reset, verify);
}

// Test function for synchronous batch SendRecv operations without work object
template <typename Fixture>
void BatchSendRecvTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  std::vector<at::Tensor> send_tensors = createSendTensors(count, dtype);
  std::vector<at::Tensor> recv_tensors = createRecvTensors(count, dtype);

  std::vector<at::Tensor> original_recv_tensors;
  original_recv_tensors.reserve(recv_tensors.size());
  for (const auto& t : recv_tensors) {
    original_recv_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto batch_op = torchcomm_->batch_op_create();
    for (size_t i = 0; i < send_tensors.size(); ++i) {
      batch_op.send(send_tensors[i], send_rank);
    }
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      batch_op.recv(recv_tensors[i], recv_rank);
    }
    batch_op.issue(false);
  };
  auto reset = [&]() {
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      recv_tensors[i].copy_(original_recv_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(recv_tensors, recv_rank); };
  run(execute, reset, verify);
}

// Test function for asynchronous batch SendRecv operations
template <typename Fixture>
void BatchSendRecvTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  std::vector<at::Tensor> send_tensors = createSendTensors(count, dtype);
  std::vector<at::Tensor> recv_tensors = createRecvTensors(count, dtype);

  std::vector<at::Tensor> original_recv_tensors;
  original_recv_tensors.reserve(recv_tensors.size());
  for (const auto& t : recv_tensors) {
    original_recv_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto batch_op = torchcomm_->batch_op_create();
    for (size_t i = 0; i < send_tensors.size(); ++i) {
      batch_op.send(send_tensors[i], send_rank);
    }
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      batch_op.recv(recv_tensors[i], recv_rank);
    }
    auto work = batch_op.issue(true);
    work->wait();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      recv_tensors[i].copy_(original_recv_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(recv_tensors, recv_rank); };
  run(execute, reset, verify);
}

// Test function for asynchronous batch SendRecv operations with early reset
template <typename Fixture>
void BatchSendRecvTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  std::vector<at::Tensor> send_tensors = createSendTensors(count, dtype);
  std::vector<at::Tensor> recv_tensors = createRecvTensors(count, dtype);

  std::vector<at::Tensor> original_recv_tensors;
  original_recv_tensors.reserve(recv_tensors.size());
  for (const auto& t : recv_tensors) {
    original_recv_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto batch_op = torchcomm_->batch_op_create();
    for (size_t i = 0; i < send_tensors.size(); ++i) {
      batch_op.send(send_tensors[i], send_rank);
    }
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      batch_op.recv(recv_tensors[i], recv_rank);
    }
    auto work = batch_op.issue(true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      recv_tensors[i].copy_(original_recv_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(recv_tensors, recv_rank); };
  run(execute, reset, verify);
}

// Test function for asynchronous batch SendRecv operations with input deleted
// after enqueue
template <typename Fixture>
void BatchSendRecvTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  auto send_tensors = std::make_shared<std::vector<at::Tensor>>(
      createSendTensors(count, dtype));
  std::vector<at::Tensor> recv_tensors = createRecvTensors(count, dtype);

  std::vector<at::Tensor> original_recv_tensors;
  original_recv_tensors.reserve(recv_tensors.size());
  for (const auto& t : recv_tensors) {
    original_recv_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto batch_op = torchcomm_->batch_op_create();
    for (size_t i = 0; i < send_tensors->size(); ++i) {
      batch_op.send((*send_tensors)[i], send_rank);
    }
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      batch_op.recv(recv_tensors[i], recv_rank);
    }
    batch_op.issue(false);
  };
  auto reset = [&]() {
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      recv_tensors[i].copy_(original_recv_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(recv_tensors, recv_rank); };
  auto cleanup = [&]() { send_tensors.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create send tensor with tensor-specific values
template <typename Fixture>
std::vector<at::Tensor> BatchSendRecvTest<Fixture>::createSendTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> tensors;
  tensors.reserve(2);
  for (int i = 0; i < 2; ++i) {
    int value = (rank_ + 1) * 10 + i;
    if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
      tensors.push_back(at::ones({count}, options) * static_cast<float>(value));
    } else if (dtype == at::kInt) {
      tensors.push_back(at::ones({count}, options) * static_cast<int>(value));
    } else if (dtype == at::kChar) {
      tensors.push_back(
          at::ones({count}, options) * static_cast<signed char>(value % 128));
    }
  }
  return tensors;
}

// Helper function to create receive tensor
template <typename Fixture>
std::vector<at::Tensor> BatchSendRecvTest<Fixture>::createRecvTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> tensors;
  tensors.reserve(2);
  for (int i = 0; i < 2; ++i) {
    tensors.push_back(at::zeros({count}, options));
  }
  return tensors;
}

// Helper function to verify results
template <typename Fixture>
void BatchSendRecvTest<Fixture>::verifyResults(
    const std::vector<at::Tensor>& recv_tensors,
    int recv_rank) {
  for (size_t i = 0; i < recv_tensors.size(); ++i) {
    int expectedValue = (recv_rank + 1) * 10 + static_cast<int>(i);
    std::string description = "recv rank " + std::to_string(recv_rank) +
        " tensor " + std::to_string(i);
    if (recv_tensors[i].scalar_type() == at::kChar) {
      expectedValue = expectedValue % 128;
    }
    verifyTensorEquality(recv_tensors[i].cpu(), expectedValue, description);
  }
}

template class BatchSendRecvTest<EagerTestFixture<BatchSendRecvParams>>;
template class BatchSendRecvTest<GraphTestFixture<BatchSendRecvParams, 1>>;
template class BatchSendRecvTest<GraphTestFixture<BatchSendRecvParams, 2>>;
