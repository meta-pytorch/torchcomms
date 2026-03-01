// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CollectiveTimeoutTest.hpp"

#include <cstdlib>
#include <functional>
#include <vector>

#include <ATen/cuda/CUDAContext.h>

#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommBatch.hpp"
#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/TorchWork.hpp"

using torch::comms::test::RankExpectation;
using torch::comms::test::TimeoutTestHelper;
using ExecMode = TimeoutTestHelper::ExecMode;

std::string CollectiveTimeoutTest::collectiveTypeName(
    const CollectiveType type) {
  switch (type) {
    case CollectiveType::kSendRecv:
      return "SendRecv";
    case CollectiveType::kBatchSendRecv:
      return "BatchSendRecv";
    case CollectiveType::kBroadcast:
      return "Broadcast";
    case CollectiveType::kAllReduce:
      return "AllReduce";
    case CollectiveType::kReduce:
      return "Reduce";
    case CollectiveType::kAllGather:
      return "AllGather";
    case CollectiveType::kAllGatherSingle:
      return "AllGatherSingle";
    case CollectiveType::kAllGatherV:
      return "AllGatherV";
    case CollectiveType::kReduceScatter:
      return "ReduceScatter";
    case CollectiveType::kReduceScatterSingle:
      return "ReduceScatterSingle";
    case CollectiveType::kReduceScatterV:
      return "ReduceScatterV";
    case CollectiveType::kAllToAllSingle:
      return "AllToAllSingle";
    case CollectiveType::kAllToAll:
      return "AllToAll";
    case CollectiveType::kBarrier:
      return "Barrier";
    case CollectiveType::kScatter:
      return "Scatter";
    case CollectiveType::kGather:
      return "Gather";
  }
  return "Unknown";
}

namespace {
at::Tensor makeTestTensor(
    const std::vector<int64_t>& sizes,
    const c10::DeviceType device_type,
    const int rank) {
  const auto device_index = rank % at::cuda::device_count();
  return at::ones(
      sizes,
      at::TensorOptions().dtype(at::kFloat).device(device_type, device_index));
}
} // namespace

void CollectiveTimeoutTest::childSetUp() {
  wrapper_ = std::make_unique<TorchCommTestWrapper>();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  if (isRunningOnCPU()) {
    device_type_ = c10::DeviceType::CPU;
  } else {
    device_type_ = c10::DeviceType::CUDA;
  }
}

void CollectiveTimeoutTest::childTearDown() {
  if (torchcomm_) {
    try {
      torchcomm_->finalize();
    } catch (...) {
    }
    torchcomm_.reset();
  }
}

void CollectiveTimeoutTest::execute(
    const CollectiveType type,
    const bool asyncOp,
    const std::chrono::milliseconds timeout) {
  constexpr int kCount = 1024;
  auto tensor = makeTestTensor({kCount}, device_type_, rank_);
  std::vector<c10::intrusive_ptr<torch::comms::TorchWork>> works;

  switch (type) {
    case CollectiveType::kSendRecv: {
      torch::comms::SendOptions send_opts;
      send_opts.timeout = timeout;
      torch::comms::RecvOptions recv_opts;
      recv_opts.timeout = timeout;
      const int dst = (rank_ + 1) % num_ranks_;
      const int src = (rank_ + num_ranks_ - 1) % num_ranks_;
      auto recv_tensor = makeTestTensor({kCount}, device_type_, rank_);
      // Alternate send/recv order by rank parity to avoid GPU-level deadlock.
      // Standalone ncclSend/ncclRecv are serialized on internal_stream_;
      // if all ranks send first, recv kernels deadlock waiting for a match.
      if (rank_ % 2 == 0) {
        works.push_back(torchcomm_->send(tensor, dst, asyncOp, send_opts));
        works.push_back(torchcomm_->recv(recv_tensor, src, asyncOp, recv_opts));
      } else {
        works.push_back(torchcomm_->recv(recv_tensor, src, asyncOp, recv_opts));
        works.push_back(torchcomm_->send(tensor, dst, asyncOp, send_opts));
      }
      break;
    }
    case CollectiveType::kBatchSendRecv: {
      // Batch API groups send+recv in ncclGroupStart/End, no deadlock risk.
      torch::comms::BatchP2POptions batch_opts;
      batch_opts.timeout = timeout;
      const int dst = (rank_ + 1) % num_ranks_;
      const int src = (rank_ + num_ranks_ - 1) % num_ranks_;
      auto recv_tensor = makeTestTensor({kCount}, device_type_, rank_);
      auto batch = torchcomm_->batch_op_create();
      batch.send(tensor, dst);
      batch.recv(recv_tensor, src);
      works.push_back(batch.issue(asyncOp, batch_opts));
      break;
    }
    case CollectiveType::kBroadcast: {
      torch::comms::BroadcastOptions opts;
      opts.timeout = timeout;
      works.push_back(torchcomm_->broadcast(tensor, 0, asyncOp, opts));
      break;
    }
    case CollectiveType::kAllReduce: {
      torch::comms::AllReduceOptions opts;
      opts.timeout = timeout;
      works.push_back(torchcomm_->all_reduce(
          tensor, torch::comms::ReduceOp::SUM, asyncOp, opts));
      break;
    }
    case CollectiveType::kReduce: {
      torch::comms::ReduceOptions opts;
      opts.timeout = timeout;
      works.push_back(torchcomm_->reduce(
          tensor, 1, torch::comms::ReduceOp::SUM, asyncOp, opts));
      break;
    }
    case CollectiveType::kAllGather: {
      torch::comms::AllGatherOptions opts;
      opts.timeout = timeout;
      std::vector<at::Tensor> outputs;
      outputs.reserve(num_ranks_);
      for (int i = 0; i < num_ranks_; ++i) {
        outputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
      }
      works.push_back(torchcomm_->all_gather(outputs, tensor, asyncOp, opts));
      break;
    }
    case CollectiveType::kAllGatherSingle: {
      torch::comms::AllGatherSingleOptions opts;
      opts.timeout = timeout;
      auto output = makeTestTensor({kCount * num_ranks_}, device_type_, rank_);
      works.push_back(
          torchcomm_->all_gather_single(output, tensor, asyncOp, opts));
      break;
    }
    case CollectiveType::kAllGatherV: {
      torch::comms::AllGatherOptions opts;
      opts.timeout = timeout;
      std::vector<at::Tensor> outputs;
      outputs.reserve(num_ranks_);
      for (int i = 0; i < num_ranks_; ++i) {
        outputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
      }
      works.push_back(torchcomm_->all_gather_v(outputs, tensor, asyncOp, opts));
      break;
    }
    case CollectiveType::kReduceScatter: {
      torch::comms::ReduceScatterOptions opts;
      opts.timeout = timeout;
      auto output = makeTestTensor({kCount}, device_type_, rank_);
      std::vector<at::Tensor> inputs;
      inputs.reserve(num_ranks_);
      for (int i = 0; i < num_ranks_; ++i) {
        inputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
      }
      works.push_back(torchcomm_->reduce_scatter(
          output, inputs, torch::comms::ReduceOp::SUM, asyncOp, opts));
      break;
    }
    case CollectiveType::kReduceScatterSingle: {
      torch::comms::ReduceScatterSingleOptions opts;
      opts.timeout = timeout;
      auto input = makeTestTensor({kCount * num_ranks_}, device_type_, rank_);
      auto output = makeTestTensor({kCount}, device_type_, rank_);
      works.push_back(torchcomm_->reduce_scatter_single(
          output, input, torch::comms::ReduceOp::SUM, asyncOp, opts));
      break;
    }
    case CollectiveType::kReduceScatterV: {
      torch::comms::ReduceScatterOptions opts;
      opts.timeout = timeout;
      auto output = makeTestTensor({kCount}, device_type_, rank_);
      std::vector<at::Tensor> inputs;
      inputs.reserve(num_ranks_);
      for (int i = 0; i < num_ranks_; ++i) {
        inputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
      }
      works.push_back(torchcomm_->reduce_scatter_v(
          output, inputs, torch::comms::ReduceOp::SUM, asyncOp, opts));
      break;
    }
    case CollectiveType::kAllToAllSingle: {
      torch::comms::AllToAllSingleOptions opts;
      opts.timeout = timeout;
      auto output = makeTestTensor({kCount * num_ranks_}, device_type_, rank_);
      auto input = makeTestTensor({kCount * num_ranks_}, device_type_, rank_);
      works.push_back(
          torchcomm_->all_to_all_single(output, input, asyncOp, opts));
      break;
    }
    case CollectiveType::kAllToAll: {
      torch::comms::AllToAllOptions opts;
      opts.timeout = timeout;
      std::vector<at::Tensor> outputs;
      std::vector<at::Tensor> inputs;
      outputs.reserve(num_ranks_);
      inputs.reserve(num_ranks_);
      for (int i = 0; i < num_ranks_; ++i) {
        outputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
        inputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
      }
      works.push_back(torchcomm_->all_to_all(outputs, inputs, asyncOp, opts));
      break;
    }
    case CollectiveType::kBarrier: {
      torch::comms::BarrierOptions opts;
      opts.timeout = timeout;
      works.push_back(torchcomm_->barrier(asyncOp, opts));
      break;
    }
    case CollectiveType::kScatter: {
      torch::comms::ScatterOptions opts;
      opts.timeout = timeout;
      auto output = makeTestTensor({kCount}, device_type_, rank_);
      std::vector<at::Tensor> inputs;
      if (rank_ == 0) {
        inputs.reserve(num_ranks_);
        for (int i = 0; i < num_ranks_; ++i) {
          inputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
        }
      }
      works.push_back(torchcomm_->scatter(output, inputs, 0, asyncOp, opts));
      break;
    }
    case CollectiveType::kGather: {
      torch::comms::GatherOptions opts;
      opts.timeout = timeout;
      std::vector<at::Tensor> outputs;
      if (rank_ == 1) {
        outputs.reserve(num_ranks_);
        for (int i = 0; i < num_ranks_; ++i) {
          outputs.push_back(makeTestTensor({kCount}, device_type_, rank_));
        }
      }
      works.push_back(torchcomm_->gather(outputs, tensor, 1, asyncOp, opts));
      break;
    }
  }

  if (asyncOp) {
    for (auto& work : works) {
      work->wait();
    }
  }
}

void CollectiveTimeoutTest::testTimeout(
    const CollectiveType type,
    const ExecMode mode) {
  if (mode != ExecMode::kEager && isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph timeout tests not supported on CPU";
  }
  // Graph mode timeout detection requires GraphEventTracker (ncclx only).
  // TODO: Port GraphEventTracker to nccl to enable graph timeout tests.
  if (mode != ExecMode::kEager) {
    const char* backend = std::getenv("TEST_BACKEND");
    if (!backend || std::string(backend) != "ncclx") {
      GTEST_SKIP()
          << "Graph mode timeout requires GraphEventTracker (ncclx only)";
    }
  }

  // Expected exit behavior per rank:
  // - Rank 0 exits cleanly or gets aborted
  // - Rank 1+ must be aborted with timeout log
  const std::vector<RankExpectation> expectations = {
      {.exitCode = 0, .signal = SIGABRT},
      {.signal = SIGABRT,
       .logMustContain = {"Aborting process due to timeout"}},
  };

  helper_.launch(
      collectiveTypeName(type),
      num_ranks_,
      [&](int /*rank*/) {
        childSetUp();

        std::vector<std::function<void()>> ops;
        ops.reserve(kNumWarmup + 1);
        for (int i = 0; i < kNumWarmup; i++) {
          ops.emplace_back([&] { execute(type); });
        }
        if (rank_ != 0) {
          ops.emplace_back([&] { execute(type, true, kTimeout); });
        }
        helper_.exec(mode, ops);

        // rank 0 skips the last op so other ranks can timeout
        if (rank_ == 0) {
          std::this_thread::sleep_for(
              kRank0Sleep); // NOLINT(facebook-hte-BadCall-sleep_for)
          _exit(0);
        } else {
          childTearDown();
        }
      },
      expectations);
}
