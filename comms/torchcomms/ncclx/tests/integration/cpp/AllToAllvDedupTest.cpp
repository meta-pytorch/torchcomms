// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllvDedupTest.hpp"
#include <gtest/gtest.h>

#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"
#include "folly/logging/xlog.h"

std::unique_ptr<TorchCommTestWrapper> AllToAllvDedupTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllToAllvDedupTest::SetUp() {
  // NCCLX alltoallvDedup requires NCCL_CTRAN_ENABLE=1
  setenv("NCCL_CTRAN_ENABLE", "1", 1);
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  num_local_ranks_ = num_ranks_;
  num_nodes_ = num_ranks_ / num_local_ranks_;

  // Get the backend and cast to TorchCommNCCLX for NCCLX-specific APIs
  auto backend = torchcomm_->getBackendImpl();
  ncclx_comm_ =
      std::dynamic_pointer_cast<torch::comms::TorchCommNCCLX>(backend);
  ASSERT_NE(ncclx_comm_, nullptr)
      << "Test requires NCCLX backend. Set TEST_BACKEND=ncclx";
}

void AllToAllvDedupTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

int AllToAllvDedupTest::expertToRank(int expert_id, int num_ranks) const {
  // experts are evenly distributed across ranks, in the order
  // of expert 0, 1->rank 0, expert 2, 3->rank 1, etc, if num_expert_per_rank is
  // 2
  const auto num_experts_per_rank = test_params_.num_experts / num_ranks;
  return expert_id / num_experts_per_rank;
}

int AllToAllvDedupTest::countNumRecvTokens() const {
  EXPECT_FALSE(all_ranks_topk_ids_.empty())
      << "all_ranks_topk_ids must be set before calling countNumRecvTokens";

  auto int32_options = at::TensorOptions().dtype(at::kInt).device(device_type_);
  at::Tensor num_recv_tokens = at::zeros({num_ranks_}, int32_options);

  for (int rank = 0; rank < num_ranks_; ++rank) {
    const auto& rank_topk_ids = all_ranks_topk_ids_[rank];
    // Flatten and iterate
    auto flat_ids = rank_topk_ids.flatten().cpu();
    auto accessor = flat_ids.accessor<int, 1>();
    for (int64_t i = 0; i < accessor.size(0); ++i) {
      int expert_id = accessor[i];
      if (expertToRank(expert_id, num_ranks_) == rank_) {
        num_recv_tokens[rank] += 1;
      }
    }
  }

  return num_recv_tokens.sum().item<int>();
}

at::Tensor AllToAllvDedupTest::getIndiceMapFromMask(
    const at::Tensor& mask) const {
  // Return a compacted index map from mask where True positions are 0..N-1 and
  // False are -1
  auto nz = at::nonzero(mask).flatten();
  auto int_options = at::TensorOptions().dtype(at::kInt).device(mask.device());
  auto indice_map = at::full({mask.size(0)}, -1, int_options);
  if (nz.numel() > 0) {
    auto indices = at::arange(nz.numel(), int_options);
    indice_map.index_put_({nz.to(at::kInt)}, indices);
  }
  return indice_map;
}

at::Tensor AllToAllvDedupTest::getIdListFromMask(const at::Tensor& mask) const {
  // Return indices of True positions in the boolean mask
  return at::nonzero(mask).flatten().to(at::kInt);
}

void AllToAllvDedupTest::setGlobalTopkIds(const at::Tensor& expert_range) {
  int num_tokens = test_params_.num_tokens;
  int topk = test_params_.topk;

  // Generate local random top-k expert IDs per token
  std::vector<at::Tensor> topk_ids_list;
  topk_ids_list.reserve(num_tokens);

  for (int i = 0; i < num_tokens; ++i) {
    // Random permutation and select top-k
    auto perm = at::randperm(expert_range.size(0), expert_range.options());
    auto selected = perm.slice(0, 0, topk);
    auto ids = expert_range.index({selected});
    topk_ids_list.push_back(ids);
  }
  at::Tensor topk_ids = at::stack(topk_ids_list).contiguous();

  // All-gather topk_ids across ranks
  std::vector<at::Tensor> all_ranks_topk_ids;
  all_ranks_topk_ids.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; ++i) {
    all_ranks_topk_ids.push_back(at::empty_like(topk_ids));
  }
  torchcomm_->all_gather(all_ranks_topk_ids, topk_ids, false);

  all_ranks_topk_ids_ = std::move(all_ranks_topk_ids);
}

at::Tensor AllToAllvDedupTest::prepareSendIndices() const {
  EXPECT_FALSE(all_ranks_topk_ids_.empty())
      << "all_ranks_topk_ids must be set before calling prepareSendIndices";

  int num_experts = test_params_.num_experts;
  int num_expert_per_node = num_experts / num_nodes_;

  std::vector<at::Tensor> send_indices_list;
  send_indices_list.reserve(num_nodes_);

  const auto& my_topk_ids = all_ranks_topk_ids_[rank_];
  at::Tensor expert_nodes =
      (my_topk_ids / num_expert_per_node).toType(at::kInt);

  for (int node = 0; node < num_nodes_; ++node) {
    // Create a mask for tokens that have at least one expert assigned to node
    at::Tensor mask = (expert_nodes == node).any(/*dim=*/1);
    at::Tensor indice_map = getIndiceMapFromMask(mask);
    send_indices_list.push_back(indice_map);
  }
  return at::stack(send_indices_list).contiguous();
}

at::Tensor AllToAllvDedupTest::prepareForwardIndices() const {
  EXPECT_FALSE(all_ranks_topk_ids_.empty())
      << "all_ranks_topk_ids must be set before calling prepareForwardIndices";

  int num_experts = test_params_.num_experts;
  int num_expert_per_rank = num_experts / num_ranks_;
  int my_local_rank = rank_ % num_local_ranks_;
  int my_node = rank_ / num_local_ranks_;

  std::vector<at::Tensor> forward_indices_list;
  forward_indices_list.reserve(num_nodes_);

  // Iterate over ranks in the cross-node rail
  for (int send_rank = my_local_rank; send_rank < num_ranks_;
       send_rank += num_local_ranks_) {
    std::vector<at::Tensor> node_forward_indices;
    node_forward_indices.reserve(num_local_ranks_);
    at::Tensor topk_ids_recv_ranks =
        (all_ranks_topk_ids_[send_rank] / num_expert_per_rank).toType(at::kInt);
    // Iterate over local ranks on the same node
    for (int recv_rank = my_node * num_local_ranks_;
         recv_rank < (my_node + 1) * num_local_ranks_;
         ++recv_rank) {
      // Create a mask for tokens that have at least one expert assigned to
      // recv_rank
      at::Tensor mask = (topk_ids_recv_ranks == recv_rank).any(/*dim=*/1);
      at::Tensor indice_map = getIndiceMapFromMask(mask);
      node_forward_indices.push_back(indice_map);
    }
    forward_indices_list.push_back(at::stack(node_forward_indices));
  }
  return at::stack(forward_indices_list).contiguous();
}

at::Tensor AllToAllvDedupTest::prepareRecvIndices() const {
  EXPECT_FALSE(all_ranks_topk_ids_.empty())
      << "all_ranks_topk_ids must be set before calling prepareRecvIndices";

  int num_experts = test_params_.num_experts;
  int num_expert_per_rank = num_experts / num_ranks_;

  std::vector<at::Tensor> recv_indices_list;
  recv_indices_list.reserve(num_expert_per_rank);

  // Iterate over all local experts
  for (int expert_id = num_expert_per_rank * rank_;
       expert_id < num_expert_per_rank * (rank_ + 1);
       ++expert_id) {
    std::vector<at::Tensor> expert_recv_indices;
    expert_recv_indices.reserve(num_ranks_);

    // Iterate over all ranks
    for (int send_rank = 0; send_rank < num_ranks_; ++send_rank) {
      // Create a mask for tokens that have at least one expert assigned to
      // local expert
      at::Tensor mask =
          (all_ranks_topk_ids_[send_rank] == expert_id).any(/*dim=*/1);
      at::Tensor indice_map = getIndiceMapFromMask(mask);
      expert_recv_indices.push_back(indice_map);
    }
    recv_indices_list.push_back(at::stack(expert_recv_indices));
  }
  return at::stack(recv_indices_list).contiguous();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AllToAllvDedupTest::prepareTensors() const {
  int num_recv_tokens = countNumRecvTokens();

  int num_tokens = test_params_.num_tokens;
  int token_numel = test_params_.token_numel;
  at::ScalarType dtype = test_params_.dtype;

  auto options = at::TensorOptions().dtype(dtype).device(device_type_);

  // Create and fill input tensor
  at::Tensor input_tensor = at::arange(num_tokens * token_numel, options)
                                .reshape({num_tokens, token_numel});

  // Fill each block with unique values
  for (int token_id = 0; token_id < num_tokens; ++token_id) {
    auto block = input_tensor[token_id];
    std::vector<int> values(token_numel);
    for (int idx = 0; idx < token_numel; ++idx) {
      values[idx] = rank_ * 10000 + token_id * 1000 + idx;
    }
    block.copy_(
        at::from_blob(
            values.data(), {token_numel}, at::TensorOptions().dtype(dtype)));
  }

  at::Tensor output_tensor =
      at::full({num_recv_tokens * token_numel}, -1, options);

  auto int32_options = at::TensorOptions().dtype(at::kInt).device(device_type_);
  at::Tensor recv_token_ids = at::full({num_recv_tokens}, -1, int32_options);

  return std::make_tuple(
      input_tensor.contiguous(), output_tensor, recv_token_ids);
}

std::vector<std::vector<at::Tensor>>
AllToAllvDedupTest::prepareExpectedRecvTokenIds() const {
  EXPECT_FALSE(all_ranks_topk_ids_.empty()) << "all_ranks_topk_ids must be set";

  int num_experts = test_params_.num_experts;
  int num_expert_per_rank = num_experts / num_ranks_;

  std::vector<std::vector<at::Tensor>> expected_recv_token_ids;
  expected_recv_token_ids.reserve(num_expert_per_rank);

  // Iterate over all local experts and all ranks
  for (int expert_id = num_expert_per_rank * rank_;
       expert_id < num_expert_per_rank * (rank_ + 1);
       ++expert_id) {
    std::vector<at::Tensor> expert_token_ids;
    expert_token_ids.reserve(num_ranks_);

    for (int send_rank = 0; send_rank < num_ranks_; ++send_rank) {
      at::Tensor mask =
          (all_ranks_topk_ids_[send_rank] == expert_id).any(/*dim=*/1);
      at::Tensor rank_token_ids = getIdListFromMask(mask);
      expert_token_ids.push_back(rank_token_ids);
    }
    expected_recv_token_ids.push_back(std::move(expert_token_ids));
  }
  return expected_recv_token_ids;
}

std::vector<std::vector<std::vector<at::Tensor>>>
AllToAllvDedupTest::prepareExpectedOutputTensor(
    const std::vector<std::vector<at::Tensor>>& expected_recv_token_ids) const {
  int token_numel = test_params_.token_numel;
  at::ScalarType dtype = test_params_.dtype;

  auto options = at::TensorOptions().dtype(dtype).device(device_type_);

  std::vector<std::vector<std::vector<at::Tensor>>> expected_tensors;
  expected_tensors.reserve(expected_recv_token_ids.size());

  for (const auto& expert_token_ids : expected_recv_token_ids) {
    std::vector<std::vector<at::Tensor>> expert_blocks;
    expert_blocks.reserve(expert_token_ids.size());

    for (int send_rank = 0;
         send_rank < static_cast<int>(expert_token_ids.size());
         ++send_rank) {
      const auto& rank_token_ids = expert_token_ids[send_rank];
      std::vector<at::Tensor> rank_blocks;

      auto rank_token_ids_cpu = rank_token_ids.cpu();
      int64_t num_tokens_from_rank = rank_token_ids_cpu.numel();
      rank_blocks.reserve(num_tokens_from_rank);

      for (int64_t i = 0; i < num_tokens_from_rank; ++i) {
        int token_id = rank_token_ids_cpu[i].item<int>();
        at::Tensor block = at::empty({token_numel}, options);
        std::vector<int> values(token_numel);
        for (int idx = 0; idx < token_numel; ++idx) {
          values[idx] = send_rank * 10000 + token_id * 1000 + idx;
        }
        block.copy_(
            at::from_blob(
                values.data(),
                {token_numel},
                at::TensorOptions().dtype(dtype)));
        rank_blocks.push_back(block);
      }
      expert_blocks.push_back(std::move(rank_blocks));
    }
    expected_tensors.push_back(std::move(expert_blocks));
  }
  return expected_tensors;
}

void AllToAllvDedupTest::checkExecOutput(
    const at::Tensor& output_tensor,
    const at::Tensor& recv_token_ids) const {
  int token_numel = test_params_.token_numel;

  auto expected_recv_token_ids = prepareExpectedRecvTokenIds();

  // Flatten expected_recv_token_ids
  std::vector<at::Tensor> flat_token_ids;
  for (const auto& expert_token_ids : expected_recv_token_ids) {
    for (const auto& rank_token_ids : expert_token_ids) {
      if (rank_token_ids.numel() > 0) {
        flat_token_ids.push_back(rank_token_ids);
      }
    }
  }

  at::Tensor expected_recv_token_ids_flat;
  if (!flat_token_ids.empty()) {
    expected_recv_token_ids_flat = at::cat(flat_token_ids);
  } else {
    expected_recv_token_ids_flat = at::empty(
        {0}, at::TensorOptions().dtype(at::kLong).device(device_type_));
  }

  // Sanity check length equal
  EXPECT_EQ(expected_recv_token_ids_flat.numel(), recv_token_ids.numel())
      << "Rank " << rank_ << ": recv_token_ids length mismatch. "
      << "Expected: " << expected_recv_token_ids_flat.numel()
      << " Got: " << recv_token_ids.numel();

  // Assert that recv_token_ids matches expected values
  EXPECT_TRUE(
      at::equal(
          recv_token_ids.to(at::kLong),
          expected_recv_token_ids_flat.to(at::kLong)))
      << "Rank " << rank_ << ": recv_token_ids mismatch";

  auto expected_output_tensors =
      prepareExpectedOutputTensor(expected_recv_token_ids);

  // Assert that output_tensor matches expected values
  int global_recv_indice = 0;
  for (size_t expert_id = 0; expert_id < expected_output_tensors.size();
       ++expert_id) {
    const auto& expert_blocks = expected_output_tensors[expert_id];
    for (size_t send_rank = 0; send_rank < expert_blocks.size(); ++send_rank) {
      const auto& rank_blocks = expert_blocks[send_rank];
      for (size_t recv_indice = 0; recv_indice < rank_blocks.size();
           ++recv_indice) {
        const auto& expected = rank_blocks[recv_indice];
        int64_t output_start = global_recv_indice * token_numel;
        int64_t output_end = output_start + token_numel;
        at::Tensor output = output_tensor.slice(0, output_start, output_end);

        EXPECT_TRUE(at::equal(output, expected))
            << "Rank " << rank_
            << ": output_tensor block mismatch at recv_indice=" << recv_indice
            << " (expert_id=" << expert_id << ", send_rank=" << send_rank
            << ") at global_recv_indice=" << global_recv_indice;

        global_recv_indice += 1;
      }
    }
  }
}

void AllToAllvDedupTest::resetRunTest(
    int num_tokens,
    int token_numel,
    int num_experts,
    int topk,
    at::ScalarType dtype) {
  all_ranks_topk_ids_.clear();
  test_params_ = DedupTestParams{
      num_tokens,
      token_numel,
      topk,
      num_experts,
      dtype,
  };
}

void AllToAllvDedupTest::runTest() {
  int num_tokens = test_params_.num_tokens;
  int token_numel = test_params_.token_numel;
  int num_experts = test_params_.num_experts;
  int topk = test_params_.topk;
  at::ScalarType dtype = test_params_.dtype;

  // Generate test specific expert range - use simple [0, 1] for debugging
  auto int_options = at::TensorOptions().dtype(at::kInt).device(device_type_);
  at::Tensor expert_range = at::arange(0, num_experts, int_options);
  XLOG(INFO) << "[Rank " << rank_
             << "] expert_range: " << tensorToString(expert_range);
  setGlobalTopkIds(expert_range);

  // Log all_ranks_topk_ids for debugging, each rank logs its own topk_ids
  XLOG(INFO) << "[Rank " << rank_ << "] all_ranks_topk_ids[" << rank_
             << "]: " << tensorToString(all_ranks_topk_ids_[rank_]);

  // Generate indices as exec input arguments
  at::Tensor send_indices = prepareSendIndices();
  at::Tensor forward_indices = prepareForwardIndices();
  at::Tensor recv_indices = prepareRecvIndices();

  // Allocate input and output tensors
  auto [input_tensor, output_tensor, recv_token_ids] = prepareTensors();

  // Log tensor shapes and values for debugging
  XLOG(INFO) << "[Rank " << rank_
             << "] send_indices: shape: " << send_indices.sizes()
             << ", value: " << tensorToString(send_indices);
  XLOG(INFO) << "[Rank " << rank_
             << "] forward_indices: shape: " << forward_indices.sizes()
             << ", value: " << tensorToString(forward_indices);
  XLOG(INFO) << "[Rank " << rank_
             << "] recv_indices: shape: " << recv_indices.sizes()
             << ", value: " << tensorToString(recv_indices);
  XLOG(INFO) << "[Rank " << rank_
             << "] input_tensor shape: " << input_tensor.sizes();
  XLOG(INFO) << "[Rank " << rank_
             << "] output_tensor shape: " << output_tensor.sizes();
  XLOG(INFO) << "[Rank " << rank_
             << "] recv_token_ids shape: " << recv_token_ids.sizes();

  // Initialize the alltoallv_dedup operation
  auto p_req = ncclx_comm_->alltoallv_dedup_init(
      num_tokens, token_numel, topk, num_experts / num_ranks_, dtype, true);
  ASSERT_NE(p_req, nullptr) << "alltoallv_dedup_init returned nullptr";

  // Execute the alltoallv_dedup operation
  auto work = ncclx_comm_->alltoallv_dedup_exec(
      output_tensor,
      recv_token_ids,
      input_tensor,
      send_indices,
      forward_indices,
      recv_indices,
      p_req);
  work->wait();

  // Verify the output
  checkExecOutput(output_tensor, recv_token_ids);
  XLOG(INFO) << "[Rank " << rank_ << "] AllToAllvDedupTest passed";
}
