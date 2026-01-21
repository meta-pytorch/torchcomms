// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

namespace py = pybind11;
using namespace torch::comms;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(_comms_ncclx, m) {
  m.doc() = "NCCLX specific python bindings for TorchComm";

  intrusive_ptr_class_<TorchCommNCCLXPersistentRequest>(
      m, "TorchCommNCCLXPersistentRequest");

  py::class_<TorchCommNCCLX, std::shared_ptr<TorchCommNCCLX>>(
      m, "TorchCommNCCLX")
      .def(
          "alltoallv_dynamic_dispatch",
          [](TorchCommNCCLX& self,
             const std::vector<at::Tensor>& output_tensor_list,
             at::Tensor& output_chunk_sizes_per_rank,
             const at::Tensor& input_tensor,
             const at::Tensor& input_chunk_sizes,
             const at::Tensor& input_chunk_indices,
             const at::Tensor& input_chunk_count_per_rank,
             int64_t hidden_dim,
             bool async_op) {
            // Scale up input_chunk_sizes by hidden_dim for C++ API
            at::Tensor scaled_input_chunk_sizes =
                input_chunk_sizes * hidden_dim;

            // Call C++ API with scaled chunk sizes
            auto work = self.alltoallv_dynamic_dispatch(
                output_tensor_list,
                output_chunk_sizes_per_rank,
                input_tensor,
                scaled_input_chunk_sizes,
                input_chunk_indices,
                input_chunk_count_per_rank,
                async_op);

            // Scale down output_chunk_sizes_per_rank by hidden_dim for Python
            // API Use floor_divide_ to maintain integer type
            output_chunk_sizes_per_rank.floor_divide_(hidden_dim);

            return work;
          },
          R"(
All-to-all dynamic dispatch operation with variable split sizes and non-contiguous indices.

This API performs an all-to-all communication where each rank can send multiple chunks
to each destination rank, with chunks potentially non-contiguous in the send buffer.

Args:
    output_tensor_list: List of output tensors (one per source rank) to receive data.
    output_chunk_sizes_per_rank: Output tensor to receive chunk size information from all ranks.
    input_tensor: Input tensor containing all data to be sent.
    input_chunk_sizes: Tensor of chunk sizes (one per chunk across all destination ranks).
    input_chunk_indices: Tensor of chunk indices indicating where each chunk is located in input_tensor.
    input_chunk_count_per_rank: Tensor indicating how many chunks are sent to each rank.
    hidden_dim: Hidden dimension size for scaling up/down chunk sizes.
    async_op: Whether to perform the operation asynchronously.

Returns:
    TorchWork object for tracking operation completion.
)",
          py::arg("output_tensor_list"),
          py::arg("output_chunk_sizes_per_rank"),
          py::arg("input_tensor"),
          py::arg("input_chunk_sizes"),
          py::arg("input_chunk_indices"),
          py::arg("input_chunk_count_per_rank"),
          py::arg("hidden_dim"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "alltoallv_dynamic_combine",
          [](TorchCommNCCLX& self,
             at::Tensor& output_tensor,
             const at::Tensor& input_tensor,
             const at::Tensor& input_chunk_sizes,
             const at::Tensor& input_chunk_indices,
             const at::Tensor& input_chunk_count_per_rank,
             int64_t hidden_dim,
             bool async_op) {
            // Scale up input_chunk_sizes by hidden_dim for C++ API
            at::Tensor scaled_input_chunk_sizes =
                input_chunk_sizes * hidden_dim;

            // Call C++ API with scaled chunk sizes
            return self.alltoallv_dynamic_combine(
                output_tensor,
                input_tensor,
                scaled_input_chunk_sizes,
                input_chunk_indices,
                input_chunk_count_per_rank,
                async_op);
          },
          R"(
All-to-all dynamic combine operation with variable split sizes and non-contiguous indices.

This API performs an all-to-all communication where each rank can send multiple chunks
to each destination rank, with chunks potentially non-contiguous in the send buffer.
This is the inverse operation of alltoallv_dynamic_dispatch.

Args:
    output_tensor: Output tensor to receive combined data.
    input_tensor: Input tensor containing all data to be sent.
    input_chunk_sizes: Tensor of chunk sizes (one per chunk across all destination ranks).
    input_chunk_indices: Tensor of chunk indices indicating where each chunk is located in input_tensor.
    input_chunk_count_per_rank: Tensor indicating how many chunks are sent to each rank.
    hidden_dim: Hidden dimension size for scaling up/down chunk sizes.
    async_op: Whether to perform the operation asynchronously.

Returns:
    TorchWork object for tracking operation completion.
)",
          py::arg("output_tensor"),
          py::arg("input_tensor"),
          py::arg("input_chunk_sizes"),
          py::arg("input_chunk_indices"),
          py::arg("input_chunk_count_per_rank"),
          py::arg("hidden_dim"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "alltoallv_dedup_init",
          [](TorchCommNCCLX& self,
             int64_t num_send_blocks,
             int64_t block_count,
             int64_t block_num_recv_buckets,
             int64_t num_recv_buckets,
             at::ScalarType dtype,
             bool async_op) {
            return self.alltoallv_dedup_init(
                static_cast<int>(num_send_blocks),
                static_cast<int>(block_count),
                static_cast<int>(block_num_recv_buckets),
                static_cast<int>(num_recv_buckets),
                dtype,
                async_op);
          },
          R"(
Initialize all-to-all deduplication operation.

This API initializes the all-to-all deduplication communication pattern by setting up
the necessary data structures and internal buffers for efficient communication with deduplication.

Args:
    num_send_blocks: Number of blocks to send (number of tokens in MoE).
    block_count: Number of elements per block (elements per token in MoE).
    block_num_recv_buckets: Number of receive buckets per block (topK in MoE).
    num_recv_buckets: Total number of receive buckets (number of experts per rank in MoE).
    dtype: Data type of the tensors.
    async_op: Whether to perform the operation asynchronously. Determines the stream used and
              must remain unchanged for the entire lifetime of the persistent operation.

Returns:
    TorchCommNCCLXPersistentRequest: Holds persistent resources for the operation.
)",
          py::arg("num_send_blocks"),
          py::arg("block_count"),
          py::arg("block_num_recv_buckets"),
          py::arg("num_recv_buckets"),
          py::arg("dtype"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "alltoallv_dedup_exec",
          [](TorchCommNCCLX& self,
             at::Tensor& output_tensor,
             at::Tensor& recv_block_ids,
             const at::Tensor& input_tensor,
             const at::Tensor& send_indices,
             const at::Tensor& forward_indices,
             const at::Tensor& recv_indices,
             at::intrusive_ptr<TorchCommNCCLXPersistentRequest> pReq) {
            return self.alltoallv_dedup_exec(
                output_tensor,
                recv_block_ids,
                input_tensor,
                send_indices,
                forward_indices,
                recv_indices,
                pReq);
          },
          R"(
Dispatch of all-to-all deduplication operation using a persistent request.

Args:
  - output_tensor: Output tensor to receive data (stores received tokens in MoE).
  - recv_block_ids: Received block IDs, one per block in output_tensor (token ID in MoE).
  - input_tensor: Input tensor containing data to be sent (stores tokens in MoE).
  - send_indices: Indices of unique blocks to be sent to each node; size = num_nodes * num_send_blocks.
  - forward_indices: Indices of unique blocks received from each node and forwarded to each local rank; size = num_nodes * num_local_ranks * num_send_blocks.
  - recv_indices: Indices of unique blocks received by each local bucket from every rank; size = num_recv_buckets * num_ranks * num_send_blocks.
  - request: Persistent request object returned by alltoallv_dedup_init.

Notes:
  - Each of send_indices, forward_indices, and recv_indices is a 1D tensor of the specified size.
  - For each of num_send_blocks elements, the value is the compacted index on the receiving side, or -1 if the block is not received.

Example:
  Suppose we have 2 nodes, 2 local ranks per node (4 ranks total), 3 send blocks (3 tokens from each rank in MoE),
  and 2 recv buckets per rank (total 8 experts distributed across 4 ranks in MoE).

  INPUT send_indices (size = num_nodes * num_send_blocks = 2 * 3 = 6):
    [0, -1, 1, 0, 1, -1]
    Meaning: To node 0, send blocks at compacted indices [0, -1, 1] (block 1 not sent, blocks 0 and 2 map to receiving indices 0 and 1);
             to node 1, send blocks at compacted indices [0, 1, -1] (block 2 not sent, blocks 0 and 1 map to receiving indices 0 and 1).

  INPUT forward_indices (size = num_nodes * num_local_ranks * num_send_blocks = 2 * 2 * 3 = 12):
    [0, 1, -1, 0, -1, 1, 0, -1, 1, -1, 0, 1]
    Meaning: From node 0 to local rank 0, forward to indices [0, 1, -1]; to local rank 1, forward to indices [0, -1, 1];
             from node 1 to local rank 0, forward to indices [0, -1, 1]; to local rank 1, forward to indices [-1, 0, 1].

  INPUT recv_indices (size = num_recv_buckets * num_ranks * num_send_blocks = 2 * 4 * 3 = 24):
    [0, -1, 1, 0, 1, -1, 0, 1, -1, -1, 0, 1, ...]
    Meaning: For bucket 0 from rank 0, receive indices [0, -1, 1]; from rank 1, receive indices [0, 1, -1];
             from rank 2, receive indices [0, 1, -1]; from rank 3, receive indices [-1, 0, 1]; etc.

  OUTPUT recv_block_ids (size = num_recv_blocks = 7):
    [0, 2, 0, 1, 0, 1, 1, 2, ...]
    Meaning: Received bucket 0 received block ids [0, 2] from rank 0; [0, 1] from rank 1; [0, 1] from rank 2; [1, 2] from rank 3; etc.

Returns:
    TorchWork object for tracking operation completion.
)",
          py::arg("output_tensor"),
          py::arg("recv_block_ids"),
          py::arg("input_tensor"),
          py::arg("send_indices"),
          py::arg("forward_indices"),
          py::arg("recv_indices"),
          py::arg("request"),
          py::call_guard<py::gil_scoped_release>());
}
