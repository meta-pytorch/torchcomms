// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_comms_ncclx, m) {
  m.doc() = "NCCLX specific python bindings for TorchComm";

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
          py::call_guard<py::gil_scoped_release>());
}
