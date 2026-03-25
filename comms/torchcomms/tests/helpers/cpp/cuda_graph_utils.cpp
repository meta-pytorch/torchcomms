// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Defined in write_addr_kernel.cu
void launchWriteAddr(const void* buf, int64_t* probe, cudaStream_t stream);

namespace py = pybind11;

namespace {

/// Launch a kernel that writes &buf into probe[0].
/// buf_ptr and probe_ptr are device pointers (from tensor.data_ptr()).
/// stream_ptr is a cudaStream_t (from torch.cuda.current_stream().cuda_stream).
void writeAddr(uintptr_t buf_ptr, uintptr_t probe_ptr, uintptr_t stream_ptr) {
  launchWriteAddr(
      reinterpret_cast<const void*>( // NOLINT(performance-no-int-to-ptr)
          buf_ptr),
      reinterpret_cast<int64_t*>( // NOLINT(performance-no-int-to-ptr)
          probe_ptr),
      reinterpret_cast<cudaStream_t>( // NOLINT(performance-no-int-to-ptr)
          stream_ptr));
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("writeAddr kernel launch failed: ") +
        cudaGetErrorString(err));
  }
}

} // namespace

PYBIND11_MODULE(cuda_graph_utils, m) {
  m.def(
      "write_addr",
      &writeAddr,
      py::arg("buf_ptr"),
      py::arg("probe_ptr"),
      py::arg("stream_ptr"),
      "Launch a kernel that writes the address of buf into probe[0].\n"
      "All arguments are raw device pointers (int from tensor.data_ptr()).");
}
