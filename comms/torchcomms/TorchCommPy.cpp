// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <c10/util/intrusive_ptr.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchWork.hpp"

namespace py = pybind11;
using namespace torch::comms;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(_comms, m) {
  m.doc() = "Python bindings for TorchComm";

  // Bind RedOpType enum
  py::enum_<ReduceOp::RedOpType>(
      m, "RedOpType", "Reduce Operation Type to perform during reduction.")
      .value("SUM", ReduceOp::RedOpType::SUM)
      .value("PRODUCT", ReduceOp::RedOpType::PRODUCT)
      .value("MIN", ReduceOp::RedOpType::MIN)
      .value("MAX", ReduceOp::RedOpType::MAX)
      .value("BAND", ReduceOp::RedOpType::BAND)
      .value("BOR", ReduceOp::RedOpType::BOR)
      .value("BXOR", ReduceOp::RedOpType::BXOR)
      .value("PREMUL_SUM", ReduceOp::RedOpType::PREMUL_SUM)
      .value("AVG", ReduceOp::RedOpType::AVG);

  // Bind ReduceOp class
  py::class_<ReduceOp>(m, "ReduceOp", "Operation to perform during reduction.")
      .def(
          py::init<ReduceOp::RedOpType>(),
          "Create default ReduceOp",
          py::arg("opType"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "type", &ReduceOp::type, "Get the type of the operation")
      .def_static(
          "PREMUL_SUM",
          &ReduceOp::make_nccl_premul_sum,
          py::arg("factor"),
          py::call_guard<py::gil_scoped_release>())
      .def_readonly_static("SUM", &ReduceOp::SUM, "Sum reduction operation")
      .def_readonly_static(
          "PRODUCT", &ReduceOp::PRODUCT, "Product reduction operation")
      .def_readonly_static("MIN", &ReduceOp::MIN, "Minimum reduction operation")
      .def_readonly_static("MAX", &ReduceOp::MAX, "Maximum reduction operation")
      .def_readonly_static(
          "BAND", &ReduceOp::BAND, "Bitwise AND reduction operation")
      .def_readonly_static(
          "BOR", &ReduceOp::BOR, "Bitwise OR reduction operation")
      .def_readonly_static(
          "BXOR", &ReduceOp::BXOR, "Bitwise XOR reduction operation")
      .def_readonly_static(
          "AVG", &ReduceOp::AVG, "Average reduction operation");

  // Bind CommOptions structure
  py::class_<CommOptions>(
      m, "CommOptions", "Options for communicator creation.")
      .def(py::init<>(), "Create default CommOptions")
      .def_readwrite(
          "abort_process_on_timeout_or_error",
          &CommOptions::abort_process_on_timeout_or_error,
          "Whether to abort process on timeout or error")
      .def_readwrite(
          "timeout",
          &CommOptions::timeout,
          "Timeout for operations (milliseconds)")
      .def_readwrite(
          "store",
          &CommOptions::store,
          "Store for communication between processes")
      .def_readwrite(
          "hints",
          &CommOptions::hints,
          "Dictionary of string hints for backend-specific options");

  py::class_<BatchP2POptions>(
      m, "BatchP2POptions", "Options for batched P2P operations.")
      .def(py::init<>(), "Create default BatchP2POptions")
      .def_readwrite("hints", &BatchP2POptions::hints, "Hints dictionary")
      .def_readwrite("timeout", &BatchP2POptions::timeout, "Timeout");

  // Bind TorchWork class
  intrusive_ptr_class_<TorchWork>(
      m,
      "TorchWork",
      R"(
TorchWork allows you to track whether an asynchronous operation has completed.

When async_op=True, the operation is enqueued on a background stream and a
TorchWork object is returned. This work object must be waited on before
using the output tensor.

This is intended to make it easier to write efficient code that can overlap
communication with computation.

Example usage:

.. code-block:: python

  tensor = ...
  # run all_reduce on a background stream and return a TorchWork object
  work = torchcomms.all_reduce(tensor, ReduceOp.SUM, async_op=True)

  # Schedule some other work on the current stream
  a = b * 2

  # block the current stream until the all_reduce is complete
  work.wait()

  # safely use the tensor after the collective completes
  tensor.sum()

  # block CPU until stream is complete
  torch.accelerator.current_stream().synchronize()
      )")
      .def(
          "is_completed",
          &TorchWork::isCompleted,
          "Check if the work is completed",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait",
          &TorchWork::wait,
          R"(
Block the current stream until the work is completed.

See https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams for more details.
          )",
          py::call_guard<py::gil_scoped_release>());

  // Bind TorchCommWindow class
  py::class_<TorchCommWindow, std::shared_ptr<TorchCommWindow>>(
      m, "TorchCommWindow")
      .def(
          "get_size",
          &TorchCommWindow::get_size,
          R"(Get the size of the window)",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "put",
          &TorchCommWindow::put,
          R"(
Put allows you to put a tensor into the previously allocated remote window.

Args:
    tensor: the tensor to put
    dst_rank: the destination rank
    target_disp: the target displacement
    async_op: if this is true, the operation is asynced and will be enqueued on a background stream and a TorchWork object is returned.


Example usage:

.. code-block:: python

  tensor = ...
  # create a window
  window = torchcomms.create_window(window_size, cpu_buf)
  # put a tensor into the window
  work = window.put(tensor, dst_rank, target_disp, async_op=True)
  work.wait()

  # on the remote side, get the tensor from the window after waiting on the remote signal
  tensor = window.get_tensor(rank, tensor_sizes, tensor_dtype, offset)

  # safely use the tensor after the collective completes
  tensor.sum()

      )",
          py::arg("tensor"),
          py::arg("dst_rank"),
          py::arg("target_disp"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "signal",
          &TorchCommWindow::signal,
          R"(
Atomic signal to notify remote peer of a change in state.

Args:
    signal_disp: the displacement in signal bugger.
    signal_val: the signal value to set for signalling.
    dst_rank: the destination rank.
    async_op: if this is true, the operation is asynced.

      )",
          py::arg("peer_rank"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait_signal",
          &TorchCommWindow::waitSignal,
          "wait for a signal from remote peer",
          py::arg("peer_rank"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_tensor",
          &TorchCommWindow::getTensor,
          "get a tensor from remote window",
          py::arg("rank"),
          py::arg("sizes"),
          py::arg("dtype"),
          py::arg("offset"),
          py::call_guard<py::gil_scoped_release>());

  // Bind BatchSendRecv::P2POp class
  py::class_<BatchSendRecv::P2POp>(
      m, "P2POp", "Represents a peer to peer operation as part of a batch.")
      .def(
          py::init<BatchSendRecv::P2POp::OpType, const at::Tensor&, int>(),
          R"(
Create P2POp.

Args:
    type: the type of the operations i.e. send/recv
    tensor: the tensor to operate on
    peer: the rank of the peer
          )",
          py::arg("type"),
          py::arg("tensor"),
          py::arg("peer"),
          py::call_guard<py::gil_scoped_release>())
      .def_readwrite("type", &BatchSendRecv::P2POp::type, "Operation type")
      .def_readwrite("tensor", &BatchSendRecv::P2POp::tensor, "Tensor")
      .def_readwrite("peer", &BatchSendRecv::P2POp::peer, "Peer rank");

  // Bind P2POp::OpType enum
  py::enum_<BatchSendRecv::P2POp::OpType>(
      m, "P2POpType", "Type of a peer to peer operation.")
      .value("SEND", BatchSendRecv::P2POp::OpType::SEND)
      .value("RECV", BatchSendRecv::P2POp::OpType::RECV);

  // Bind BatchSendRecv class
  py::class_<BatchSendRecv>(
      m,
      "BatchSendRecv",
      R"(
BatchSendRecv allows you to run multiple send/recv operations concurrently
unlike the standard send/recv APIs which only allow you to have one inflight at
a time.
      )")
      .def(
          "send",
          &BatchSendRecv::send,
          R"(
Add send operation to batch. Must be paired with a corresponding
recv operation on a different rank.

Args:
    tensor: the tensor to send
    dst: the destination rank
          )",
          py::arg("tensor"),
          py::arg("dst"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recv",
          &BatchSendRecv::recv,
          R"(
Add recv operation to batch. Must be paired with a corresponding send operation
on a different rank.

Args:
    tensor: the tensor to receive into
    src: the source rank
          )",
          py::arg("tensor"),
          py::arg("src"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "issue",
          [](BatchSendRecv& self,
             bool async_op,
             const BatchP2POptions& options) {
            return self.issue(async_op, options);
          },
          "Issues the batched operations",
          py::arg("async_op"),
          py::arg("options") = BatchP2POptions(),
          py::call_guard<py::gil_scoped_release>())
      .def_readwrite(
          "ops",
          &BatchSendRecv::ops,
          "List of P2P operations",
          py::call_guard<py::gil_scoped_release>());

  m.def(
      "new_comm",
      [](const std::string& backend,
         const c10::Device& device,
         const std::string& name,
         std::optional<bool> abort_process_on_timeout_or_error,
         std::optional<std::chrono::milliseconds> timeout,
         std::optional<bool> high_priority_stream,
         std::optional<c10::intrusive_ptr<c10d::Store>> store,
         std::optional<std::unordered_map<std::string, std::string>> hints) {
        py::module_ torchcomms = py::module_::import("torchcomms");
        torchcomms.attr("_load_backend")(backend);

        {
          py::gil_scoped_release release{};

          CommOptions opts;
          if (abort_process_on_timeout_or_error) {
            opts.abort_process_on_timeout_or_error =
                *abort_process_on_timeout_or_error;
          }
          if (timeout) {
            opts.timeout = *timeout;
          }
          if (high_priority_stream) {
            opts.high_priority_stream = *high_priority_stream;
          }
          if (store) {
            opts.store = *store;
          }
          if (hints) {
            opts.hints = *hints;
          }

          return new_comm(backend, device, name, opts);
        }
      },
      R"(
Create a new communicator.

This requires all ranks that will be part of the commmunicator call this
function simultaneously.

Ranks and world size will be derived from environment variables set by launchers
such as torchrun (i.e. ``RANK``, ``WORLD_SIZE``).

Backends typically use a store to initialize which can either be provided or
automatically instantiated from environment variables such as ``MASTER_ADDR``
and ``MASTER_PORT``. Backends are not required to use the store to initialize if
more performant options are available.

Subcommunicators can be instantiated by using the ``split`` method.

Args:
  backend (str): The backend to use for the communicator.
  device (torch.device): The device to use for the communicator.
  name (str): The name of the communicator. This must be unique within the process.
  abort_process_on_timeout_or_error (bool): Whether to abort process on timeout or error.
  timeout (timedelta): Timeout for initialization.
  high_priority_stream (bool): Whether to use high priority stream.
  store (torch.distributed.Store): Store used to initialize the communicator between processes.
  hints (dict): Dictionary of string hints for backend-specific options.
      )",
      py::arg("backend"),
      py::arg("device"),
      py::arg("name"),
      py::arg("abort_process_on_timeout_or_error") = std::nullopt,
      py::arg("timeout") = std::nullopt,
      py::arg("high_priority_stream") = std::nullopt,
      py::arg("store") = nullptr,
      py::arg("hints") = std::nullopt);

  py::class_<TorchCommBackend, std::shared_ptr<TorchCommBackend>>(
      m,
      "TorchCommBackend",
      "Abstract class that all torchcomms Backends implement.");

  // Bind TorchComm class
  py::class_<TorchComm, std::shared_ptr<TorchComm>>(m, "TorchComm")
      .def(
          "finalize",
          &TorchComm::finalize,
          "Finalize and free all resources. This must be called prior to destruction.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_rank",
          &TorchComm::getRank,
          "Get the rank of this process",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_size",
          &TorchComm::getSize,
          "Get the world size",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_name",
          &TorchComm::getCommName,
          "Get the name of the communicator",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_options",
          &TorchComm::getOptions,
          "Get the communicator options",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_device",
          &TorchComm::getDevice,
          "Get the communicator device",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_backend",
          &TorchComm::getBackend,
          "Get communicator backend name",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "unsafe_get_backend",
          &TorchComm::unsafeGetBackend,
          R"(
Get communicator backend implementation.

WARNING: This is intended as an escape hatch for experimentation and
development. Direct backend access provides no backwards compatibility
guarantees. Users depending on unsafe_get_backend should expect their code to
break as interfaces change.
          )",
          py::call_guard<py::gil_scoped_release>())

      // Point-to-Point Operations
      .def(
          "send",
          [](TorchComm& self,
             const at::Tensor& tensor,
             int dst,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            SendOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.send(tensor, dst, async_op, opts);
          },
          R"(
Send tensor to destination rank.

This will not run concurrently with other operations (including send/recv) on
the same stream.

Args:
    tensor: the tensor to send
    dst: the destination rank
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("tensor"),
          py::arg("dst"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recv",
          [](TorchComm& self,
             at::Tensor& tensor,
             int src,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            RecvOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.recv(tensor, src, async_op, opts);
          },
          R"(
Receive tensor from source rank.

This will not run concurrently with other operations (including send/recv) on
the same stream.

Args:
    tensor: the tensor to receive into
    src: the source rank
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("tensor"),
          py::arg("src"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())

      // Collective Operations
      .def(
          "broadcast",
          [](TorchComm& self,
             at::Tensor& tensor,
             int root,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            BroadcastOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.broadcast(tensor, root, async_op, opts);
          },
          R"(
Broadcast tensor to all ranks in the communicator.

Args:
    tensor: the tensor to broadcast if root or receive into if not root
    root: the root rank
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("tensor"),
          py::arg("root"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_reduce",
          [](TorchComm& self,
             at::Tensor& tensor,
             ReduceOp op,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllReduceOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_reduce(tensor, op, async_op, opts);
          },
          R"(
Reduce a tensor across all ranks in the communicator.

Args:
    tensor: the tensor to all-reduce
    op: the reduction operation
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("tensor"),
          py::arg("op"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reduce",
          [](TorchComm& self,
             const at::Tensor& tensor,
             int root,
             ReduceOp op,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            ReduceOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.reduce(tensor, root, op, async_op, opts);
          },
          R"(
Reduce a tensor from all ranks to a single rank in the communicator.

Output will only be available on the root rank.

Args:
    tensor: the tensor to reduce
    root: the root rank
    op: the reduction operation
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("tensor"),
          py::arg("root"),
          py::arg("op"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_gather",
          [](TorchComm& self,
             const std::vector<at::Tensor>& tensor_list,
             const at::Tensor& tensor,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllGatherOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_gather(tensor_list, tensor, async_op, opts);
          },
          R"(
Gather a tensor from all ranks in the communicator.

Output will be available on all ranks.

Args:
    tensor_list: the list of tensors to gather into
    tensor: the input tensor to share
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("tensor_list"),
          py::arg("tensor"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_gather_v",
          [](TorchComm& self,
             const std::vector<at::Tensor>& tensor_list,
             const at::Tensor& tensor,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllGatherOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_gather_v(tensor_list, tensor, async_op, opts);
          },
          R"(
Gather a tensor from all ranks in the communicator, supporting variable tensor sizes per rank.

Output will be available on all ranks.

Args:
    tensor_list: the list of tensors to gather into; the list is the same on all ranks, but tensor sizes may differ between indices.
    tensor: the input tensor to share; size may differ per rank.
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("tensor_list"),
          py::arg("tensor"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_gather_single",
          [](TorchComm& self,
             at::Tensor& output,
             const at::Tensor& input,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllGatherSingleOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_gather_single(output, input, async_op, opts);
          },
          R"(
Gather a single tensor from all ranks in the communicator.

The output tensor must be of size (world_size * input.numel()).

Args:
    output: the output tensor to gather into
    input: the input tensor to share
    async_op: whether to perform the operation asynchronously
    hints: dictionary of string hints for backend-specific options
    timeout: timeout for the operation
          )",
          py::arg("output"),
          py::arg("input"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reduce_scatter",
          [](TorchComm& self,
             at::Tensor& output,
             const std::vector<at::Tensor>& input_list,
             ReduceOp op,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            ReduceScatterOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.reduce_scatter(output, input_list, op, async_op, opts);
          },
          R"(
Reduce, then scatter a list of tensors to all ranks.

Args:
    output: Output tensor.
    input_list: List of tensors to reduce and scatter.
    op: Reduction operation.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output"),
          py::arg("input_list"),
          py::arg("op"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reduce_scatter_v",
          [](TorchComm& self,
             at::Tensor& output,
             const std::vector<at::Tensor>& input_list,
             ReduceOp op,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            ReduceScatterOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.reduce_scatter_v(
                output, input_list, op, async_op, opts);
          },
          R"(
Reduce, then scatter a list of tensors to all ranks, supporting variable tensor sizes per rank.

Args:
    output: Output tensor on each rank; size may differ per rank.
    input_list: List of tensors to reduce and scatter; the list is the same on all ranks, but tensor sizes may differ between indices.
    op: Reduction operation.
    async_op: Whether to perform the operation asynchronously
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output"),
          py::arg("input_list"),
          py::arg("op"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reduce_scatter_single",
          [](TorchComm& self,
             at::Tensor& output,
             const at::Tensor& input,
             ReduceOp op,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            ReduceScatterSingleOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.reduce_scatter_single(
                output, input, op, async_op, opts);
          },
          R"(
Reduce, then scatter a single tensor to all ranks.

The input tensor must be of size (world_size * output.numel()).

Args:
    output: Output tensor.
    input: Input tensor to reduce and scatter.
    op: Reduction operation.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output"),
          py::arg("input"),
          py::arg("op"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_to_all_single",
          [](TorchComm& self,
             at::Tensor& output,
             const at::Tensor& input,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllToAllSingleOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_to_all_single(output, input, async_op, opts);
          },
          R"(
Split input tensor and then scatter the split list to all ranks.

Later the received tensors are concatenated and returned as a single
output tensor.

The input and output tensor sizes must a multiple of world_size.

Args:
    output: Output tensor.
    input: Input tensor to split and scatter.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output"),
          py::arg("input"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_to_all_v_single",
          [](TorchComm& self,
             at::Tensor& output,
             const at::Tensor& input,
             const std::vector<uint64_t>& output_split_sizes,
             const std::vector<uint64_t>& input_split_sizes,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllToAllvSingleOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_to_all_v_single(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                async_op,
                opts);
          },
          R"(
All-to-all single tensor operation with variable split sizes.

Args:
    output: Output tensor.
    input: Input tensor to split and scatter.
    output_split_sizes: List of output split sizes.
    input_split_sizes: List of input split sizes.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output"),
          py::arg("input"),
          py::arg("output_split_sizes"),
          py::arg("input_split_sizes"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_to_all",
          [](TorchComm& self,
             const std::vector<at::Tensor>& output_tensor_list,
             const std::vector<at::Tensor>& input_tensor_list,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllToAllOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_to_all(
                output_tensor_list, input_tensor_list, async_op, opts);
          },
          R"(
Scatter the split list to all ranks.

Args:
    output_tensor_list: Output tensor list.
    input_tensor_list: Input tensor list to scatter.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output_tensor_list"),
          py::arg("input_tensor_list"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "barrier",
          [](TorchComm& self,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            BarrierOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.barrier(async_op, opts);
          },
          R"(
Block until all ranks have reached this call.

Args:
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())

      // Scatter and Gather Operations
      .def(
          "scatter",
          [](TorchComm& self,
             at::Tensor& output_tensor,
             const std::vector<at::Tensor>& input_tensor_list,
             int root,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            ScatterOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.scatter(
                output_tensor, input_tensor_list, root, async_op, opts);
          },
          R"(
Scatter the split list to all ranks from the root.

Args:
    output_tensor: Output tensor.
    input_tensor_list: Input tensor list to scatter.
    root: The root rank.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output_tensor"),
          py::arg("input_tensor_list"),
          py::arg("root"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "gather",
          [](TorchComm& self,
             const std::vector<at::Tensor>& output_tensor_list,
             const at::Tensor& input_tensor,
             int root,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            GatherOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.gather(
                output_tensor_list, input_tensor, root, async_op, opts);
          },
          R"(
Gather the input tensor from all ranks to the root.

Args:
    output_tensor_list: Output tensor list. Will be empty on non-root ranks.
    input_tensor: Input tensor to gather.
    root: The root rank.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.
          )",
          py::arg("output_tensor_list"),
          py::arg("input_tensor"),
          py::arg("root"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())

      // window operations
      .def(
          "window_allocate",
          [](TorchComm& self,
             const size_t window_size,
             bool cpu_buf = false,
             const size_t signal_size = 256) {
            return self.window_allocate(window_size, cpu_buf, signal_size);
          },
          R"(
Allocate a shared Window with current TorchComm Communicator.

Args:
    window_size: The byte size of the window to be created.
    cpu_buf: buffer is allocated on the CPU, or on the same device as the assigned communicator.

Returns: The window object.
          )",
          py::arg("window_size"),
          py::arg("cpu_buf") = false,
          py::arg("signal_size") = 256,
          py::call_guard<py::gil_scoped_release>())

      // Communicator Management
      .def(
          "split",
          [](TorchComm& self,
             const std::vector<int>& ranks,
             const std::string& name,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            CommOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.split(ranks, name, opts);
          },
          R"(
Split communicator into a subgroup.

Args:
    ranks: List of ranks to include in the new subgroup. If the list is empty,
        None will be returned. If the list is non-empty but does not include
        the current rank, an exception will be thrown.
    name: Name for the new communicator.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.

Returns: A new communicator for the subgroup, or None if the ranks list is empty.

Raises: RuntimeError if the ranks list is non-empty and the current rank is not included.
          )",
          py::arg("ranks"),
          py::arg("name"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())

      // Batch Operations
      .def(
          "batch_op_create",
          &TorchComm::batch_op_create,
          "Create a batch operation object for batched P2P operations.",
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "mem_allocator",
          &TorchComm::getMemAllocator,
          "Get the communication-specific memory allocator");

  intrusive_ptr_class_<BackendWrapper, c10d::Backend>(m, "_BackendWrapper")
      .def(
          py::init<std::shared_ptr<TorchComm>>(),
          "Create BackendWrapper around a TorchCommBackend",
          py::arg("comm"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_comm",
          &BackendWrapper::getComm,
          "Get the underlying TorchComm instance",
          py::call_guard<py::gil_scoped_release>());
  intrusive_ptr_class_<WorkWrapper, c10d::Work>(m, "WorkWrapper");

  m.def(
      "_get_store",
      [](const std::string& backend_name,
         const std::string& name,
         std::chrono::milliseconds timeout) {
        return StoreManager::get().getStore(backend_name, name, timeout);
      },
      R"(
      Return a new store object that's unique to the given backend and
      communicator name.
      )",
      py::arg("backend_name"),
      py::arg("name"),
      py::arg("timeout") = std::chrono::milliseconds(60000),
      py::call_guard<py::gil_scoped_release>());
}
