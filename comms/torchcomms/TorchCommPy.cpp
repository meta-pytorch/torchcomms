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
#include "comms/torchcomms/hooks/FlightRecorderPy.h"

namespace py = pybind11;
using namespace torch::comms;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

// helper to create a pybind11 class with a custom metaclass for torch.compile
// support if needed.
template <typename... Types, typename... Extra>
auto py_opaque_class(py::module_& m, const char* name, Extra&&... extra) {
  py::object opaque_metaclass = py::none();
  try {
    py::module_ sys = py::module_::import("sys");
    py::dict modules = sys.attr("modules");
    if (modules.contains("torchcomms._opaque_meta")) {
      opaque_metaclass =
          modules["torchcomms._opaque_meta"].attr("OpaqueBaseMeta");
    }
  } catch (...) {
    opaque_metaclass = py::none();
  }

  if (opaque_metaclass.is_none()) {
    return py::class_<Types...>(m, name, std::forward<Extra>(extra)...);
  } else {
    return py::class_<Types...>(
        m,
        name,
        std::forward<Extra>(extra)...,
        py::metaclass(opaque_metaclass));
  }
}

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

  // Bind ReduceOp class (with custom metaclass for torch.compile support)
  auto reduce_op_class = py_opaque_class<ReduceOp>(m, "ReduceOp");
  reduce_op_class
      .def(
          py::init<ReduceOp::RedOpType>(),
          "Create default ReduceOp",
          py::arg("opType"),
          py::call_guard<py::gil_scoped_release>())
      .def("__copy__", [](const ReduceOp& self) { return self; })
      .def(
          "__deepcopy__",
          [](const ReduceOp& self, const py::dict& memo) {
            auto self_obj = py::cast(self);
            auto self_id =
                py::cast(reinterpret_cast<uintptr_t>(self_obj.ptr()));
            if (memo.contains(self_id)) {
              return memo[self_id].cast<ReduceOp>();
            }
            auto copy = self;
            memo[self_id] = py::cast(copy);
            return copy;
          })
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

  py::enum_<TorchCommWinAccessType>(
      m, "TorchCommWinAccessType", "Window attribute.")
      .value(
          "WIN_ACCESS_TYPE_UNIFIED",
          TorchCommWinAccessType::WIN_ACCESS_TYPE_UNIFIED)
      .value(
          "WIN_ACCESS_TYPE_SEPARATE",
          TorchCommWinAccessType::WIN_ACCESS_TYPE_SEPARATE);

  // Bind OpName enum for hooks
  py::enum_<OpName>(m, "OpName", "Collective operation name for hooks.")
      .value("send", OpName::send)
      .value("recv", OpName::recv)
      .value("broadcast", OpName::broadcast)
      .value("all_reduce", OpName::all_reduce)
      .value("reduce", OpName::reduce)
      .value("all_gather", OpName::all_gather)
      .value("all_gather_v", OpName::all_gather_v)
      .value("all_gather_single", OpName::all_gather_single)
      .value("reduce_scatter", OpName::reduce_scatter)
      .value("reduce_scatter_v", OpName::reduce_scatter_v)
      .value("reduce_scatter_single", OpName::reduce_scatter_single)
      .value("all_to_all_single", OpName::all_to_all_single)
      .value("all_to_all_v_single", OpName::all_to_all_v_single)
      .value("all_to_all", OpName::all_to_all)
      .value("barrier", OpName::barrier)
      .value("scatter", OpName::scatter)
      .value("gather", OpName::gather)
      .value("split", OpName::split)
      .value("new_window", OpName::new_window);

  // Bind RemovableHandle class for hook management
  py::class_<RemovableHandle, std::unique_ptr<RemovableHandle>>(
      m,
      "RemovableHandle",
      R"(
Handle for removing a registered hook.

Call remove() to unregister the hook. The hook will be automatically
unregistered when the handle is garbage collected if remove() was not called.
      )")
      .def(
          "remove",
          &RemovableHandle::remove,
          "Unregister the hook associated with this handle.");

  // Bind PreHookArgs struct for pre-hook callbacks
  py::class_<TorchComm::PreHookArgs>(
      m,
      "PreHookArgs",
      R"(
Arguments passed to pre-hook callbacks.

Pre-hooks are called before each collective operation starts.
      )")
      .def_readonly("name", &TorchComm::PreHookArgs::name, "Operation name")
      .def_readonly(
          "async_op",
          &TorchComm::PreHookArgs::async_op,
          "Whether the operation is asynchronous")
      .def_readonly(
          "root", &TorchComm::PreHookArgs::root, "Root rank for rooted ops")
      .def_readonly(
          "op_id",
          &TorchComm::PreHookArgs::op_id,
          "Unique operation ID for correlation with post-hook");

  // Bind PostHookArgs struct for post-hook callbacks
  py::class_<TorchComm::PostHookArgs>(
      m,
      "PostHookArgs",
      R"(
Arguments passed to post-hook callbacks.

Post-hooks are called after each collective operation completes.
      )")
      .def_readonly("name", &TorchComm::PostHookArgs::name, "Operation name")
      .def_readonly(
          "op_id",
          &TorchComm::PostHookArgs::op_id,
          "Unique operation ID for correlation with pre-hook");

  py::class_<TorchCommWindowAttr, std::shared_ptr<TorchCommWindowAttr>>(
      m, "TorchCommWindowAttr", "Window attributes.")
      .def(py::init<>(), "Create default TorchCommWindowAttr")
      .def_readwrite(
          "access_type",
          &TorchCommWindowAttr::accessType,
          "Window access type");

  // Bind TorchCommWindow class
  py_opaque_class<TorchCommWindow, std::shared_ptr<TorchCommWindow>>(
      m, "TorchCommWindow")
      .def(
          "__copy__",
          [](const std::shared_ptr<TorchCommWindow>& self) { return self; })
      .def(
          "__deepcopy__",
          [](const std::shared_ptr<TorchCommWindow>& self,
             const py::dict& memo) {
            auto self_obj = py::cast(self);
            auto self_id =
                py::cast(reinterpret_cast<uintptr_t>(self_obj.ptr()));
            if (memo.contains(self_id)) {
              return memo[self_id].cast<std::shared_ptr<TorchCommWindow>>();
            }

            auto new_window = self->clone();

            auto original_tensor = self->get_tensor();
            auto cloned_tensor = new_window->get_tensor();
            if (original_tensor.has_value() && cloned_tensor.has_value()) {
              auto original_tensor_obj = py::cast(original_tensor.value());
              memo[py::cast(
                  reinterpret_cast<uintptr_t>(original_tensor_obj.ptr()))] =
                  py::cast(cloned_tensor.value());
            }

            memo[self_id] = py::cast(new_window);
            return new_window;
          })
      .def(
          "tensor_register",
          [](TorchCommWindow& self, const at::Tensor& tensor) {
            self.tensor_register(tensor);
          },
          R"(
Register a tensor buffer with the window for RMA operations.

Args:
    tensor (torch.Tensor): Contiguous tensor to register. Must be allocated
        within a memory pool created via ``torchcomms.get_mem_allocator()``.

Raises:
    RuntimeError: If tensor is not contiguous or a buffer is already registered.

Example:

.. code-block:: python

    import torchcomms

    allocator = torchcomms.get_mem_allocator(comm.get_backend())
    pool = torch.cuda.MemPool(allocator)
    with torch.cuda.use_mem_pool(pool):
        buffer = torch.ones([size], dtype=dtype, device=device)

    window = comm.new_window()
    window.tensor_register(buffer)

      )",
          py::arg("tensor"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "tensor_deregister",
          &TorchCommWindow::tensor_deregister,
          R"(
Deregister the tensor buffer and free window resources.

This is a collective operation with internal barriers ensuring all ranks
finish using the window before deregistration.

Raises:
    RuntimeError: If no tensor is currently registered.

Note:
    Any tensors from ``map_remote_tensor`` become invalid after this call.

      )",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_size",
          &TorchCommWindow::get_size,
          R"(
Get the size of the registered window buffer in bytes.

Returns:
    int: Size in bytes, or 0 if no tensor is registered.

      )",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_tensor",
          [](const TorchCommWindow& self) -> std::optional<at::Tensor> {
            return self.get_tensor();
          },
          R"(
Get the registered tensor buffer, if any.

Returns:
    Optional[torch.Tensor]: The registered tensor, or None if no tensor is registered.

      )")
      .def_property_readonly(
          "dtype",
          [](TorchCommWindow& self) {
            return py::reinterpret_steal<py::object>(
                THPDtype_New(self.getDtype(), "torch"));
          },
          R"(The dtype of the registered buffer tensor.

Returns:
    torch.dtype: The dtype of the registered buffer, e.g. torch.float32.

Note:
    This is primarily used by torch.compile's meta kernel to determine
    the output tensor dtype for map_remote_tensor() operations.
          )")
      .def_property_readonly(
          "shape",
          [](TorchCommWindow& self) { return self.getShape(); },
          R"(The shape of the registered buffer tensor.

Returns:
    list[int]: The shape of the registered buffer as a list of dimensions.

Note:
    This is primarily used by torch.compile's meta kernel to determine
    the output tensor shape for map_remote_tensor() operations.
          )")
      .def_property_readonly(
          "device",
          [](TorchCommWindow& self) { return self.getDevice(); },
          R"(The device of the registered buffer tensor.

Returns:
    torch.device: The device of the registered buffer.

Note:
    This is primarily used by torch.compile's meta kernel to determine
    the output tensor device for map_remote_tensor() operations.
          )")
      .def(
          "put",
          [](TorchCommWindow& self,
             const at::Tensor& tensor,
             int dst_rank,
             size_t target_offset_nelems,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            PutOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.put(
                tensor, dst_rank, target_offset_nelems, async_op, opts);
          },

          R"(
Write data to a remote rank's window buffer (one-sided put).

The receiver does not participate; use ``signal``/``wait_signal`` to
synchronize before accessing the data.

Args:
    tensor (torch.Tensor): Data to write. Must match the window buffer dtype.
    dst_rank (int): Destination rank.
    target_offset_nelems (int): Offset in destination buffer (in elements).
    async_op (bool): if this is true, the operation is asynced and will be enqueued on a background stream and a TorchWork object is returned.
    hints (Dict[str, str], optional): Backend-specific hints.
    timeout (timedelta, optional): Operation timeout.

Returns:
    TorchWork: Work object for synchronization.

Raises:
    RuntimeError: If data exceeds window buffer size.

Example:

.. code-block:: python

    # Sender (rank 0)
    work = window.put(data, dst_rank=1, target_offset_nelems=0, async_op=True)
    work.wait()
    window.signal(peer_rank=1, async_op=False)

    # Receiver (rank 1)
    window.wait_signal(peer_rank=0, async_op=False)
    received = buffer[:data.numel()]

      )",
          py::arg("tensor"),
          py::arg("dst_rank"),
          py::arg("target_offset_nelems"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "signal",
          [](TorchCommWindow& self,
             int peer_rank,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            SignalOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.signal(peer_rank, async_op, opts);
          },
          R"(
Atomically send a signal to notify a remote peer that data is ready.

Used with ``wait_signal`` to synchronize.

Args:
    peer_rank (int): Destination rank to signal.
    async_op (bool): If True, returns TorchWork for async completion.
    hints (Dict[str, str], optional): Backend-specific hints.
    timeout (timedelta, optional): Operation timeout.

Returns:
    TorchWork: Work object for synchronization.

Example:

.. code-block:: python

    window.put(data, dst_rank=1, ...)
    window.signal(peer_rank=1, async_op=False)

      )",
          py::arg("peer_rank"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait_signal",
          [](TorchCommWindow& self,
             int peer_rank,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            WaitSignalOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.wait_signal(peer_rank, async_op, opts);
          },
          R"(
Wait for a signal from a remote peer.

Blocks until the peer calls ``signal``. Used to synchronize before
accessing data written by ``put``.

Args:
    peer_rank (int): Rank to wait for signal from.
    async_op (bool): If True, returns TorchWork for async completion.
    hints (Dict[str, str], optional): Backend-specific hints.
    timeout (timedelta, optional): Operation timeout.

Returns:
    TorchWork: Work object for synchronization.

Example:

.. code-block:: python

    window.wait_signal(peer_rank=0, async_op=False)
    received = buffer[:size]

      )",
          py::arg("peer_rank"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_attr",
          &TorchCommWindow::get_attr,
          R"(
Get window attributes for a peer rank.

Args:
    peer_rank (int): Rank to query attributes for.

Returns:
    TorchCommWindowAttr: Contains ``access_type``:
        - ``WIN_ACCESS_TYPE_UNIFIED``: Direct memory access (NVLink)
        - ``WIN_ACCESS_TYPE_SEPARATE``: Network-based access

Example:

.. code-block:: python

    attr = window.get_attr(peer_rank=1)
    if attr.access_type == torchcomms.TorchCommlWinAccessType.WIN_ACCESS_TYPE_UNIFIED:
        remote = window.map_remote_tensor(rank=1)

      )",
          py::arg("peer_rank"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "map_remote_tensor",
          &TorchCommWindow::map_remote_tensor,
          R"(
Map a remote rank's window buffer as a local tensor.

Only supported when ``get_attr(rank)`` returns ``WIN_ACCESS_TYPE_UNIFIED``
(direct memory access via NVLink). Network-based access (``WIN_ACCESS_TYPE_SEPARATE``)
is not yet supported.

Args:
    rank (int): Remote rank whose buffer to map.

Returns:
    torch.Tensor: View of the remote rank's window buffer.

Raises:
    RuntimeError: If direct memory access is not available.

Note:
    Tensor becomes invalid after ``tensor_deregister()``.

Example:

.. code-block:: python

    attr = window.get_attr(peer_rank=1)
    if attr.access_type == torchcomms.TorchCommlWinAccessType.WIN_ACCESS_TYPE_UNIFIED:
        remote = window.map_remote_tensor(rank=1)

      )",
          py::arg("rank"),
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
  py_opaque_class<BatchSendRecv, std::shared_ptr<BatchSendRecv>>(
      m, "BatchSendRecv", R"(
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
  py_opaque_class<TorchComm, std::shared_ptr<TorchComm>>(m, "TorchComm")
      // NOTE: copy/deepcopy return the same object (not a clone).
      // Actually cloning the underlying communicator would be extremely
      // expensive (requires collective operations to create new comm groups).
      .def(
          "__copy__",
          [](const std::shared_ptr<TorchComm>& self) { return self; })
      .def(
          "__deepcopy__",
          [](const std::shared_ptr<TorchComm>& self, const py::dict& memo) {
            auto self_obj = py::cast(self);
            auto self_id =
                py::cast(reinterpret_cast<uintptr_t>(self_obj.ptr()));
            if (memo.contains(self_id)) {
              return memo[self_id].cast<std::shared_ptr<TorchComm>>();
            }
            memo[self_id] = py::cast(self);
            return self;
          })
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
          "get_backend_impl",
          &TorchComm::getBackendImpl,
          R"(
Get communicator backend implementation.

WARNING: This is intended as an escape hatch for experimentation and
development. Direct backend access provides no backwards compatibility
guarantees. Users depending on get_backend_impl should expect their code to
break as interfaces change.
          )",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "unsafe_get_backend",
          [](TorchComm& self) {
            PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "unsafe_get_backend() is deprecated, "
                "use get_backend_impl() instead.",
                1);
            return self.getBackendImpl();
          },
          R"(
Deprecated: Use get_backend_impl() instead.
          )")

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
             const ReduceOp& op,
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
             const ReduceOp& op,
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
             const ReduceOp& op,
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
             const ReduceOp& op,
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
             const ReduceOp& op,
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
          "new_window",
          [](TorchComm& self, const std::optional<at::Tensor>& tensor) {
            return self.new_window(tensor);
          },
          R"(
Create a new window object for Remote Memory Access (RMA) operations.

Windows enable one-sided communication where data can be written directly
to a remote rank's buffer without receiver-side participation.

Args:
    tensor (torch.Tensor, optional): Contiguous tensor to register with the
        window. Must be allocated within a memory pool created via
        ``torchcomms.get_mem_allocator()``. If provided, the tensor will be
        registered immediately during window creation. If not provided, use
        ``tensor_register()`` later.

Raises:
    RuntimeError: If tensor is provided and a buffer is already registered
        (double registration is not allowed).

Returns:
    TorchCommWindow: Window object, registered if tensor was provided.

Note:
    Requires ``ncclx`` backend.

Example:

.. code-block:: python

    import torchcomms

    allocator = torchcomms.get_mem_allocator(comm.get_backend())
    pool = torch.cuda.MemPool(allocator)
    with torch.cuda.use_mem_pool(pool):
        buffer = torch.ones([size], dtype=dtype, device=device)

    # Option 1: Create window with tensor registration in one step
    window = comm.new_window(buffer)

    # Option 2: Create window and register buffer separately
    window = comm.new_window()
    window.tensor_register(buffer)

    # Sender
    window.put(data, dst_rank=1, target_offset_nelems=0, async_op=False)
    window.signal(peer_rank=1, async_op=False)

    # Receiver
    window.wait_signal(peer_rank=0, async_op=False)
    received = buffer[:size]

    # Cleanup
    window.tensor_deregister()

      )",
          py::arg("tensor") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())

      // Persistent AllGather operations
      .def(
          "all_gather_p_init",
          [](TorchComm& self,
             at::Tensor& output,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllGatherPInitOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_gather_p_init(output, opts);
          },
          R"(
Initialize a persistent AllGather operation.

This is a SM free collective operation where the memory is pre-registered and uses 
Copy Engine or DMA to move data from one rank to the other.

Args:
    output: Pre-allocated output tensor of size (world_size * input_size).
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.

Returns:
    An opaque handle to use with all_gather_p_exec and all_gather_p_free.

Note:
    Requires ``rcclx`` backend.

Example:

.. code-block:: python

    # Initialize once
    handle = comm.all_gather_p_init(output_tensor)

    # Execute many times
    for input_tensor in inputs:
        work = comm.all_gather_p_exec(handle, input_tensor, async_op=True)
        work.wait()

    # Free when done
    comm.all_gather_p_free(handle)

          )",
          py::arg("output"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_gather_p_exec",
          [](TorchComm& self,
             TorchCommBackend::AllGatherPHandle handle,
             const at::Tensor& input,
             bool async_op,
             std::optional<std::unordered_map<std::string, std::string>> hints,
             std::optional<std::chrono::milliseconds> timeout) {
            AllGatherPExecOptions opts;
            if (hints) {
              opts.hints = *hints;
            }
            if (timeout) {
              opts.timeout = *timeout;
            }
            return self.all_gather_p_exec(handle, input, async_op, opts);
          },
          R"(
Execute a persistent AllGather operation.

Args:
    handle: Handle returned by all_gather_p_init.
    input: Input tensor to gather from all ranks.
    async_op: Whether to perform the operation asynchronously.
    hints: Dictionary of string hints for backend-specific options.
    timeout: Timeout for the operation.

Returns:
    TorchWork: Work object for synchronization.

          )",
          py::arg("handle"),
          py::arg("input"),
          py::arg("async_op"),
          py::arg("hints") = std::nullopt,
          py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "all_gather_p_free",
          [](TorchComm& self, TorchCommBackend::AllGatherPHandle handle) {
            self.all_gather_p_free(handle);
          },
          R"(
Free a persistent AllGather handle.

Args:
    handle: Handle returned by all_gather_p_init.

          )",
          py::arg("handle"),
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
      // NOTE: This property is kept temporarily to avoid breaking upstream
      // callers. The allocator is actually a global static per backend
      // (accessed via get_mem_allocator(backend)), not tied to the comm
      // instance lifetime. Future code should use the global function directly.
      .def_property_readonly(
          "mem_allocator",
          [](TorchComm& self) { return get_mem_allocator(self.getBackend()); },
          "Get the communication-specific memory allocator")

      // Hook registration methods
      .def(
          "register_pre_hook",
          [](TorchComm& self, const py::function& callback) {
            auto hook = [callback](TorchComm::PreHookArgs args) {
              py::gil_scoped_acquire acquire;
              callback(args);
            };
            return self.registerPreHook(std::move(hook));
          },
          R"doc(
Register a pre-hook callback that is called before each collective operation.

The callback receives a PreHookArgs object containing operation metadata.

Args:
    callback: A callable that takes a PreHookArgs argument.

Returns:
    RemovableHandle: A handle that can be used to unregister the hook.

Example::

    def my_pre_hook(args):
        print(f"Starting {args.name}")
    handle = comm.register_pre_hook(my_pre_hook)
    # ... run operations ...
    handle.remove()  # Unregister when done

Note:
    Hooks are not thread-safe and must not be modified while a collective
    operation is in progress.
          )doc",
          py::arg("callback"))
      .def(
          "register_post_hook",
          [](TorchComm& self, const py::function& callback) {
            auto hook = [callback](const TorchComm::PostHookArgs& args) {
              py::gil_scoped_acquire acquire;
              callback(args);
            };
            return self.registerPostHook(std::move(hook));
          },
          R"doc(
Register a post-hook callback that is called after each collective operation.

The callback receives a PostHookArgs object containing operation metadata.
The op_id can be used to correlate with the corresponding pre-hook call.

Args:
    callback: A callable that takes a PostHookArgs argument.

Returns:
    RemovableHandle: A handle that can be used to unregister the hook.

Example::

    def my_post_hook(args):
        print(f"Completed {args.name}")
    handle = comm.register_post_hook(my_post_hook)
    # ... run operations ...
    handle.remove()  # Unregister when done

Note:
    Hooks are not thread-safe and must not be modified while a collective
    operation is in progress.
          )doc",
          py::arg("callback"))
      .def(
          "register_abort_hook",
          [](TorchComm& self, const py::function& callback) {
            auto hook = [callback]() {
              py::gil_scoped_acquire acquire;
              callback();
            };
            return self.registerAbortHook(std::move(hook));
          },
          R"doc(
Register an abort hook callback that is called before process abort.

This hook is called when a collective operation times out or fails and the
process is about to abort. Use this to capture debug information.

Args:
    callback: A callable with no arguments.

Returns:
    RemovableHandle: A handle that can be used to unregister the hook.

Example::

    def my_abort_hook():
        print("About to abort, saving debug info...")
        save_debug_state()
    handle = comm.register_abort_hook(my_abort_hook)

Note:
    Hooks are not thread-safe and must not be modified while a collective
    operation is in progress.
          )doc",
          py::arg("callback"));

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
          py::call_guard<py::gil_scoped_release>())
      .def("name", &BackendWrapper::getBackendName)
      .def_property_readonly(
          "options",
          &BackendWrapper::getOptions,
          R"(Return the options used to create the torchComm under the hood.)",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_verify_work_timeout",
          &BackendWrapper::verifyWorkTimeoutForTest,
          R"(
Verify that a work object has the expected timeout.
Used for testing timeout propagation.

Args:
    work: The work object to verify.
    timeout: The expected timeout.

Returns:
    bool: True if the work object has the expected timeout, False otherwise.
          )",
          py::arg("work"),
          py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_default_timeout",
          &BackendWrapper::setTimeout,
          R"(
Set the default timeout for this backend.

Args:
    timeout: The timeout value to set.
          )",
          py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>());
  intrusive_ptr_class_<WorkWrapper, c10d::Work>(m, "WorkWrapper");
  // Register the backend Options
  intrusive_ptr_class_<BackendWrapper::Options, c10d::Backend::Options>(
      m, "_BackendWrapperOptions");

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

  m.def(
      "get_mem_allocator",
      [](const std::string& backend) { return get_mem_allocator(backend); },
      R"(
      Get the global memory allocator for the specified backend.

      This allocator is static per backend and not tied to any specific
      communicator instance. Memory allocated with this allocator can be
      shared across multiple communicators of the same backend.

      Args:
          backend: The backend name (e.g., "nccl", "ncclx")

      Returns:
          A c10::Allocator object for the specified backend.
      )",
      py::arg("backend"),
      py::call_guard<py::gil_scoped_release>());

  // Add Flight Recorder submodule
  auto hooks_submodule = m.def_submodule("hooks", "TorchComm module for hooks");
  torch::comms::fr::initFlightRecorderPyBindings(hooks_submodule);
}
