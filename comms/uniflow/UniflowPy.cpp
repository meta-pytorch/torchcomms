// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "comms/uniflow/Connection.h"
#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/Uniflow.h"

namespace py = pybind11;

namespace uniflow {
namespace {

// Type-erased Result for Python: holds either a py::object value or an Err.
class PyResult {
 public:
  explicit PyResult(py::object val) : val_(std::move(val)) {}
  explicit PyResult(Err err) : err_(std::move(err)) {}

  bool hasValue() const {
    return !err_.has_value();
  }
  bool hasError() const {
    return err_.has_value();
  }

  py::object value() const {
    if (err_) {
      throw std::runtime_error("Result contains error: " + err_->toString());
    }
    return val_;
  }

  const Err& error() const {
    if (!err_) {
      throw std::runtime_error("Result contains a value, not an error");
    }
    return *err_;
  }

 private:
  py::object val_;
  std::optional<Err> err_;
};

// Convert Result<T> → PyResult. Must be called with GIL held.
template <typename T>
PyResult toResult(Result<T> result) {
  if (result) {
    return PyResult(py::cast(std::move(result).value()));
  }
  return PyResult(std::move(result).error());
}

// Overload for Result<void> (Status) — no value() method.
inline PyResult toResult(Status status) {
  if (status) {
    return PyResult(py::none());
  }
  return PyResult(status.error());
}

// Overload for Result<vector<uint8_t>> — convert to py::bytes, not list.
inline PyResult toResult(Result<std::vector<uint8_t>> result) {
  if (result) {
    auto& v = result.value();
    return PyResult(
        py::bytes(reinterpret_cast<const char*>(v.data()), v.size()));
  }
  return PyResult(std::move(result).error());
}

// Type-erased future: virtual interface for wait/get operations.
class FutureBase {
 public:
  virtual ~FutureBase() = default;
  virtual bool done() = 0;
  virtual bool wait_for(int timeoutMs) = 0;
  virtual PyResult get() = 0;
};

template <typename T>
class FutureImpl : public FutureBase {
 public:
  explicit FutureImpl(std::future<Result<T>> fut) : fut_(std::move(fut)) {}

  bool done() override {
    return fut_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

  bool wait_for(int timeoutMs) override {
    py::gil_scoped_release release;
    return fut_.wait_for(std::chrono::milliseconds(timeoutMs)) ==
        std::future_status::ready;
  }

  PyResult get() override {
    if (cached_) {
      return *cached_;
    }
    py::gil_scoped_release release;
    auto result = fut_.get();
    py::gil_scoped_acquire acquire;
    cached_ = toResult(result);
    return *cached_;
  }

 private:
  std::future<Result<T>> fut_;
  std::optional<PyResult> cached_;
};

// Single Python-visible future class.
class UniflowFuture {
 public:
  template <typename T>
  explicit UniflowFuture(std::future<Result<T>>&& fut)
      : impl_(std::make_shared<FutureImpl<T>>(std::move(fut))) {}

  bool done() {
    return impl_->done();
  }
  PyResult get() {
    return impl_->get();
  }
  bool wait_for(int timeoutMs) {
    return impl_->wait_for(timeoutMs);
  }

 private:
  std::shared_ptr<FutureBase> impl_;
};

/// Convert py::bytes to std::span<const uint8_t> (zero-copy).
std::span<const uint8_t> bytesToSpan(std::string_view sv) {
  return {reinterpret_cast<const uint8_t*>(sv.data()), sv.size()};
}

} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for UniFlow";

  // ---------------------------------------------------------------------------
  // Enums
  // ---------------------------------------------------------------------------

  py::enum_<ErrCode>(m, "ErrCode", "UniFlow error codes.")
      .value("NotImplemented", ErrCode::NotImplemented)
      .value("DriverError", ErrCode::DriverError)
      .value("TopologyDisconnect", ErrCode::TopologyDisconnect)
      .value("InvalidArgument", ErrCode::InvalidArgument)
      .value("NotConnected", ErrCode::NotConnected)
      .value("TransportError", ErrCode::TransportError)
      .value("ConnectionFailed", ErrCode::ConnectionFailed)
      .value("MemoryRegistrationFailed", ErrCode::MemoryRegistrationFailed)
      .value("Timeout", ErrCode::Timeout)
      .value("ResourceExhausted", ErrCode::ResourceExhausted);

  py::enum_<MemoryType>(m, "MemoryType", "Type of memory segment.")
      .value("DRAM", MemoryType::DRAM, "Host memory (CPU RAM)")
      .value("VRAM", MemoryType::VRAM, "GPU memory (HBM/GDDR)")
      .value("NVME", MemoryType::NVME, "NVMe storage");

  py::enum_<TransportType>(m, "TransportType", "Transport backend type.")
      .value("NVLink", TransportType::NVLink, "NVLink for intra-node or MNNVL")
      .value("RDMA", TransportType::RDMA, "InfiniBand or RoCE RDMA")
      .value("TCP", TransportType::TCP, "TCP/IP fallback");

  // ---------------------------------------------------------------------------
  // Err
  // ---------------------------------------------------------------------------

  py::class_<Err>(m, "Err", "UniFlow error with code and message.")
      .def_property_readonly("code", &Err::code, "Error code.")
      .def_property_readonly("message", &Err::message, "Error message.")
      .def("__str__", &Err::toString)
      .def("__repr__", &Err::toString);

  // ---------------------------------------------------------------------------
  // Result
  // ---------------------------------------------------------------------------

  py::class_<PyResult>(m, "Result", "Holds either a value or an Err.")
      .def("has_value", &PyResult::hasValue, "True if the result has a value.")
      .def("has_error", &PyResult::hasError, "True if the result has an error.")
      .def("__bool__", &PyResult::hasValue)
      .def("value", &PyResult::value, "Get the value, or raise if error.")
      .def("error", &PyResult::error, "Get the Err, or raise if value.");

  // ---------------------------------------------------------------------------
  // Segment
  // ---------------------------------------------------------------------------

  py::class_<Segment>(m, "Segment", "A contiguous memory region for transfer.")
      .def(
          py::init([](uintptr_t ptr,
                      size_t length,
                      MemoryType memType,
                      int deviceId) {
            return Segment(
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<void*>(ptr),
                length,
                memType,
                deviceId);
          }),
          "Create a Segment from a raw pointer.",
          py::arg("ptr"),
          py::arg("length"),
          py::arg("mem_type") = MemoryType::DRAM,
          py::arg("device_id") = -1)
      .def_property_readonly(
          "data_ptr",
          [](const Segment& s) {
            return reinterpret_cast<uintptr_t>(s.data());
          },
          "Raw data pointer as integer.")
      .def_property_readonly(
          "length",
          [](const Segment& s) { return s.len(); },
          "Length in bytes.")
      .def_property_readonly(
          "mem_type",
          [](const Segment& s) { return s.memType(); },
          "Memory type.")
      .def_property_readonly(
          "device_id",
          [](const Segment& s) { return s.deviceId(); },
          "Device ID.");

  // ---------------------------------------------------------------------------
  // RegisteredSegment
  // ---------------------------------------------------------------------------

  py::class_<RegisteredSegment>(
      m, "RegisteredSegment", "A locally registered memory segment.")
      .def(
          "export_id",
          [](RegisteredSegment& s) { return toResult(s.exportId()); },
          "Export registration for sharing with remote peers.")
      .def(
          "span",
          [](RegisteredSegment& s, size_t offset, size_t length) {
            return s.span(offset, length);
          },
          py::keep_alive<0, 1>(),
          py::arg("offset"),
          py::arg("length"),
          "Create a sub-span.")
      .def_property_readonly(
          "length",
          [](const RegisteredSegment& s) { return s.len(); },
          "Length in bytes.");

  py::class_<RegisteredSegment::Span>(
      m, "RegisteredSegmentSpan", "A view into a RegisteredSegment.")
      .def_property_readonly(
          "size",
          [](const RegisteredSegment::Span& s) { return s.size(); },
          "Size in bytes.")
      .def_property_readonly(
          "mem_type",
          [](const RegisteredSegment::Span& s) { return s.memType(); },
          "Memory type.")
      .def_property_readonly(
          "device_id",
          [](const RegisteredSegment::Span& s) { return s.deviceId(); },
          "Device ID.");

  // ---------------------------------------------------------------------------
  // RemoteRegisteredSegment
  // ---------------------------------------------------------------------------

  py::class_<RemoteRegisteredSegment>(
      m,
      "RemoteRegisteredSegment",
      "A remotely registered memory segment imported for put/get.")
      .def(
          "span",
          [](RemoteRegisteredSegment& s, size_t offset, size_t length) {
            return s.span(offset, length);
          },
          py::keep_alive<0, 1>(),
          py::arg("offset"),
          py::arg("length"),
          "Create a sub-span.")
      .def_property_readonly(
          "length",
          [](const RemoteRegisteredSegment& s) { return s.len(); },
          "Length in bytes.");

  py::class_<RemoteRegisteredSegment::Span>(
      m,
      "RemoteRegisteredSegmentSpan",
      "A view into a RemoteRegisteredSegment.")
      .def_property_readonly(
          "size",
          [](const RemoteRegisteredSegment::Span& s) { return s.size(); },
          "Size in bytes.")
      .def_property_readonly(
          "mem_type",
          [](const RemoteRegisteredSegment::Span& s) { return s.memType(); },
          "Memory type.")
      .def_property_readonly(
          "device_id",
          [](const RemoteRegisteredSegment::Span& s) { return s.deviceId(); },
          "Device ID.");

  // ---------------------------------------------------------------------------
  // TransferRequest
  // ---------------------------------------------------------------------------

  py::class_<TransferRequest>(
      m, "TransferRequest", "A paired local-remote span for batch put/get.")
      .def(
          py::init<RegisteredSegment::Span, RemoteRegisteredSegment::Span>(),
          py::arg("local"),
          py::arg("remote"));

  // ---------------------------------------------------------------------------
  // RequestOptions
  // ---------------------------------------------------------------------------

  py::class_<RequestOptions>(
      m, "RequestOptions", "Options for data-transfer operations.")
      .def(
          py::init([](std::optional<uintptr_t> stream,
                      std::optional<int> timeoutMs) {
            RequestOptions opts;
            if (stream) {
              // NOLINTNEXTLINE(performance-no-int-to-ptr)
              opts.stream = reinterpret_cast<void*>(*stream);
            }
            if (timeoutMs) {
              opts.timeout = std::chrono::milliseconds(*timeoutMs);
            }
            return opts;
          }),
          py::arg("stream") = py::none(),
          py::arg("timeout_ms") = py::none());

  // ---------------------------------------------------------------------------
  // Futures
  // ---------------------------------------------------------------------------

  py::class_<UniflowFuture>(
      m, "UniflowFuture", "Handle for an async operation.")
      .def("done", &UniflowFuture::done, "Check if the operation completed.")
      .def(
          "get", &UniflowFuture::get, "Block until complete. Returns a Result.")
      .def(
          "wait_for",
          &UniflowFuture::wait_for,
          py::arg("timeout_ms"),
          "Wait up to timeout_ms. Returns True if ready, False if timed out.");

  // ---------------------------------------------------------------------------
  // UniflowAgentConfig
  // ---------------------------------------------------------------------------

  py::class_<UniflowAgentConfig>(
      m, "UniflowAgentConfig", "Configuration for UniflowAgent.")
      .def(
          py::init([](int deviceId,
                      const std::string& name,
                      const std::string& listenAddress,
                      int connectRetries,
                      int connectTimeoutMs) {
            return UniflowAgentConfig{
                .deviceId = deviceId,
                .name = name,
                .listenAddress = listenAddress,
                .connectRetries = connectRetries,
                .connectTimeoutMs = connectTimeoutMs,
            };
          }),
          py::arg("device_id") = -1,
          py::arg("name") = "",
          py::arg("listen_address") = "",
          py::arg("connect_retries") = 10,
          py::arg("connect_timeout_ms") = 1000)
      .def_readwrite("device_id", &UniflowAgentConfig::deviceId)
      .def_readwrite("name", &UniflowAgentConfig::name)
      .def_readwrite("listen_address", &UniflowAgentConfig::listenAddress)
      .def_readwrite("connect_retries", &UniflowAgentConfig::connectRetries)
      .def_readwrite(
          "connect_timeout_ms", &UniflowAgentConfig::connectTimeoutMs);

  // ---------------------------------------------------------------------------
  // MultiTransport
  // ---------------------------------------------------------------------------

  py::class_<MultiTransport>(
      m,
      "MultiTransport",
      "Manages multiple transport backends for a single connection.")
      .def(
          "bind",
          [](MultiTransport& t) { return toResult(t.bind()); },
          "Bind all transports and return serialized transport info.")
      .def(
          "connect",
          [](MultiTransport& t, const py::bytes& info) {
            auto span = bytesToSpan(info);
            py::gil_scoped_release release;
            auto status = t.connect(span);
            py::gil_scoped_acquire acquire;
            return toResult(std::move(status));
          },
          "Connect transports using remote peer's transport info.",
          py::arg("info"))
      .def(
          "put",
          [](MultiTransport& t,
             const std::vector<TransferRequest>& reqs,
             const RequestOptions& opts) {
            return UniflowFuture(t.put(reqs, opts));
          },
          "Initiate a batch put (write to remote).",
          py::arg("requests"),
          py::arg("options") = RequestOptions{})
      .def(
          "get",
          [](MultiTransport& t,
             const std::vector<TransferRequest>& reqs,
             const RequestOptions& opts) {
            return UniflowFuture(t.get(reqs, opts));
          },
          "Initiate a batch get (read from remote).",
          py::arg("requests"),
          py::arg("options") = RequestOptions{})
      .def(
          "send",
          [](MultiTransport& t,
             RegisteredSegment::Span src,
             const RequestOptions& opts) {
            return UniflowFuture(t.send(src, opts));
          },
          "Send a registered segment span (zero-copy).",
          py::arg("src"),
          py::arg("options") = RequestOptions{})
      .def(
          "recv",
          [](MultiTransport& t,
             RegisteredSegment::Span dst,
             const RequestOptions& opts) {
            return UniflowFuture(t.recv(dst, opts));
          },
          "Receive into a registered segment span (zero-copy).",
          py::arg("dst"),
          py::arg("options") = RequestOptions{})
      .def(
          "transfer_count",
          &MultiTransport::transferCount,
          "Number of transfers dispatched to a given transport type.",
          py::arg("transport_type"))
      .def(
          "shutdown",
          [](MultiTransport& t) {
            py::gil_scoped_release release;
            t.shutdown();
          },
          "Shut down all transports.");

  // ---------------------------------------------------------------------------
  // MultiTransportFactory
  // ---------------------------------------------------------------------------

  py::class_<MultiTransportFactory, std::shared_ptr<MultiTransportFactory>>(
      m,
      "MultiTransportFactory",
      "Factory for creating transports and registering memory segments.")
      .def(
          py::init([](int deviceId, const std::string& nicFilter) {
            if (nicFilter.empty()) {
              return std::make_shared<MultiTransportFactory>(deviceId);
            }
            return std::make_shared<MultiTransportFactory>(
                deviceId, NicFilter(nicFilter));
          }),
          "Create a MultiTransportFactory for the given device.",
          py::arg("device_id"),
          py::arg("nic_filter") = "")
      .def(
          "register_segment",
          [](MultiTransportFactory& f, Segment& seg) {
            return toResult(f.registerSegment(seg));
          },
          "Register a local memory segment.",
          py::arg("segment"))
      .def(
          "import_segment",
          [](MultiTransportFactory& f, const py::bytes& exportId) {
            return toResult(f.importSegment(bytesToSpan(exportId)));
          },
          "Import a remote segment from an export ID.",
          py::arg("export_id"))
      .def(
          "create_transport",
          [](MultiTransportFactory& f, const py::bytes& peerTopology) {
            return toResult(f.createTransport(bytesToSpan(peerTopology)));
          },
          "Create a MultiTransport from peer topology.",
          py::arg("peer_topology"))
      .def(
          "get_topology",
          [](MultiTransportFactory& f) {
            auto v = f.getTopology();
            return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
          },
          "Get the local topology as serialized bytes.");

  // ---------------------------------------------------------------------------
  // Connection
  // ---------------------------------------------------------------------------

  py::class_<Connection>(
      m,
      "Connection",
      "A connected pair of local and remote transports for data transfer.")
      .def(
          "shutdown",
          [](Connection& c) {
            py::gil_scoped_release release;
            c.shutdown();
          },
          "Gracefully shut down the connection.")
      .def("__enter__", [](Connection& c) -> Connection& { return c; })
      .def(
          "__exit__",
          [](Connection& c, py::args) {
            py::gil_scoped_release release;
            c.shutdown();
          })
      .def(
          "send_ctrl_msg",
          [](Connection& c, const py::bytes& payload) {
            auto span = bytesToSpan(payload);
            py::gil_scoped_release release;
            auto status = c.sendCtrlMsg(span);
            py::gil_scoped_acquire acquire;
            return toResult(std::move(status));
          },
          "Send a control message to the peer.",
          py::arg("payload"))
      .def(
          "recv_ctrl_msg",
          [](Connection& c) {
            std::vector<uint8_t> payload;
            py::gil_scoped_release release;
            auto result = c.recvCtrlMsg(payload);
            py::gil_scoped_acquire acquire;
            if (result) {
              return PyResult(
                  py::bytes(
                      reinterpret_cast<const char*>(payload.data()),
                      payload.size()));
            }
            return PyResult(std::move(result).error());
          },
          "Receive a control message from the peer.")
      .def(
          "put",
          [](Connection& c,
             const std::vector<TransferRequest>& reqs,
             const RequestOptions& opts) {
            return UniflowFuture(c.put(reqs, opts));
          },
          "Initiate a batch put (write to remote).",
          py::arg("requests"),
          py::arg("options") = RequestOptions{})
      .def(
          "get",
          [](Connection& c,
             const std::vector<TransferRequest>& reqs,
             const RequestOptions& opts) {
            return UniflowFuture(c.get(reqs, opts));
          },
          "Initiate a batch get (read from remote).",
          py::arg("requests"),
          py::arg("options") = RequestOptions{})
      .def(
          "send",
          [](Connection& c,
             RegisteredSegment::Span src,
             const RequestOptions& opts) {
            return UniflowFuture(c.send(src, opts));
          },
          "Send a registered segment span (zero-copy).",
          py::arg("src"),
          py::arg("options") = RequestOptions{})
      .def(
          "recv",
          [](Connection& c,
             RegisteredSegment::Span dst,
             const RequestOptions& opts) {
            return UniflowFuture(c.recv(dst, opts));
          },
          "Receive into a registered segment span (zero-copy).",
          py::arg("dst"),
          py::arg("options") = RequestOptions{});

  // ---------------------------------------------------------------------------
  // UniflowAgent
  // ---------------------------------------------------------------------------

  py::class_<UniflowAgent>(
      m,
      "UniflowAgent",
      R"(High-level agent for establishing connections and transferring data.

Creates a TcpServer on the listen_address and a TcpClient for outgoing
connections. Use get_unique_id() to obtain the connectable address, then
exchange it with peers to establish connections via accept()/connect().
      )")
      .def(
          py::init([](const UniflowAgentConfig& config) {
            return std::make_unique<UniflowAgent>(config);
          }),
          "Create a UniflowAgent from configuration.",
          py::arg("config"))
      .def(
          "get_unique_id",
          [](UniflowAgent& a) { return toResult(a.getUniqueId()); },
          "Get the connectable address (ip:port) for this agent.")
      .def(
          "register_segment",
          [](UniflowAgent& a, Segment& seg) {
            py::gil_scoped_release release;
            auto result = a.registerSegment(seg);
            py::gil_scoped_acquire acquire;
            return toResult(std::move(result));
          },
          "Register a local memory segment for data transfer.",
          py::arg("segment"))
      .def(
          "import_segment",
          [](UniflowAgent& a, const py::bytes& exportId) {
            auto span = bytesToSpan(exportId);
            py::gil_scoped_release release;
            auto result = a.importSegment(span);
            py::gil_scoped_acquire acquire;
            return toResult(std::move(result));
          },
          "Import a remote segment from an export ID.",
          py::arg("export_id"))
      .def(
          "accept",
          [](UniflowAgent& a) {
            py::gil_scoped_release release;
            auto result = a.accept();
            py::gil_scoped_acquire acquire;
            return toResult(std::move(result));
          },
          "Accept an incoming connection from a peer.")
      .def(
          "connect",
          [](UniflowAgent& a, const std::string& peerId) {
            py::gil_scoped_release release;
            auto result = a.connect(peerId);
            py::gil_scoped_acquire acquire;
            return toResult(std::move(result));
          },
          "Connect to a remote peer by its unique ID.",
          py::arg("peer_id"));
}

} // namespace uniflow
