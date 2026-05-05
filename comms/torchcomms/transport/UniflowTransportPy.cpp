// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Exposes uniflow's MultiTransport layer as torchcomms._transport.
// Replaces the previous RdmaTransport-only bindings with uniflow's
// multi-backend transport (NVLink + RDMA).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/Segment.h"

namespace py = pybind11;
using namespace uniflow;

namespace {

// ---------------------------------------------------------------------------
// Type-erased Result for Python: holds either a py::object value or an Err.
// ---------------------------------------------------------------------------

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

  // Pythonic accessor: returns the value on success, raises a Python
  // exception on error. Allows callers to use idiomatic try/except
  // instead of has_value()/value()/error() inspection.
  py::object unwrap() const {
    if (err_) {
      throw std::runtime_error(err_->toString());
    }
    return val_;
  }

 private:
  py::object val_;
  std::optional<Err> err_;
};

template <typename T>
PyResult toResult(Result<T> result) {
  if (result) {
    return PyResult(py::cast(std::move(result).value()));
  }
  return PyResult(std::move(result).error());
}

inline PyResult toResult(Status status) {
  if (status) {
    return PyResult(py::none());
  }
  return PyResult(status.error());
}

inline PyResult toResult(Result<std::vector<uint8_t>> result) {
  if (result) {
    auto& v = result.value();
    return PyResult(
        py::bytes(reinterpret_cast<const char*>(v.data()), v.size()));
  }
  return PyResult(std::move(result).error());
}

// ---------------------------------------------------------------------------
// Type-erased Future for Python
// ---------------------------------------------------------------------------

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

class TransportFuture {
 public:
  template <typename T>
  explicit TransportFuture(std::future<Result<T>>&& fut)
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

PYBIND11_MODULE(_transport, m) {
  m.doc() = "Uniflow MultiTransport python bindings for TorchComm";

  // -------------------------------------------------------------------------
  // Enums
  // -------------------------------------------------------------------------

  py::enum_<ErrCode>(m, "ErrCode")
      .value("NotImplemented", ErrCode::NotImplemented)
      .value("DriverError", ErrCode::DriverError)
      .value("TopologyDisconnect", ErrCode::TopologyDisconnect)
      .value("InvalidArgument", ErrCode::InvalidArgument)
      .value("NotConnected", ErrCode::NotConnected)
      .value("TransportError", ErrCode::TransportError)
      .value("ConnectionFailed", ErrCode::ConnectionFailed)
      .value("MemoryRegistrationError", ErrCode::MemoryRegistrationError)
      .value("Timeout", ErrCode::Timeout)
      .value("ResourceExhausted", ErrCode::ResourceExhausted);

  py::enum_<MemoryType>(m, "MemoryType")
      .value("DRAM", MemoryType::DRAM)
      .value("VRAM", MemoryType::VRAM)
      .value("NVME", MemoryType::NVME);

  py::enum_<TransportType>(m, "TransportType")
      .value("NVLink", TransportType::NVLink)
      .value("RDMA", TransportType::RDMA)
      .value("TCP", TransportType::TCP);

  // -------------------------------------------------------------------------
  // Err / Result
  // -------------------------------------------------------------------------

  py::class_<Err>(m, "Err")
      .def_property_readonly("code", &Err::code)
      .def_property_readonly("message", &Err::message)
      .def("__str__", &Err::toString)
      .def("__repr__", &Err::toString);

  py::class_<PyResult>(m, "Result")
      .def("has_value", &PyResult::hasValue)
      .def("has_error", &PyResult::hasError)
      .def("__bool__", &PyResult::hasValue)
      .def("value", &PyResult::value)
      .def("error", &PyResult::error)
      .def("unwrap", &PyResult::unwrap);

  // -------------------------------------------------------------------------
  // Future
  // -------------------------------------------------------------------------

  py::class_<TransportFuture>(m, "TransportFuture")
      .def("done", &TransportFuture::done)
      .def("get", &TransportFuture::get)
      .def("wait_for", &TransportFuture::wait_for, py::arg("timeout_ms"));

  // -------------------------------------------------------------------------
  // Segment
  // -------------------------------------------------------------------------

  py::class_<Segment>(m, "Segment")
      .def(
          py::init([](uintptr_t ptr,
                      size_t length,
                      MemoryType memType,
                      int deviceId) {
            return Segment(
                reinterpret_cast<void*>(ptr), length, memType, deviceId);
          }),
          py::arg("ptr"),
          py::arg("length"),
          py::arg("mem_type") = MemoryType::DRAM,
          py::arg("device_id") = -1)
      .def(
          "__init__",
          [](Segment& self, const at::Tensor& tensor) {
            MemoryType mt =
                tensor.is_cuda() ? MemoryType::VRAM : MemoryType::DRAM;
            int devId = tensor.is_cuda() ? tensor.get_device() : -1;
            new (&self) Segment(tensor.data_ptr(), tensor.nbytes(), mt, devId);
          },
          py::arg("tensor"))
      .def_property_readonly(
          "data_ptr",
          [](const Segment& s) {
            return reinterpret_cast<uintptr_t>(s.data());
          })
      .def_property_readonly("length", [](const Segment& s) { return s.len(); })
      .def_property_readonly(
          "mem_type", [](const Segment& s) { return s.memType(); })
      .def_property_readonly(
          "device_id", [](const Segment& s) { return s.deviceId(); });

  // -------------------------------------------------------------------------
  // RegisteredSegment
  // -------------------------------------------------------------------------

  py::class_<RegisteredSegment>(m, "RegisteredSegment")
      .def(
          "export_id",
          [](RegisteredSegment& s) { return toResult(s.exportId()); })
      .def(
          "span",
          [](RegisteredSegment& s, size_t offset, size_t length) {
            return s.span(offset, length);
          },
          py::keep_alive<0, 1>(),
          py::arg("offset"),
          py::arg("length"))
      .def_property_readonly(
          "length", [](const RegisteredSegment& s) { return s.len(); });

  py::class_<RegisteredSegment::Span>(m, "RegisteredSegmentSpan")
      .def_property_readonly(
          "size", [](const RegisteredSegment::Span& s) { return s.size(); });

  // -------------------------------------------------------------------------
  // RemoteRegisteredSegment
  // -------------------------------------------------------------------------

  py::class_<RemoteRegisteredSegment>(m, "RemoteRegisteredSegment")
      .def(
          "span",
          [](RemoteRegisteredSegment& s, size_t offset, size_t length) {
            return s.span(offset, length);
          },
          py::keep_alive<0, 1>(),
          py::arg("offset"),
          py::arg("length"))
      .def_property_readonly(
          "length", [](const RemoteRegisteredSegment& s) { return s.len(); });

  py::class_<RemoteRegisteredSegment::Span>(m, "RemoteRegisteredSegmentSpan")
      .def_property_readonly(
          "size",
          [](const RemoteRegisteredSegment::Span& s) { return s.size(); });

  // -------------------------------------------------------------------------
  // TransferRequest
  // -------------------------------------------------------------------------

  py::class_<TransferRequest>(m, "TransferRequest")
      .def(
          py::init<RegisteredSegment::Span, RemoteRegisteredSegment::Span>(),
          py::arg("local"),
          py::arg("remote"));

  // -------------------------------------------------------------------------
  // RequestOptions
  // -------------------------------------------------------------------------

  py::class_<RequestOptions>(m, "RequestOptions")
      .def(
          py::init([](std::optional<uintptr_t> stream,
                      std::optional<int> timeoutMs) {
            RequestOptions opts;
            if (stream) {
              opts.stream = reinterpret_cast<void*>(*stream);
            }
            if (timeoutMs) {
              opts.timeout = std::chrono::milliseconds(*timeoutMs);
            }
            return opts;
          }),
          py::arg("stream") = py::none(),
          py::arg("timeout_ms") = py::none());

  // -------------------------------------------------------------------------
  // MultiTransportFactory
  // -------------------------------------------------------------------------

  py::class_<MultiTransportFactory, std::shared_ptr<MultiTransportFactory>>(
      m, "MultiTransportFactory")
      .def(
          py::init([](int deviceId) {
            return std::make_shared<MultiTransportFactory>(deviceId);
          }),
          py::arg("device_id"))
      .def(
          "register_segment",
          [](MultiTransportFactory& f, Segment& seg) {
            return toResult(f.registerSegment(seg));
          },
          py::arg("segment"))
      .def(
          "import_segment",
          [](MultiTransportFactory& f, const py::bytes& exportId) {
            return toResult(f.importSegment(bytesToSpan(exportId)));
          },
          py::arg("export_id"))
      .def(
          "create_transport",
          [](MultiTransportFactory& f, const py::bytes& peerTopology) {
            return toResult(f.createTransport(bytesToSpan(peerTopology)));
          },
          py::arg("peer_topology"))
      .def("get_topology", [](MultiTransportFactory& f) {
        auto v = f.getTopology();
        return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
      });

  // -------------------------------------------------------------------------
  // MultiTransport
  // -------------------------------------------------------------------------

  py::class_<MultiTransport>(m, "MultiTransport")
      .def("bind", [](MultiTransport& t) { return toResult(t.bind()); })
      .def(
          "connect",
          [](MultiTransport& t, const py::bytes& info) {
            std::string infoStr = info.cast<std::string>();
            std::vector<uint8_t> infoVec(infoStr.begin(), infoStr.end());
            py::gil_scoped_release release;
            auto status = t.connect(infoVec);
            py::gil_scoped_acquire acquire;
            return toResult(std::move(status));
          },
          py::arg("info"))
      .def(
          "put",
          [](MultiTransport& t,
             const std::vector<TransferRequest>& reqs,
             const RequestOptions& opts) {
            return TransportFuture(t.put(reqs, opts));
          },
          py::arg("requests"),
          py::arg("options") = RequestOptions{})
      .def(
          "get",
          [](MultiTransport& t,
             const std::vector<TransferRequest>& reqs,
             const RequestOptions& opts) {
            return TransportFuture(t.get(reqs, opts));
          },
          py::arg("requests"),
          py::arg("options") = RequestOptions{})
      .def(
          "transfer_count",
          &MultiTransport::transferCount,
          py::arg("transport_type"))
      .def("shutdown", [](MultiTransport& t) {
        py::gil_scoped_release release;
        t.shutdown();
      });
}
