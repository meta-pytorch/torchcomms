#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/hccl/TorchCommHCCL.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_comms_hccl, m) {
  m.doc() = "HCCL specific python bindings for TorchComm";

  py::class_<TorchCommHCCL, std::shared_ptr<TorchCommHCCL>>(m, "TorchCommHCCL");
}
