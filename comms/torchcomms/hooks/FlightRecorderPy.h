// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <pybind11/pybind11.h>

namespace torch::comms::fr {

void initFlightRecorderPyBindings(pybind11::module_& m);

} // namespace torch::comms::fr
