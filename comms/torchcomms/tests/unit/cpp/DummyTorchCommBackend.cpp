// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchCommBackend.hpp> // @manual=//comms/torchcomms:torchcomms-headers-cpp
#include <comms/torchcomms/TorchCommDummy.hpp> // @manual=//comms/torchcomms:torchcomms-headers-cpp

static torch::comms::TorchCommBackend* new_comm_impl() {
  return new torch::comms::TorchCommDummy();
}

static void destroy_comm_impl(torch::comms::TorchCommBackend* comm) {
  delete comm;
}

static const char* get_supported_version_impl() {
  return torch::comms::TORCHCOMM_BACKEND_ABI_VERSION;
}

extern "C" torch::comms::DynamicLoaderInterface
create_dynamic_loader_dummy_test() {
  torch::comms::DynamicLoaderInterface interface{
      .new_comm = new_comm_impl,
      .destroy_comm = destroy_comm_impl,
      .get_supported_version = get_supported_version_impl,
  };
  return interface;
}
