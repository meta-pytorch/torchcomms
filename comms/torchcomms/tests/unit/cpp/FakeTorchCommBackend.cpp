// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchCommBackend.hpp> // @manual=//comms/torchcomms:torchcomms-headers-cpp
#include <comms/torchcomms/fake/TorchCommFake.hpp> // @manual=//comms/torchcomms/fake:torchcomms-fake-cpp

#ifdef TORCHCOMMS_LEGACY_DIRECT_REGISTRATION_TEST
#include <comms/torchcomms/TorchCommFactory.hpp> // @manual=//comms/torchcomms:torchcomms-headers-cpp
#endif

static torch::comms::TorchCommBackend* new_comm_impl() {
  return new torch::comms::TorchCommFake();
}

static void destroy_comm_impl(torch::comms::TorchCommBackend* comm) {
  delete comm;
}

static const char* get_supported_version_impl() {
  return torch::comms::TORCHCOMM_BACKEND_ABI_VERSION;
}

extern "C" torch::comms::DynamicLoaderInterface
create_dynamic_loader_fake_test() {
  torch::comms::DynamicLoaderInterface interface{
      .new_comm = new_comm_impl,
      .destroy_comm = destroy_comm_impl,
      .get_supported_version = get_supported_version_impl,
  };
  return interface;
}

#ifdef TORCHCOMMS_LEGACY_DIRECT_REGISTRATION_TEST
namespace {

bool factory_called = false;
bool mismatched_new_comm_called = false;

torch::comms::TorchCommBackend* mismatched_new_comm_impl() {
  mismatched_new_comm_called = true;
  return nullptr;
}

const char* mismatched_supported_version_impl() {
  return "incompatible-test-version";
}

class LegacyDirectRegistration {
 public:
  LegacyDirectRegistration() {
    torch::comms::TorchCommFactory::get().register_backend(
        "legacy_direct_registration_test", []() {
          factory_called = true;
          return std::shared_ptr<torch::comms::TorchCommBackend>();
        });
  }
};

LegacyDirectRegistration legacy_direct_registration;

} // namespace

extern "C" bool legacy_direct_registration_factory_called() {
  return factory_called;
}

extern "C" bool mismatched_dynamic_new_comm_called() {
  return mismatched_new_comm_called;
}

extern "C" torch::comms::DynamicLoaderInterface
create_dynamic_loader_abi_mismatch_test() {
  return {
      .new_comm = mismatched_new_comm_impl,
      .destroy_comm = destroy_comm_impl,
      .get_supported_version = mismatched_supported_version_impl,
  };
}
#endif
