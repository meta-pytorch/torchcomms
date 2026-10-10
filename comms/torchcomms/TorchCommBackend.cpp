// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommBackend.hpp"

#include <glog/logging.h>

#include <cstdlib>

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
// Copied from https://fburl.com/code/tu9hg6gf
namespace google::glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace google::glog_internal_namespace_

namespace torch::comms {

void TorchCommBackend::ensureLoggingInit() {
  // This trick can only be used on UNIX platforms.
  if (::google::glog_internal_namespace_::IsGoogleLoggingInitialized()) {
    return;
  }
  ::google::InitGoogleLogging("torchcomm");
  // Default to stderr for torchrun-style runs but let users override
  // via standard glog env vars (GLOG_logtostderr, GLOG_log_dir, etc.).
  if (std::getenv("GLOG_log_dir") == nullptr &&
      std::getenv("GLOG_logtostderr") == nullptr &&
      std::getenv("GLOG_alsologtostderr") == nullptr) {
    FLAGS_logtostderr = true;
  }

  // This will trigger a kernel panic on GB200 NVIDIA driver
  // temporarily disable signal handler until NVIDIA releases the new driver
  // in late Jan.
#if !defined(__aarch64__)
  ::google::InstallFailureSignalHandler();
#endif
}

TorchCommBackend::TorchCommBackend() {
  ensureLoggingInit();
}

} // namespace torch::comms
