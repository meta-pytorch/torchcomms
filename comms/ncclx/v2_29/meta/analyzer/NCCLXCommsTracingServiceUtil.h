// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace ncclx {

class NCCLXCommsTracingServiceUtil {
 public:
  // Starts the CommsTracingService for analyzer debugging
  static void startService();

  // Stops the CommsTracingService
  static void stopService();

  // Retrieve the port number of the CommsTracingService
  static int getPort();
};

} // namespace ncclx
