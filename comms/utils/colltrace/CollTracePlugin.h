// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

// Abstract class for interfaces to implement for plugin of colltrace.
// How the plugin is used:
// 1. The plugin is registered in the colltrace library.
// 2. The following callbacks will be triggered in the colltrace library in
// order:
//    beforeCollKernelScheduled -> afterCollKernelScheduled ->
//    afterCollKernelStart -> [whenCollKernelHang] -> afterCollKernelEnd
// Please note that beforeCollKernelScheduled and afterCollKernelScheduled will
// be triggered in the calling thread, while the rest will be triggered in the
// colltrace thread.
class ICollTracePlugin {
 public:
  virtual ~ICollTracePlugin() = default;

  // For now the failures returned will be ignored. In the future, we may
  // consider to recreate the plugin if the failure is unrecoverable.

  // Get the name of the current plugin
  virtual std::string_view getName() const noexcept = 0;

  // ----- Callbacks below will be triggered in the calling (main) thread -----

  // Callback that will be called before a collective is scheduled. For cuda
  // event based tracking, this function will be called after the cuda event is
  // inserted into the stream.
  virtual CommsMaybeVoid beforeCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept = 0;
  // Callback that will be called after a collective is scheduled. For cuda
  // event based tracking, this function will be called before the cuda event is
  // inserted into the stream.
  virtual CommsMaybeVoid afterCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept = 0;

  // ----- Callbacks below will be triggered in the colltrace thread -----

  virtual CommsMaybeVoid afterCollKernelStart(
      CollTraceEvent& curEvent) noexcept = 0;

  // Every maxCheckCancelInterval, this function will be called for each plugin
  // to give the plugin a chance to perform some checks like async error and
  // timeout. This will happen both when waiting for the collective to start and
  // when waiting for the collective to end.
  virtual CommsMaybeVoid collEventProgressing(
      CollTraceEvent& curEvent) noexcept = 0;

  virtual CommsMaybeVoid afterCollKernelEnd(
      CollTraceEvent& curEvent) noexcept = 0;
};

} // namespace meta::comms::colltrace
