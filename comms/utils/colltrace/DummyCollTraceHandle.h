// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

// Dummy implementation of ICollTraceHandle that returns success for all
// operations
class DummyCollTraceHandle : public ICollTraceHandle {
 public:
  DummyCollTraceHandle() = default;
  ~DummyCollTraceHandle() override = default;

  CommsMaybeVoid trigger(CollTraceHandleTriggerState state) noexcept override {
    return folly::Unit{};
  }

  CommsMaybeVoid triggerPlugin(
      std::string pluginName,
      folly::dynamic params) noexcept override {
    return folly::Unit{};
  }

  CommsMaybe<std::shared_ptr<ICollRecord>> getCollRecord() noexcept override {
    return std::shared_ptr<ICollRecord>{nullptr};
  }

  CommsMaybeVoid invalidate() noexcept override {
    return folly::Unit{};
  }
};

} // namespace meta::comms::colltrace
