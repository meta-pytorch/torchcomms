// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/CollTraceHandle.h"
#include "meta/colltrace/CollTraceEvent.h"

namespace meta::comms::colltrace {

class CollTraceLegacyHandle : public ICollTraceHandle {
 public:
  enum class HandleType { baseline, ctran };
  CollTraceLegacyHandle(
      CtranComm* comm,
      std::unique_ptr<::CollTraceEvent> event,
      HandleType type);

  CommsMaybeVoid trigger(CollTraceHandleTriggerState state) noexcept override;
  CommsMaybeVoid triggerPlugin(
      std::string pluginName,
      folly::dynamic params) noexcept override;
  CommsMaybe<std::shared_ptr<ICollRecord>> getCollRecord() noexcept override;
  CommsMaybeVoid invalidate() noexcept override;

  bool isValid() const noexcept;

 private:
  CtranComm* comm_{nullptr};
  std::unique_ptr<::CollTraceEvent> event_;
  std::shared_ptr<ICollRecord> collRecord_;
  HandleType handleType_;
  // These are old CpuWaitEvents. Specify they are in the global scope as we
  // also have implementation for them in the new CollTrace.
  ::CpuWaitEvent* cpuStartEvent_{nullptr};
  ::CpuWaitEvent* cpuEndEvent_{nullptr};
};

} // namespace meta::comms::colltrace
