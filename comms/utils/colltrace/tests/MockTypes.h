// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/colltrace/CollTracePlugin.h"
#include "comms/utils/colltrace/CollWaitEvent.h"

namespace meta::comms::colltrace {
// Mock CollTrace class for testing CollTraceHandle
class MockCollTrace : public ICollTrace {
 public:
  MockCollTrace() {}

  MOCK_METHOD(
      CommsMaybeVoid,
      triggerEventState,
      (CollTraceEvent & collEvent, CollTraceHandleTriggerState state),
      (noexcept, override));

  MOCK_METHOD(
      CommsMaybe<std::shared_ptr<CollTraceHandle>>,
      recordCollective,
      (std::unique_ptr<ICollMetadata> metadata,
       std::unique_ptr<ICollWaitEvent> waitEvent),
      (noexcept, override));

  MOCK_METHOD(
      ICollTracePlugin*,
      getPluginByName,
      (std::string name),
      (noexcept, override));
};

// Mock ICollTracePlugin for testing
class MockCollTracePlugin : public ICollTracePlugin {
 public:
  MOCK_METHOD(std::string_view, getName, (), (const, noexcept, override));
  MOCK_METHOD(
      CommsMaybeVoid,
      beforeCollKernelScheduled,
      (CollTraceEvent & curEvent),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybeVoid,
      afterCollKernelScheduled,
      (CollTraceEvent & curEvent),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybeVoid,
      afterCollKernelStart,
      (CollTraceEvent & curEvent),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybeVoid,
      collEventProgressing,
      (CollTraceEvent & curEvent),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybeVoid,
      afterCollKernelEnd,
      (CollTraceEvent & curEvent),
      (noexcept, override));
};

// Mock CollWaitEvent for testing
class MockCollWaitEvent : public ICollWaitEvent {
 public:
  MOCK_METHOD(
      CommsMaybeVoid,
      beforeCollKernelScheduled,
      (),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybeVoid,
      afterCollKernelScheduled,
      (),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybe<bool>,
      waitCollStart,
      (std::chrono::milliseconds sleepTimeMs),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybe<bool>,
      waitCollEnd,
      (std::chrono::milliseconds sleepTimeMs),
      (noexcept, override));
  MOCK_METHOD(CommsMaybeVoid, signalCollStart, (), (noexcept, override));
  MOCK_METHOD(CommsMaybeVoid, signalCollEnd, (), (noexcept, override));
  MOCK_METHOD(
      CommsMaybe<system_clock_time_point>,
      getCollEnqueueTime,
      (),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybe<system_clock_time_point>,
      getCollStartTime,
      (),
      (noexcept, override));
  MOCK_METHOD(
      CommsMaybe<system_clock_time_point>,
      getCollEndTime,
      (),
      (noexcept, override));
};

// Mock ICollMetadata for testing
class MockCollMetadata : public ICollMetadata {
 public:
  MockCollMetadata() = default;
  ~MockCollMetadata() override = default;

  MOCK_METHOD(std::size_t, hash, (), (const, override));
  MOCK_METHOD(
      bool,
      equals,
      (const ICollMetadata& other),
      (const, noexcept, override));
  MOCK_METHOD(
      std::string_view,
      getMetadataType,
      (),
      (const, noexcept, override));
  MOCK_METHOD(folly::dynamic, toDynamic, (), (const, noexcept, override));
  MOCK_METHOD(
      void,
      fromDynamic,
      (const folly::dynamic& d),
      (noexcept, override));
};

} // namespace meta::comms::colltrace
