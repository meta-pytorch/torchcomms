// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <concepts>

template <typename T>
concept PinnedHostItem = requires(T t) {
  { t.reset() } -> std::same_as<void>;
  { T::name() } -> std::same_as<const char*>;
  { t.inUse() } -> std::same_as<bool>;
  { t.onPop() } -> std::same_as<void>;
};

template <PinnedHostItem T>
class PinnedHostPool;
