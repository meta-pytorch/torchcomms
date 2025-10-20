// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

struct SQueues {};

namespace ctran {

struct CtranTcpDmConfig {};

class CtranTcpDmRequest {
 public:
  explicit CtranTcpDmRequest() {}

  bool isComplete() {
    return false;
  }

  void complete() {}
};

} // namespace ctran
