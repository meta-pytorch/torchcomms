// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ScubaLoggerTestMixin.h"

#include <cstdlib>

#include <folly/testing/TestUtil.h>

#include "param.h" // @manual

void ScubaLoggerTestMixin::SetUp() {
  // Initialize the logging directory to be a tmpdir
  tmpDir_ = std::make_unique<folly::test::TemporaryDirectory>();
  setenv("NCCL_SCUBA_LOG_FILE_PREFIX", tmpDir_->path().string().c_str(), 1);
  setenv("NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED", "True", 1);
  initEnv();
}

const folly::test::TemporaryDirectory& ScubaLoggerTestMixin::scubaDir() const {
  return *tmpDir_;
}
