// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/SkipDestroyUtil.h"

namespace ctran::utils {
namespace {
bool skipDestroyCtran = false;
}

void setSkipDestroyCtran(bool newVal) {
  skipDestroyCtran = newVal;
}

bool getSkipDestroyCtran() {
  return skipDestroyCtran;
}

} // namespace ctran::utils
