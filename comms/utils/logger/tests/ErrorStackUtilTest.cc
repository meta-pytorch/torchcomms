// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/ErrorStackUtil.h"

#include <folly/debugging/symbolizer/Symbolizer.h>
#include <folly/portability/GTest.h>

using meta::comms::logger::captureNativeErrorStack;
using meta::comms::logger::detail::normalizeStackTrace;

// This target exercises the real symbolizer path. If folly is ever configured
// without one here, the capture test below would silently pass against the
// inline stub instead, so assert the capability rather than infer it.
static_assert(
    FOLLY_HAVE_ELF && FOLLY_HAVE_DWARF,
    "buck2 is expected to build folly with a real symbolizer; without it the "
    "captureNativeErrorStack test would exercise the empty-string stub");

TEST(ErrorStackUtilTest, EmptyTraceYieldsEmptyVector) {
  // Both the no-symbolizer stub and the real implementation return "" when no
  // trace is available; splitting it would yield one bogus frame.
  EXPECT_TRUE(normalizeStackTrace("").empty());
}

TEST(ErrorStackUtilTest, SplitsFramesOnNewlines) {
  const std::vector<std::string> expected = {"frame_one", "frame_two"};
  EXPECT_EQ(normalizeStackTrace("frame_one\nframe_two"), expected);
}

TEST(ErrorStackUtilTest, IgnoresBlankLines) {
  // A trailing newline must not produce an empty trailing frame.
  const std::vector<std::string> expected = {"frame_one", "frame_two"};
  EXPECT_EQ(normalizeStackTrace("frame_one\n\nframe_two\n"), expected);
}

TEST(ErrorStackUtilTest, DropsLeadingPlumbingFrames) {
  // Leading logging / Scuba frames are stripped so the stack starts at the
  // real error site.
  const std::vector<std::string> expected = {"real_error_site", "caller"};
  EXPECT_EQ(
      normalizeStackTrace(
          "folly::symbolizer::getStackTraceStr()\n"
          "NcclScubaSample::setError\n"
          "real_error_site\n"
          "caller"),
      expected);
}

TEST(ErrorStackUtilTest, KeepsInternalMarkerAppearingAfterRealFrame) {
  // Only *leading* plumbing is dropped -- a marker deeper in the stack is a
  // genuine frame.
  const std::vector<std::string> expected = {
      "real_error_site", "logErrorToScuba"};
  EXPECT_EQ(normalizeStackTrace("real_error_site\nlogErrorToScuba"), expected);
}

TEST(ErrorStackUtilTest, CaptureProducesNoEmptyFrames) {
  // Symbolizer.h documents that even the real implementation may return "",
  // so the result is allowed to be empty; what must never happen is a frame
  // that is itself empty.
  for (const auto& frame : captureNativeErrorStack()) {
    EXPECT_FALSE(frame.empty());
  }
}
