// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

/*
 * The greppable abort-log contract, spelled out for tests.
 *
 * Two decisions here, and they pull in opposite directions:
 *
 * 1. These are literals, NOT `FT_ABORT_FIRST_WRITER_` / `FT_ABORT_SITE_` from
 *    `Abort.h`. A test that built its expectation from the macro would follow
 *    it anywhere it went, including somewhere no existing log search would
 *    find it. These are the strings oncall greps for, so changing one should
 *    fail a test rather than silently retarget it.
 * 2. There is one copy, not one per test file. The rejected alternative was to
 *    spell them out separately in `AbortDeviceUT.cc` and `AbortUT.cc`; that
 *    keeps property 1 but adds a second failure mode, where editing one file
 *    leaves the other asserting a stale string and still passing.
 */

namespace comms::fault_tolerance::testing {

// Prefixes the one line emitted by whichever writer won the reason CAS.
constexpr const char* kFirstWriterMarker = "COMMS FT ABORT FIRST WRITER: ";

// Prefixes the line an `FT_ABORT_*` macro adds naming where the abort was
// observed. Separate from the marker above because an observation is not a
// transition; counting first-writer markers must stay a count of transitions.
constexpr const char* kSiteMarker = "COMMS FT ABORT SITE: ";

} // namespace comms::fault_tolerance::testing
