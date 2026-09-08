// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

/*
 * The greppable abort-log contract, spelled out for tests.
 *
 * Deliberately NOT derived from `FT_ABORT_FIRST_WRITER_` / `FT_ABORT_SITE_` in
 * `Abort.h`. A test that built its expectation from the macro would follow it
 * anywhere it went, including somewhere no existing log search would find it;
 * these are the strings oncall greps for, so a change to either should fail a
 * test rather than silently retarget it.
 *
 * Shared between `AbortDeviceUT.cc` and `AbortUT.cc` rather than spelled twice.
 * Two independent copies keep the same property against the macro while adding
 * a new way to drift: an edit to one file leaves the other asserting a stale
 * string, and the stale one still passes.
 */

namespace comms::fault_tolerance::testing {

// Prefixes the one line emitted by whichever writer won the reason CAS.
constexpr const char* kFirstWriterMarker = "COMMS FT ABORT FIRST WRITER: ";

// Prefixes the line an `FT_ABORT_*` macro adds naming where the abort was
// observed. Separate from the marker above because an observation is not a
// transition; counting first-writer markers must stay a count of transitions.
constexpr const char* kSiteMarker = "COMMS FT ABORT SITE: ";

} // namespace comms::fault_tolerance::testing
