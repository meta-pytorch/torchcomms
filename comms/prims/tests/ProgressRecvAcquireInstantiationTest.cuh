// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace comms::prims::test {

// Returns true once the translation unit holding the acquire/release
// instantiations has linked in. See the .cu for what it covers.
bool progress_recv_acquire_instantiations_linked();

} // namespace comms::prims::test
