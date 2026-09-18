// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

#include "comms/utils/collstats/CollStatsAtomics.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// The per-communicator, double-buffered aggregate. Two banks of per-key values
// and one atomic epoch word: finalizers write bank[epoch]; at the readout
// boundary the reader flips the epoch, then copies and zeroes the retired bank.
// Each bank holds exactly one window's totals, so the reader needs no host
// delta subtraction. Key identity lives on the host, in the registry that
// assigned the slots; only the values are double-buffered.
//
// The flip and the zeroing live in CollStatsReader.cu, not here: the banks are
// device memory, so the flip is a one-thread kernel on the reader stream and
// the reset is a cudaMemsetAsync. There is deliberately no host-side helper for
// either -- one taking a bare CollStatValue* would invite a host memset of a
// device pointer.
//
// One value slot is reserved past the key capacity for the catch-all, so
// observations whose key did not fit the registry still contribute a
// distribution rather than vanishing.

namespace meta::comms::collstats {

struct CollStatDoubleBank {
  uint32_t numKeys; // key-index capacity; catch-all lives at index numKeys
  uint64_t epoch; // finalizers read it, the reader flips it; see below
  CollStatValue* values[2]; // each [numKeys + 1]; [numKeys] is the catch-all
};

// The value array finalizers must write for the current epoch.
//
// This load carries no ordering of its own, and deliberately so: what keeps a
// finalizer out of the bank being retired is CUDA stream gating, not the
// memory model. The reader records an event on every instrumented stream
// before flipping and makes those streams wait on the flip afterwards, so a
// finalizer either ran entirely before the flip or launches entirely after it
// (see gateOldFinalizers/gateNewLaunches in CollStatsReader.cu). Adding an
// acquire fence per finalize would cost the hot path and buy nothing.
//
// The one ungated path -- the teardown flush -- device-synchronizes first,
// which gives the same guarantee.
COLLSTATS_HD inline CollStatValue* collStatCurrentValues(
    CollStatDoubleBank* bank) {
  const uint64_t e = collStatAtomicLoad(&bank->epoch);
  return bank->values[e & 1u];
}

} // namespace meta::comms::collstats
