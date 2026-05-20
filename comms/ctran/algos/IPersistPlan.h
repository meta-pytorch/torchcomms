// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace ctran::algos {

// Key identifying which algorithm owns a persistent plan.
// Add new entries here when registering plans for additional algorithms.
enum class PersistPlanKey {
  kAllgatherCtsrd,
};

struct PersistPlanKeyHash {
  size_t operator()(PersistPlanKey key) const {
    return static_cast<size_t>(key);
  }
};

// Type-erasure base for algorithm-specific persistent plans.
//
// Each algorithm defines its own PersistPlan in its own namespace
// (e.g., ctran::allgather::ctsrd::PersistPlan) inheriting from this class.
// Plans are stored in CtranAlgo::persistPlans_ via getOrCreatePersistPlan(),
// created once on first collective call and reused for the communicator's
// lifetime. Callers downcast via static_cast to the concrete type.
//
// To add a persistent plan for a new algorithm:
//   1. Add a key to PersistPlanKey above
//   2. Define YourPersistPlan : public IPersistPlan in your algorithm's
//   namespace
//   3. Call algo->getOrCreatePersistPlan(key, factory) in your impl
class IPersistPlan {
 public:
  virtual ~IPersistPlan() = default;
};

} // namespace ctran::algos
