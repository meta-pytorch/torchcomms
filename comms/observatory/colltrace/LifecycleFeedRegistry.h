// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "comms/observatory/colltrace/LifecycleFeedTypes.h"

// The process-wide set of lifecycle-event feeds. There must be exactly one;
// see comms/observatory/README.md before changing how this is built or linked.

namespace meta::comms::colltrace {

// What a caller can do with a feed, supplied by whoever registers it.
struct LifecycleFeedOps {
  // The id this feed stamps on its events, and all a caller holding one of
  // those events knows about where it came from.
  uint64_t commId{0};
  // Held weakly and never dereferenced: it decides whether the feed is still
  // alive, so a caller that dies without deregistering is reaped rather than
  // drained through a dangling callable.
  std::weak_ptr<void> alive;
  std::function<uint64_t()> requestFlush;
  // Waits for the generation `requestFlush` returned, up to the deadline the
  // caller passes. False means the deadline came first, not that the feed is
  // broken.
  std::function<bool(uint64_t, std::chrono::nanoseconds)> waitFlush;
  std::function<std::vector<LifecycleEventRecord>()> drainUnread;
  // The highest collective id this feed has stamped so far. A consumer binds
  // its own record of a collective to the ids the feed is handing out, and it
  // has no other way to learn where the feed has reached.
  std::function<uint64_t()> latestCollId;
  // What a collective captured into a graph is. Answered by the feed that
  // reported it, not by whoever owns the communicator now: a comm that
  // rebuilds its tracer keeps reporting the old one's replays, and the new
  // tracer has never heard of them.
  std::function<std::optional<CapturedCollDescription>(uint64_t)>
      describeCaptured;
};

struct LifecycleFeedSource {
  // Pins the feed alive for as long as the caller holds the snapshot, so a
  // drain can request a flush and wait for it without the owner retiring in
  // between.
  std::shared_ptr<void> alive;
  // Shared, not copied: a snapshot takes a reference, so no callable a
  // registrant supplied is ever constructed or destroyed inside this library.
  std::shared_ptr<const LifecycleFeedOps> ops;
};

// Re-registering an owner replaces its feed.
//
// Returns false, having registered nothing, when `ops.alive` is already empty:
// the first snapshot would reap such a feed and every drain after that would
// return a subset with nothing to say it had. That is the failure this library
// exists to prevent, and a registrant can get it by moving the owning
// `shared_ptr` before registering rather than after.
//
// `commIdCollision`, when given, reports whether another live feed already
// stamps `ops.commId`. Consumers key on that id, so two feeds sharing one merge
// into a single identity: their collectives collide and each retires the
// other's. It is answered under the same lock as the registration, so it cannot
// go stale between the two. A zero id is the unset default and collides freely;
// only a real id means anything here.
bool registerLifecycleFeed(
    const void* owner,
    LifecycleFeedOps ops,
    bool* commIdCollision = nullptr);

void deregisterLifecycleFeed(const void* owner);

// Every live feed, pinned. Feeds whose owner has gone are dropped here, which
// is the only place they are reaped.
std::vector<LifecycleFeedSource> snapshotLifecycleFeeds();

// Every feed's unread events, from every backend that registered one, in
// timestamp order.
//
// This lives here rather than in a backend because the set it walks is
// process-wide: a drain written inside one backend is the whole process's
// drain by accident, and a process running only the other backend has none.
//
// Flushing is two passes so the requests overlap instead of running back to
// back, and each generation travels with the feed that issued it so the pairing
// cannot drift.
//
// Destructive: an event is delivered once, to whoever asks first. A backend
// that also drains its own feed directly is a second consumer of the same
// queue, and the two will split the stream between them.
//
// `budget` bounds the whole call, not each feed: the waits run one after
// another, so a per-feed limit would still let several stalled backends sum.
// A feed that runs out contributes what it had already published and is counted
// in `lifecycleFeedFailures()`.
std::vector<LifecycleEventRecord> drainAllLifecycleEvents(
    std::chrono::nanoseconds budget = std::chrono::seconds{1});

// How many times a feed has thrown, or missed a deadline, for the life of the
// process. A skipped feed contributes nothing and says nothing otherwise, which
// is the silent subset this library exists to prevent, one feed at a time.
// An unset callable is allowed and is not counted.
uint64_t lifecycleFeedFailures() noexcept;

// Describe a collective captured into a graph, asked of the feed that stamps
// `commId` on its events. Empty when no such feed is registered, or when the
// feed does not recognise the id.
std::optional<CapturedCollDescription> describeCapturedCollective(
    uint64_t commId,
    uint64_t capturedCollId);

// The highest collective id the feed stamping `commId` has reached. Keyed on
// the id, not the communicator, so a caller needs nothing of the backend that
// owns it. Empty when no live feed stamps the id, when the feed cannot answer,
// or when asking it raises; a zero id is the unset default several feeds may
// carry, so it answers empty too.
std::optional<uint64_t> lifecycleLatestCollIdForCommId(uint64_t commId);

// Drops every feed. Tests only: a process has one registry, so a test that
// leaves entries behind changes the next one.
void resetLifecycleFeedRegistryForTest();

} // namespace meta::comms::colltrace
