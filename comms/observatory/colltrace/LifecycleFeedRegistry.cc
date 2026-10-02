// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/observatory/colltrace/LifecycleFeedRegistry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace meta::comms::colltrace {

namespace {
// Held by shared_ptr so that nothing a registrant supplied is constructed or
// destroyed inside this library. A LifecycleFeedOps is four std::functions, and
// copying one runs the copy constructor of every captured object; sharing means
// a snapshot takes a refcount instead.
using SharedOps = std::shared_ptr<const LifecycleFeedOps>;

struct RegistryHolder {
  std::mutex mutex;
  std::vector<std::pair<const void*, SharedOps>> entries;
};

RegistryHolder& holder() {
  // Deliberately never destroyed. Its __cxa_atexit entry is registered on the
  // first call at runtime, so it would run ahead of any backend global built at
  // load time -- and a global whose destructor deregisters from here would then
  // lock a destroyed mutex, inside a .so that may already be unmapped. Leaking
  // keeps the lazy thread-safe init and drops only the destruction. The pointer
  // stays reachable, so LeakSanitizer does not report it.
  static auto* instance = new RegistryHolder();
  return *instance;
}

std::atomic<uint64_t>& failureCount() {
  static std::atomic<uint64_t> count{0};
  return count;
}

void countFailure() {
  failureCount().fetch_add(1, std::memory_order_relaxed);
}
} // namespace

uint64_t lifecycleFeedFailures() noexcept {
  return failureCount().load(std::memory_order_relaxed);
}

uint64_t getNextLifecycleFeedCommId() noexcept {
  static std::atomic<uint64_t> nextCommId{1};
  return nextCommId.fetch_add(1, std::memory_order_relaxed);
}

// Every function below that drops an entry moves it into a local first, so the
// last reference goes once the lock is released. A callable can capture
// anything, and one that reaches back into the registry as it dies would
// deadlock on this non-recursive mutex. Those locals look unused; they are not.

bool registerLifecycleFeed(
    const void* owner,
    LifecycleFeedOps ops,
    bool* commIdCollision) {
  // Refuse a feed nobody owns. Registering it would succeed, the first snapshot
  // would reap it, and every drain after that would quietly return a subset.
  //
  // Pinned rather than merely tested: an owner dropped between the test and the
  // lock would leave exactly the dead entry this refuses. Holding it to the end
  // of the call closes that window.
  const auto pin = ops.alive.lock();
  if (pin == nullptr) {
    if (commIdCollision != nullptr) {
      *commIdCollision = false;
    }
    return false;
  }
  // Built before the lock: this is the one copy of the registrant's callables
  // that this function makes.
  auto shared = std::make_shared<const LifecycleFeedOps>(std::move(ops));
  SharedOps retired;
  {
    std::lock_guard lock(holder().mutex);
    auto& entries = holder().entries;
    if (commIdCollision != nullptr) {
      // Answered under the same lock as the insert, so a second registrant on
      // another thread cannot slip in between the two and have both see none.
      const auto commId = shared->commId;
      // Zero is the unset default, and several feeds carrying it are not two
      // registrants claiming one identity. Only a real id can collide.
      *commIdCollision = commId != 0 &&
          std::any_of(entries.begin(),
                      entries.end(),
                      [commId, owner](const auto& entry) {
                        // A feed whose owner is gone stamps nothing further,
                        // and it may sit here until the next snapshot reaps it.
                        // Reporting it would fail a registrant reusing the id
                        // of a communicator that has left.
                        return entry.first != owner &&
                            entry.second->commId == commId &&
                            !entry.second->alive.expired();
                      });
    }
    const auto existing = std::find_if(
        entries.begin(), entries.end(), [owner](const auto& entry) {
          return entry.first == owner;
        });
    if (existing != entries.end()) {
      retired = std::exchange(existing->second, std::move(shared));
    } else {
      entries.emplace_back(owner, std::move(shared));
    }
  }
  return true;
}

void deregisterLifecycleFeed(const void* owner) {
  std::vector<SharedOps> retired;
  {
    std::lock_guard lock(holder().mutex);
    auto& entries = holder().entries;
    const auto removed = std::stable_partition(
        entries.begin(), entries.end(), [owner](const auto& entry) {
          return entry.first != owner;
        });
    for (auto entry = removed; entry != entries.end(); ++entry) {
      retired.push_back(std::move(entry->second));
    }
    entries.erase(removed, entries.end());
  }
}

std::vector<LifecycleFeedSource> snapshotLifecycleFeeds() {
  std::vector<LifecycleFeedSource> sources;
  std::vector<SharedOps> reaped;
  {
    std::lock_guard lock(holder().mutex);
    auto& entries = holder().entries;
    sources.reserve(entries.size());
    for (const auto& entry : entries) {
      // Locking the owner here is what reaps a feed nobody deregistered.
      if (auto alive = entry.second->alive.lock()) {
        sources.push_back(
            LifecycleFeedSource{
                .alive = std::move(alive), .ops = entry.second});
      }
    }
    // A second walk, and one erase, rather than erasing inside the first: a
    // vector erase shifts everything behind it, so reaping n feeds in the walk
    // would be quadratic with this lock held. Anything the walk above pinned is
    // live and stays live, so the two passes cannot disagree.
    const auto dead = std::stable_partition(
        entries.begin(), entries.end(), [](const auto& entry) {
          return !entry.second->alive.expired();
        });
    for (auto entry = dead; entry != entries.end(); ++entry) {
      reaped.push_back(std::move(entry->second));
    }
    entries.erase(dead, entries.end());
  }
  return sources;
}

std::vector<LifecycleEventRecord> drainAllLifecycleEvents(
    std::chrono::nanoseconds budget) {
  // One allowance for the whole call; a clock read per feed would restart it.
  const auto deadline = std::chrono::steady_clock::now() + budget;

  // The snapshot pins every feed, so none can retire between the flush request
  // and the wait for it.
  const auto sources = snapshotLifecycleFeeds();

  // The ops fields are optional, and calling an empty std::function throws. A
  // feed that cannot answer is treated as one with nothing to say, so a single
  // backend cannot take the drain away from the rest.
  std::vector<std::pair<const LifecycleFeedSource*, uint64_t>> pending;
  pending.reserve(sources.size());
  for (const auto& source : sources) {
    if (source.ops->requestFlush == nullptr) {
      continue;
    }
    try {
      pending.emplace_back(&source, source.ops->requestFlush());
    } catch (...) {
      countFailure();
    }
  }
  for (const auto& [source, generation] : pending) {
    if (source->ops->waitFlush == nullptr) {
      continue;
    }
    // Clamped, not skipped: a feed that already flushed answers at zero, and
    // skipping it would drop its events for nothing.
    const auto remaining = std::max(
        std::chrono::nanoseconds::zero(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            deadline - std::chrono::steady_clock::now()));
    try {
      if (!source->ops->waitFlush(generation, remaining)) {
        // Same consequence as a throw: the drain returns short and silent.
        countFailure();
      }
    } catch (...) {
      countFailure();
    }
  }

  std::vector<LifecycleEventRecord> events;
  for (const auto& source : sources) {
    if (source.ops->drainUnread == nullptr) {
      continue;
    }
    std::vector<LifecycleEventRecord> unread;
    try {
      unread = source.ops->drainUnread();
    } catch (...) {
      countFailure();
      continue;
    }
    events.insert(
        events.end(),
        std::make_move_iterator(unread.begin()),
        std::make_move_iterator(unread.end()));
  }
  // Stable, so two events sharing a timestamp keep the order their feed
  // reported them in.
  std::stable_sort(
      events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timestamp < rhs.timestamp;
      });
  return events;
}

std::optional<CapturedCollDescription> describeCapturedCollective(
    uint64_t commId,
    uint64_t capturedCollId) {
  // An empty answer means both "never captured this id" and "cannot answer at
  // all", and the two are indistinguishable from here, so every feed stamping
  // the id is asked rather than just the first.
  for (const auto& source : snapshotLifecycleFeeds()) {
    if (source.ops->commId != commId ||
        source.ops->describeCaptured == nullptr) {
      continue;
    }
    // describeCaptured is registrant code, and this loop walks every backend in
    // the process. Letting a throw escape would skip the feeds behind it, so
    // one backend's bug would take describe away from the other's
    // communicators. A feed that raises is treated as one with nothing to say.
    try {
      if (auto described = source.ops->describeCaptured(capturedCollId)) {
        return described;
      }
    } catch (...) {
      countFailure();
    }
  }
  return std::nullopt;
}

std::optional<uint64_t> lifecycleLatestCollIdForCommId(uint64_t commId) {
  if (commId == 0) {
    return std::nullopt;
  }
  for (const auto& source : snapshotLifecycleFeeds()) {
    if (source.ops->commId != commId || source.ops->latestCollId == nullptr) {
      continue;
    }
    try {
      return source.ops->latestCollId();
    } catch (...) {
      countFailure();
    }
  }
  return std::nullopt;
}

void resetLifecycleFeedRegistryForTest() {
  decltype(holder().entries) dropped;
  {
    std::lock_guard lock(holder().mutex);
    dropped.swap(holder().entries);
  }
  failureCount().store(0, std::memory_order_relaxed);
}

} // namespace meta::comms::colltrace
