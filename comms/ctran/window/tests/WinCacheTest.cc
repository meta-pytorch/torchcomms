// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>

#include <gtest/gtest.h>

#include "comms/ctran/window/WinCache.h"

using ctran::CtranWin;
using ctran::WinCache;

namespace {

// WinCache never dereferences the window pointer, so opaque non-null addresses
// stand in for real CtranWin objects.
CtranWin* dummyWin(uintptr_t tag) {
  return reinterpret_cast<CtranWin*>(tag);
}

const void* addr(uintptr_t value) {
  return reinterpret_cast<const void*>(value);
}

// Inserts and asserts success, returning the AVL handle for the new entry.
void* insertWin(
    WinCache& cache,
    const void* base,
    std::size_t len,
    CtranWin* win) {
  void* hdl = nullptr;
  EXPECT_EQ(cache.insert(base, len, win, &hdl), commSuccess);
  return hdl;
}

} // namespace

TEST(WinCacheTest, FindExactBase) {
  WinCache cache;
  auto* win = dummyWin(0x1000);
  insertWin(cache, addr(0x10000), 0x100, win);
  EXPECT_EQ(cache.find(addr(0x10000), 0x100), win);
}

TEST(WinCacheTest, FindSubrange) {
  WinCache cache;
  auto* win = dummyWin(0x1000);
  insertWin(cache, addr(0x10000), 0x100, win);
  // A strict subrange inside [0x10000, 0x10100) resolves to the window.
  EXPECT_EQ(cache.find(addr(0x10040), 0x10), win);
}

TEST(WinCacheTest, FindMissEmptyMap) {
  WinCache cache;
  EXPECT_EQ(cache.find(addr(0x10000), 0x10), nullptr);
}

TEST(WinCacheTest, FindMissBeforeAfterAndGap) {
  WinCache cache;
  auto* winA = dummyWin(0x1000);
  auto* winB = dummyWin(0x2000);
  insertWin(cache, addr(0x10000), 0x100, winA);
  insertWin(cache, addr(0x20000), 0x100, winB);
  // Before all ranges.
  EXPECT_EQ(cache.find(addr(0x0FFFF), 0x1), nullptr);
  // In the gap between the two ranges.
  EXPECT_EQ(cache.find(addr(0x10200), 0x10), nullptr);
  // After all ranges.
  EXPECT_EQ(cache.find(addr(0x30000), 0x10), nullptr);
}

TEST(WinCacheTest, FindBoundaryContainmentAndOnePast) {
  WinCache cache;
  auto* win = dummyWin(0x1000);
  insertWin(cache, addr(0x10000), 0x100, win);
  // addr + bytes == base + len is fully contained.
  EXPECT_EQ(cache.find(addr(0x100F0), 0x10), win);
  // One byte past the end is not contained.
  EXPECT_EQ(cache.find(addr(0x100F1), 0x10), nullptr);
}

TEST(WinCacheTest, InsertNullWinRejected) {
  WinCache cache;
  void* hdl = reinterpret_cast<void*>(0xdead);
  EXPECT_EQ(
      cache.insert(addr(0x10000), 0x100, nullptr, &hdl), commInvalidArgument);
  // On error the handle is cleared.
  EXPECT_EQ(hdl, nullptr);
  EXPECT_EQ(cache.find(addr(0x10000), 0x100), nullptr);
}

TEST(WinCacheTest, InsertZeroLenSkipped) {
  WinCache cache;
  auto* win = dummyWin(0x1000);
  void* hdl = reinterpret_cast<void*>(0xdead);
  EXPECT_EQ(cache.insert(addr(0x10000), 0, win, &hdl), commSuccess);
  // Zero-length range is not cached and yields no handle.
  EXPECT_EQ(hdl, nullptr);
  EXPECT_EQ(cache.find(addr(0x10000), 0x1), nullptr);
}

TEST(WinCacheTest, OverlapPartialBothStored) {
  WinCache cache;
  auto* first = dummyWin(0x1000);
  auto* second = dummyWin(0x2000);
  // [0x10000, 0x10100) and [0x10080, 0x10180) partially overlap; both cached.
  insertWin(cache, addr(0x10000), 0x100, first);
  insertWin(cache, addr(0x10080), 0x100, second);
  // A point only in first.
  EXPECT_EQ(cache.find(addr(0x10000), 0x10), first);
  // A point only in second (past first's end): second is cached, so resolvable.
  EXPECT_EQ(cache.find(addr(0x10100), 0x10), second);
  // A point in the overlap resolves to one of the containing windows.
  auto* found = cache.find(addr(0x10080), 0x10);
  EXPECT_TRUE(found == first || found == second);
}

TEST(WinCacheTest, OverlapNestedBothStored) {
  WinCache cache;
  auto* outer = dummyWin(0x1000);
  auto* inner = dummyWin(0x2000);
  insertWin(cache, addr(0x10000), 0x100, outer);
  // [0x10040, 0x10050) is fully inside outer; both cached.
  insertWin(cache, addr(0x10040), 0x10, inner);
  // A sub-range inside inner resolves to a containing window.
  auto* found = cache.find(addr(0x10040), 0x10);
  EXPECT_TRUE(found == outer || found == inner);
  // A point only in outer resolves to outer.
  EXPECT_EQ(cache.find(addr(0x10000), 0x10), outer);
}

TEST(WinCacheTest, OverlapExactDuplicateBothStored) {
  WinCache cache;
  auto* first = dummyWin(0x1000);
  auto* second = dummyWin(0x2000);
  insertWin(cache, addr(0x10000), 0x100, first);
  // Same range registered twice; both cached, a lookup returns one of them.
  insertWin(cache, addr(0x10000), 0x100, second);
  auto* found = cache.find(addr(0x10000), 0x100);
  EXPECT_TRUE(found == first || found == second);
}

TEST(WinCacheTest, AdjacentRangesBothCached) {
  WinCache cache;
  auto* winA = dummyWin(0x1000);
  auto* winB = dummyWin(0x2000);
  insertWin(cache, addr(0x10000), 0x100, winA);
  // base + len == next base: touching but not overlapping, both cached.
  insertWin(cache, addr(0x10100), 0x100, winB);
  EXPECT_EQ(cache.find(addr(0x100F0), 0x10), winA);
  EXPECT_EQ(cache.find(addr(0x10100), 0x10), winB);
}

TEST(WinCacheTest, EraseByHandleIsIdempotent) {
  WinCache cache;
  auto* win = dummyWin(0x1000);
  void* hdl = insertWin(cache, addr(0x10000), 0x100, win);
  // Erase by handle removes the entry.
  cache.erase(hdl);
  EXPECT_EQ(cache.find(addr(0x10000), 0x100), nullptr);
  // Erasing again with the same (stale) handle is a safe no-op.
  cache.erase(hdl);
  EXPECT_EQ(cache.find(addr(0x10000), 0x100), nullptr);
  // A null handle is a safe no-op.
  cache.erase(nullptr);
}

// Two overlapping windows register the same buffer. Erasing one by its handle
// must remove only that window's entry, leaving the other resolvable. Range
// lookup could resolve to the wrong entry, so handle-based erase is required to
// avoid a dangling (use-after-free) entry for the still-alive window.
TEST(WinCacheTest, EraseByHandleAmongOverlapsRemovesOnlyTarget) {
  WinCache cache;
  auto* first = dummyWin(0x1000);
  auto* second = dummyWin(0x2000);
  // Identical range: `first` lands in the AVL tree, `second` in the overlap
  // list. A containment range lookup cannot distinguish them.
  void* firstHdl = insertWin(cache, addr(0x10000), 0x100, first);
  void* secondHdl = insertWin(cache, addr(0x10000), 0x100, second);

  // Erase the list-resident window by handle. A range-based erase would
  // misresolve the containment search to `first` (the tree node) and leave
  // `second` dangling.
  cache.erase(secondHdl);
  // Erase the tree window too; if `second` were still dangling, find() would
  // now resolve to the freed `second`.
  cache.erase(firstHdl);

  EXPECT_EQ(cache.find(addr(0x10000), 0x100), nullptr);
}
