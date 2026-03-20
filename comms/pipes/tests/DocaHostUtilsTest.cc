// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <unistd.h>

#include "comms/pipes/DocaHostUtils.h"

namespace comms::pipes::tests {

static const size_t kPageSize = sysconf(_SC_PAGESIZE);

// Helper: round addr down to page boundary
static uintptr_t pageFloor(uintptr_t addr) {
  return addr & ~(kPageSize - 1);
}

// Helper: round size up to page boundary
static size_t pageCeil(size_t size) {
  return ((size + kPageSize - 1) / kPageSize) * kPageSize;
}

// Helper: verify the alignment invariants that every result must satisfy
static void verifyInvariants(
    const DmaBufAlignment& result,
    uintptr_t allocBase,
    size_t allocSize,
    void* ptr) {
  auto alignedBaseAddr = reinterpret_cast<uintptr_t>(result.alignedBase);
  // Base must be page-aligned and <= allocBase
  EXPECT_EQ(alignedBaseAddr % kPageSize, 0u);
  EXPECT_LE(alignedBaseAddr, allocBase);
  // Size must be page-aligned
  EXPECT_EQ(result.alignedSize % kPageSize, 0u);
  // Aligned range must cover the full allocation
  EXPECT_GE(alignedBaseAddr + result.alignedSize, allocBase + allocSize);
  // dmabufOffset must equal ptr - alignedBase
  EXPECT_EQ(
      result.dmabufOffset, reinterpret_cast<uintptr_t>(ptr) - alignedBaseAddr);
}

TEST(DmaBufAlignmentTest, AlreadyAligned) {
  // Both base and size are already page-aligned
  uintptr_t allocBase = kPageSize * 4;
  size_t allocSize = kPageSize * 2;
  void* ptr = reinterpret_cast<void*>(allocBase);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
  EXPECT_EQ(result.alignedBase, reinterpret_cast<void*>(allocBase));
  EXPECT_EQ(result.alignedSize, allocSize);
  EXPECT_EQ(result.dmabufOffset, 0u);
}

TEST(DmaBufAlignmentTest, BaseNotAligned) {
  // allocBase is mid-page (e.g., cudaMalloc sub-allocation)
  uintptr_t allocBase = kPageSize * 4 + kPageSize / 2;
  size_t allocSize = kPageSize / 4;
  void* ptr = reinterpret_cast<void*>(allocBase);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(result.alignedBase), kPageSize * 4);
  EXPECT_EQ(result.dmabufOffset, kPageSize / 2);
}

TEST(DmaBufAlignmentTest, SizeNotAligned) {
  // Base is aligned but size is tiny (e.g., 8-byte sink buffer)
  uintptr_t allocBase = kPageSize * 4;
  size_t allocSize = 8;
  void* ptr = reinterpret_cast<void*>(allocBase);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
  EXPECT_EQ(result.alignedSize, kPageSize);
  EXPECT_EQ(result.dmabufOffset, 0u);
}

TEST(DmaBufAlignmentTest, BothUnaligned) {
  // Neither base nor size is aligned
  uintptr_t allocBase = kPageSize * 3 + 0x345;
  size_t allocSize = 0x100;
  void* ptr = reinterpret_cast<void*>(allocBase);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
}

TEST(DmaBufAlignmentTest, PtrWithinAllocation) {
  // User pointer is offset within a large allocation
  uintptr_t allocBase = kPageSize * 4;
  size_t allocSize = kPageSize * 8;
  void* ptr = reinterpret_cast<void*>(allocBase + kPageSize * 2);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
  EXPECT_EQ(result.dmabufOffset, kPageSize * 2);
}

TEST(DmaBufAlignmentTest, ExactPageBoundary) {
  // Allocation exactly fills one page
  uintptr_t allocBase = kPageSize * 4;
  size_t allocSize = kPageSize;
  void* ptr = reinterpret_cast<void*>(allocBase);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
  EXPECT_EQ(result.alignedSize, kPageSize);
  EXPECT_EQ(result.dmabufOffset, 0u);
}

TEST(DmaBufAlignmentTest, AllocationSpansPageBoundary) {
  // Allocation starts near end of one page and crosses into next
  uintptr_t allocBase = kPageSize * 5 - kPageSize / 4;
  size_t allocSize = kPageSize / 2; // crosses the boundary

  void* ptr = reinterpret_cast<void*>(allocBase);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
  // Must span at least 2 pages
  EXPECT_EQ(result.alignedSize, 2 * kPageSize);
}

TEST(DmaBufAlignmentTest, LargeAllocationUnalignedBase) {
  // Large allocation with unaligned base — realistic cudaMalloc scenario
  uintptr_t allocBase = kPageSize * 100 + 0x200;
  size_t allocSize = kPageSize * 16;
  void* ptr = reinterpret_cast<void*>(allocBase + kPageSize);

  auto result = compute_dmabuf_alignment(allocBase, allocSize, ptr, kPageSize);

  verifyInvariants(result, allocBase, allocSize, ptr);
}

} // namespace comms::pipes::tests
