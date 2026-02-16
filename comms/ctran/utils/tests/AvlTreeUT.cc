// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_set>

#include "comms/ctran/utils/CtranAvlTree.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"

class Range {
 public:
  Range(uintptr_t addr, size_t len) : addr(addr), len(len) {};
  ~Range() = default;

  bool isOverlap(Range& other) {
    if (addr + len < other.addr || other.addr + other.len < addr) {
      return false;
    } else {
      return true;
    }
  }

 public:
  uintptr_t addr{0};
  size_t len{0};
};

class RangeRegistration {
 public:
  RangeRegistration(Range& range, void* val) {
    this->addr = reinterpret_cast<void*>(range.addr);
    this->len = range.len;
    this->val = val;
  }
  ~RangeRegistration() = default;

 public:
  void* addr{nullptr};
  size_t len{0};
  void* val{nullptr};
  void* hdl{nullptr};
};

#define MAX_BUF_LEN (1024)

static inline bool assignNonOverlapRange(
    class Range& range,
    std::vector<class Range>& existingRanges) {
  int maxTry = 10000;
  while (maxTry--) {
    range.addr = reinterpret_cast<uintptr_t>(
        ((rand() % UINTPTR_MAX) / MAX_BUF_LEN) * MAX_BUF_LEN);
    range.len = rand() % MAX_BUF_LEN;

    bool overlap = false;
    for (auto& r : existingRanges) {
      if (range.isOverlap(r)) {
        overlap = true;
        break; // retry
      }
    }

    // found, return
    if (!overlap) {
      return true;
    }
  }

  // failed to find non-overlap range
  return false;
}

static inline bool assignOverlapRange(
    class Range& range,
    std::vector<class Range>& existingRanges) {
  int maxTry = 10000;
  while (maxTry--) {
    int r = rand() % existingRanges.size(); // pick a random existing range
    range.addr = existingRanges[r].addr +
        rand() % MAX_BUF_LEN / 2; // pick a random starting offset
    range.len = rand() % MAX_BUF_LEN; // pick a random length
    if (range.isOverlap(existingRanges[r])) {
      return true;
    }
  }
  // failed to find non-overlap range
  return false;
}

class CtranUtilsAvlTreeTest : public ::testing::Test {
 public:
  CtranUtilsAvlTreeTest() = default;

  // Generate a list of non-overlapping buffer ranges, since this is the
  // majority of the use case. A test can specify hint to insert some
  // overlapping ranges at random position, 0 means no overlapping ranges. It
  // returns the actual number of generated overlapping ranges by updating
  // numOverlapsHint.
  void genBufRanges(const int maxNumBufs, int* numOverlapsHint) {
    std::unordered_set<int> overlapIdx;

    // avoid resizing copy
    bufRanges.reserve(maxNumBufs);
    // clear any previously generated ranges
    bufRanges.clear();

    // pick some random indexes to fill with overlapping ranges
    while (overlapIdx.size() < *numOverlapsHint) {
      int idx = rand() % maxNumBufs;
      // Skip 0th index, since cannot use any existing range to generate an
      // overlapping one
      if (idx == 0) {
        continue;
      }
      overlapIdx.insert(idx);
    }

    int numOverlaps = overlapIdx.size();
    for (int i = 0; i < maxNumBufs; i++) {
      auto range = Range(0, 0);
      if (overlapIdx.find(i) != overlapIdx.end()) {
        // If failed to assign an overlapping range at given index, just leave
        // the non-overlapping range as is
        if (!assignOverlapRange(range, bufRanges)) {
          numOverlaps--;
        }
      } else {
        // Assign non-overlapping range in other index; if fails, just leave the
        // overlapping range as is
        if (!assignNonOverlapRange(range, bufRanges)) {
          numOverlaps++;
        }
      }
      bufRanges.push_back(range);
    }

    if (*numOverlapsHint != numOverlaps) {
      printf(
          "WARNING: Only mixed %d overlapping ranges in %ld total ranges, but planed %d/%d\n",
          numOverlaps,
          this->bufRanges.size(),
          *numOverlapsHint,
          maxNumBufs);
    }
    *numOverlapsHint = numOverlaps;
  }

  void SetUp() override {
    // WARN: for now this will default to display WARN level+ messages only.
    // Change NcclLoggerInitConfig.logLevel to change
    NcclLogger::init();
  }

  void TearDown() override {
    NcclLogger::close();
  }

 public:
  std::vector<Range> bufRanges;
};

TEST_F(CtranUtilsAvlTreeTest, MixedRangeInsertRemoveFromHead) {
  auto tree = std::make_unique<CtranAvlTree>();

  const int maxNumBufs = 10000, numOverlaps = 200;
  int numOverlapsHint = numOverlaps;
  this->genBufRanges(maxNumBufs, &numOverlapsHint);

  // Check insertion
  std::unordered_map<void*, std::unique_ptr<RangeRegistration>>
      hdlToRangeRegistMap;
  std::vector<void*> hdlList;
  for (int i = 0; i < maxNumBufs; i++) {
    auto rangeRegist = std::make_unique<RangeRegistration>(
        this->bufRanges[i], reinterpret_cast<void*>(static_cast<uintptr_t>(i)));

    rangeRegist->hdl = tree->insert(
        reinterpret_cast<void*>(rangeRegist->addr),
        rangeRegist->len,
        rangeRegist->val);
    ASSERT_NE(rangeRegist->hdl, nullptr);
    ASSERT_EQ(tree->validateHeight(), true);
    ASSERT_EQ(tree->isBalanced(), true);

    auto hdl = rangeRegist->hdl;
    hdlList.push_back(hdl);
    hdlToRangeRegistMap[hdl] = std::move(rangeRegist);
  }

  ASSERT_EQ(tree->size(), hdlToRangeRegistMap.size());

  // Check allElem returns list of handles and each is correct
  auto allElems = tree->getAllElems();
  ASSERT_EQ(allElems.size(), hdlToRangeRegistMap.size());
  for (auto& hdl : allElems) {
    // Expect matchs with hdlToRangeRegistMap
    auto it = hdlToRangeRegistMap.find(hdl);
    EXPECT_TRUE(it != hdlToRangeRegistMap.end());

    // Expect the lookup result also matchs record in hdlToRangeRegistMap
    auto& rangeRegist = it->second;
    void* val = tree->lookup(hdl);
    EXPECT_EQ(val, rangeRegist->val);
  }

  // Check removal from head
  size_t remaining = hdlList.size();
  for (int i = 0; i < hdlList.size() - 1; i++) {
    tree->remove(hdlList[i]);

    ASSERT_EQ(tree->size(), --remaining);
    ASSERT_EQ(tree->validateHeight(), true);

    // FIXME: it is known issue that the tree may be imbalanced after removal
    // ASSERT_EQ(tree->isBalanced(), true);
  }
}

TEST_F(CtranUtilsAvlTreeTest, MixedRangeInsertRemoveFromEnd) {
  auto tree = std::make_unique<CtranAvlTree>();

  const int maxNumBufs = 10000, numOverlaps = 200;
  int numOverlapsHint = numOverlaps;
  this->genBufRanges(maxNumBufs, &numOverlapsHint);

  // Check insertion
  std::unordered_map<void*, std::unique_ptr<RangeRegistration>>
      hdlToRangeRegistMap;
  std::vector<void*> hdlList;
  for (int i = 0; i < maxNumBufs; i++) {
    auto rangeRegist = std::make_unique<RangeRegistration>(
        this->bufRanges[i], reinterpret_cast<void*>(static_cast<uintptr_t>(i)));

    rangeRegist->hdl = tree->insert(
        reinterpret_cast<void*>(rangeRegist->addr),
        rangeRegist->len,
        rangeRegist->val);
    ASSERT_NE(rangeRegist->hdl, nullptr);
    ASSERT_EQ(tree->validateHeight(), true);
    ASSERT_EQ(tree->isBalanced(), true);

    auto hdl = rangeRegist->hdl;
    hdlList.push_back(hdl);
    hdlToRangeRegistMap[hdl] = std::move(rangeRegist);
  }

  ASSERT_EQ(tree->size(), hdlToRangeRegistMap.size());

  // Check allElem returns list of handles and each is correct
  auto allElems = tree->getAllElems();
  ASSERT_EQ(allElems.size(), hdlToRangeRegistMap.size());
  for (auto& hdl : allElems) {
    // Expect matchs with hdlToRangeRegistMap
    auto it = hdlToRangeRegistMap.find(hdl);
    EXPECT_TRUE(it != hdlToRangeRegistMap.end());

    // Expect the lookup result also matchs record in hdlToRangeRegistMap
    auto& rangeRegist = it->second;
    void* val = tree->lookup(hdl);
    EXPECT_EQ(val, rangeRegist->val);
  }

  // Check removal from end
  size_t remaining = allElems.size();
  for (int i = hdlList.size() - 1; i >= 0; i--) {
    tree->remove(hdlList[i]);

    ASSERT_EQ(tree->size(), --remaining);
    ASSERT_EQ(tree->validateHeight(), true);

    // FIXME: it is known issue that the tree may be imbalanced after removal
    // ASSERT_EQ(tree->isBalanced(), true);
  }
}

// Test only non overlap ranges since Pytorch ensures all registered buffers are
// non-overlapping.
TEST_F(CtranUtilsAvlTreeTest, SearchNonOverlapRanges) {
  auto tree = std::make_unique<CtranAvlTree>();

  // Generate random ranges
  const int maxNumBufs = 10000, numOverlaps = 0;
  int numOverlapsHint = numOverlaps;
  this->genBufRanges(maxNumBufs, &numOverlapsHint);

  // Insert all ranges
  std::vector<RangeRegistration> rangeRegistList;
  for (int i = 0; i < maxNumBufs; i++) {
    auto rangeRegist = RangeRegistration(
        this->bufRanges[i], reinterpret_cast<void*>(static_cast<uintptr_t>(i)));

    rangeRegist.hdl = tree->insert(
        reinterpret_cast<void*>(rangeRegist.addr),
        rangeRegist.len,
        rangeRegist.val);

    rangeRegistList.push_back(rangeRegist);
  }

  // Search randomly and check search result
  const int searchIter = 100000;
  std::unordered_set<int> idxSet;
  for (int i = 0; i < searchIter; i++) {
    void *hdl, *val;
    int idx = rand() % maxNumBufs;
    idxSet.insert(idx);
    hdl = tree->search(rangeRegistList[idx].addr, rangeRegistList[idx].len);
    val = tree->lookup(hdl);

    ASSERT_EQ(rangeRegistList[idx].hdl, hdl);
    ASSERT_EQ(rangeRegistList[idx].val, val);
  }

  // Remove all ranges
  for (int i = 0; i < rangeRegistList.size(); i++) {
    tree->remove(rangeRegistList[i].hdl);
  }
  rangeRegistList.clear();
}

// Test ToString
TEST_F(CtranUtilsAvlTreeTest, ToString) {
  auto tree = std::make_unique<CtranAvlTree>();

  // Generate random ranges
  const int maxNumBufs = 1000, numOverlaps = 200;
  int numOverlapsHint = numOverlaps;
  this->genBufRanges(maxNumBufs, &numOverlapsHint);

  // Insert all ranges
  std::vector<RangeRegistration> rangeRegistList;
  for (int i = 0; i < maxNumBufs; i++) {
    auto rangeRegist = RangeRegistration(
        this->bufRanges[i], reinterpret_cast<void*>(static_cast<uintptr_t>(i)));

    rangeRegist.hdl = tree->insert(
        reinterpret_cast<void*>(rangeRegist.addr),
        rangeRegist.len,
        rangeRegist.val);

    rangeRegistList.push_back(rangeRegist);
  }

  // Get the string representation of the tree
  std::string treeString = tree->toString();

  // Randomly search some ranges in the returned string
  const int searchIter = 100000;
  for (int i = 0; i < searchIter; i++) {
    int idx = rand() % maxNumBufs;
    std::string rangeStr = CtranAvlTree::rangeToString(
        rangeRegistList[idx].addr, rangeRegistList[idx].len);
    ASSERT_TRUE(treeString.find(rangeStr) != std::string::npos)
        << "Cannot find " << rangeStr << std::endl;
  }

  // Remove all ranges
  for (int i = 0; i < rangeRegistList.size(); i++) {
    tree->remove(rangeRegistList[i].hdl);
  }
  rangeRegistList.clear();
}

TEST_F(CtranUtilsAvlTreeTest, DoubleRemove) {
  auto tree = std::make_unique<CtranAvlTree>();

  const int maxNumBufs = 1, numOverlaps = 0;
  int numOverlapsHint = numOverlaps;
  this->genBufRanges(maxNumBufs, &numOverlapsHint);

  // Check insertion
  RangeRegistration rangeRegist = RangeRegistration(
      this->bufRanges[0], reinterpret_cast<void*>(static_cast<uintptr_t>(0)));

  rangeRegist.hdl = tree->insert(
      reinterpret_cast<void*>(rangeRegist.addr),
      rangeRegist.len,
      rangeRegist.val);
  ASSERT_NE(rangeRegist.hdl, nullptr);
  ASSERT_EQ(tree->validateHeight(), true);
  ASSERT_EQ(tree->isBalanced(), true);

  ASSERT_EQ(tree->size(), 1);

  // Expect lookup is fine
  ASSERT_EQ(tree->lookup(rangeRegist.hdl), rangeRegist.val);

  // Check removal from head
  auto res = tree->remove(rangeRegist.hdl);
  ASSERT_EQ(res, commSuccess);
  ASSERT_EQ(tree->size(), 0);
  ASSERT_EQ(tree->validateHeight(), true);

  // Expect lookup should not crash and return nullptr with an invalid handle
  ASSERT_EQ(tree->lookup(rangeRegist.hdl), nullptr);

  // Check double removal
  res = tree->remove(rangeRegist.hdl);
  ASSERT_EQ(res, commInvalidUsage);
  ASSERT_EQ(tree->size(), 0);
}

// Test searchRange with 5 contiguous 20MB segments.
// Query [30MB, 70MB) should find segments 2, 3, 4
// Segments layout:
//   Segment 1: [0MB,  20MB)
//   Segment 2: [20MB, 40MB)  <- overlaps [30MB, 70MB) at [30MB, 40MB)
//   Segment 3: [40MB, 60MB)  <- fully inside [30MB, 70MB)
//   Segment 4: [60MB, 80MB)  <- overlaps [30MB, 70MB) at [60MB, 70MB)
//   Segment 5: [80MB, 100MB)
TEST_F(CtranUtilsAvlTreeTest, SearchRangePartialRange) {
  auto tree = std::make_unique<CtranAvlTree>();

  constexpr size_t MB = 1024 * 1024;
  constexpr size_t segmentSize = 20 * MB;
  constexpr int numSegments = 5;

  // Insert 5 contiguous 20MB segments
  std::vector<void*> handles(numSegments);
  for (int i = 0; i < numSegments; i++) {
    uintptr_t addr = i * segmentSize;
    void* val = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
    handles[i] = tree->insert(reinterpret_cast<void*>(addr), segmentSize, val);
    ASSERT_NE(handles[i], nullptr);
  }
  ASSERT_EQ(tree->size(), numSegments);

  // Query range [30MB, 70MB)
  uintptr_t rangeStart = 30 * MB;
  uintptr_t rangeEnd = 70 * MB;
  auto overlapping = tree->searchRange(
      reinterpret_cast<void*>(rangeStart), rangeEnd - rangeStart);

  // Should find exactly 3 segments
  ASSERT_EQ(overlapping.size(), 3);

  // Verify we found the correct handles
  std::unordered_set<void*> expectedHandles = {
      handles[1], handles[2], handles[3]};
  std::unordered_set<void*> foundHandles(
      overlapping.begin(), overlapping.end());
  EXPECT_EQ(expectedHandles, foundHandles);

  // Cleanup
  for (auto hdl : handles) {
    tree->remove(hdl);
  }
}

// Test searchRange with 5 contiguous 20MB segments.
// Query [20MB, 80MB) should find segments 2, 3, 4
// This tests exact boundary alignment where:
//   - rangeStart (20MB) equals segment 2's start
//   - rangeEnd (80MB) equals segment 5's start (exclusive)
// Segments layout:
//   Segment 1: [0MB,  20MB)  <- ends at rangeStart, no overlap
//   Segment 2: [20MB, 40MB)  <- starts at rangeStart, overlaps
//   Segment 3: [40MB, 60MB)  <- fully inside
//   Segment 4: [60MB, 80MB)  <- ends at rangeEnd, overlaps
//   Segment 5: [80MB, 100MB) <- starts at rangeEnd, no overlap (half-open)
TEST_F(CtranUtilsAvlTreeTest, SearchRangeExactBoundary) {
  auto tree = std::make_unique<CtranAvlTree>();

  constexpr size_t MB = 1024 * 1024;
  constexpr size_t segmentSize = 20 * MB;
  constexpr int numSegments = 5;

  // Insert 5 contiguous 20MB segments
  std::vector<void*> handles(numSegments);
  for (int i = 0; i < numSegments; i++) {
    uintptr_t addr = i * segmentSize;
    void* val = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
    handles[i] = tree->insert(reinterpret_cast<void*>(addr), segmentSize, val);
    ASSERT_NE(handles[i], nullptr);
  }
  ASSERT_EQ(tree->size(), numSegments);

  // Query range [20MB, 80MB) - exact segment boundaries
  // Should find segments 2, 3, 4
  uintptr_t rangeStart = 20 * MB;
  uintptr_t rangeEnd = 80 * MB;
  auto overlapping = tree->searchRange(
      reinterpret_cast<void*>(rangeStart), rangeEnd - rangeStart);

  // Should find exactly 3 segments
  ASSERT_EQ(overlapping.size(), 3);

  // Verify we found the correct handles
  std::unordered_set<void*> expectedHandles = {
      handles[1], handles[2], handles[3]};
  std::unordered_set<void*> foundHandles(
      overlapping.begin(), overlapping.end());
  EXPECT_EQ(expectedHandles, foundHandles);

  // Cleanup
  for (auto hdl : handles) {
    tree->remove(hdl);
  }
}

// Test that searchRange correctly prunes the left subtree when the search range
// starts after a node's start address. This leverages the fact that segments
// don't overlap, so all nodes in the left subtree must end before nodeStart.
//
// Tree structure (ordered by address):
//   Segment 1: [100, 150)   <- left child
//   Segment 2: [200, 250)   <- root
//   Segment 3: [300, 350)   <- right child
//
// Query [220, 280): starts after root's start (200), so left subtree should be
// pruned. Only segments 2 and 3 should be checked/returned.
TEST_F(CtranUtilsAvlTreeTest, SearchRangeLeftSubtreePruning) {
  auto tree = std::make_unique<CtranAvlTree>();

  // Insert segments in order that creates a balanced tree
  // Insert middle first (becomes root), then left, then right
  void* hdl2 = tree->insert(
      reinterpret_cast<void*>(200), 50, reinterpret_cast<void*>(2));
  void* hdl1 = tree->insert(
      reinterpret_cast<void*>(100), 50, reinterpret_cast<void*>(1));
  void* hdl3 = tree->insert(
      reinterpret_cast<void*>(300), 50, reinterpret_cast<void*>(3));

  ASSERT_EQ(tree->size(), 3);

  // Query [220, 280) - starts at 220 which is > root's start (200)
  // Since segments don't overlap, segment 1 [100, 150) cannot overlap with
  // [220, 280) because it ends at 150 < 200 (root's start).
  // The pruning condition rangeStart < nodeStart (220 < 200 = false) should
  // skip the left subtree entirely.
  auto overlapping =
      tree->searchRange(reinterpret_cast<void*>(220), 60); // [220, 280)

  // Should find segment 2 [200, 250) which overlaps at [220, 250)
  // Should NOT find segment 1 [100, 150) - no overlap
  // Should NOT find segment 3 [300, 350) - no overlap
  ASSERT_EQ(overlapping.size(), 1);

  void* foundVal = tree->lookup(overlapping[0]);
  EXPECT_EQ(foundVal, reinterpret_cast<void*>(2));

  // Query [350, 400) - beyond all segments
  auto noOverlap =
      tree->searchRange(reinterpret_cast<void*>(350), 50); // [350, 400)
  EXPECT_EQ(noOverlap.size(), 0);

  // Query [0, 50) - before all segments
  auto beforeAll = tree->searchRange(reinterpret_cast<void*>(0), 50); // [0, 50)
  EXPECT_EQ(beforeAll.size(), 0);

  // Cleanup
  tree->remove(hdl1);
  tree->remove(hdl2);
  tree->remove(hdl3);
}
