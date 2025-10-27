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
