// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_AVL_TREE_H_
#define CTRAN_AVL_TREE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>
#include "comms/utils/commSpecs.h"

/**
 * AVL tree.
 * It supports both non-overlapping address ranges and overlapping address
 * ranges. Since most of the usecase would be non-overlapping ranges, we
 * optimize it using an intenral AVL tree structure (TreeElem* root_) which
 * provides O(logN) insert, search, and remove complexity. For any overlapping
 * ranges, we maintain them using a list (std::vector<TreeElem*> list_).
 */
class CtranAvlTree {
 public:
  CtranAvlTree() = default;
  ~CtranAvlTree();

  // Insert a new element into the tree and return the corresponding handle.
  // If the new element range overlaps with any existing element, insertion
  // fails and nullptr is returned.
  void* insert(const void* addr, std::size_t len, void* val);

  // Remove an element from the tree by searching the provided handle.
  commResult_t remove(void* hdl);

  // Search for an element in the tree, handle is returned if found; otherwise
  // return nullptr.
  void* search(const void* addr, std::size_t len) const;

  // Lookup the value of the provided handle.
  void* lookup(void* hdl) const;

  // format a given range to a string with consistent format
  static std::string rangeToString(const void* addr, std::size_t len);

  // Print all elements in the tree into a string.
  std::string toString() const;

  // Get all elements in the tree.
  std::vector<void*> getAllElems() const;

  // Search for all elements whose address range overlaps with [addr, addr+len).
  // Returns handles to all overlapping elements.
  std::vector<void*> searchRange(const void* addr, std::size_t len) const;

  // Get total number of elements.
  size_t size() const;

  // Validate if all elements in the tree is with the correct height.
  bool validateHeight() const;

  // Check all nodes in the tree are balanced (i.e., the height difference of
  // left and right sub trees is <= 1)
  bool isBalanced() const;

 private:
  class TreeElem;
  class TreeElem* root_{nullptr};
  std::vector<TreeElem*> list_;
  // Store all valid handles for fast handle validation.
  // Each handle is a key of the AVL TreeElem or entry in the list.
  std::unordered_set<void*> handles_;
  mutable std::mutex mutex_;
};

#endif
