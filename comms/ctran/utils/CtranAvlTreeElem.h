// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_AVL_TREE_ELEM_H_
#define CTRAN_AVL_TREE_ELEM_H_

#include <vector>
#include "comms/ctran/utils/CtranAvlTree.h"

/**
 * AVL tree internal implementation.
 * Restriction: It supports only non-overlapping address ranges; otherwise
 * return nullptr handle at insertion.
 */
class CtranAvlTree::TreeElem {
 public:
  TreeElem(uintptr_t addr, std::size_t len, void* val)
      : addr(addr), len(len), val(val) {};
  ~TreeElem(void);

  // Insert a new element into the tree and rebalance internally.
  // If the new element range overlaps with any existing element, no insertion
  // would happen and hdl is set to nullptr. After a successful insertion, it
  // triggers internal rebalance to keep the tree balanced and then returns
  // the new root and the handle of the inserted element.
  TreeElem* insert(uintptr_t addr, std::size_t len, void* val, TreeElem** hdl);

  // Remove the element from the tree.
  // Set removed flag to true if found and removed, otherwise set to false.
  // After a successful removal, it triggers internal rebalance to keep the tree
  // balanced and return the new root.
  TreeElem* remove(TreeElem* e, bool* removed);

  // Append all elements in the subtree under this element from the given indent
  // into a string
  void treeToString(int indent, std::stringstream& ss);

  // Get total number of elements under this root
  size_t size(void);

  // Check if the tree is balanced (i.e., the height difference of left and
  // right sub trees is <= 1)
  bool isBalanced();

  // Validate if all elements in the tree is with the correct height
  bool validateHeight();

  uintptr_t addr{0};
  std::size_t len{0};
  void* val{nullptr};
  TreeElem* left{nullptr};
  TreeElem* right{nullptr};

 private:
  TreeElem* leftRotate(void);
  TreeElem* rightRotate(void);
  TreeElem* balance(void);
  TreeElem* removeSelf(void);
  void updateHeight(void);

  uint32_t height_{1};
};

#endif
