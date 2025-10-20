// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/CtranAvlTreeElem.h"

#include <unistd.h>
#include <algorithm>
#include <deque>
#include <iostream>
#include <sstream>

#include "comms/ctran/utils/ExtUtils.h"

CtranAvlTree::TreeElem::~TreeElem(void) {
  if (this->left) {
    delete this->left;
  }
  if (this->right) {
    delete this->right;
  }
}

void CtranAvlTree::TreeElem::updateHeight(void) {
  uint32_t lHeight, rHeight;

  if (this->left) {
    lHeight = this->left->height_;
  } else {
    lHeight = 0;
  }

  if (this->right) {
    rHeight = this->right->height_;
  } else {
    rHeight = 0;
  }

  this->height_ = std::max(lHeight, rHeight) + 1;
}

void CtranAvlTree::TreeElem::treeToString(int indent, std::stringstream& ss) {
  if (indent && this->left == nullptr && this->right == nullptr) {
    return;
  }

  // print indent
  for (int i = 0; i < indent; i++) {
    ss << "    ";
  }

  // print myself and children
  ss << CtranAvlTree::rangeToString(
            reinterpret_cast<const void*>(this->addr), this->len)
     << "(" << this->height_ << ")";
  ss << " L ";
  if (this->left) {
    ss << CtranAvlTree::rangeToString(
              reinterpret_cast<const void*>(this->left->addr), this->left->len)
       << "(" << this->left->height_ << ")";
  } else {
    ss << "(null)";
  }
  ss << " R ";
  if (this->right) {
    ss << CtranAvlTree::rangeToString(
              reinterpret_cast<const void*>(this->right->addr),
              this->right->len)
       << "(" << this->right->height_ << ")";
  } else {
    ss << "(null)";
  }
  ss << std::endl;

  // left subtree
  if (this->left) {
    this->left->treeToString(indent + 1, ss);
  }

  // right subtree
  if (this->right) {
    this->right->treeToString(indent + 1, ss);
  }
}

CtranAvlTree::TreeElem* CtranAvlTree::TreeElem::leftRotate(void) {
  if (this->right == nullptr) {
    return nullptr;
  }

  CtranAvlTree::TreeElem* newroot = this->right;
  this->right = newroot->left;
  this->updateHeight();

  newroot->left = this;
  newroot->updateHeight();

  return newroot;
}

CtranAvlTree::TreeElem* CtranAvlTree::TreeElem::rightRotate(void) {
  if (this->left == nullptr) {
    return nullptr;
  }

  CtranAvlTree::TreeElem* newroot = this->left;
  this->left = newroot->right;
  this->updateHeight();

  newroot->right = this;
  newroot->updateHeight();

  return newroot;
}

CtranAvlTree::TreeElem* CtranAvlTree::TreeElem::balance(void) {
  uint32_t leftHeight = this->left ? this->left->height_ : 0;
  uint32_t rightHeight = this->right ? this->right->height_ : 0;

  if (leftHeight > rightHeight + 1) {
    uint32_t leftLeftHeight = this->left->left ? this->left->left->height_ : 0;
    uint32_t leftRightHeight =
        this->left->right ? this->left->right->height_ : 0;

    if (leftLeftHeight > leftRightHeight) {
      return this->rightRotate();
    } else {
      this->left = this->left->leftRotate();
      return this->rightRotate();
    }
  } else if (rightHeight > leftHeight + 1) {
    uint32_t rightLeftHeight =
        this->right->left ? this->right->left->height_ : 0;
    uint32_t rightRightHeight =
        this->right->right ? this->right->right->height_ : 0;

    if (rightRightHeight > rightLeftHeight) {
      return this->leftRotate();
    } else {
      this->right = this->right->rightRotate();
      return this->leftRotate();
    }
  }

  return this;
}

CtranAvlTree::TreeElem* CtranAvlTree::TreeElem::insert(
    uintptr_t addr,
    std::size_t len,
    void* val,
    TreeElem** hdl) {
  CtranAvlTree::TreeElem* newroot = this;
  *hdl = nullptr;

  // If we have overlapping buffers, return immediately with nullptr handle
  if ((this->addr >= addr && this->addr < addr + len) ||
      (this->addr <= addr && this->addr + this->len > addr)) {
    return newroot;
  }

  if (this->addr > addr) {
    // Insert into left side and rebalance left subtree
    if (this->left == nullptr) {
      *hdl = new CtranAvlTree::TreeElem(addr, len, val);
      this->left = *hdl;
    } else {
      this->left = this->left->insert(addr, len, val, hdl);
    }
    this->left->updateHeight();
    this->updateHeight();
    newroot = this->balance();
  } else if (this->addr < addr) {
    // Insert into right side and rebalance right subtree
    if (this->right == nullptr) {
      *hdl = new CtranAvlTree::TreeElem(addr, len, val);
      this->right = *hdl;
    } else {
      this->right = this->right->insert(addr, len, val, hdl);
    }
    this->right->updateHeight();
    this->updateHeight();
    newroot = this->balance();
  }

  return newroot;
}

CtranAvlTree::TreeElem* CtranAvlTree::TreeElem::removeSelf(void) {
  CtranAvlTree::TreeElem* newroot;
  std::deque<CtranAvlTree::TreeElem*> updateElems;

  if (this->left == nullptr && this->right == nullptr) {
    newroot = nullptr;
  } else if (this->left == nullptr) {
    newroot = this->right;
  } else if (this->right == nullptr) {
    newroot = this->left;
  } else if (this->left->height_ >= this->right->height_) {
    // Left subtree is higher, find the largest node from the left as the new
    // root

    if (!this->left->right) {
      // - Left subtree doesn't have a right subtree, thus it is the largest
      this->left->right = this->right;
      newroot = this->left;
    } else {
      // - Find the right-most node which is the largest
      CtranAvlTree::TreeElem* tmp = this->left->right;
      CtranAvlTree::TreeElem* prev = this->left;
      updateElems.push_back(prev);
      while (tmp->right) {
        prev = tmp;
        tmp = tmp->right;
        updateElems.push_back(prev);
      }
      newroot = tmp;
      prev->right = tmp->removeSelf();
      newroot->left = this->left;
      newroot->right = this->right;
    }
  } else {
    // Right subtree is higher, find the smallest node from the right subtree as
    // the new root

    if (!this->right->left) {
      // - Right subtree doesn't have a left subtree, thus it is the smallest
      this->right->left = this->left;
      newroot = this->right;
    } else {
      // - Find the left-most node which is the smallest
      CtranAvlTree::TreeElem* tmp = this->right->left;
      CtranAvlTree::TreeElem* prev = this->right;
      updateElems.push_back(prev);
      while (tmp->left) {
        prev = tmp;
        tmp = tmp->left;
        updateElems.push_back(prev);
      }
      newroot = tmp;
      prev->left = tmp->removeSelf();
      newroot->left = this->left;
      newroot->right = this->right;
    }
  }

  // this node can be dislinked now
  this->left = nullptr;
  this->right = nullptr;

  // Update height for the updated path from bottom to top
  while (!updateElems.empty()) {
    auto temp = updateElems.back();
    updateElems.pop_back();
    temp->updateHeight();
  }

  // Last update newroot
  if (newroot) {
    newroot->updateHeight();
  }

  return newroot;
}

CtranAvlTree::TreeElem* CtranAvlTree::TreeElem::remove(
    CtranAvlTree::TreeElem* e,
    bool* removed) {
  std::deque<CtranAvlTree::TreeElem*> updateElems;
  CtranAvlTree::TreeElem* newroot = this;
  bool removed_ = false;

  if (this == e) {
    // Remove root
    newroot = this->removeSelf();
    removed_ = true;
  } else {
    // Depth-first search a node from subtree and remove;
    // exit if removed, or not found but have traversed to bottom
    CtranAvlTree::TreeElem* temp = this;
    while (temp && !removed_) {
      // Found a matching node, remove it
      if (temp->left == e) {
        updateElems.push_back(temp);
        temp->left = temp->left->removeSelf();
        removed_ = true;
      } else if (temp->right == e) {
        updateElems.push_back(temp);
        temp->right = temp->right->removeSelf();
        removed_ = true;
      }
      // Not found, search the child subtree
      else if (temp->addr < e->addr) {
        updateElems.push_back(temp);
        temp = temp->right;
      } else {
        updateElems.push_back(temp);
        temp = temp->left;
      }
    }
  }

  if (removed_) {
    // Update height for the updated path from bottom to top
    while (!updateElems.empty()) {
      auto temp = updateElems.back();
      updateElems.pop_back();
      temp->updateHeight();
    }
  }

  // FIXEME: Tree may be unbalanced after removal, we need rebalance it
  // But we don't have a good way to do it unless balance all recursively
  // which can be expensive. Skip it for now to avoid heavy removal
  // overhead
  *removed = removed_;
  return newroot;
}

size_t CtranAvlTree::TreeElem::size() {
  size_t size = 0;
  std::deque<CtranAvlTree::TreeElem*> pendingElems;

  // Count total amount of nodes via breadth first traversal from top
  pendingElems.push_back(this);
  while (!pendingElems.empty()) {
    auto temp = dequeFront(pendingElems);
    size++;

    if (temp->left) {
      pendingElems.push_back(temp->left);
    }
    if (temp->right) {
      pendingElems.push_back(temp->right);
    }
  }

  return size;
}

bool CtranAvlTree::TreeElem::isBalanced() {
  bool balanced = true;
  std::deque<CtranAvlTree::TreeElem*> pendingElems;

  // validate balance of every subtree with two children via breadth
  // first traversal from top
  pendingElems.push_back(this);
  while (!pendingElems.empty()) {
    auto temp = dequeFront(pendingElems);

    int lHeight = 0, rHeight = 0;
    if (temp->left) {
      pendingElems.push_back(temp->left);
      lHeight = temp->left->height_;
    }
    if (temp->right) {
      pendingElems.push_back(temp->right);
      rHeight = temp->right->height_;
    }

    balanced &= (abs(lHeight - rHeight) <= 1);
    if (!balanced) {
      break;
    }
  }

  return balanced;
}

bool CtranAvlTree::TreeElem::validateHeight() {
  bool correct = true;
  std::deque<CtranAvlTree::TreeElem*> pendingElems;

  // validate height correctness of every node with two children via breadth
  // first traversal from top
  pendingElems.push_back(this);
  while (!pendingElems.empty()) {
    auto temp = dequeFront(pendingElems);

    int lHeight = 0, rHeight = 0;
    if (temp->left) {
      pendingElems.push_back(temp->left);
      lHeight = temp->left->height_;
    }
    if (temp->right) {
      pendingElems.push_back(temp->right);
      rHeight = temp->right->height_;
    }

    correct &= (temp->height_ == std::max(lHeight, rHeight) + 1);
    if (!correct) {
      break;
    }
  }
  return correct;
}
