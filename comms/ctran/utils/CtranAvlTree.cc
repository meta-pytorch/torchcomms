// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <iostream>
#include <sstream>

#include "comms/ctran/utils/CtranAvlTree.h"
#include "comms/ctran/utils/CtranAvlTreeElem.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/logger/LogUtils.h"

CtranAvlTree::~CtranAvlTree() {
  if (this->root_) {
    delete this->root_;
  }
}

void* CtranAvlTree::insert(const void* addr_, std::size_t len, void* val) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(addr_);
  CtranAvlTree::TreeElem* hdl = nullptr;
  std::lock_guard<std::mutex> lock(this->mutex_);

  if (this->root_ == nullptr) {
    this->root_ = new CtranAvlTree::TreeElem(addr, len, val);
    hdl = this->root_;
  } else {
    // Try to insert into AVL tree for fast search
    this->root_ = this->root_->insert(addr, len, val, &hdl);

    // If fails to be handled by AVL tree, insert into list as fallback
    if (hdl == nullptr) {
      auto e = new CtranAvlTree::TreeElem(addr, len, val);
      this->list_.push_back(e);
      hdl = e;
    }
  }
  handles_.insert(hdl);
  return hdl;
}

commResult_t CtranAvlTree::remove(void* hdl) {
  CtranAvlTree::TreeElem* e = reinterpret_cast<CtranAvlTree::TreeElem*>(hdl);
  std::lock_guard<std::mutex> lock(this->mutex_);

  auto it = handles_.find(hdl);
  if (it == handles_.end()) {
    CLOGF(
        ERR,
        "CTRAN-AVL-TREE: Trying to remove hdl {} while the handle is not in the cache, likely double freeing",
        (void*)hdl);
    return commInvalidUsage;
  }

  // First try to remove from AVL tree
  bool removed = false;
  this->root_ = this->root_->remove(e, &removed);
  if (removed) {
    delete e;
  } else {
    // Not found in AVL tree, thus try to remove from list
    auto it = std::find(this->list_.begin(), this->list_.end(), e);
    if (it != this->list_.end()) {
      this->list_.erase(
          std::remove(this->list_.begin(), this->list_.end(), e),
          this->list_.end());
      delete e;
    }
  }
  handles_.erase(it);
  return commSuccess;
}

void* CtranAvlTree::search(const void* addr_, std::size_t len) const {
  uintptr_t addr = reinterpret_cast<uintptr_t>(const_cast<void*>(addr_));
  std::lock_guard<std::mutex> lock(this->mutex_);

  // First try to search in AVL tree
  CtranAvlTree::TreeElem* r = this->root_;
  while (r) {
    if (r->addr > addr) {
      r = r->left;
      continue;
    } else if (r->addr + r->len < addr + len) {
      r = r->right;
      continue;
    } else {
      // Found the matching node
      break;
    }
  }

  // If not found in AVL tree, search in list
  if (!r) {
    for (auto e : this->list_) {
      if (e->addr <= addr && e->addr + e->len >= addr + len) {
        r = e;
        break;
      }
    }
  }
  return r;
}

void* CtranAvlTree::lookup(void* hdl) const {
  std::lock_guard<std::mutex> lock(this->mutex_);
  // Invalid handle
  if (handles_.find(hdl) == handles_.end()) {
    return nullptr;
  }
  return reinterpret_cast<CtranAvlTree::TreeElem*>(hdl)->val;
}

std::vector<void*> CtranAvlTree::getAllElems() const {
  std::vector<void*> ret;
  std::deque<CtranAvlTree::TreeElem*> pendingList;
  std::lock_guard<std::mutex> lock(this->mutex_);

  if (this->root_ != nullptr) {
    pendingList.push_back(this->root_);
  }

  // Enqueue all element in the tree via breadth first traversal
  while (!pendingList.empty()) {
    auto r = dequeFront(pendingList);
    ret.push_back(r);

    if (r->left) {
      pendingList.push_back(r->left);
    }
    if (r->right) {
      pendingList.push_back(r->right);
    }
  }

  for (auto e : this->list_) {
    ret.push_back(e);
  }

  return ret;
}

std::vector<void*> CtranAvlTree::getAllElemVals() const {
  std::vector<void*> ret;
  std::deque<CtranAvlTree::TreeElem*> pendingList;
  std::lock_guard<std::mutex> lock(this->mutex_);

  if (this->root_ != nullptr) {
    pendingList.push_back(this->root_);
  }

  // Enqueue all element values in the tree via breadth first traversal
  while (!pendingList.empty()) {
    auto r = dequeFront(pendingList);
    ret.push_back(r->val);

    if (r->left) {
      pendingList.push_back(r->left);
    }
    if (r->right) {
      pendingList.push_back(r->right);
    }
  }

  for (auto e : this->list_) {
    ret.push_back(e->val);
  }

  return ret;
}

std::vector<void*> CtranAvlTree::searchRange(const void* addr, std::size_t len)
    const {
  std::vector<void*> result;
  std::lock_guard<std::mutex> lock(this->mutex_);

  uintptr_t rangeStart = reinterpret_cast<uintptr_t>(addr);
  uintptr_t rangeEnd = rangeStart + len;

  // Helper lambda for recursive tree traversal with pruning
  std::function<void(TreeElem*)> traverse = [&](TreeElem* node) {
    if (node == nullptr) {
      return;
    }

    uintptr_t nodeStart = node->addr;
    uintptr_t nodeEnd = node->addr + node->len;

    // Check if this node overlaps with the search range
    bool overlaps = (nodeStart < rangeEnd) && (nodeEnd > rangeStart);
    if (overlaps) {
      result.push_back(node);
    }

    // Prune left subtree: only traverse if search range could overlap with
    // nodes having smaller addresses. Since segments don't overlap, all nodes
    // in the left subtree have ranges that end before nodeStart. So we only
    // need to check left if rangeStart < nodeStart.
    if (node->left && rangeStart < nodeStart) {
      traverse(node->left);
    }

    // Prune right subtree: only traverse if search range extends past this
    // node's start (could overlap with larger addresses)
    if (node->right && rangeEnd > nodeStart) {
      traverse(node->right);
    }
  };

  // Traverse the AVL tree with pruning
  traverse(this->root_);

  // Also check the fallback list for overlapping ranges
  for (auto e : this->list_) {
    uintptr_t nodeStart = e->addr;
    uintptr_t nodeEnd = e->addr + e->len;
    bool overlaps = (nodeStart < rangeEnd) && (nodeEnd > rangeStart);
    if (overlaps) {
      result.push_back(e);
    }
  }

  return result;
}

std::string CtranAvlTree::rangeToString(const void* addr, std::size_t len) {
  std::stringstream ss;
  ss << "[" << addr << ", " << len << "]";
  return ss.str();
}

std::string CtranAvlTree::toString() const {
  std::stringstream ss;
  std::lock_guard<std::mutex> lock(this->mutex_);

  ss << "Internal AVL tree:" << std::endl;
  if (this->root_ != nullptr) {
    this->root_->treeToString(0, ss);
    ss << std::endl;
  }

  ss << "Internal list:" << std::endl;
  bool first = true;
  for (auto e : this->list_) {
    if (!first) {
      ss << ",";
    }
    ss << this->rangeToString(reinterpret_cast<const void*>(e->addr), e->len);
    first = false;
  }
  ss << std::endl;
  return ss.str();
}

size_t CtranAvlTree::size() const {
  std::lock_guard<std::mutex> lock(this->mutex_);
  return handles_.size();
}

bool CtranAvlTree::validateHeight() const {
  std::lock_guard<std::mutex> lock(this->mutex_);
  // If root is null, it is balanced
  if (!this->root_) {
    return true;
  }

  // Check if all tree elements are with correct height
  return this->root_->validateHeight();
}

bool CtranAvlTree::isBalanced() const {
  std::lock_guard<std::mutex> lock(this->mutex_);
  // If root is null, it is balanced
  if (!this->root_) {
    return true;
  }

  // Check if the tree is balanced
  return this->root_->isBalanced();
}
