// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

namespace uniflow {

/// A move-only, type-erased callable wrapper for void() signatures.
///
/// Unlike std::function, Func supports move-only callables (e.g., lambdas
/// capturing std::unique_ptr). Uses small-buffer optimization to avoid
/// heap allocation for callables up to kInlineSize bytes.
///
/// Similar to folly::Function<void()> but standalone with no dependencies.
class Func {
 public:
  /// Small-buffer size: callables up to this size are stored inline.
  /// 48 bytes accommodates most lambda captures without heap allocation.
  static constexpr size_t kInlineSize = 6 * sizeof(void*);

  Func() noexcept = default;

  /* implicit */ Func(std::nullptr_t) noexcept {}

  /// Construct from any callable that is invocable as void().
  template <typename F>
    requires(
        !std::same_as<std::decay_t<F>, Func> &&
        std::is_nothrow_invocable_r_v<void, std::decay_t<F>>)
  /* implicit */ Func(F&& f) {
    using Decay = std::decay_t<F>;
    if constexpr (
        sizeof(Decay) <= kInlineSize &&
        alignof(Decay) <= alignof(std::max_align_t) &&
        std::is_nothrow_move_constructible_v<Decay>) {
      ::new (&storage_) Decay(std::forward<F>(f));
      vtable_ = &vtableFor<Decay>;
      isInline_ = true;
    } else {
      auto* p = ::new Decay(std::forward<F>(f));
      *reinterpret_cast<Decay**>(&storage_) = p;
      vtable_ = &heapVtableFor<Decay>;
      isInline_ = false;
    }
  }

  ~Func() {
    if (vtable_) {
      vtable_->destroy(this);
    }
  }

  Func(Func&& other) noexcept : isInline_(other.isInline_) {
    if (other.vtable_) {
      other.vtable_->moveConstruct(&other, this);
      vtable_ = other.vtable_;
      other.vtable_->destroy(&other);
      other.vtable_ = nullptr;
    }
  }

  Func& operator=(Func&& other) noexcept {
    if (this != &other) {
      if (vtable_) {
        vtable_->destroy(this);
      }
      vtable_ = nullptr;
      isInline_ = other.isInline_;
      if (other.vtable_) {
        other.vtable_->moveConstruct(&other, this);
        vtable_ = other.vtable_;
        other.vtable_->destroy(&other);
        other.vtable_ = nullptr;
      }
    }
    return *this;
  }

  Func(const Func&) = delete;
  Func& operator=(const Func&) = delete;

  /// Returns true if this Func holds a callable.
  explicit operator bool() const noexcept {
    return vtable_ != nullptr;
  }

  /// Invoke the contained callable. After invocation, the Func is empty.
  void operator()() {
    if (!vtable_) {
      throw std::bad_function_call();
    }
    auto invoke = InvokeHelper(this, vtable_);
    vtable_ = nullptr;
    invoke();
  }

 private:
  struct VTable {
    void (*invoke)(Func* self);
    void (*moveConstruct)(Func* from, Func* to);
    void (*destroy)(Func* self);
  };

  class InvokeHelper {
   public:
    explicit InvokeHelper(Func* self, const VTable* vtable)
        : self_(self), vtable_(vtable) {}
    ~InvokeHelper() {
      vtable_->destroy(self_);
    }
    void operator()() {
      vtable_->invoke(self_);
    }

   private:
    Func* self_{nullptr};
    const VTable* vtable_{nullptr};
  };

  // VTable for inline (SBO) storage
  template <typename F>
  static const VTable vtableFor;

  // VTable for heap-allocated storage
  template <typename F>
  static const VTable heapVtableFor;

  template <typename F>
  F& getInline() noexcept {
    return *std::launder(reinterpret_cast<F*>(&storage_));
  }

  template <typename F>
  F* getHeap() noexcept {
    return *reinterpret_cast<F**>(&storage_);
  }

  alignas(std::max_align_t) unsigned char storage_[kInlineSize];
  const VTable* vtable_{nullptr};
  bool isInline_{false};
};

// --- VTable implementations ---

template <typename F>
const Func::VTable Func::vtableFor = {
    // invoke
    [](Func* self) { self->getInline<F>()(); },
    // moveConstruct
    [](Func* from, Func* to) {
      ::new (&to->storage_) F(std::move(from->getInline<F>()));
    },
    // destroy
    [](Func* self) { self->getInline<F>().~F(); },
};

template <typename F>
const Func::VTable Func::heapVtableFor = {
    // invoke
    [](Func* self) { (*self->getHeap<F>())(); },
    // moveConstruct
    [](Func* from, Func* to) {
      *reinterpret_cast<F**>(&to->storage_) = from->getHeap<F>();
      *reinterpret_cast<F**>(&from->storage_) = nullptr;
    },
    // destroy
    [](Func* self) { ::delete self->getHeap<F>(); },
};

} // namespace uniflow
