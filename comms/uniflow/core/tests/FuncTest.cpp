// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include <gtest/gtest.h>

#include "comms/uniflow/core/Func.h"

using namespace uniflow;

// --- Reusable test callables ---

// Small callable that fits in SBO (inline storage).
// Tracks constructor/destructor calls via cnt for leak detection.
auto makeInlineFunc(int& cnt, int& out, int value) {
  struct Small {
    int* cnt;
    int* out;
    int value;
    void operator()() const noexcept {
      *out = value;
    }
    Small(int* cnt, int* out, int value) : cnt(cnt), out(out), value(value) {
      *cnt += 1;
    }
    Small& operator=(const Small& other) = delete;
    Small(const Small& other) = delete;
    Small& operator=(Small&& other) noexcept = delete;
    Small(Small&& other) noexcept
        : cnt(other.cnt), out(other.out), value(other.value) {
      *cnt += 1;
      other.out = nullptr;
    }
    ~Small() {
      *cnt -= 1;
    }
  };
  return Small(&cnt, &out, value);
}

// Large callable that exceeds kInlineSize, forcing heap allocation.
// Tracks constructor/destructor calls via cnt for leak detection.
auto makeHeapFunc(int& cnt, int& out, int value) {
  struct Large {
    char padding[128]{};
    int* cnt;
    int* out;
    int value;
    void operator()() const noexcept {
      *out = value;
    }
    Large(int* cnt, int* out, int value) : cnt(cnt), out(out), value(value) {
      *cnt += 1;
    }
    Large(const Large& other) = delete;
    Large& operator=(const Large& other) = delete;
    Large& operator=(Large&& other) noexcept = delete;
    Large(Large&& other) noexcept
        : cnt(other.cnt), out(other.out), value(other.value) {
      *cnt += 1;
      other.out = nullptr;
    }
    ~Large() {
      *cnt -= 1;
    }
  };
  return Large(&cnt, &out, value);
}

// --- Parameterized test fixture ---

enum class FuncType { Inline, Heap };

class FuncParamTest : public testing::TestWithParam<FuncType> {
 protected:
  void SetUp() override {
    cnt_ = 0;
    out_ = 0;
  }
  Func makeFunc(int value) {
    switch (GetParam()) {
      case FuncType::Inline:
        return Func(makeInlineFunc(cnt_, out_, value));
      case FuncType::Heap:
        return Func(makeHeapFunc(cnt_, out_, value));
      default:
        return Func();
    }
  }

  int cnt_{};
  int out_{};
};

INSTANTIATE_TEST_SUITE_P(
    InlineAndHeap,
    FuncParamTest,
    testing::Values(FuncType::Inline, FuncType::Heap),
    [](const testing::TestParamInfo<FuncType>& info) {
      return info.param == FuncType::Inline ? "Inline" : "Heap";
    });

// --- Parameterized tests (run for both inline and heap) ---

TEST_P(FuncParamTest, Invoke) {
  Func f = makeFunc(42);
  EXPECT_TRUE(static_cast<bool>(f));
  f();
  EXPECT_EQ(out_, 42);
  EXPECT_EQ(cnt_, 0);
}

TEST_P(FuncParamTest, EmptyAfterInvoke) {
  Func f = makeFunc(1);
  EXPECT_TRUE(static_cast<bool>(f));
  f();
  EXPECT_FALSE(static_cast<bool>(f));
  EXPECT_EQ(cnt_, 0);
}

TEST_P(FuncParamTest, MoveConstruct) {
  Func f1 = makeFunc(1);
  Func f2(std::move(f1));
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(static_cast<bool>(f1));
  EXPECT_TRUE(static_cast<bool>(f2));
  f2();
  EXPECT_EQ(out_, 1);
  EXPECT_EQ(cnt_, 0);
}

TEST_P(FuncParamTest, MoveAssign) {
  int cnt = 0;
  int out = 0;
  Func f1 = makeFunc(1);
  Func f2;
  switch (GetParam()) {
    case FuncType::Inline:
      f2 = Func(makeInlineFunc(cnt, out, 2));
      break;
    case FuncType::Heap:
      f2 = Func(makeHeapFunc(cnt, out, 2));
      break;
  }
  f2 = std::move(f1);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(static_cast<bool>(f1));
  EXPECT_TRUE(static_cast<bool>(f2));
  f2();
  EXPECT_EQ(out, 0);
  EXPECT_EQ(out_, 1);
  EXPECT_EQ(cnt_, 0);
  EXPECT_EQ(cnt, 0);
}

TEST_P(FuncParamTest, MoveAssignToEmpty) {
  Func f1 = makeFunc(1);
  Func f2;
  f2 = std::move(f1);
  EXPECT_TRUE(static_cast<bool>(f2));
  f2();
  EXPECT_EQ(out_, 1);
  EXPECT_EQ(cnt_, 0);
}

TEST_P(FuncParamTest, MoveAssignFromEmpty) {
  Func f1 = makeFunc(1);
  Func f2;
  f1 = std::move(f2);
  EXPECT_FALSE(static_cast<bool>(f1));
  EXPECT_EQ(cnt_, 0);
}

TEST_P(FuncParamTest, DestroyWithoutInvoke) {
  {
    Func f = makeFunc(1);
  }
  EXPECT_EQ(cnt_, 0);
}

// --- Non-parameterized tests ---

TEST(FuncTest, DefaultConstructedIsEmpty) {
  Func f;
  EXPECT_FALSE(static_cast<bool>(f));
}

TEST(FuncTest, NullptrConstructedIsEmpty) {
  Func f(nullptr);
  EXPECT_FALSE(static_cast<bool>(f));
}

TEST(FuncTest, MoveOnlyCapture) {
  auto ptr = std::make_unique<int>(99);
  int result = 0;
  Func f([p = std::move(ptr), &result]() noexcept { result = *p; });
  EXPECT_TRUE(static_cast<bool>(f));
  f();
  EXPECT_EQ(result, 99);
}

TEST(FuncTest, MoveOnlyCaptureHeap) {
  struct BigMoveOnly {
    std::unique_ptr<int> ptr;
    char padding[128]{};
  };
  BigMoveOnly bmo;
  bmo.ptr = std::make_unique<int>(55);
  int result = 0;
  Func f([b = std::move(bmo), &result]() noexcept { result = *b.ptr; });
  f();
  EXPECT_EQ(result, 55);
}

TEST(FuncTest, FunctionPointer) {
  static int sideEffect = 0;
  Func f(+[]() noexcept { sideEffect = 123; });
  f();
  EXPECT_EQ(sideEffect, 123);
}
