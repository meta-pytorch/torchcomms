//===-- test_target_class_deepcpy.cpp -------------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) demonstrating the deepcpy functionality on gpu
//===----------------------------------------------------------------------===//

#include <iostream>
#include "ompvv.h"
#include <omp.h>

#define N 1024*512
#pragma omp requires unified_shared_memory

class A {
  int *Val = nullptr;
  public:
    void SetVal(int x) {
      for (int i = 0; i < N; ++i) {
        Val[i] = x;
      }
    }
    int GetVal(int Indx) {
      return Val[Indx];
    }
    A(int x) {
      Val = new int[N];
      for (int i = 0; i < N; ++i) {
        Val[i] = x;
      }
    }
    // Deep copy constructor
    A(A  &a) {
      this->Val = new int[N];
      for (int i = 0; i < N; ++i) {
        Val[i] = a.GetVal(i);
      }
    }
    ~A() {
      delete[] Val;
    }
};

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  A a(123);
  A b = a;
#pragma omp target
  {
    b.SetVal(999);
  }
  // verifying the results
  int count = 0;
  for (int i = 0; i < N; ++i) {
    if (b.GetVal(i) != 999) {
      count++;
    }
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (count != 0));
  OMPVV_REPORT_AND_RETURN(errors);
}
