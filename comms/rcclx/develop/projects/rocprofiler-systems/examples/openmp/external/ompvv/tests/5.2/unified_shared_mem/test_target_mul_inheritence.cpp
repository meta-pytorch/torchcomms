//===-- test_target_mul_inheritence.cpp ----------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) Testing multiple inheritence on gpu
//===----------------------------------------------------------------------===//

#include <iostream>
#include <omp.h>
#include "ompvv.h"

#define N 1024*512

#pragma omp requires unified_shared_memory

class A {
  int *Ax = new int[N];
        public:
  int GetAx(int Indx) {
    return Ax[Indx];
  }
  A() {
    for (int i = 0; i < N; ++i) {
      Ax[i] = 10;
    }
  }

  void SetAx(int Val) {
    for (int i = 0; i < N; ++i) {
      Ax[i] = Val;
    }
  }
};

class B {
  int *Bx = new int[N];
  public:
  int GetBx(int Indx) {
    return Bx[Indx];
  }

  B() {
    for (int i = 0; i < N; ++i) {
      Bx[i] = 20;
    }
  }

  void SetBx(int Val) {
    for (int i = 0; i < N; ++i) {
      Bx[i] = Val;
    }
  }
};

class C : public B, public A {
  int Cx;
        public:
  int GetCx() {
    return Cx;
  }
  C(int x) {
    Cx = x;
  }
};


int main() {
  OMPVV_TEST_OFFLOADING;
  C cx(30);
  int Errors = 0, errors = 0;
  int TotGpus = omp_get_num_devices();
  for (int dev = 0; dev < TotGpus; ++dev) {
#pragma omp target data map(tofrom: Errors) device(dev)
#pragma omp target device(dev)
    {
      if (cx.GetCx() != 30) {
        Errors++;
      }
      // Setting value to an array in class A through class C object
      cx.SetAx(123);
      // Verifying if the value is set
      if ((cx.GetAx(0) != 123) || (cx.GetAx(N/2) != 123) ||
          (cx.GetAx(N-1) != 123)) {
        Errors++;
      }
      cx.SetBx(333);
      // Verifying if the value is set
      if ((cx.GetBx(0) != 333) || (cx.GetBx(N/2) != 333) ||
          (cx.GetBx(N-1) != 333)) {
        Errors++;
      }
    }
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (Errors != 0));

  OMPVV_REPORT_AND_RETURN(errors);
}
