//===-- test_target_mul_inheritence1.cpp ---------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) Working of diamond pattern of inheritence on gpu
//===----------------------------------------------------------------------===//

#include <iostream>
#include <omp.h>
#include "ompvv.h"

#define N 1024*10

#pragma omp requires unified_shared_memory

class A {
  int *Ax = new int[N];
  public:
  static int StatVar;
  int GetAx(int Indx) {
    return Ax[Indx];
  }
  static void IncrStatVar(int x = 1) {
    StatVar += x;
  }
  void SetAx(int Val) {
    for (int i = 0; i < N; ++i) {
      Ax[i] = Val;
        }
  }
  A() {
    IncrStatVar();
    for (int i = 0; i < N; ++i) {
      Ax[i] = 10;
    }
  }
  A(int x) {
    IncrStatVar(x);
    for (int i = 0; i < N; ++i) {
      Ax[i] = 10;
    }
  }

  ~A() {
    delete[] Ax;
  }
};
int A::StatVar = 0;

class B : virtual public A {
  int *Bx = new int[N];
        public:
  int GetBx(int Indx) {
    return Bx[Indx];
  }

  void SetBx(int Val) {
    for (int i = 0; i < N; ++i) {
      Bx[i] = Val;
    }
  }

  B() {
    for (int i = 0; i < N; ++i) {
      Bx[i] = 20;
    }
  }
  ~B() {
    delete[] Bx;
  }
};

class C : virtual public A {
  int *Cx = new int[N];
        public:
  C() {
    for (int i = 0; i < N; ++i) {
      Cx[i] = 30;
    }
  }
  ~C() {
    delete[] Cx;
  }

  int GetCx(int Indx) {
    return Cx[Indx];
  }
  void SetCx(int Val) {
    for (int i = 0; i < N; ++i) {
      Cx[i] = Val;
    }
  }
};

class D : public B, public C  {
  int Dx;
        public:
  int GetDx() {
    return Dx;
  }
  D(int x) {
    Dx = x;
  }
};

class D_PC : public B, public C  {
  int Dx;
        public:
  int GetDx() {
    return Dx;
  }
  D_PC(int x):A(x) {
    Dx = x;
  }
};


int main() {
  OMPVV_TEST_OFFLOADING;
  D dx(40);
  int Errors = 0, errors = 0;
  int TotGpus = omp_get_num_devices();
  for (int dev = 0; dev < TotGpus; ++dev) {
  #pragma omp target data map(tofrom: Errors) device(dev)
  #pragma omp target device(dev)
    {
      if (dx.GetDx() != 40) {
        Errors++;
      }
      // Setting value to an array in class A through class C object
      dx.SetAx(123);
      // Verifying if the value is set
      if ((dx.GetAx(0) != 123) || (dx.GetAx(N/2) != 123) ||
          (dx.GetAx(N-1) != 123)) {
        Errors++;
      }
      dx.SetBx(333);
      // Verifying if the value is set
      if ((dx.GetBx(0) != 333) || (dx.GetBx(N/2) != 333) ||
          (dx.GetBx(N-1) != 333)) {
          Errors++;
      }
      dx.SetCx(444);
      // Verifying if the value is set
      if ((dx.GetCx(0) != 444) || (dx.GetCx(N/2) != 444) ||
          (dx.GetCx(N-1) != 444)) {
          Errors++;
      }
      if (A::StatVar != 1) {
        Errors++;
      }
    }
  }

  D_PC DPCx(999);
  for (int dev = 0; dev < TotGpus; ++dev) {
  #pragma omp target data map(tofrom: Errors) device(dev)
  #pragma omp target device(dev)
    {
      if (A::StatVar != 999) {
        Errors++;
      }
    }
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (Errors != 0));

  OMPVV_REPORT_AND_RETURN(errors);
}
