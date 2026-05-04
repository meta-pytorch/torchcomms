//===--test_target_PureVirDestr.cpp -------------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) Testing pure virtual destructor on gpu
//===----------------------------------------------------------------------===//

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <omp.h>
#include "ompvv.h"

#define N 1024*1024

#pragma omp requires unified_shared_memory
#pragma omp begin declare target
class base {
  int *x = nullptr;
  public:
    int *BaseMem = nullptr;
    int *ChildMem = nullptr;
    base(int *ptr1 = nullptr, int *ptr2 = nullptr) {
      x = new int[N];
      for (int i = 0; i < N; ++i) {
        x[i] = 10;
      }
    }
    virtual ~base() = 0;
};

base::~base() {
  delete[] x;
  if (*ChildMem == 1) {
    *BaseMem = 1;
  } else {
    *BaseMem = -1;
  }
}

class derived: public base {
  int *y = nullptr;
  public:
    derived(int *ptr1 = nullptr, int *ptr2 = nullptr) {
      BaseMem = ptr1;
      *BaseMem = 0;
      ChildMem = ptr2;
      *ChildMem = 0;
      y = new int[N];
      for (int i = 0; i < N; ++i) {
        y[i] = 20;
      }
    }
    ~derived() {
      delete[] y;
      *ChildMem = 1;
    }
};
#pragma omp end declare target

int main() {
  OMPVV_TEST_OFFLOADING;
  int *ForChild = nullptr, *ForBase = nullptr, errors = 0;
  int TotGpus = omp_get_num_devices(), Errs;
  for (int dev = 0; dev < TotGpus; ++dev) {
    Errs = -17;
    ForChild = reinterpret_cast<int*>(omp_target_alloc(sizeof(int), dev));
    ForBase = reinterpret_cast<int*>(omp_target_alloc(sizeof(int), dev));

#pragma omp target device(dev) map(tofrom: Errs)
    {
      derived *d = new derived(ForBase, ForChild);
      base *b = d;
      delete b;
      Errs = *ForChild;
    }
    omp_target_free(ForChild, dev);
    omp_target_free(ForBase, dev);
    OMPVV_TEST_AND_SET_VERBOSE(errors, (Errs != 1));
  }

  OMPVV_REPORT_AND_RETURN(errors);
}
