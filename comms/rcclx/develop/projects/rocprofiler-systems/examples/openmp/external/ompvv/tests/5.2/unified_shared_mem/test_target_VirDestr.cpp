//===--  test_target_VirDestr.cpp-------------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) Testing virtual destructor on gpu
//===----------------------------------------------------------------------===//


#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <omp.h>
#include "ompvv.h"

#define N 1024*512

#pragma omp requires unified_shared_memory
class base {
  int *x = nullptr;
  public:
    static int StatVar;
    static void SetStatVal() {
      StatVar = 999;
    }
    base() {
      x = new int[N];
      for (int i = 0; i < N; ++i) {
        x[i] = 10;
      }
    }
    virtual ~base() {
      delete[] x;
    }
};

int base::StatVar = 0;
class derived: public base {
  int *y = nullptr;
  public:
    derived() {
      y = new int[N];
      for (int i = 0; i < N; ++i) {
        y[i] = 20;
      }
    }
    ~derived() {
      delete[] y;
      StatVar = 999;
    }
};

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
#pragma omp target
    {
      derived *d = new derived();
      base *b = d;
      delete b;
    }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (base::StatVar != 999));

  OMPVV_REPORT_AND_RETURN(errors);
}
