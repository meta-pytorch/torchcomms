//===--- test_target_update_to.c ------------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks target update with the 'to' clause by checking that
// after a target update to is applied to an array, the device reads and
// uses the expected, new value rather than the previous value.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include "ompvv.h"

#define N 1024

int a[N];
int b[N];
int c[N];

void update_b() {
  int i;
  for (i = 0; i < N; i++) {
    b[i] = b[i] * 2;
  }
}

// Test for OpenMP 4.5 target update with to
int main() {
  int errors= 0;
  int i = 0;

  OMPVV_TEST_OFFLOADING;

  for (i = 0; i < N; i++) {
    a[i] = 10;
    b[i] = 2;
    c[i] = 0;
  }

#pragma omp target data map(to: a[:N], b[:N]) map(from: c)
  {
#pragma omp target
    {
      int j = 0;
      for (j = 0; j < N; j++) {
        c[j] = (a[j] + b[j]);        // c = 12
      }
    }

    update_b();

#pragma omp target update to(b[:N])  // update b = 4

#pragma omp target
    {
      int j = 0;
      for (j = 0; j < N; j++) {
        c[j] = (c[j] + b[j]);        // c = 16
      }
    }
  }

  for (i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != 16);
  }

  OMPVV_REPORT_AND_RETURN(errors);
}
