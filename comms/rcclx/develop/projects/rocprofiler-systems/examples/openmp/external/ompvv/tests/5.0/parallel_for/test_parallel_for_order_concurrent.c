//===--- test_parallel_for_order_concurrent.c -------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the parallel for directive with the order(concurrent)
// The test performs simple operations on an int array which are then
// checked for correctness. The specification indicates only that iterations
// in the loop may be executed concurrently, so no particular order or lack
// thereof can be required by this test.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_parallel_for_order_concurrent() {
  OMPVV_INFOMSG("test_parallel_for_order_concurrent");
  int errors = 0;
  int x[N];
  int y[N];
  int z[N];

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = i + 1;
    z[i] = 2*(i + 1);
  }

#pragma omp parallel for order(concurrent) num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z)
  for (int i = 0; i < N; i++) {
    x[i] += y[i]*z[i];
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != 1 + (y[i]*z[i]));
  }

  return errors;
}


int main() {
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_parallel_for_order_concurrent());

  OMPVV_REPORT_AND_RETURN(errors);
}
