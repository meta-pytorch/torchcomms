//===--- test_parallel_master_device.c --------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the parallel master directive in a target context. The
// test performs simple operations on an int array which are then checked for
// correctness.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_parallel_master_device() {
  OMPVV_INFOMSG("test_parallel_master_device");
  int errors = 0;
  int num_threads = -1;
  int x[N];
  int y[N];
  int z[N];

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = i + 1;
    z[i] = 2*(i + 1);
  }

#pragma omp target map(tofrom: x, num_threads) map(to: y, z)
  {
#pragma omp parallel master num_threads(OMPVV_NUM_THREADS_DEVICE) shared(x, y, z, num_threads)
    {
#pragma omp taskloop
      for (int i = 0; i < N; i++) {
        x[i] += y[i]*z[i];
      }
      num_threads = omp_get_num_threads();
    }
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != 1 + (y[i]*z[i]));
  }

  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of parallel master with taskloop can't be guaranteed.");
  OMPVV_ERROR_IF(num_threads < 1, "Test returned an invalid number of threads.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);

  return errors;
}


int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_parallel_master_device());

  OMPVV_REPORT_AND_RETURN(errors);
}
