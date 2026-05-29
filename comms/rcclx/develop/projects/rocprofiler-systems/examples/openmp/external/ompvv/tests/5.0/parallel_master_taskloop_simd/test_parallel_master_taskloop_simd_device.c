//===--- test_parallel_master_taskloop_simd_device.c ------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the parallel master taskloop simd directive. The test
// performs simple operations on an int array which are then checked for
// correctness. This test checks the construct in a target context.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_parallel_master_taskloop_simd_device() {
  OMPVV_INFOMSG("test_parallel_master_taskloop_simd_device");
  int errors = 0;
  int x[N];
  int y[N];
  int z[N];
  int num_threads = -1;

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = i + 1;
    z[i] = 2*(i + 1);
  }

#pragma omp target map(tofrom: x, num_threads) map(to: y, z)
  {
#pragma omp parallel master taskloop simd num_threads(OMPVV_NUM_THREADS_DEVICE) shared(x, y, z, num_threads)
    for (int i = 0; i < N; i++) {
      x[i] += y[i]*z[i];
      if (i == 0) {
         num_threads = omp_get_num_threads();
      }
    }
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != 1 + (y[i]*z[i]));
  }

  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of parallel master with taskloop can't be guaranteed.");
  OMPVV_ERROR_IF(num_threads < 1, "Test returned an invalid number of threads.");
  OMPVV_INFOMSG("This test does not guarantee vector instructions were generated for the simd construct.");

  return errors;
}


int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_parallel_master_taskloop_simd_device());

  OMPVV_REPORT_AND_RETURN(errors);
}
