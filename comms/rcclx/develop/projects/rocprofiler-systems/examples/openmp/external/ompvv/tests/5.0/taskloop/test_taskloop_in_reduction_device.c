//===--- test_taskloop_in_reduction_device.c ------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the taskloop directive with the `in_reduction` reduction
// participation clause. It performs simple array operations which are added
// to a reduction variable in a taskloop with the in_reduction clause. This
// test checks the above in a target context.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_taskloop_in_reduction() {
  OMPVV_INFOMSG("test_taskloop_in_reduction_device");
  int errors = 0;
  int num_threads = -1;
  int y[N];
  int z[N];
  int sum = 0;
  int expected_sum = 0;

  for (int i = 0; i < N; i++) {
    y[i] = i + 1;
    z[i] = 2*(i + 1);
  }

#pragma omp target parallel reduction(task, +: sum) num_threads(OMPVV_NUM_THREADS_DEVICE) shared(y, z, num_threads) defaultmap(tofrom)
  {
#pragma omp master
    {
#pragma omp taskloop in_reduction(+: sum)
      for (int i = 0; i < N; i++) {
        sum += y[i]*z[i];
      }
      num_threads = omp_get_num_threads();
    }
  }

  for (int i = 0; i < N; i++) {
    expected_sum += y[i]*z[i];
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, sum != expected_sum);

  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of taskloop can't be guaranteed.");
  OMPVV_ERROR_IF(num_threads < 1, "Test returned an invalid number of threads.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);

  return errors;
}


int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_taskloop_in_reduction());

  OMPVV_REPORT_AND_RETURN(errors);
}
