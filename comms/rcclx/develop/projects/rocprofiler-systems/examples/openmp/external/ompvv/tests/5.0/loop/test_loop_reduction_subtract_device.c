//===--- test_loop_reduction_subtract_device.c --------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the reduction clause on a loop directive, testing that the
// variable in the reduction clause is properly reduced using the subtract
// operator. This test checks the above in a target context.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int test_subtraction() {
  int a[N];
  int b[N];
  int total = 0;
  int host_total = 0;
  int errors = 0;
  int num_threads[N];

  for (int x = 0; x < N; ++x) {
    a[x] = 1;
    b[x] = x;
    num_threads[x] = -x;
  }

#pragma omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: total, a, b, num_threads)
  {
#pragma omp loop reduction(-:total)
    for (int x = 0; x < N; ++x) {
      total -= a[x] + b[x];
    }
#pragma omp for
    for (int x = 0; x < N; ++x) {
      num_threads[x] = omp_get_num_threads();
    }
  }

  for (int x = 0; x < N; ++x) {
    host_total -= a[x] + b[x];
  }

  for (int x = 1; x < N; ++x) {
    OMPVV_WARNING_IF(num_threads[x - 1] != num_threads[x], "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.");
  }
  OMPVV_WARNING_IF(num_threads[0] == 1, "Test operated with one thread.  Reduction clause cannot be tested.");
  OMPVV_WARNING_IF(num_threads[0] <= 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.");

  OMPVV_TEST_AND_SET_VERBOSE(errors, host_total != total);
  OMPVV_ERROR_IF(host_total != total, "Total from loop directive is %d but expected total is %d.", total, host_total);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_subtraction() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
