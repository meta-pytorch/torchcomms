//===--- test_loop_reduction_bitor_device.c ---------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the reduction clause on a loop directive, testing that the
// variable in the reduction clause is properly reduced using the bitor
// operator. This test checks the above in a target context.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024
#define THRESHOLD 1024

int test_bitor() {
  int a[N];
  // See the 'and' operator test for an exaplantion of this math.
  double true_margin = pow(exp(1), log(.5)/N);
  int errors = 0;
  int num_threads[N];
  int num_attempts = 0;
  int have_true = 0, have_false = 0;
  srand(1);

  for (int x = 0; x < N; ++x) {
    a[x] = 0;
  }
  while ((!have_true || !have_false) && (num_attempts < THRESHOLD)) {
    have_true = 0;
    have_false = 0;
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < 16; ++y) {
        if (rand() / (double) RAND_MAX > true_margin) {
          a[x] += (1 << y);
          have_true = 1;
        } else {
          have_false = 1;
        }
      }
      num_threads[x] = -x;
    }
    num_attempts++;
  }

  OMPVV_WARNING_IF(!have_true, "No true bits were generated to test");
  OMPVV_WARNING_IF(!have_false, "No false bits were generated to test");

  unsigned int b = 0;

#pragma omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: a, b, num_threads)
  {
#pragma omp loop reduction(|:b)
    for (int x = 0; x < N; ++x) {
      b = b | a[x];
    }
#pragma omp for
    for (int x = 0; x < N; ++x) {
      num_threads[x] = omp_get_num_threads();
    }
  }

  unsigned int host_b = 0;

  for (int x = 0; x < N; ++x) {
    host_b = host_b | a[x];
  }

  for (int x = 1; x < N; ++x) {
    OMPVV_WARNING_IF(num_threads[x - 1] != num_threads[x], "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.");
  }
  OMPVV_WARNING_IF(num_threads[0] == 1, "Test operated with one thread.  Reduction clause cannot be tested.");
  OMPVV_WARNING_IF(num_threads[0] <= 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.");

  OMPVV_TEST_AND_SET_VERBOSE(errors, b != host_b);
  OMPVV_ERROR_IF(host_b != b, "Bit from loop directive is %d but expected bit is %d.", b, host_b);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_bitor() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
