//===--- test_loop_reduction_multiply_device.c ------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the reduction clause on a loop directive, testing that the
// variable in the reduction clause is properly reduced using the multiply
// operator. This test checks the above in a target context.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int test_multiply() {
  int a[N];
  int errors = 0;
  int num_threads[N];
  srand(1);

  for (int x = 0; x < N; ++x) {
    a[x] = 1 + (int) rand() / (double) RAND_MAX;
    num_threads[x] = -x;
  }

  int result = 1;
  int host_result;

  for (int x = 0; x < N; x = x + 16) {
    result = 1;
#pragma omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: result, a, num_threads)
    {
#pragma omp loop reduction(*:result)
      for (int y = 0; y < 16; ++y) {
        result *= a[x + y];
      }
#pragma omp for
      for (int y = 0; y < 16; ++y) {
        num_threads[x + y] = omp_get_num_threads();
      }
    }
    host_result = 1;
    for (int y = 0; y < 16; ++y) {
      host_result *= a[x + y];
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, host_result != result);
    OMPVV_INFOMSG_IF(host_result != result, "Loop directive result is %d and expected result is %d.", result, host_result);
  }

  for (int x = 1; x < N; ++x) {
    OMPVV_WARNING_IF(num_threads[x - 1] != num_threads[x], "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.");
  }
  OMPVV_WARNING_IF(num_threads[0] == 1, "Test operated with one thread.  Reduction clause cannot be tested.");
  OMPVV_WARNING_IF(num_threads[0] <= 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.");

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_multiply() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
