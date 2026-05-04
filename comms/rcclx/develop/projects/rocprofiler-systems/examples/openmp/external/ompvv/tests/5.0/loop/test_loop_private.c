//===--- test_loop_private.c ------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the private clause on a loop directive to indicate that the
// variable in the private clause should be made private to each thread
// executing the loop region.  The test then operates on the privatized
// variable in such a way that would most likely cause competing operations
// if the variable is not privatized.  If the computation completes without
// errors, we assume that the privatization occured.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE 1024

int main() {
  int a[SIZE];
  int b[SIZE];
  int c[SIZE];
  int d[SIZE];
  int privatized;
  int errors = 0;
  int num_threads = -1;

  for (int x = 0; x < SIZE; ++x) {
    a[x] = 1;
    b[x] = x;
    c[x] = 2*x;
    d[x] = 0;
  }

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
  {
#pragma omp loop private(privatized)
    for (int x = 0; x < SIZE; ++x) {
      privatized = 0;
      for (int y = 0; y < a[x] + b[x]; ++y) {
        privatized++;
      }
      d[x] = c[x] * privatized;
    }
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
  }

  for (int x = 0; x < SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, d[x] != (1 + x)*2*x);
    if (d[x] != (1 + x)*2*x) {
      break;
    }
  }

  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread. Results of private test are inconclusive.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);

  OMPVV_REPORT_AND_RETURN(errors);
}
