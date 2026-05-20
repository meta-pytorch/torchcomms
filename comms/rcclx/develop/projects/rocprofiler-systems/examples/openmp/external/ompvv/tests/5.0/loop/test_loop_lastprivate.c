//===--- test_loop_lastprivate.c ----------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the lastprivate clause with a loop directive. According to
// specification, the list items on a lastprivate clause in this context may
// only contain loop iteration variables of loops associated with the loop
// directive. This test checks that the loop iteration variables associated
// with a loop directive and a loop directive with collapse(2) have valid
// values after the parallel region containing the loop.
//
////===------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE 1024
#define SIZE2 512

int test_one_loop_level() {
  int a[SIZE];
  int b[SIZE];
  int errors = 0;
  int lp_errors = 0;
  int x = 0;

  for (x = 0; x < SIZE; ++x) {
    a[x] = 1;
    b[x] = x;
  }

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
  {
#pragma omp loop lastprivate(x)
    for (x = 0; x < SIZE; ++x) {
      a[x] += b[x];
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(lp_errors, x != SIZE);
  OMPVV_ERROR_IF(lp_errors, "Loop iteration variable in loop construct ended with invalid value.");

  for (x = 0; x < SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, a[x] - b[x] != 1);
  }

  return errors + lp_errors;
}

int test_two_loop_levels() {
  int a[SIZE][SIZE2];
  int b[SIZE][SIZE2];
  int errors = 0;
  int lp_errors_x = 0;
  int lp_errors_y = 0;
  int x = 0;
  int y = 0;

  for (x = 0; x < SIZE; ++x) {
    for (y = 0; y < SIZE2; ++y) {
      a[x][y] = 1;
      b[x][y] = x + y;
    }
  }

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
  {
#pragma omp loop lastprivate(x, y) collapse(2)
    for (x = 0; x < SIZE; ++x) {
      for (y = 0; y < SIZE2; ++y) {
        a[x][y] += b[x][y];
      }
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(lp_errors_x, x != SIZE);
  OMPVV_TEST_AND_SET_VERBOSE(lp_errors_y, y != SIZE2);
  OMPVV_ERROR_IF(lp_errors_x, "Outer loop iteration variable in loop directive with collapse ended with invalid value.");
  OMPVV_ERROR_IF(lp_errors_y, "Inner loop iteration variable in loop directive with collapse ended with invalid value.");

  for (x = 0; x < SIZE; ++x) {
    for (y = 0; y < SIZE2; ++y) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, a[x][y] - b[x][y] != 1);
    }
  }

  return errors + lp_errors_x + lp_errors_y;
}

int main() {
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_one_loop_level());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_two_loop_levels());

  OMPVV_REPORT_AND_RETURN(errors);
}
