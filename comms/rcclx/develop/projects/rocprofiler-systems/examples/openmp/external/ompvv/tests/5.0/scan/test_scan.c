//===--- test_scan.c --------------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the scan directive with both the inclusive and exclusive
// clause, as well as the parallel for simd directive with the
// reduction(inscan) clause. The test peforms the scan operation with the add
// operator and then checks the results against an expected result from a
// sequentially-computed prefix sum.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_scan_inclusive() {
  OMPVV_INFOMSG("test_scan_inclusive");
  int errors = 0;
  int x = 0;
  int expected_x = 0;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 0;
  }

#pragma omp parallel for simd reduction(inscan, +: x) num_threads(OMPVV_NUM_THREADS_HOST)
  for (int i = 0; i < N; i++) {
    x += a[i];
#pragma omp scan inclusive(x)
    b[i] = x;
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      expected_x += a[j];
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, b[i] != expected_x);
    expected_x = 0;
  }

  return errors;
}

int test_scan_exclusive() {
  OMPVV_INFOMSG("test_scan_exclusive");
  int errors = 0;
  int x = 0;
  int expected_x = 0;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 0;
  }

#pragma omp parallel for simd reduction(inscan, +: x) num_threads(OMPVV_NUM_THREADS_HOST)
  for (int i = 0; i < N; i++) {
    b[i] = x;
#pragma omp scan exclusive(x)
    x += a[i];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      expected_x += a[j];
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, b[i] != expected_x);
    expected_x = 0;
  }

  return errors;
}

int main() {
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_scan_inclusive());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_scan_exclusive());

  OMPVV_REPORT_AND_RETURN(errors);
}
