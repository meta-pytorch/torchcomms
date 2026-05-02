//===--- test_declare_variant.c ---------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the declare variant clause to create two variants of a
// simple base function, one for use in a parallel region, and one for use
// in a target region. The function sets each element of an array to its
// index. Each variant is called on a separate array and the results are
// checked.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

void p_fn(int *arr);
void t_fn(int *arr);

#pragma omp declare variant(p_fn) match(construct = {parallel})
#pragma omp declare variant(t_fn) match(construct = {target})
void fn(int *arr) {              // base for use on the host in sequence
  for (int i = 0; i < N; i++) {
    arr[i] = i;
  }
}

void p_fn(int *arr) {            // variant for use on host in parallel
#pragma omp for
  for (int i = 0; i < N; i++) {
    arr[i] = i + 1;
  }
}

#pragma omp declare target
void t_fn(int *arr) {            // variant for use on target
#pragma omp distribute
  for (int i = 0; i < N; i++) {
    arr[i] = i + 2;
  }
}
#pragma omp end declare target

int main() {
  OMPVV_TEST_OFFLOADING;

  int default_errors = 0;
  int p_errors = 0;
  int t_errors = 0;
  int a[N], b[N], c[N];

  for (int i = 0; i < N; i++) {
    a[i] = 0;
    b[i] = 0;
    c[i] = 0;
  }

  fn(a);

#pragma omp parallel
  {
    fn(b);
  }

#pragma omp target teams map(tofrom: c[0:N])
  {
    fn(c);
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(default_errors, a[i] != i);
    OMPVV_TEST_AND_SET_VERBOSE(p_errors, b[i] != (i + 1));
    OMPVV_TEST_AND_SET_VERBOSE(t_errors, c[i] != (i + 2));
  }

  OMPVV_ERROR_IF(default_errors, "Did not use default variant of test function when expected.");
  OMPVV_ERROR_IF(p_errors, "Did not use parallel variant of test function when expected.");
  OMPVV_ERROR_IF(t_errors, "Did not use target variant of test function when expected.");

  OMPVV_REPORT_AND_RETURN(default_errors + p_errors + t_errors);
}
