//===------ test_declare_target_end_declare_target.c ----------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// The test verifies that the variable and function enclosed between declare 
// target and end declare target are available inside the target region.
// Updates using the enclosed function are made inside the target region and
// results are verified on the host.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp declare target
int aint = 10;
void compute_array(int a[N], int b[N], int c[N]) {
  for (int i = 0; i < N; i++) {
    a[i] = b[i]*c[i] + aint * i;
  }
  return;
}
#pragma omp end declare target

int test_declare_target() {

  OMPVV_INFOMSG("test_declare_target_end_declare_target");

  int errors = 0;
  int x[N];
  int y[N];
  int z[N];

  for (int i = 0; i < N; i++) {
    x[i] = 0;
    y[i] = 1;
    z[i] = i;
  }

#pragma omp target map(tofrom: x) map(to:y, z)
  {
    compute_array(x, y, z);
  }

  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != (y[i] * z[i] + 10 * i));
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target() != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
