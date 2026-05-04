//===--- test_target_teams_distribute_depend_out_out.c------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks for dependency between all combinations of out and inout
// by checking order-dependent results from pairs of possibly asynchronous
// loops. The test fails if any required dependency is broken.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_target_teams_distribute_depend_out_out() {
  int isOffloading = 0;
  int a[N];
  int b[N];
  int c[N];
  int d[N];
  int out_out_errors = 0;
  int inout_out_errors = 0;
  int out_inout_errors = 0;
  int inout_inout_errors = 0;

  for (int x = 0; x < N; ++x) {
    a[x] = x;
    b[x] = 2 * x;
    c[x] = 0;
    d[x] = 0;
  }

#pragma omp target data map(to: a[0:N], b[0:N]) map(alloc: c[0:N]) map(from: d[0:N])
  {
#pragma omp target teams distribute nowait depend(out: c) map(alloc: a[0:N], b[0:N], c[0:N])
    for (int x = 0; x < N; ++x) {
      c[x] = a[x] + b[x];
    }
#pragma omp target teams distribute nowait depend(out: c) map(alloc: b[0:N], c[0:N], d[0:N])
    for (int x = 0; x < N; ++x) {
      d[x] = c[x] + b[x];
    }
#pragma omp taskwait
  }

  for (int x = 0; x < N; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(out_out_errors, d[x] != 5*x);
  }

#pragma omp target data map(to: a[0:N], b[0:N]) map(alloc: c[0:N]) map(from: d[0:N])
  {
#pragma omp target teams distribute nowait depend(out: c) map(alloc: a[0:N], b[0:N], c[0:N])
    for (int x = 0; x < N; ++x) {
      c[x] = a[x] + b[x];
    }
#pragma omp target teams distribute nowait depend(inout: c) map(alloc: a[0:N], c[0:N], d[0:N])
    for (int x = 0; x < N; ++x) {
      d[x] = c[x] + a[x];
    }
#pragma omp taskwait
  }

  for (int x = 0; x < N; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(out_inout_errors, d[x] != 4*x);
  }

#pragma omp target data map(to: a[0:N], b[0:N]) map(alloc: c[0:N]) map(from: d[0:N])
  {
#pragma omp target teams distribute nowait depend(inout: c) map(alloc: a[0:N], b[0:N], c[0:N])
    for (int x = 0; x < N; ++x) {
      c[x] = a[x] + b[x];
    }
#pragma omp target teams distribute nowait depend(out: c) map(alloc: b[0:N], c[0:N], d[0:N])
    for (int x = 0; x < N; ++x) {
      d[x] = c[x] + b[x];
    }
#pragma omp taskwait
  }

  for (int x = 0; x < N; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(inout_out_errors, d[x] != 5*x);
  }

#pragma omp target data map(to: a[0:N], b[0:N]) map(alloc: c[0:N]) map(from: d[0:N])
  {
#pragma omp target teams distribute nowait depend(inout: c) map(alloc: a[0:N], b[0:N], c[0:N])
    for (int x = 0; x < N; ++x) {
      c[x] = a[x] + b[x];
    }
#pragma omp target teams distribute nowait depend(inout: c) map(alloc: a[0:N], c[0:N], d[0:N])
    for (int x = 0; x < N; ++x) {
      d[x] = c[x] + a[x];
    }
#pragma omp taskwait
  }

  for (int x = 0; x < N; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(inout_inout_errors, d[x] != 4*x);
  }

  OMPVV_ERROR_IF(out_out_errors == 1, "Test of depend(out) task becoming dependent task of depend(out) task did not pass with offloading %s", (isOffloading ? "enabled" : "disabled"));
  OMPVV_ERROR_IF(inout_out_errors == 1, "Test of depend(out) task becoming dependent task of depend(inout) task did not pass with offloading %s", (isOffloading ? "enabled" : "disabled"));
  OMPVV_ERROR_IF(out_inout_errors == 1, "Test of depend(inout) task becoming dependent task of depend(out) task did not pass with offloading %s", (isOffloading ? "enabled" : "disabled"));
  OMPVV_ERROR_IF(inout_inout_errors == 1, "Test of depend(inout) task becoming dependent task of depend(inout) task did not pass with offloading %s", (isOffloading ? "enabled" : "disabled"));

  return inout_inout_errors + inout_out_errors + out_inout_errors + out_out_errors;
}

int main() {
  int errors = 0;
  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  errors += test_target_teams_distribute_depend_out_out();
  OMPVV_INFOMSG_IF(errors != 0, "Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
  OMPVV_REPORT_AND_RETURN(errors);
}
