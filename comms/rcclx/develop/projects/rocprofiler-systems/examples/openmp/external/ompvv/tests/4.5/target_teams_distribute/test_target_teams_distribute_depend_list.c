//===--- test_target_teams_distribute_depend_list.c---------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks for dependency between multiple out-dependent tasks by
// checking order-dependent results from pairs of possibly asynchronous loops
// The test fails if any required dependency is broken.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_target_teams_distribute_depend_list() {
  int isOffloading = 0;
  int errors = 0;
  int a[N];
  int b[N];
  int c[N];
  int d[N];
  int e[N];
  int f[N];
  int g[N];

  for (int x = 0; x < N; ++x) {
    a[x] = x;
    b[x] = 2 * x;
    c[x] = 0;
    d[x] = 0;
    e[x] = 0;
    f[x] = 0;
    g[x] = 0;
  }

#pragma omp target data map(to: a[0:N], b[0:N]) map(alloc: c[0:N], d[0:N], e[0:N]) map(from: f[0:N], g[0:N])
  {
#pragma omp target teams distribute nowait depend(out: c) map(alloc: a[0:N], b[0:N], c[0:N])
    for (int x = 0; x < N; ++x) {
      c[x] = a[x] + b[x];
    }
#pragma omp target teams distribute nowait depend(out: d) map(alloc: a[0:N], b[0:N], d[0:N])
    for (int x = 0; x < N; ++x) {
      d[x] = a[x] + b[x] + x;
    }
#pragma omp target teams distribute nowait depend(out: c, d, e) map(alloc: c[0:N], d[0:N], e[0:N])
    for (int x = 0; x < N; ++x) {
      e[x] = c[x] + d[x];
    }
#pragma omp target teams distribute nowait depend(out: e) map(alloc: a[0:N], e[0:N], f[0:N])
    for (int x = 0; x < N; ++x) {
      f[x] = e[x] + a[x];
    }
#pragma omp target teams distribute nowait depend(out: e) map(alloc: b[0:N], e[0:N], g[0:N])
    for (int x = 0; x < N; ++x) {
      g[x] = e[x] + b[x];
    }
#pragma omp taskwait
  }

  for (int x = 0; x < N; ++x) {
    OMPVV_ERROR_IF(f[x] != 8*x || g[x] != 9*x, "Test of depend clause using multiple dependencies did not pass with offloading %s", (isOffloading ? "enabled" : "disabled"));
    OMPVV_TEST_AND_SET_VERBOSE(errors, f[x] != 8*x || g[x] != 9*x);
  }

  return errors;
}

int main() {
  int errors = 0;
  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  errors += test_target_teams_distribute_depend_list();
  OMPVV_INFOMSG_IF(errors != 0, "Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
  OMPVV_REPORT_AND_RETURN(errors);
}
