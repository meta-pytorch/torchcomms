//===---- test_target_teams_distribute_parallel_for_map_default.c -----------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// test the mapping of arrays by default. The expected behavior is that all 
// the arrays are mapped tofrom by default.
//
//===------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define N 2000

int test_target_teams_distribute_parallel_for_map_default() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_devices");
  
  int a[N];
  int b[N];
  int c[N];
  int d[N];
  int scalar = 20;
  int scalar2 = -1;
  int errors = 0;
  int i, j, dev;

  // array initialization
  for (i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = i;
    c[i] = 2*i;
    d[i] = 0;
  }


#pragma omp target teams distribute parallel for
  for (j = 0; j < N; ++j) {
    // scalar is firstprivate for the target region, but 
    // in a parallel construct, if not default clause is present
    // the variable is shared. Hence scalar = any other value 
    // could cause a data race.
    // See page 182, line 1.
    d[j] += c[j] * (a[j] + b[j] + scalar);
#pragma omp atomic write
    scalar2 = j;
  } // atomic prevents indeterminacy from simultaneous writes
    // since scalar2 is shared implicitly.

  for (i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, d[i] != (1 + i + 20) * 2*i);
  }

  return errors;
}

// Test for OpenMP 4.5 target enter data with if
int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_map_default());

  OMPVV_REPORT_AND_RETURN(errors);
}
