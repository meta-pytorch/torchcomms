//===--- test_target_teams_distribute_parallel_for_map_from.c ---------------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// Testing the map from of scalars and arrays when used with target teams 
// distrbute parallel for.
//
//===------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define N 1024

int test_target_teams_distribute_parallel_for_map_from() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_map_from");
  
  int a[N];
  int scalar = 0;
  int errors = 0;
  int i,j, dev;

  scalar = 0;
  // array initialization
  for (i = 0; i < N; ++i) {
    a[i] = 1;
  }


#pragma omp target teams distribute parallel for map(from: a, scalar)
  for (j = 0; j < N; ++j) {
#pragma omp atomic write
    scalar = 20;
    a[j] = 10;
  }

  // check the results
  OMPVV_TEST_AND_SET(errors, scalar != 20);
  for (i = 0; i < N; ++i) {
    OMPVV_TEST_AND_SET(errors, a[i] != 10);
  }

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_map_from());

  OMPVV_REPORT_AND_RETURN(errors);
}
