//===--- test_target_teams_distribute_parallel_for_simd_atomic.c ------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks that the atomic construct can be used within the target
// teams distribute parallel for construct with simd to avoid a race
// condition in updating a shared variable, whose value is checked after
// updating.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_target_teams_distribute_parallel_for_simd_atomic() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_simd_atomic");
  int errors = 0, x = 0;

#pragma omp target teams distribute parallel for simd map(tofrom: x) shared(x) num_teams(OMPVV_NUM_TEAMS_DEVICE) num_threads(OMPVV_NUM_THREADS_DEVICE)
  for (int i = 0; i < N; i++) {
#pragma omp atomic update
    x += 1;
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, x != N);

  return errors;
}


int main() {
  int errors = 0;
  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_simd_atomic());

  OMPVV_REPORT_AND_RETURN(errors);
}
