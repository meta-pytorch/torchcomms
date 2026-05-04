//===---- test_target_teams_distribute_parallel_for.c - combined consutrct -===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// testing the combined construct target teams distribute parallel for
//
//===----------------------------------------------------------------------------------===//
//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define ARRAY_SIZE 1024

int test_target_teams_distribute_parallel_for() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for");

  // Variable for errors counting
  int errors = 0;

  int a[ARRAY_SIZE];
  int b[ARRAY_SIZE];
  int c[ARRAY_SIZE];
  int num_teams = 0;
  int num_threads[ARRAY_SIZE];
  int alert_num_threads = 0;
  int i;

  // a and b array initialization
  for (i = 0; i < ARRAY_SIZE; ++i) {
    a[i] = 1;
    b[i] = i;
    c[i] = 2 * i;
    num_threads[i] = 0;
  }


#pragma omp target teams distribute parallel for map(from:num_teams, num_threads) num_teams(OMPVV_NUM_TEAMS_DEVICE) num_threads(OMPVV_NUM_THREADS_DEVICE)
  for (i = 0; i < ARRAY_SIZE; ++i) {
#pragma omp atomic write
    num_teams = omp_get_num_teams();
    num_threads[i] = omp_get_num_threads();
    a[i] += b[i] * c[i];
  }


  for (i = 0; i < ARRAY_SIZE; ++i) {
    OMPVV_TEST_AND_SET(errors, (a[i] != 1 + (b[i] * c[i])));
    if (num_threads[i] == 1) {
      alert_num_threads++;
    }
  }

  // Rise lack of parallelism alerts
  if (num_teams == 1) {
    OMPVV_WARNING("Test operated with one team.  Parallelism of teams distribute can't be guaranteed.");
  }
  if (alert_num_threads == ARRAY_SIZE) {
    OMPVV_WARNING("Test operated with one thread in all the teams. Parallel clause had no effect");
  }

  return errors;
}

// Test for OpenMP 4.5 target data with if
int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for());

  OMPVV_REPORT_AND_RETURN(errors);
}
