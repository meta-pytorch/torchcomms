//===---- test_target_teams_distribute_parallel_for_schedule_private.c - private per thread combined construct-===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// private variables are private to the running thread. When using the scheduler clause it should be possible to
// have a private variable that shares its memory space across multiple iterations
//
//
//===---------------------------------------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

// This has been selected for the num teams and num threads below
#define SIZE_N 2048

int test_target_teams_distribute_parallel_for_sched_private() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_schedule_private");
  
  int a[SIZE_N];
  int firstprivatized=1;
  int reported_num_teams[SIZE_N];
  int reported_team_num[SIZE_N];
  int reported_num_threads[SIZE_N];
  int errors = 0;

  // array initialization
  for (int i = 0; i < SIZE_N; i++) {
    a[i] = 0;
  }

  // check multiple sizes. 
#pragma omp target teams distribute parallel for firstprivate(firstprivatized) num_teams(OMPVV_NUM_TEAMS_DEVICE) num_threads(OMPVV_NUM_THREADS_DEVICE) schedule(static,8)
    for (int j = 0; j < SIZE_N; ++j) {
      reported_num_teams[j] = omp_get_num_teams();
      reported_num_threads[j] = omp_get_num_threads();
      reported_team_num[j] = omp_get_team_num();
      if (j%8 == 0)
        firstprivatized = 0;
      firstprivatized++;
      a[j] += firstprivatized;
    }

  // Checking for reported teams and number of threads
  OMPVV_WARNING_IF(reported_num_teams[0] == 1, "Number of teams reported was 1, test cannot assert privatization across teams");
  // To have a single warning for threds. diff teams could have different number of threads, we warn if all are 1
  int warning_threads = 0;
  for (int i = 0; i < SIZE_N; i++) {
    if (reported_num_threads[i] == 1)
      warning_threads++;
    if (i > 0) {
      OMPVV_ERROR_IF(reported_num_teams[i] != reported_num_teams[i-1], "Discrepancy in the reported number of teams across teams");
      if (reported_team_num[i] == reported_team_num[i-1] && reported_num_threads[i] != reported_num_threads[i-1])
        OMPVV_ERROR("Discrepancy in the reported number of threads inside a single team");
    }
  }
  OMPVV_WARNING_IF(warning_threads == SIZE_N, "Number of threads was 1 for all teams. test cannot assert privatization across teams");
  // Checking for the expected values 
  for (int i = 0; i < SIZE_N; i+=8) {
    for (int j = 0; j < 8; j++) { 
      OMPVV_TEST_AND_SET(errors, a[i + j] != j+1);
    }
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_sched_private());

  OMPVV_REPORT_AND_RETURN(errors);
}
