//===---- test_target_teams_distribute_parallel_for_firstprivate.c - combined consutrct -===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// This test check for a private variable within a pragma omp target teams distribute 
// parallel for that is initialized from the host through firstprivate clause
// We use a private variable within a for loop and asign it every iteration
// hoping that we won't get into data races. We do this multiple times to improve
// test
//
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define SIZE_N 1024

int test_target_teams_distribute_parallel_for_firstprivate() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_firstprivate");
  
  int a[SIZE_N];
  int b[SIZE_N];
  int c[SIZE_N];
  int d[SIZE_N];
  int firstized = 10;
  int privatized;
  int reported_num_teams[SIZE_N];
  int reported_team_num[SIZE_N];
  int reported_num_threads[SIZE_N];
  int errors = 0;
  int i, j, dev;

  // array initialization
  for (i = 0; i < SIZE_N; i++) {
    a[i] = 1;
    b[i] = i;
    c[i] = 2*i;
    d[i] = 0;
  }

  // check multiple sizes. 
#pragma omp target data map(to: a[0:SIZE_N], b[0:SIZE_N], c[0:SIZE_N])
  {
#pragma omp target teams distribute parallel for firstprivate(privatized, firstized, i) num_teams(OMPVV_NUM_TEAMS_DEVICE) num_threads(OMPVV_NUM_THREADS_DEVICE)
      for (j = 0; j < SIZE_N; ++j) {
        reported_num_teams[j] = omp_get_num_teams();
        reported_num_threads[j] = omp_get_num_threads();
        reported_team_num[j] = omp_get_team_num();
        privatized = 0;
        for (i = 0; i < a[j] + b[j]; ++i) {
          privatized++;
        }
        privatized += firstized;
        d[j] = c[j] * privatized;
      }
  }

  // Checking for reported teams and number of threads
  OMPVV_WARNING_IF(reported_num_teams[0] == 1, "Number of teams reported was 1, test cannot assert privatization across teams");
  // To have a single warning for threds. diff teams could have different number of threads, we warn if all are 1
  int warning_threads = 0;
  for (i = 0; i < SIZE_N; i++) {
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
  for (i = 0; i < SIZE_N; i++) {
    // 10 = initial value of privatized + 1 initial value of a[i] 
    // + i initial value of b[i]
    OMPVV_TEST_AND_SET(errors, d[i] != (10 + 1 + i)*2*i);
  }

  return errors;
}

// Test for OpenMP 4.5 target enter data with if
int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_firstprivate());

  OMPVV_REPORT_AND_RETURN(errors);
}
