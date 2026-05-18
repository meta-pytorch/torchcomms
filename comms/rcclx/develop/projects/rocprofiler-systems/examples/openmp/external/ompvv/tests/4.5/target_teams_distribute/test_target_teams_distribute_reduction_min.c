//===--- test_target_teams_distribute_reduction_min.c----------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the reduction clause on a target teams distribute directive,
// testing that the variable in the reduction clause is properly reduced using
// the min operator.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int test_min() {
  int a[N];
  int b[N];
  int errors = 0;
  int num_teams[N];
  srand(1);

  for (int x = 0; x < N; ++x) {
    a[x] = (int) rand() / (double) (RAND_MAX / 100);
    b[x] = (int) rand() / (double) (RAND_MAX / 100);
    num_teams[x] = -x;
  }

  int result = a[0] + b[0];

#pragma omp target teams distribute reduction(min:result) defaultmap(tofrom:scalar)
  for (int x = 0; x < N; ++x) {
    num_teams[x] = omp_get_num_teams();
    result = fmin(result, a[x] + b[x]);
  }

  int host_min = a[0] + b[0];

  for (int x = 0; x < N; ++x) {
    host_min = fmin(host_min, a[x] + b[x]);
  }

  for (int x = 1; x < N; ++x) {
    OMPVV_WARNING_IF(num_teams[x - 1] != num_teams[x], "Kernel reported differing numbers of teams.  Validity of testing of reduction clause cannot be guaranteed.");
  }
  OMPVV_WARNING_IF(num_teams[0] == 1, "Test operated with one team.  Reduction clause cannot be tested.");
  OMPVV_WARNING_IF(num_teams[0] <= 0, "Test reported invalid number of teams.  Validity of testing of reduction clause cannot be guaranteed.");

  OMPVV_TEST_AND_SET_VERBOSE(errors, host_min != result);
  OMPVV_ERROR_IF(host_min != result, "Min on device is %d but expected min from host is %d.", result, host_min);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_min() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
