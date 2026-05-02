//===--- test_target_teams_distribute_reduction_add.c-------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the reduction clause on a target teams distribute directive,
// testing that the variable in the reduction clause is properly reduced using
// the add operator.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int test_add() {
  int a[N];
  int b[N];
  int total = 0;
  int host_total = 0;
  int errors = 0;
  int num_teams[N];

  for (int x = 0; x < N; ++x) {
    a[x] = 1;
    b[x] = x;
    num_teams[x] = -1;
  }

#pragma omp target teams distribute reduction(+:total) defaultmap(tofrom:scalar)
  for (int x = 0; x < N; ++x) {
    num_teams[x] = omp_get_num_teams();
    total += a[x] + b[x];
  }

  for (int x = 0; x < N; ++x) {
    host_total += a[x] + b[x];
  }

  for (int x = 1; x < N; ++x) {
    OMPVV_WARNING_IF(num_teams[x - 1] != num_teams[x], "Kernel reported differing numbers of teams.  Validity of testing of reduction clause cannot be guaranteed.");
  }
  OMPVV_WARNING_IF(num_teams[0] == 1, "Test operated with one team.  Reduction clause cannot be tested.");
  OMPVV_WARNING_IF(num_teams[0] <= 0, "Test reported invalid number of teams.  Validity of testing of reduction clause cannot be guaranteed.");

  OMPVV_TEST_AND_SET_VERBOSE(errors, host_total != total);
  OMPVV_ERROR_IF(host_total != total, "Total on device is %d but expected total from host is %d.", total, host_total);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_add() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
