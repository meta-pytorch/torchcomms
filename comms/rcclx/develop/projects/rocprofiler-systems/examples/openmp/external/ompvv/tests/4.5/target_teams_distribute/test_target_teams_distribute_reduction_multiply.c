//===--- test_target_teams_distribute_reduction_multiply.c-------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the reduction clause on a target teams distribute directive,
// testing that the variable in the reduction clause is properly reduced using
// the multiply operator.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int test_multiply() {
  int a[N];
  int errors = 0;
  int num_teams[N];
  srand(1);

  for (int x = 0; x < N; ++x) {
    a[x] = 1 + (int) rand() / (double) RAND_MAX;
    num_teams[x] = -x;
  }

  int result = 1;
  int host_result;

  for (int x = 0; x < N; x = x + 16) {
    result = 1;
#pragma omp target teams distribute reduction(*:result) defaultmap(tofrom:scalar)
    for (int y = 0; y < 16; ++y) {
      result *= a[x + y];
      num_teams[x + y] = omp_get_num_teams();
    }
    host_result = 1;
    for (int y = 0; y < 16; ++y) {
      host_result *= a[x + y];
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, host_result != result);
    OMPVV_INFOMSG_IF(host_result != result, "Device result is %d and host result is %d.", result, host_result);
  }

  for (int x = 1; x < N; ++x) {
    OMPVV_WARNING_IF(num_teams[x - 1] != num_teams[x], "Kernel reported differing numbers of teams.  Validity of testing of reduction clause cannot be guaranteed.");
  }
  OMPVV_WARNING_IF(num_teams[0] == 1, "Test operated with one team.  Reduction clause cannot be tested.");
  OMPVV_WARNING_IF(num_teams[0] <= 0, "Test reported invalid number of teams.  Validity of testing of reduction clause cannot be guaranteed.");

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_multiply() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
