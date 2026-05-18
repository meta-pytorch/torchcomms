//===--- test_target_teams_distribute_reduction_bitxor.c----------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the reduction clause on a target teams distribute directive,
// testing that the variable in the reduction clause is properly reduced using
// the bitxor operator.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int test_bitxor() {
  unsigned int a[N];
  int errors = 0;
  int num_teams[N];
  srand(1);

  for (int x = 0; x < N; ++x) {
    a[x] = (unsigned int) rand() / (double) (RAND_MAX / 2);
    num_teams[x] = -x;
  }

  unsigned int b = 0;

#pragma omp target teams distribute reduction(^:b) defaultmap(tofrom:scalar)
  for (int x = 0; x < N; ++x) {
    num_teams[x] = omp_get_num_teams();
    b = (b ^ a[x]);
  }

  unsigned int host_b = 0;

  for (int x = 0; x < N; ++x) {
    host_b = (host_b ^ a[x]);
  }

  for (int x = 1; x < N; ++x) {
    OMPVV_WARNING_IF(num_teams[x - 1] != num_teams[x], "Kernel reported differing numbers of teams.  Validity of testing of reduction clause cannot be guaranteed.");
  }
  OMPVV_WARNING_IF(num_teams[0] == 1, "Test operated with one team.  Reduction clause cannot be tested.");
  OMPVV_WARNING_IF(num_teams[0] <= 0, "Test reported invalid number of teams.  Validity of testing of reduction clause cannot be guaranteed.");

  OMPVV_TEST_AND_SET_VERBOSE(errors, b != host_b);
  OMPVV_ERROR_IF(host_b != b, "Bit on device is %d but expected bit from host is %d.", b, host_b);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_bitxor() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
