//===--- test_target_teams_distribute_shared.c-------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the shared clause on a target teams distribute directive and
// tests in a few ways that the variable is shared between the teams.  In the
// first test, the atomic directive is used to indicate that all operations on
// the variable should be done atomically.  If the value is the correct value
// at the end of the region, then all teams operated on the same variable, and
// the variable was not privatized.
//
// The second test, instead of writing to the variable, only reads from the
// variable.  This tests that the value of the shared variable has not been
// initialized improperly.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE 1024

int main() {
  int is_offloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
  int a[SIZE];
  int share = 0;
  int errors = 0;
  int num_teams;

  for (int x = 0; x < SIZE; ++x) {
    a[x] = x;
  }

  // The defaultmap(tofrom:scalar) is used here because the OpenMP 4.5 specification
  // forbids the use of map and data-sharing clauses on the same list item in the
  // same construct. See pg. 218, lines 15-16.
#pragma omp target teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) shared(share, num_teams) map(to: a[0:SIZE]) defaultmap(tofrom:scalar)
  for (int x = 0; x < SIZE; ++x) {
#pragma omp atomic write
    num_teams = omp_get_num_teams();
#pragma omp atomic
    share = share + a[x];
  }

  for (int x = 0; x < SIZE; ++x) {
    share = share - x;
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, (share != 0));
  OMPVV_ERROR_IF(errors != 0, "The value of share is = %d but expected 0.", share);

  share = 5;

#pragma omp target data map(tofrom: a[0:SIZE]) map(tofrom: share)
  {
#pragma omp target teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) shared(share)
    for (int x = 0; x < SIZE; ++x) {
      a[x] = a[x] + share;
    }
  }

  for (int x = 0; x < SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, (a[x] - 5 != x));
  }

  if (num_teams == 1) {
    OMPVV_WARNING("Test operated on one team, results of default shared test are inconclusive.");
  }

  OMPVV_REPORT_AND_RETURN(errors);
}
