//===--- test_teams_distribute_default_none.c----------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This tests uses the default(none) clause on a teams directive.
// The test aims to validate that all values will not have default data sharing
// attributes.
//
////===------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 128

int main() {
  int a[N];
  int b[N];
  int c[N];
  int d[N];
  int x;
  int privatized;
  int num_teams;
  int share = 0;
  int errors = 0;

  OMPVV_WARNING("Test only uses default(none) clause and does not guarantee that the default(none) is enforced.");

  for (int x = 0; x < N; ++x) {
    a[x] = 1;
    b[x] = x;
    c[x] = 2*x;
    d[x] = 0;
  }

#pragma omp teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) default(none) shared(a, b, c, d, num_teams) private(privatized)
    for (int x = 0; x < N; ++x) {
      privatized = 0;
      for (int y = 0; y < a[x] + b[x]; ++y) {
        privatized++;
      }
      d[x] = c[x] * privatized;
      if (omp_get_team_num() == 0) {
        num_teams = omp_get_num_teams();
      }
    }

  OMPVV_WARNING_IF(num_teams == 1, "The number of teams was 1. This is not a specification error but we could not guarantee parallelism of teams.");

  for (x = 0; x < N; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, (d[x] != (1 + x)*2*x));
    if (d[x] != (1 + x)*2*x) {
      break;
    }
  }

#pragma omp teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) default(none) shared(share, b, num_teams)
    for (int x = 0; x < N; ++x) {
#pragma omp atomic update
      share = share + b[x];
      if (omp_get_team_num() == 0) {
        num_teams = omp_get_num_teams();
      }
    }

  OMPVV_WARNING_IF(num_teams == 1, "The number of teams was 1. This is not a specification error but we could not guarantee parallelism of teams.");

  for (int x = 0; x < N; ++x) {
    share = share - x;
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (share != 0));

  OMPVV_REPORT_AND_RETURN(errors);
}
