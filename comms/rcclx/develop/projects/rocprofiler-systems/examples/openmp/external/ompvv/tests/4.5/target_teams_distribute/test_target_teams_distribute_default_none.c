//===--- test_target_teams_distribute_default_none.c-------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This tests uses the default(none) clause on a target teams distribute test.
// The test aims to validate that all values will not have default data sharing
// attributes.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int main() {
  int is_offloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
  OMPVV_WARNING("Test only uses default(none) clause and does not guarantee that the default(none) is enforced.");
  int a[N];
  int b[N];
  int c[N];
  int d[N];
  int privatized;
  int num_teams;
  int share = 0;
  int x;
  int errors = 0;

  for (int x = 0; x < N; ++x) {
    a[x] = 1;
    b[x] = x;
    c[x] = 2*x;
    d[x] = 0;
  }

#pragma omp target data map(from: d[0:N], num_teams) map(to: a[0:N], b[0:N], c[0:N])
  {
#pragma omp target teams distribute default(none) shared(a, b, c, d, num_teams) private(x, privatized) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    for (x = 0; x < N; ++x) {
      privatized = 0;
      for (int y = 0; y < a[x] + b[x]; ++y) {
        privatized++;
      }
      d[x] = c[x] * privatized;
      if (omp_get_team_num() == 0) {
        num_teams = omp_get_num_teams();
      }
    }
  }

  OMPVV_WARNING_IF(num_teams == 1, "The number of teams was 1. This is not a specification error but we could not guarantee parallelism of teams.");

  for (x = 0; x < N; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, (d[x] != (1 + x)*2*x));
    if (d[x] != (1 + x)*2*x) {
      break;
    }
  }

#pragma omp target data map(from: num_teams) map(to: b[0:N])
  {
#pragma omp target teams distribute default(none) private(x) shared(share, b, num_teams) defaultmap(tofrom:scalar) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    for (x = 0; x < N; ++x) {
#pragma omp atomic update
      share = share + b[x];
      if (omp_get_team_num() == 0) {
        num_teams = omp_get_num_teams();
      }
    }
  }

  OMPVV_WARNING_IF(num_teams == 1, "The number of teams was 1. This is not a specification error but we could not guarantee parallelism of teams.");

  for (int x = 0; x < N; ++x) {
    share = share - x;
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (share != 0));

  OMPVV_REPORT_AND_RETURN(errors);
}
