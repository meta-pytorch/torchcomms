//===--- test_target_teams_distribute_private.c------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the private clause on a target teams distribute directive to
// indicate that the variable in the private clause should be made private to
// each team executing the teams distribute region.  The test then operates on
// the privatized variable in such a way that would most likely cause competing
// operations if the variable is not privatized.  If the computation completes
// without errors, we assume that the privatization occured.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE 1024

int main() {
  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  int a[SIZE];
  int b[SIZE];
  int c[SIZE];
  int d[SIZE];
  int privatized;
  int errors = 0;
  int num_teams = -1;

  for (int x = 0; x < SIZE; ++x) {
    a[x] = 1;
    b[x] = x;
    c[x] = 2*x;
    d[x] = 0;
  }

#pragma omp target data map(from: d[0:SIZE]) map(to: a[0:SIZE], b[0:SIZE], c[0:SIZE])
  {
#pragma omp target teams distribute private(privatized) map(alloc: a[0:SIZE], b[0:SIZE], c[0:SIZE], d[0:SIZE]) map(tofrom: num_teams) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    for (int x = 0; x < SIZE; ++x) {
      if (omp_get_team_num() == 0) {
        num_teams = omp_get_num_teams();
      }
      privatized = 0;
      for (int y = 0; y < a[x] + b[x]; ++y) {
        privatized++;
      }
      d[x] = c[x] * privatized;
    }
  }

  for (int x = 0; x < SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, d[x] != (1 + x)*2*x);
    if (d[x] != (1 + x)*2*x) {
      break;
    }
  }

  OMPVV_WARNING_IF(num_teams == 1, "Test ran with one team. Results of private test are inconclusive.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams < 1);

  OMPVV_REPORT_AND_RETURN(errors);
}
