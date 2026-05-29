//===--- test_target_teams_distribute.c--------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the target teams distribute directive and tests to validate
// that computation inside the region executes properly.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int main() {
  int a[N];
  int b[N];
  int num_teams[N];
  int errors = 0;
  int is_offloading;

  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
  OMPVV_TEST_SHARED_ENVIRONMENT

  // a and b array initialization
  for (int x = 0; x < N; ++x) {
    a[x] = 1;
    b[x] = x;
    num_teams[x] = -1;
  }

#pragma omp target teams distribute map(tofrom: a[0:N], num_teams[0:N]) map(to: b[0:N])
  for (int x = 0; x < N; ++x) {
    num_teams[x] = omp_get_num_teams();
    a[x] += b[x];
  }

  if (num_teams[0] == 1) {
    OMPVV_WARNING("Test operated with one team.  Parallelism of teams distribute can't be guaranteed.");
  } else if (num_teams[0] < 1) {
    OMPVV_ERROR("omp_get_num_teams() reported a value below one.");
  }

  for (int x = 1; x < N; ++x) {
    if (num_teams[x] != num_teams[x - 1]) {
      OMPVV_ERROR("Test reported an inconsistent number of teams between loop iterations.");
      errors++;
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, (a[x] != 1 + b[x]));
    if (a[x] != 1 + b[x]){
      errors++;
      break;
    }
  }

  OMPVV_INFOMSG_IF(!errors, "Test passed with %d teams.", num_teams[0]);

  OMPVV_REPORT_AND_RETURN(errors);
}
