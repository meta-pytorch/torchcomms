//===------ test_target_teams_distribute_dist_schedule.c ------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks that the dist_schedule clause (which must have kind
// static) correctly causes CHUNK_SIZE iterations to be split among the
// number of teams the test is run with (in a round-robin fashion in order
// of the team number) when a chunk size is given. The test also confirms
// that when no chunk size is given, that each team receives no more than
// one "chunk" of implementation-defined size.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
#define CHUNK_SIZE 64

int test_dist_schedule() {
  int errors = 0;
  int num_teams;
  int a[N];
  int b[N];

  for (int i = 0; i < N; ++i) {
    a[i] = -1;
    b[i] = -1;
  }

#pragma omp target teams distribute map(from: num_teams) map(tofrom: a[0:N]) dist_schedule(static, CHUNK_SIZE)
  for (int i = 0; i < N; ++i) {
    if (omp_get_team_num() == 0) {
      num_teams = omp_get_num_teams();
    }
    a[i] = omp_get_team_num();
  }

  OMPVV_WARNING_IF(num_teams == 1, "Cannot test dist_schedule(static, chunk_size) because num_teams was 1.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams < 1);

  int counter = -1;
  for (int i = 0; i < N; ++i) {
    if (i % CHUNK_SIZE == 0) {
      counter = (counter + 1) % num_teams;
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != counter);
    OMPVV_ERROR_IF(a[i] != counter, "Loop iterations were not properly scheduled with specified chunk_size of %d.", CHUNK_SIZE);
  }

  num_teams = -1;

#pragma omp target teams distribute map(from: num_teams) map(tofrom: b[0:N]) dist_schedule(static)
  for (int i = 0; i < N; ++i) {
    if (omp_get_team_num() == 0) {
      num_teams = omp_get_num_teams();
    }
    b[i] = omp_get_team_num();
  }

  OMPVV_WARNING_IF(num_teams == 1, "Cannot test dist_schedule(static, chunk_size) because num_teams was 1.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams < 1);

  counter = 1;

  int err_cond = 0;
  for (int i = 1; i < N; ++i) {
    err_cond = a[i] < a[i - 1] || a[i] > (a[i - 1] + 1);
    OMPVV_TEST_AND_SET_VERBOSE(errors, err_cond);
    OMPVV_ERROR_IF(err_cond, "Loop iterations were not properly sheduled with unspecified chunk_size.");
    if (err_cond) {
      break;
    }
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_SHARED_ENVIRONMENT;

  int errors = 0;

  errors = test_dist_schedule();

  OMPVV_REPORT_AND_RETURN(errors);
}
