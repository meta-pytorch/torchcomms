//===--- test_loop_nested.c -------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the loop directive without any clauses with nested loops.
// The construct is applied in a teams and a parallel construct and the
// correctness of array operations in the nested loops in the loop construct
// is checked. Compare with test_loop_bind.c, which performs the same
// operations in a nested loop with a bind clause.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 32

int test_loop_nested_teams() {
  OMPVV_INFOMSG("test_loop_nested_teams");
  int errors = 0;
  int x[N][N];
  int y[N];
  int z[N];
  int num_teams = -1;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      x[i][j] = 1;
    }
    y[i] = i;
    z[i] = 2*i;
  }

#pragma omp teams num_teams(OMPVV_NUM_TEAMS_DEVICE) thread_limit(OMPVV_NUM_THREADS_HOST)
  {
#pragma omp loop
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        x[i][j] += y[i]*z[i];
      }
    }
    #pragma omp parallel if(0)
    if (omp_get_team_num() == 0) {
      num_teams = omp_get_num_teams();
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, x[i][j] != 1 + (y[i]*z[i]));
    }
  }


  OMPVV_WARNING_IF(num_teams == 1, "Test ran with one team, so parallelism of loop construct can't be guaranteed.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams < 1);
  OMPVV_ERROR_IF(num_teams < 1, "omp_get_num_teams() returned an invalid number of teams.");

  return errors;
}

int test_loop_nested_parallel() {
  OMPVV_INFOMSG("test_loop_nested_parallel");
  int errors = 0;
  int x[N][N];
  int y[N];
  int z[N];
  int num_threads = -1;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      x[i][j] = 1;
    }
    y[i] = i;
    z[i] = 2*i;
  }

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
  {
#pragma omp loop
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        x[i][j] += y[i]*z[i];
      }
    }
    if (omp_get_thread_num() == 0 && omp_get_team_num() == 0) {
      num_threads = omp_get_num_threads();
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, x[i][j] != 1 + (y[i]*z[i]));
    }
  }

  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of loop construct can't be guaranteed.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);
  OMPVV_ERROR_IF(num_threads < 1, "omp_get_num_threads() returned an invalid number of threads.");

  return errors;
}

int main() {
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_loop_nested_teams());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_loop_nested_parallel());

  OMPVV_REPORT_AND_RETURN(errors);
}
