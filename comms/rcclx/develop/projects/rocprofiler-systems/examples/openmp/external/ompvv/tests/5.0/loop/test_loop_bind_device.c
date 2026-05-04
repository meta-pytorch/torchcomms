//===--- test_loop_bind_device.c --------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the loop directive with the bind(binding) clause. The bind
// clause indicates that the loop construct should apply in the context of the
// given binding, one of teams, parallel, or thread. Each of these bindings
// is tested in an appropriate context and the correctness of results of
// array operations in the nested loop is checked. This test checks these
// directives in a target context.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 32

int test_loop_bind_teams() {
  OMPVV_INFOMSG("test_loop_bind_teams");
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

#pragma omp target teams num_teams(OMPVV_NUM_TEAMS_DEVICE) thread_limit(OMPVV_NUM_THREADS_HOST) map(tofrom: x, y, z, num_teams)
  {
#pragma omp loop bind(teams)
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


  OMPVV_WARNING_IF(num_teams == 1, "Test ran with one team, so parallelism of loop construct with bind(teams) can't be guaranteed.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams < 1);
  OMPVV_ERROR_IF(num_teams < 1, "omp_get_num_teams() returned an invalid number of teams.");

  return errors;
}

int test_loop_bind_parallel() {
  OMPVV_INFOMSG("test_loop_bind_parallel");
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

#pragma omp target parallel num_threads(OMPVV_NUM_THREADS_HOST) map(tofrom: x, y, z, num_threads)
  {
#pragma omp loop bind(parallel)
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

int test_loop_bind_thread_teams() {
  OMPVV_INFOMSG("test_loop_bind_thread_teams");
  int errors = 0;
  int y[N];
  int z[N];
  int result[N][N][OMPVV_NUM_TEAMS_DEVICE];
  int num_teams = -1;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < OMPVV_NUM_TEAMS_DEVICE; k++) {
        result[i][j][k] = 1;
      }
    }
    y[i] = i;
    z[i] = 2*i;
  }

#pragma omp target teams num_teams(OMPVV_NUM_TEAMS_DEVICE) thread_limit(OMPVV_NUM_THREADS_HOST) map(tofrom: result, y, z, num_teams)
  {
    int x[N][N];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        x[i][j] = 1;
      }
    }
#pragma omp loop bind(thread)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        x[i][j] += y[i]*z[i];
      }
    }
    #pragma omp parallel if(0)
    if (omp_get_team_num() == 0) {
      num_teams = omp_get_num_teams();
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        result[i][j][omp_get_team_num()] = x[i][j];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < num_teams; k++) {
        OMPVV_TEST_AND_SET_VERBOSE(errors, result[i][j][k] != (1 + (y[i]*z[i])));
      }
    }
  }

  OMPVV_WARNING_IF(num_teams == 1, "Test ran with one team, so parallelism of loop construct can't be guaranteed.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams < 1);
  OMPVV_ERROR_IF(num_teams < 1, "omp_get_num_teams() returned an invalid number of teams.");

  return errors;
}

int test_loop_bind_thread_parallel() {
  OMPVV_INFOMSG("test_loop_bind_thread_parallel");
  int errors = 0;
  int y[N];
  int z[N];
  int result[N][N][OMPVV_NUM_THREADS_HOST];
  int num_threads = -1;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < OMPVV_NUM_THREADS_HOST; k++) {
        result[i][j][k] = 1;
      }
    }
    y[i] = i;
    z[i] = 2*i;
  }

#pragma omp target parallel shared(result) num_threads(OMPVV_NUM_THREADS_HOST) map(tofrom: result, y, z, num_threads)
  {
    int x[N][N];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        x[i][j] = 1;
      }
    }
#pragma omp loop bind(thread)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        x[i][j] += y[i]*z[i];
      }
    }
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        result[i][j][omp_get_thread_num()] = x[i][j];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < num_threads; k++) {
        OMPVV_TEST_AND_SET_VERBOSE(errors, result[i][j][k] != (1 + (y[i]*z[i])));
      }
    }
  }

  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of loop construct can't be guaranteed.");
  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);
  OMPVV_ERROR_IF(num_threads < 1, "omp_get_num_threads() returned an invalid number of threads.");

  return errors;
}

int main() {
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_loop_bind_teams());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_loop_bind_parallel());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_loop_bind_thread_teams());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_loop_bind_thread_parallel());

  OMPVV_REPORT_AND_RETURN(errors);
}
