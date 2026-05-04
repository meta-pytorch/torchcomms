//===--- test_loop_collapse.c ------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the collapse clause with the loop directive and tests that
// for loops out of the scope of the collapsed loops are not parallelized.
// This test tests using one and two collapsed loops.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 128 //Array Size of 128 uses 16MB target memory and
                       //scales n^3 in test_collapse2()

int test_collapse1() {

  int * a_mem = (int *)malloc(N*N*sizeof(int));
  int * b_mem = (int *)malloc(N*(N+1)*sizeof(int));
  int (*a)[N] = (int (*)[N])a_mem;
  int (*b)[N + 1] = (int (*)[N+1])b_mem;
  int errors = 0;

  // a and b array initialization
  for (int x = 0; x < N; ++x) {
    b[x][0] = 0;
    for (int y = 0; y < N; ++y) {
      a[x][y] = x + y;
      b[x][y+1] = 0;
    }
  }

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
  {
#pragma omp loop collapse(1)
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < N; ++y) {
        b[x][y + 1] = b[x][y] + a[x][y];
      }
    }
  }

  int temp_total;
  for (int x = 0; x < N; ++x) {
    temp_total = 0;
    for (int y = 0; y < N+1; ++y) {
      OMPVV_TEST_AND_SET(errors, ((temp_total - b[x][y]) != 0));
      if (y != N) {
        temp_total = temp_total + a[x][y];
      }
    }
  }
  free(a_mem);
  free(b_mem);
  return errors;
}

int test_collapse2() {
  int * a_mem = (int *)malloc(N*N*N*sizeof(int));
  int * b_mem = (int *)malloc(N*N*(N+1)*sizeof(int));
  int (*a)[N][N] = (int (*)[N][N])a_mem;
  int (*b)[N][N + 1] = (int (*)[N][N+1])b_mem;
  int errors = 0;
  int num_threads = 0;

  // a and b array initialization
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      b[x][y][0] = 0;
      for (int z = 0; z < N; ++z) {
        a[x][y][z] = x + y + z;
        b[x][y][z+1] = 0;
      }
    }
  }

#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
#pragma omp loop collapse(2)
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < N; ++y) {
        for (int z = 0; z < N; ++z) {
          b[x][y][z + 1] = b[x][y][z] + a[x][y][z];
        }
      }
    }
  }

  int temp_total;
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      temp_total = 0;
      for (int z = 0; z < N + 1; ++z) {
        OMPVV_TEST_AND_SET(errors, ((temp_total - b[x][y][z]) != 0));
        if (z != N) {
          temp_total = temp_total + a[x][y][z];
        }
      }
    }
  }

  if (num_threads == 1) {
    OMPVV_WARNING("Test operated with one thread.  Parallelism of loop directive in parallel region can't be guaranteed.");
  }

  free(a_mem);
  free(b_mem);
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_collapse1() != 0);
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_collapse2() != 0);

  OMPVV_REPORT_AND_RETURN(errors);

}
