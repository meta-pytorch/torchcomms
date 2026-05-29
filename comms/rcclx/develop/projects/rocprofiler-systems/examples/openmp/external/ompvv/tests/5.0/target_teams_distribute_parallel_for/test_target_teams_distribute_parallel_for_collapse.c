//===---- test_target_teams_distribute_parallel_for_collapse.c - combined consutrct -===//
// 
// OpenMP API Version 5.0 Nov 2018
// 
// Testing loop collapse with multiple loops. Trying to mimic lsms kernel.
//
//===--------------------------------------------------------------------------------===//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define SIZE_N 10
#define SIZE_M 12

int test_target_teams_distribute_parallel_for_collapse() {
  OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_collapse");
  
  int a[SIZE_N][SIZE_M][SIZE_N][SIZE_M];
  int errors = 0;
  int i,j,k,l;

  // a array initialization
  for (i = 0; i < SIZE_N; i++) {
    for (j = 0; j < SIZE_M; j++) {
      for (k = 0; k < SIZE_N; k++) {
        for (l = 0; l < SIZE_M; l++) {
          a[i][j][k][l] = 1;
        }
      }
    }
  }

#pragma omp target teams distribute parallel for collapse(4) map(tofrom: a) private(i,j,k,l)
  for (i = 0; i < SIZE_N; i++) {
    for (j = 0; j < SIZE_M; j++) {
      for (k = i; k < SIZE_N; k++) {
        for (l = 0; l < SIZE_M; l++) {
          a[i][j][k][l] += i+2*j+3*k+4*l;
        }
      }
    }
  }

  for (i = 0; i < SIZE_N; i++) {
    for (j = 0; j < SIZE_M; j++) {
      for (k = i; k < SIZE_N; k++) {
        for (l = 0; l < SIZE_M; l++) {
          int temp = 1+ i + 2*j + 3*k + 4*l;
          OMPVV_TEST_AND_SET(errors, a[i][j][k][l] != temp);
        }
      }
    }
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_collapse());

  OMPVV_REPORT_AND_RETURN(errors);
}
