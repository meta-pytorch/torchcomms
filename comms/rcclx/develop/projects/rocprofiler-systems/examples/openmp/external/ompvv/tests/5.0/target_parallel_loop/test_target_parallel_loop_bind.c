//===---------------test_target_parallel_loop_bind.c-----------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Description - This is a test program to demonstrate how bind(parallel)
// clause is used with target parallel loop construct.
//
////===-------------------------------------------------------------------===//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define DIM_1 10
#define DIM_2 10
#define DIM_3 12

int test_target_parallel_loop_bind() {
  OMPVV_INFOMSG("test_target_parallel_loop_bind(parallel)");
  int arr[DIM_1][DIM_2][DIM_3];
  int errors = 0;
  int i, j, k;
  int temp = 0;
  int num_threads = -1;

  // arr initialization
  for (i = 0; i < DIM_1; i++) {
    for (j = 0; j < DIM_2; j++) {
      for (k = 0; k < DIM_3; k++) {
          arr[i][j][k] = 1;
      }
    }
  }

#pragma omp target parallel loop bind(parallel) map(tofrom: arr) \
                       num_threads(OMPVV_NUM_THREADS_DEVICE) private(j,k)
  for (i = 0; i < DIM_1; i++) {
    for (j = 0; j < DIM_2; j++) {
      for (k = i; k < DIM_3; k++) {
          arr[i][j][k] += i+2*j+3*k;
        }
      }
    }
  
  // validation
  for (i = 0; i < DIM_1; i++) {
    for (j = 0; j < DIM_2; j++) {
      for (k = i; k < DIM_3; k++) {
          temp = 1 + i + 2*j + 3*k;
          if (arr[i][j][k] != temp)
            errors++;
      }
    }
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_parallel_loop_bind());
  OMPVV_REPORT_AND_RETURN(errors);
}
