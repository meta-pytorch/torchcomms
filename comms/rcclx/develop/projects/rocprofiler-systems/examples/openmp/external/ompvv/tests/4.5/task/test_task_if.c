//===-- test_task_if.c ------------------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// Description
// testTaskWithIf():
// Create an undeferred task followed by a deferred task
// in a parallel code section. Ensure that the undeferred
// task is always executed before deferred task.
//===----------------------------------------------------------------------===//



#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include "ompvv.h"

int testTaskWithIf(int num_threads, int val, int inc) {
  int errors = 0;
  int *A = (int*) (malloc(num_threads*sizeof(int)));
  omp_set_num_threads(num_threads);
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    A[id] = val;
#pragma omp task shared(A) if(0)
    {
#pragma omp critical
      {
        A[id] += inc;
      }
    }
#pragma omp task shared(A) if(1)
    {
#pragma omp critical
      {
        A[id] *= inc;
      }
    }
  }
  int ret = 0;
  for (int i = 0; i < num_threads; i++) {
    if (A[i] != ((val + inc)*inc)) {
      ret = 1;
      break;
    }
  }
  free(A);
  OMPVV_TEST_AND_SET_VERBOSE(errors, ret != 0);
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithIf(4, 5, 2));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithIf(8, 5, 2));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithIf(16, 5, 2));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithIf(32, 5, 2));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithIf(64, 5, 2));
  OMPVV_REPORT_AND_RETURN(errors);
}
