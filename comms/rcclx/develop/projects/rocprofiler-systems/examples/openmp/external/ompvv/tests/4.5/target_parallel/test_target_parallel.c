//===---- test_target_parallel.c - combined consutrct target and parallel  -------------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// This test checks for the combined construct target and parallel. It allows to create a
// parallel region inside of the target device. 
//
//===----------------------------------------------------------------------------------===//
//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ompvv.h"

int test_target_parallel() {
  OMPVV_INFOMSG("test_target_parallel");

  int num_threads[OMPVV_NUM_THREADS_DEVICE];
  int errors = 0;

#pragma omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(from:num_threads[0:OMPVV_NUM_THREADS_DEVICE])
  {
    int thread_id = omp_get_thread_num();
    num_threads[thread_id] = omp_get_num_threads();
  }

  //Warning if only 1 thread in region 2, parallel does nothing
  OMPVV_WARNING_IF(num_threads[0] == 1, "The number of threads in the parallel region was 1. This is not a specifications error but we could not confirm the parallel region.");

  //Error if num_threads is inconsistent between threads
  for (int i = 1; i < num_threads[0]; ++i) {
    OMPVV_TEST_AND_SET(errors, num_threads[i] != num_threads[0]);
    OMPVV_ERROR_IF(num_threads[i] != num_threads[0], "The number of threads recorded by thread %d was %d. Expected was %d.", i, num_threads[i], num_threads[0]);
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_parallel());

  OMPVV_REPORT_AND_RETURN(errors);
}
