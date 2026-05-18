//===-------- test_target_and_task_nowait.c - test task after target offload  ------------===//
// 
// OpenMP API Version 4.5 Nov 2015
//
// This test checks if dependence expressed on target and task 
// regions are honoured in the presense of nowait.
// This test is motivated by OpenMP usage in QMCPack.
//
////===----------------------------------------------------------------------===//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define N 1000

int test_target_and_task_nowait() {
  int errors = 0, i;
  int sum = 0, a = 0;

#pragma omp target map(tofrom: a, sum) depend(out: a) nowait
  {
    for (i = 0; i < N; i++) {
       sum++;
    }
    a += 1;
  }

#pragma omp task depend(in: a) shared(a,errors)
  {
     if(a != 1) {
       errors += 1; 
     }
  }
  #pragma omp taskwait
  if (sum != N) {
    errors++;
  }
  return(errors);
}

int main() {

  OMPVV_TEST_OFFLOADING;

  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_and_task_nowait());

  OMPVV_REPORT_AND_RETURN(errors);
}

