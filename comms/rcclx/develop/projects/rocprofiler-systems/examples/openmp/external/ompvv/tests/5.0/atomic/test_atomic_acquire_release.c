//===--- test_atomic_acquire_release.c - test write release and read aquire--===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Adapted from OpenMP examples acquire_release.2.c
// When the atomic read operation on thread 1 reads a non-zero value from y,
// this results in a release/acquire synchronization that in turn implies that 
// the assignment to x on thread 0 happens before the read of x on thread 1. 
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

int test_atomic_acquire_release() {
  OMPVV_INFOMSG("test_atomic_acquire_release");

  int x = 0, y = 0;
  int errors = 0;

#pragma omp parallel num_threads(2)
   {
      int thrd = omp_get_thread_num();
       if (thrd == 0) {
          x = 10;
          #pragma omp atomic write release // or seq_cst
          y = 1;
       } else {
          int tmp = 0;
          while (tmp == 0) {
            #pragma omp atomic read acquire // or seq_cst
            tmp = y;
          }
          OMPVV_TEST_AND_SET(errors, x != 10);
       }
   }
   return errors;
}


int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_acquire_release());

  OMPVV_REPORT_AND_RETURN(errors);
}
