//===--- test_atomic_fail_relaxed.c -----------------------------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// Utilizes an compare w/ an atomic release to ensure the implicit
// setting of x=10. Restrictions on atomic fail state they must be used
// on an atomic compare.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_atomic_fail_relaxed() {
  OMPVV_INFOMSG("test_atomic_fail_relaxed");

  int x = 0, y = 0;
  int errors = 0;

#pragma omp parallel num_threads(2)
   {
      int thrd = omp_get_thread_num();
       if (thrd == 0) {
          y = 1;
          #pragma omp atomic write seq_cst
          x = 10;
       } else {
          while (y != 5) {
            #pragma omp atomic compare fail(relaxed)
            if(y == 1){
               y = 5;
            }
          }
          OMPVV_TEST_AND_SET(errors, x != 10);
          OMPVV_TEST_AND_SET(errors, y != 5);
       }
   }
   return errors;
}


int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_fail_relaxed());

  OMPVV_REPORT_AND_RETURN(errors);
}
