//===---test_target_requires_atomic_default_mem_order_seq_cst.c--------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks for support of the atomic_default_mem_order clause on the 
// requires directive. This clause determines the default memory behavior for
// atomic constructs. These behaviors are seq_cst, acq_rel, and relaxed.
// This test checks the seq_cst behavior, which is also the default.
//
// Adapted from 5.0 OpenMP example acquire_release.2.c
////===----------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires atomic_default_mem_order(seq_cst)

int test_target_atomic_seq_cst() {
  OMPVV_INFOMSG("test_target_atomic_seq_cst");

  int x = 0, y = 0;
  int errors = 0;

#pragma omp target parallel num_threads(2) map(tofrom: x, y, errors)
   {
      int thrd = omp_get_thread_num();
       if (thrd == 0) {
          x = 10;
          #pragma omp atomic write 
          y = 1;
       } else {
          int tmp = 0;
          while (tmp == 0) {
            #pragma omp atomic read 
            tmp = y;
          }
          OMPVV_TEST_AND_SET(errors, x != 10);
       }
   }
   OMPVV_ERROR_IF(errors > 0, "Requires atomic_default_mem_order(seq_cst) test failed");
   return errors;
}

int main() {

  int errors = 0;

  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_atomic_seq_cst());

  OMPVV_REPORT_AND_RETURN(errors);
}
