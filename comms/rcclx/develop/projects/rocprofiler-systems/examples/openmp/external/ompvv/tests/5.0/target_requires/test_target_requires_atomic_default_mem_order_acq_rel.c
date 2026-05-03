//===---test_target_requires_atomic_default_mem_order_acq_rel.c--------------===//
//
// OpenMP API Version 5.0 Nov 2018
// 
// This test checks for support of the atomic_default_mem_order clause on the 
// requires directive. This clause determines the default memory behavior for
// atomic constructs. These behaviors are seq_cst, acq_rel, and relaxed.
// This test checks the for acq_rel as the memory-order-clause.
//
////===----------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires atomic_default_mem_order(acq_rel)

int test_target_requires_atomic_acq_rel() {
  OMPVV_INFOMSG("test_target_requires_atomic_acq_rel");

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
   OMPVV_ERROR_IF(errors > 0, "Requires atomic_default_mem_order(acq_rel) test failed");
   return errors;
}

int main() {

  int errors = 0;

  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_requires_atomic_acq_rel());

  OMPVV_REPORT_AND_RETURN(errors);
}
