//===---test_requires_atomic_default_mem_order_acq_rel.c---------------------===//
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

#pragma omp requires atomic_default_mem_order(acq_rel)

int test_requires_atomic_acq_rel() {
  OMPVV_INFOMSG("test_requires_atomic_acq_rel");

  int x = 0, y = 0;
  int errors = 0;
  
  omp_set_dynamic(0);
  omp_set_num_threads(2);

#pragma omp parallel
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
   return errors;
}


int main() {

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_requires_atomic_acq_rel());

  OMPVV_REPORT_AND_RETURN(errors);
}


